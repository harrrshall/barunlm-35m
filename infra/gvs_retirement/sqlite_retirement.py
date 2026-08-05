"""Crash-durable compare-and-append storage for GVS population retirement.

This module is the storage boundary expected by
``gvs_authorization.claim_gvs_population_once``.  It accepts the callback's exact
canonical event bytes, independently checks their structural commitments and HMAC
through a caller-supplied custody boundary, and commits one whole-population claim
in a ``BEGIN IMMEDIATE`` SQLite transaction.  The callback returns only after a
``synchronous=FULL`` commit.

The database must live in a private directory outside the source repository.  No
secret is accepted or stored here: an external authenticator verifies HMAC payloads
without returning key material.  This module also does not provide the external
signer, prompt/label encryption, host isolation, backups, or rollback-resistant
replication required before private-label access can be authorized.

The authenticator is still a Python object in this process.  Sealing the ledger's
ordinary attributes prevents accidental callback replacement, not mutation by
hostile same-process code.  A separately isolated signer/authenticator and service
account remain mandatory; this storage primitive is deliberately nonauthorizing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

CLAIM_EVENT_VERSION = "barun-gvs-population-claim-event-v3"
LEDGER_SCHEMA_VERSION = "barun-gvs-retirement-sqlite-v1"
ZERO_SHA256 = "0" * 64

_EVENT_HASH_DOMAIN = b"barun-gvs-population-claim-event-v3/hash"
_EVENT_HMAC_DOMAIN = b"barun-gvs-population-claim-event-v3/hmac"
_MAX_EVENT_BYTES = 2 * 1024 * 1024
_MAX_LEDGER_BYTES = 2 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 100_000
_MAX_PURPOSE_BYTES = 4096
_MAX_EVENTS = 512
_BUSY_TIMEOUT_MILLISECONDS = 30_000
_APPLICATION_ID = 0x42525631  # ASCII "BRV1".
_USER_VERSION = 1
_SOURCE_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")
_PHASE_ORDER = ("d_support", "selection", "confirmation")
_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "sequence",
        "claim_id",
        "phase",
        "run_id",
        "population_manifest_sha256",
        "prompt_collection_sha256",
        "label_commitment_root_sha256",
        "authorization_receipt_sha256",
        "scoring_session_id",
        "accessor_id",
        "purpose",
        "claimed_at_utc",
        "previous_event_sha256",
        "key_id",
        "partial_retry_authorized",
        "authorizes_model_cuda_training_or_jarvis",
        "authorizes_private_label_access",
        "proves_external_atomicity",
        "authorization_runtime_sha256",
        "event_sha256",
        "event_hmac_sha256",
    }
)
_SHA256_FIELDS = (
    "population_manifest_sha256",
    "prompt_collection_sha256",
    "label_commitment_root_sha256",
    "authorization_receipt_sha256",
    "previous_event_sha256",
    "authorization_runtime_sha256",
    "event_sha256",
    "event_hmac_sha256",
)
_IDENTIFIER_FIELDS = (
    "claim_id",
    "run_id",
    "scoring_session_id",
    "accessor_id",
    "key_id",
)

_CREATE_STATE_SQL = """
CREATE TABLE ledger_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version TEXT NOT NULL,
    head_sha256 TEXT NOT NULL,
    next_sequence INTEGER NOT NULL CHECK (next_sequence >= 0),
    ledger_key_id TEXT,
    last_claimed_at_utc TEXT
)
""".strip()

_CREATE_EVENTS_SQL = """
CREATE TABLE claim_events (
    sequence INTEGER PRIMARY KEY CHECK (sequence >= 0),
    event_sha256 TEXT NOT NULL UNIQUE,
    claim_id TEXT NOT NULL UNIQUE,
    phase TEXT NOT NULL,
    run_id TEXT NOT NULL,
    population_manifest_sha256 TEXT NOT NULL UNIQUE,
    prompt_collection_sha256 TEXT NOT NULL UNIQUE,
    label_commitment_root_sha256 TEXT NOT NULL UNIQUE,
    previous_event_sha256 TEXT NOT NULL,
    claimed_at_utc TEXT NOT NULL UNIQUE,
    key_id TEXT NOT NULL,
    canonical_event BLOB NOT NULL UNIQUE,
    UNIQUE (run_id, phase)
)
""".strip()


class RetirementServiceError(RuntimeError):
    """A retirement event or durable ledger invariant failed closed."""


@runtime_checkable
class ClaimEventAuthenticator(Protocol):
    """External secret-custody boundary used to authenticate one event.

    Implementations receive the already domain-separated HMAC payload and must
    compare the presented digest without returning or logging secret material.
    Passing an implementation directly to this module is an interface contract,
    not proof of process isolation or secret custody.
    """

    def verify_event_hmac(
        self,
        *,
        key_id: str,
        authenticated_payload: bytes,
        presented_hmac_sha256: str,
    ) -> bool:
        """Return exact ``True`` only when the event HMAC is authentic."""


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    """Small detached view of the current durable ledger head."""

    schema_version: str
    head_sha256: str
    next_sequence: int
    ledger_key_id: str | None
    last_claimed_at_utc: str | None


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as error:
        raise RetirementServiceError("claim event is not strict canonical JSON") from error


def _strict_json_snapshot(
    value: object,
    *,
    depth: int = 0,
    nodes: list[int] | None = None,
) -> object:
    if depth > _MAX_JSON_DEPTH:
        raise RetirementServiceError("claim event exceeds the JSON depth bound")
    counter = [0] if nodes is None else nodes
    counter[0] += 1
    if counter[0] > _MAX_JSON_NODES:
        raise RetirementServiceError("claim event exceeds the JSON node bound")
    if value is None or type(value) in {bool, int, float, str}:
        if type(value) is float and not (-float("inf") < value < float("inf")):
            raise RetirementServiceError("claim event contains a non-finite number")
        if type(value) is str:
            try:
                value.encode("utf-8", errors="strict")
            except UnicodeEncodeError as error:
                raise RetirementServiceError("claim event contains invalid UTF-8 text") from error
        return value
    if type(value) is list:
        return [_strict_json_snapshot(item, depth=depth + 1, nodes=counter) for item in value]
    if type(value) is dict:
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str or key in result:
                raise RetirementServiceError("claim event contains an invalid JSON key")
            result[key] = _strict_json_snapshot(item, depth=depth + 1, nodes=counter)
        return result
    raise RetirementServiceError("claim event must contain exact built-in JSON values")


def _load_exact_event(event_bytes: object) -> dict[str, Any]:
    if type(event_bytes) is not bytes:
        raise RetirementServiceError("claim event must be exact immutable bytes")
    if not event_bytes or len(event_bytes) > _MAX_EVENT_BYTES:
        raise RetirementServiceError("claim event exceeds its exact byte bound")
    try:
        text = event_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RetirementServiceError("claim event is not strict UTF-8") from error

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RetirementServiceError("claim event contains a duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise RetirementServiceError(f"claim event contains forbidden constant {token!r}")

    try:
        loaded = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except RetirementServiceError:
        raise
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as error:
        raise RetirementServiceError("claim event is not strict JSON") from error
    snapshot = _strict_json_snapshot(loaded)
    if type(snapshot) is not dict:
        raise RetirementServiceError("claim event must be one exact JSON object")
    if set(snapshot) != _EVENT_FIELDS:
        raise RetirementServiceError("claim event fields differ from the frozen v3 contract")
    if _canonical_bytes(snapshot) != event_bytes:
        raise RetirementServiceError("claim event bytes are not the exact canonical JSON encoding")
    return snapshot


def _strict_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise RetirementServiceError(f"{label} must be one lowercase SHA-256")
    return value


def _strict_identifier(value: object, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise RetirementServiceError(f"{label} must be one bounded portable identifier")
    if unicodedata.normalize("NFC", value) != value:
        raise RetirementServiceError(f"{label} must be NFC-normalized")
    return value


def _strict_timestamp(value: object, *, label: str) -> datetime:
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        raise RetirementServiceError(f"{label} must be one canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise RetirementServiceError(f"{label} is not a real timestamp") from error
    canonical = parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if parsed.microsecond == 0:
        canonical = canonical.replace(".000000Z", "Z")
    else:
        canonical = canonical[:-1].rstrip("0") + "Z"
    if value != canonical:
        raise RetirementServiceError(f"{label} is not canonically encoded")
    return parsed


def _validated_event(event_bytes: bytes) -> tuple[dict[str, Any], bytes]:
    event = _load_exact_event(event_bytes)
    if event["schema_version"] != CLAIM_EVENT_VERSION:
        raise RetirementServiceError("unsupported population-claim event version")
    if type(event["sequence"]) is not int or not 0 <= event["sequence"] < _MAX_EVENTS:
        raise RetirementServiceError("claim event sequence is outside the durable ledger bound")
    for field in _IDENTIFIER_FIELDS:
        _strict_identifier(event[field], label=field)
    if event["phase"] not in _PHASE_ORDER:
        raise RetirementServiceError("claim event phase is unsupported")
    purpose = event["purpose"]
    if type(purpose) is not str or not purpose or purpose != purpose.strip():
        raise RetirementServiceError("claim event purpose must be bounded nonempty text")
    try:
        purpose_bytes = purpose.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise RetirementServiceError("claim event purpose is not valid UTF-8") from error
    if len(purpose_bytes) > _MAX_PURPOSE_BYTES or unicodedata.normalize("NFC", purpose) != purpose:
        raise RetirementServiceError("claim event purpose must be bounded NFC text")
    if any(
        character != " "
        and (character.isspace() or unicodedata.category(character).startswith("C"))
        for character in purpose
    ):
        raise RetirementServiceError(
            "claim event purpose contains forbidden whitespace or controls"
        )
    _strict_timestamp(event["claimed_at_utc"], label="claimed_at_utc")
    for field in _SHA256_FIELDS:
        _strict_sha256(event[field], label=field)
    for field in (
        "partial_retry_authorized",
        "authorizes_model_cuda_training_or_jarvis",
        "authorizes_private_label_access",
        "proves_external_atomicity",
    ):
        if event[field] is not False:
            raise RetirementServiceError(f"claim event {field} must remain false")

    hash_payload = dict(event)
    hash_payload["event_sha256"] = ZERO_SHA256
    hash_payload["event_hmac_sha256"] = ZERO_SHA256
    expected_event_hash = hashlib.sha256(
        _EVENT_HASH_DOMAIN + b"\x00" + _canonical_bytes(hash_payload)
    ).hexdigest()
    if event["event_sha256"] != expected_event_hash:
        raise RetirementServiceError("claim event self-hash mismatch")
    mac_payload = dict(event)
    mac_payload["event_hmac_sha256"] = ZERO_SHA256
    authenticated_payload = _EVENT_HMAC_DOMAIN + b"\x00" + _canonical_bytes(mac_payload)
    return event, authenticated_payload


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def _effective_user_id() -> int:
    get_effective_uid = getattr(os, "geteuid", None)
    if get_effective_uid is None:
        raise RetirementServiceError(
            "retirement storage requires a POSIX effective-user ownership check"
        )
    effective_uid = get_effective_uid()
    if type(effective_uid) is not int or effective_uid < 0:
        raise RetirementServiceError("effective-user ownership could not be established")
    return effective_uid


def _validate_private_parent(parent: Path) -> None:
    try:
        parent_stat = parent.lstat()
    except OSError as error:
        raise RetirementServiceError("retirement state directory could not be inspected") from error
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise RetirementServiceError("retirement database parent must be one directory")
    if parent_stat.st_uid != _effective_user_id():
        raise RetirementServiceError(
            "retirement state directory must be owned by the effective user"
        )
    parent_mode = stat.S_IMODE(parent_stat.st_mode)
    if parent_mode != 0o700:
        raise RetirementServiceError(
            "retirement state directory must have exact mode 0700 and must not permit "
            "group/world access"
        )


def _validate_database_stat(database_stat: os.stat_result) -> tuple[int, int]:
    if not stat.S_ISREG(database_stat.st_mode):
        raise RetirementServiceError("retirement database must be one regular file")
    if database_stat.st_uid != _effective_user_id():
        raise RetirementServiceError("retirement database must be owned by the effective user")
    if stat.S_IMODE(database_stat.st_mode) != 0o600:
        raise RetirementServiceError("retirement database must have exact mode 0600")
    if database_stat.st_nlink != 1:
        raise RetirementServiceError("retirement database must have exactly one hard link")
    return database_stat.st_dev, database_stat.st_ino


def _database_file_identity(database_path: Path) -> tuple[int, int]:
    try:
        database_stat = database_path.lstat()
    except OSError as error:
        raise RetirementServiceError("retirement database could not be inspected") from error
    if stat.S_ISLNK(database_stat.st_mode):
        raise RetirementServiceError("retirement database must not be a symlink")
    return _validate_database_stat(database_stat)


def _database_path(database_path: Path, *, repository_root: Path) -> Path:
    if not isinstance(database_path, Path) or not database_path.is_absolute():
        raise RetirementServiceError("retirement database path must be one absolute pathlib.Path")
    if not isinstance(repository_root, Path) or not repository_root.is_absolute():
        raise RetirementServiceError("repository_root must be one absolute pathlib.Path")
    if database_path.name in {"", ".", ".."}:
        raise RetirementServiceError("retirement database path must name one file")
    try:
        resolved_parent = database_path.parent.resolve(strict=True)
        resolved_repository = repository_root.resolve(strict=True)
        resolved_actual_repository = _SOURCE_REPOSITORY_ROOT.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise RetirementServiceError(
            "retirement database parent and repository must exist"
        ) from error
    if not resolved_parent.is_dir():
        raise RetirementServiceError("retirement database parent must be one directory")
    if database_path.parent != resolved_parent:
        raise RetirementServiceError("retirement database parent must not traverse a symlink")
    resolved_path = resolved_parent / database_path.name
    if _is_within(resolved_path, resolved_actual_repository):
        raise RetirementServiceError(
            "retirement database must live outside the actual source repository"
        )
    if _is_within(resolved_path, resolved_repository):
        raise RetirementServiceError(
            "retirement database must live outside the declared source repository"
        )
    _validate_private_parent(resolved_parent)
    try:
        existing_stat = resolved_path.lstat()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise RetirementServiceError("retirement database could not be inspected") from error
    else:
        if stat.S_ISLNK(existing_stat.st_mode):
            raise RetirementServiceError("retirement database must not be a symlink")
    return resolved_path


class SQLiteRetirementLedger:
    """SQLite-backed global GVS whole-population retirement ledger.

    This object seals ordinary attribute assignment after initialization to avoid
    accidental authenticator replacement.  Python objects in one address space are
    not a security boundary: hostile same-process code can mutate collaborators or
    bypass language-level sealing.  External process/account isolation, signer
    custody, backups, and anti-rollback persistence remain required before use.
    """

    __slots__ = ("_authenticator", "_database_identity", "_database_path", "_sealed")

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("retirement ledger attributes are sealed after initialization")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        database_path: Path,
        authenticator: ClaimEventAuthenticator,
        repository_root: Path | None = None,
    ) -> None:
        self._sealed = False
        if not isinstance(authenticator, ClaimEventAuthenticator):
            raise RetirementServiceError("authenticator does not implement the custody protocol")
        default_repository = Path(__file__).resolve().parents[2]
        root = default_repository if repository_root is None else repository_root
        self._database_path = _database_path(database_path, repository_root=root)
        self._authenticator = authenticator
        self._database_identity = self._create_private_database_file()
        self._initialize_schema()
        self._sealed = True

    @property
    def database_path(self) -> Path:
        """Return the resolved external-state path (never event or key contents)."""

        return self._database_path

    def _create_private_database_file(self) -> tuple[int, int]:
        open_flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(
                self._database_path,
                os.O_CREAT | os.O_EXCL | open_flags,
                0o600,
            )
        except FileExistsError:
            expected_identity = _database_file_identity(self._database_path)
            try:
                descriptor = os.open(self._database_path, open_flags)
            except OSError as error:
                raise RetirementServiceError(
                    "existing retirement database could not be opened safely"
                ) from error
            try:
                descriptor_identity = _validate_database_stat(os.fstat(descriptor))
                observed_identity = _database_file_identity(self._database_path)
                if (
                    descriptor_identity != expected_identity
                    or observed_identity != expected_identity
                ):
                    raise RetirementServiceError(
                        "retirement database changed during safe existing-file open"
                    )
            finally:
                os.close(descriptor)
            return expected_identity
        except OSError as error:
            raise RetirementServiceError("retirement database could not be created") from error
        else:
            try:
                os.fchmod(descriptor, 0o600)
                identity = _validate_database_stat(os.fstat(descriptor))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                parent_flags = (
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                parent_descriptor = os.open(self._database_path.parent, parent_flags)
                try:
                    parent_stat = os.fstat(parent_descriptor)
                    if not stat.S_ISDIR(parent_stat.st_mode):
                        raise RetirementServiceError(
                            "retirement state directory changed during sync"
                        )
                    _validate_private_parent(self._database_path.parent)
                    os.fsync(parent_descriptor)
                finally:
                    os.close(parent_descriptor)
            except RetirementServiceError:
                raise
            except OSError as error:
                raise RetirementServiceError(
                    "retirement state directory could not be synced"
                ) from error
            observed_identity = _database_file_identity(self._database_path)
            if observed_identity != identity:
                raise RetirementServiceError("retirement database changed during creation")
            return identity

    def _connect(self) -> sqlite3.Connection:
        expected_identity = _database_file_identity(self._database_path)
        if expected_identity != self._database_identity:
            raise RetirementServiceError(
                "retirement database identity changed after initialization"
            )
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._database_path,
                timeout=_BUSY_TIMEOUT_MILLISECONDS / 1000,
                isolation_level=None,
            )
            observed_identity = _database_file_identity(self._database_path)
            if observed_identity != expected_identity:
                connection.close()
                raise RetirementServiceError("retirement database changed during SQLite open")
            connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MILLISECONDS}")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA fullfsync=ON")
            journal_mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()
            if journal_mode is None or str(journal_mode[0]).lower() != "delete":
                connection.close()
                raise RetirementServiceError("SQLite rollback-journal durability is unavailable")
            if connection.execute("PRAGMA synchronous").fetchone() != (2,):
                connection.close()
                raise RetirementServiceError("SQLite synchronous=FULL could not be established")
            if _database_file_identity(self._database_path) != expected_identity:
                connection.close()
                raise RetirementServiceError(
                    "retirement database changed while SQLite durability was configured"
                )
            return connection
        except RetirementServiceError:
            if connection is not None:
                try:
                    connection.close()
                except sqlite3.Error:
                    pass
            raise
        except sqlite3.Error as error:
            if connection is not None:
                try:
                    connection.close()
                except sqlite3.Error:
                    pass
            raise RetirementServiceError("retirement database connection failed") from error

    def _initialize_schema(self) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            application_id = connection.execute("PRAGMA application_id").fetchone()
            existing_objects = connection.execute(
                "SELECT type, name FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            ).fetchall()
            if application_id not in {(0,), (_APPLICATION_ID,)}:
                raise RetirementServiceError("database belongs to another application")
            if application_id == (0,) and existing_objects:
                raise RetirementServiceError(
                    "unidentified nonempty database is not a retirement ledger"
                )
            connection.execute(
                _CREATE_STATE_SQL.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1)
            )
            connection.execute(
                _CREATE_EVENTS_SQL.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1)
            )
            connection.execute(
                "INSERT OR IGNORE INTO ledger_state "
                "(singleton, schema_version, head_sha256, next_sequence, ledger_key_id, "
                "last_claimed_at_utc) VALUES (1, ?, ?, 0, NULL, NULL)",
                (LEDGER_SCHEMA_VERSION, ZERO_SHA256),
            )
            connection.execute(f"PRAGMA application_id={_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version={_USER_VERSION}")
            self._validate_schema(connection)
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        application_id = connection.execute("PRAGMA application_id").fetchone()
        user_version = connection.execute("PRAGMA user_version").fetchone()
        if application_id != (_APPLICATION_ID,) or user_version != (_USER_VERSION,):
            raise RetirementServiceError("retirement database version metadata is invalid")
        objects = connection.execute(
            "SELECT type, name FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        if objects != [("table", "claim_events"), ("table", "ledger_state")]:
            raise RetirementServiceError("retirement database contains unexpected schema objects")
        state_rows = connection.execute("SELECT COUNT(*) FROM ledger_state").fetchone()
        if state_rows != (1,):
            raise RetirementServiceError("retirement database has no unique ledger head")

    def inspect(self) -> LedgerSnapshot:
        """Read a detached head snapshot without exposing any event body."""

        connection = self._connect()
        try:
            connection.execute("BEGIN")
            self._validate_schema(connection)
            row = connection.execute(
                "SELECT schema_version, head_sha256, next_sequence, ledger_key_id, "
                "last_claimed_at_utc FROM ledger_state WHERE singleton = 1"
            ).fetchone()
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        if row is None:
            raise RetirementServiceError("retirement database has no ledger head")
        schema_version, head, sequence, key_id, timestamp = row
        if schema_version != LEDGER_SCHEMA_VERSION:
            raise RetirementServiceError("retirement ledger schema version changed")
        _strict_sha256(head, label="durable ledger head")
        if type(sequence) is not int or not 0 <= sequence <= _MAX_EVENTS:
            raise RetirementServiceError("durable ledger sequence is invalid")
        if key_id is not None:
            _strict_identifier(key_id, label="durable ledger key_id")
        if timestamp is not None:
            _strict_timestamp(timestamp, label="durable ledger last timestamp")
        return LedgerSnapshot(schema_version, head, sequence, key_id, timestamp)

    def _validated_ledger_rows(
        self,
        connection: sqlite3.Connection,
        *,
        state: tuple[str, int, str | None, str | None],
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            "SELECT sequence, event_sha256, claim_id, phase, run_id, "
            "population_manifest_sha256, prompt_collection_sha256, "
            "label_commitment_root_sha256, previous_event_sha256, claimed_at_utc, "
            "key_id, canonical_event FROM claim_events ORDER BY sequence"
        ).fetchall()
        durable_head, next_sequence, durable_key_id, last_timestamp = state
        if len(rows) != next_sequence:
            raise RetirementServiceError("retirement event count does not match the durable head")
        events: list[dict[str, Any]] = []
        previous_head = ZERO_SHA256
        previous_time: datetime | None = None
        seen_claims: set[str] = set()
        seen_populations: set[str] = set()
        seen_prompts: set[str] = set()
        seen_labels: set[str] = set()
        run_phases: dict[str, str] = {}
        for expected_sequence, row in enumerate(rows):
            raw = row[-1]
            if type(raw) is not bytes:
                raise RetirementServiceError("stored retirement event is not immutable bytes")
            event, authenticated_payload = _validated_event(raw)
            expected_metadata = (
                event["sequence"],
                event["event_sha256"],
                event["claim_id"],
                event["phase"],
                event["run_id"],
                event["population_manifest_sha256"],
                event["prompt_collection_sha256"],
                event["label_commitment_root_sha256"],
                event["previous_event_sha256"],
                event["claimed_at_utc"],
                event["key_id"],
            )
            if row[:-1] != expected_metadata:
                raise RetirementServiceError("stored retirement metadata differs from event bytes")
            if event["sequence"] != expected_sequence:
                raise RetirementServiceError("stored retirement sequence is not contiguous")
            if event["previous_event_sha256"] != previous_head:
                raise RetirementServiceError("stored retirement hash chain is broken")
            if durable_key_id is not None and event["key_id"] != durable_key_id:
                raise RetirementServiceError("stored retirement key_id changed")
            try:
                authenticated = self._authenticator.verify_event_hmac(
                    key_id=event["key_id"],
                    authenticated_payload=authenticated_payload,
                    presented_hmac_sha256=event["event_hmac_sha256"],
                )
            except Exception:  # noqa: BLE001 - custody implementations are an untrusted boundary.
                raise RetirementServiceError("stored event authentication failed closed") from None
            if authenticated is not True:
                raise RetirementServiceError("stored event authentication failed closed")
            for value, seen, label in (
                (event["claim_id"], seen_claims, "claim"),
                (event["population_manifest_sha256"], seen_populations, "population"),
                (event["prompt_collection_sha256"], seen_prompts, "prompt collection"),
                (event["label_commitment_root_sha256"], seen_labels, "label commitment"),
            ):
                if value in seen:
                    raise RetirementServiceError(f"stored {label} retirement is duplicated")
                seen.add(value)
            event_time = _strict_timestamp(event["claimed_at_utc"], label="claimed_at_utc")
            if previous_time is not None and event_time <= previous_time:
                raise RetirementServiceError("stored retirement timestamps are not increasing")
            prior_phase = run_phases.get(event["run_id"])
            if prior_phase is not None and _PHASE_ORDER.index(event["phase"]) <= _PHASE_ORDER.index(
                prior_phase
            ):
                raise RetirementServiceError("stored retirement phases do not advance")
            if event["phase"] == "confirmation" and prior_phase != "selection":
                raise RetirementServiceError("stored confirmation lacks selection retirement")
            run_phases[event["run_id"]] = event["phase"]
            events.append(event)
            previous_head = event["event_sha256"]
            previous_time = event_time
        if previous_head != durable_head:
            raise RetirementServiceError("stored events do not reproduce the durable ledger head")
        if rows and last_timestamp != events[-1]["claimed_at_utc"]:
            raise RetirementServiceError("stored events do not reproduce the durable timestamp")
        if not rows and state != (ZERO_SHA256, 0, None, None):
            raise RetirementServiceError("empty durable ledger has nonempty state")
        return events

    def compare_and_append(self, event_bytes: bytes, expected_previous_head: str) -> str:
        """Atomically append exact event bytes when the durable head still matches.

        This method has the exact two-argument shape required by
        ``claim_gvs_population_once``.  Any lost acknowledgement is fail-closed: a
        subsequent read will show the population retired, while replay is rejected.
        """

        expected_head = _strict_sha256(expected_previous_head, label="expected previous head")
        event, authenticated_payload = _validated_event(event_bytes)
        if event["previous_event_sha256"] != expected_head:
            raise RetirementServiceError("event and callback disagree on the previous head")
        try:
            authenticated = self._authenticator.verify_event_hmac(
                key_id=event["key_id"],
                authenticated_payload=authenticated_payload,
                presented_hmac_sha256=event["event_hmac_sha256"],
            )
        except Exception:  # noqa: BLE001 - custody implementations are an untrusted boundary.
            raise RetirementServiceError("external event authentication failed closed") from None
        if authenticated is not True:
            raise RetirementServiceError("external event authentication failed closed")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._validate_schema(connection)
            state = connection.execute(
                "SELECT schema_version, head_sha256, next_sequence, ledger_key_id, "
                "last_claimed_at_utc FROM ledger_state WHERE singleton = 1"
            ).fetchone()
            if state is None or state[0] != LEDGER_SCHEMA_VERSION:
                raise RetirementServiceError("durable ledger state is unavailable")
            _, durable_head, next_sequence, durable_key_id, last_timestamp = state
            self._validated_ledger_rows(
                connection,
                state=(durable_head, next_sequence, durable_key_id, last_timestamp),
            )
            if durable_head != expected_head:
                raise RetirementServiceError("durable ledger compare-and-append head mismatch")
            if event["previous_event_sha256"] != durable_head:
                raise RetirementServiceError("population claim hash chain is broken")
            if event["sequence"] != next_sequence:
                raise RetirementServiceError("population claim sequence is not contiguous")
            if next_sequence >= _MAX_EVENTS:
                raise RetirementServiceError("retirement ledger reached its frozen event bound")
            if durable_key_id is not None and event["key_id"] != durable_key_id:
                raise RetirementServiceError("population claim ledger key_id changed")
            event_time = _strict_timestamp(event["claimed_at_utc"], label="claimed_at_utc")
            if last_timestamp is not None and event_time <= _strict_timestamp(
                last_timestamp, label="durable ledger last timestamp"
            ):
                raise RetirementServiceError("population claim timestamps must strictly increase")
            self._validate_run_phase(connection, event)
            current_bytes = connection.execute(
                "SELECT COALESCE(SUM(length(canonical_event)), 0), COUNT(*) FROM claim_events"
            ).fetchone()
            if current_bytes is None:
                raise RetirementServiceError("retirement ledger size could not be determined")
            payload_bytes, event_count = current_bytes
            projected_bytes = payload_bytes + len(event_bytes) + event_count + 2
            if projected_bytes > _MAX_LEDGER_BYTES:
                raise RetirementServiceError("retirement ledger exceeds the canonical export bound")
            try:
                connection.execute(
                    "INSERT INTO claim_events "
                    "(sequence, event_sha256, claim_id, phase, run_id, "
                    "population_manifest_sha256, prompt_collection_sha256, "
                    "label_commitment_root_sha256, previous_event_sha256, claimed_at_utc, "
                    "key_id, canonical_event) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event["sequence"],
                        event["event_sha256"],
                        event["claim_id"],
                        event["phase"],
                        event["run_id"],
                        event["population_manifest_sha256"],
                        event["prompt_collection_sha256"],
                        event["label_commitment_root_sha256"],
                        event["previous_event_sha256"],
                        event["claimed_at_utc"],
                        event["key_id"],
                        sqlite3.Binary(event_bytes),
                    ),
                )
            except sqlite3.IntegrityError:
                raise RetirementServiceError(
                    "population, prompt, label commitment, claim, phase, or event is already retired"
                ) from None
            updated = connection.execute(
                "UPDATE ledger_state SET head_sha256 = ?, next_sequence = ?, "
                "ledger_key_id = COALESCE(ledger_key_id, ?), last_claimed_at_utc = ? "
                "WHERE singleton = 1 AND head_sha256 = ? AND next_sequence = ?",
                (
                    event["event_sha256"],
                    next_sequence + 1,
                    event["key_id"],
                    event["claimed_at_utc"],
                    expected_head,
                    next_sequence,
                ),
            )
            if updated.rowcount != 1:
                raise RetirementServiceError("durable compare-and-append lost its head race")
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return event["event_sha256"]

    @staticmethod
    def _validate_run_phase(connection: sqlite3.Connection, event: Mapping[str, Any]) -> None:
        prior = connection.execute(
            "SELECT phase FROM claim_events WHERE run_id = ? ORDER BY sequence",
            (event["run_id"],),
        ).fetchall()
        current_position = _PHASE_ORDER.index(event["phase"])
        if prior:
            prior_position = _PHASE_ORDER.index(prior[-1][0])
            if current_position <= prior_position:
                raise RetirementServiceError(
                    "population claim phases must strictly advance per run"
                )
        if event["phase"] == "confirmation" and (not prior or prior[-1][0] != "selection"):
            raise RetirementServiceError(
                "confirmation retirement requires an earlier same-run selection retirement"
            )

    def read_ledger(self) -> list[dict[str, Any]]:
        """Return the validated canonical event list for authorization refresh."""

        connection = self._connect()
        try:
            connection.execute("BEGIN")
            self._validate_schema(connection)
            state = connection.execute(
                "SELECT head_sha256, next_sequence, ledger_key_id, last_claimed_at_utc "
                "FROM ledger_state WHERE singleton = 1"
            ).fetchone()
            if state is None:
                raise RetirementServiceError("retirement database has no ledger head")
            events = self._validated_ledger_rows(connection, state=state)
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return events
