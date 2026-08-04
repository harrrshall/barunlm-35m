"""Build a local, nonauthoritative redacted view of the immutable W&B v0 evidence.

The public W&B v0 artifact is immutable and remains the authoritative distribution.  This tool
does not import W&B, use a network API, or mutate the source tree.  It verifies the exact pinned
local release manifest and its evidence allowlist, then creates a fresh destination containing
byte-for-byte copies except for narrowly classified workstation paths and unrelated protected
JarvisLabs resource identities.  Credentials are rejected, never redacted.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

PINNED_V0_MANIFEST_SHA256 = "cbb29c4921855031bfeee1c1f5e9ed1a33a932b902a875c21f255fac793c2165"
PINNED_V0_PUBLIC_FILE_COUNT = 296
PINNED_V0_PUBLIC_TOTAL_BYTES = 135_926_068
PINNED_V0_WANDB_DIGEST = "b5e2cb35698e36bc1a97095c6c2d8565"
MANIFEST_SCHEMA_VERSION = "barunaction-wandb-release-manifest-v1"
RECEIPT_SCHEMA_VERSION = "barunaction-wandb-redacted-evidence-view-v1"
EXPECTED_PROJECT = "barunaction-35m"
EXPECTED_ARTIFACT_NAME = "barunaction-35m-candidate-v2-evidence"
EXPECTED_ARTIFACT_TYPE = "evaluation"
RECEIPT_NAME = "redaction-receipt.json"
RELEASE_MANIFEST_OUTPUT_NAME = "release-manifest.json"

_ALLOWED_TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".log", ".md", ".txt"})
_ALLOWED_EXTENSIONLESS = frozenset({"LICENSE", "NOTICE"})
_FORBIDDEN_DIRECTORY_NAMES = frozenset(
    {
        ".aws",
        ".cache",
        ".git",
        ".gnupg",
        ".ssh",
        "credentials",
        "secrets",
        "wandb",
    }
)
_FORBIDDEN_FILE_NAMES = frozenset(
    {
        ".netrc",
        "credentials",
        "credentials.json",
        "id_ed25519",
        "id_rsa",
        "netrc",
        "secrets",
        "secrets.json",
        "wandb_api_key",
    }
)
_FORBIDDEN_FILE_SUFFIXES = (".key", ".p12", ".pem", ".pfx")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

# These roots are workstation-local or ephemeral orchestration locations.  Remote training paths
# such as /home/barun-artifacts and /root/... are scientific provenance and are intentionally not
# changed.  The conservative alphabet makes an unfamiliar path fail instead of guessing its end.
_PATH_SEGMENT = rb"[A-Za-z0-9._@%+=~-]+"
_LOCAL_PATH_PATTERN = re.compile(
    rb"(?:(?<![A-Za-z0-9:/])|(?<=file://))(?:"
    rb"/Users/" + _PATH_SEGMENT + rb"(?:/" + _PATH_SEGMENT + rb")*|"
    rb"/private/tmp/" + _PATH_SEGMENT + rb"(?:/" + _PATH_SEGMENT + rb")*|"
    rb"/private/var/folders/" + _PATH_SEGMENT + rb"(?:/" + _PATH_SEGMENT + rb")*|"
    rb"/tmp/" + _PATH_SEGMENT + rb"(?:/" + _PATH_SEGMENT + rb")*|"
    rb"/var/folders/" + _PATH_SEGMENT + rb"(?:/" + _PATH_SEGMENT + rb")*"
    rb")"
)
_LOCAL_PREFIX_PATTERN = re.compile(rb"/(?:Users/|private/(?:tmp/|var/folders/)|tmp/|var/folders/)")

# Only unrelated resources from the durable denylist are redacted.  Project evidence instance
# IDs (for example the Qwen comparison instance) remain untouched because they are scientific
# provenance.  Numeric sentinels preserve JSON scalar types and cross-reference equality.
_PROTECTED_RESOURCE_IDS = (
    (b"463058", b"900000001", "unrelated_resource_01"),
    (b"463697", b"900000002", "unrelated_resource_02"),
    (b"463719", b"900000003", "unrelated_resource_03"),
    (b"463904", b"900000004", "unrelated_resource_04"),
    (b"463912", b"900000005", "unrelated_resource_05"),
    (b"463936", b"900000006", "unrelated_resource_06"),
    (b"463964", b"900000007", "unrelated_resource_07"),
    (b"464346", b"900000008", "unrelated_resource_08"),
    (b"464367", b"900000009", "unrelated_resource_09"),
)
_PROTECTED_RESOURCE_NAMES = (
    (
        re.compile(rb"(?i)(?<![A-Za-z0-9])(?:kroda|cruda)(?![A-Za-z0-9])"),
        b"PROTECTED_RESOURCE_01",
        "unrelated_resource_01",
    ),
    (
        re.compile(rb"(?i)(?<![A-Za-z0-9])kimi-k3-jl-node-a-20260803-1724(?![A-Za-z0-9])"),
        b"PROTECTED_RESOURCE_03",
        "unrelated_resource_03",
    ),
    (
        re.compile(rb"(?i)(?<![A-Za-z0-9])kimi-k3-jl-node-opt-20260804(?![A-Za-z0-9])"),
        b"PROTECTED_RESOURCE_04",
        "unrelated_resource_04",
    ),
    (
        re.compile(rb"(?i)(?<![A-Za-z0-9])kimi-k3-jl-node-opt128-20260804(?![A-Za-z0-9])"),
        b"PROTECTED_RESOURCE_07",
        "unrelated_resource_07",
    ),
)

_CREDENTIAL_PATTERNS = (
    re.compile(rb"wandb_v1_[A-Za-z0-9_-]{16,}", re.IGNORECASE),
    re.compile(rb"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{16,}", re.IGNORECASE),
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"(?:authorization\s*:\s*bearer|bearer\s+)[A-Za-z0-9._~+/-]{16,}", re.IGNORECASE),
    re.compile(rb"WANDB_API_KEY\s*[:=]", re.IGNORECASE),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(
        rb"https?://[^\x00-\x20\"'<>]+[?&](?:access_token|api_key|auth|authorization|key|"
        rb"signature|sig|token|x-amz-credential|x-amz-security-token|x-amz-signature)=",
        re.IGNORECASE,
    ),
)


class RedactedEvidenceError(RuntimeError):
    """The source or requested distribution operation failed closed."""


def _fail(message: str) -> NoReturn:
    raise RedactedEvidenceError(message)


@dataclass(frozen=True)
class _Match:
    kind: str
    start: int
    end: int
    replacement: bytes
    stable_identity: str


@dataclass(frozen=True)
class _SourceFile:
    relative: str
    payload: bytes
    sha256: str
    mode: int


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_constant(value: str) -> NoReturn:
    _fail(f"JSON contains non-finite constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            _fail(f"JSON contains duplicate key {key!r}")
        value[key] = child
    return value


def _load_strict_json(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except RedactedEvidenceError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RedactedEvidenceError(f"{label} is not strict UTF-8 JSON") from error


def _validate_jsonl(payload: bytes, *, label: str) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise RedactedEvidenceError(f"{label} is not UTF-8") from error
    lines = text.splitlines(keepends=True)
    if not lines or any(not line.strip() for line in lines):
        _fail(f"{label} must contain nonblank JSONL rows")
    if not text.endswith("\n"):
        _fail(f"{label} must end with a newline")
    for number, line in enumerate(lines, start=1):
        _load_strict_json(line.encode("utf-8"), label=f"{label} line {number}")


def _contains_credential(payload: bytes) -> bool:
    return any(pattern.search(payload) is not None for pattern in _CREDENTIAL_PATTERNS)


def _canonical_relative(value: Any, *, label: str, allow_directory: bool = False) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(f"{label} is not a canonical POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.parts != tuple(
        part for part in path.parts if part not in {".", ".."}
    ):
        _fail(f"{label} is not a canonical POSIX relative path")
    if str(path) != value:
        _fail(f"{label} is not a canonical POSIX relative path")
    for part in path.parts:
        lowered = part.lower()
        if lowered in _FORBIDDEN_DIRECTORY_NAMES or lowered in _FORBIDDEN_FILE_NAMES:
            _fail(f"{label} contains forbidden credential/runtime name {part!r}")
    if path.name.lower().endswith(_FORBIDDEN_FILE_SUFFIXES):
        _fail(f"{label} has a forbidden credential suffix")
    if (
        not allow_directory
        and path.name not in _ALLOWED_EXTENSIONLESS
        and path.suffix.lower() not in _ALLOWED_TEXT_SUFFIXES
    ):
        _fail(f"{label} is not an allowlisted text evidence file")
    return value


def _read_regular_nofollow(path: Path, *, label: str) -> tuple[bytes, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RedactedEvidenceError(f"cannot safely open {label}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after or len(payload) != before.st_size:
            _fail(f"{label} changed while being read")
        return payload, stat.S_IMODE(before.st_mode)
    finally:
        os.close(descriptor)


def _walk_exact_tree(root: Path) -> tuple[set[str], set[str]]:
    if root.is_symlink() or not root.is_dir():
        _fail("evidence source root must be a non-symlink directory")
    files: set[str] = set()
    directories: set[str] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise RedactedEvidenceError("cannot enumerate evidence source tree") from error
        for entry in entries:
            relative = Path(entry.path).relative_to(root).as_posix()
            if entry.is_symlink():
                _fail(f"evidence source contains symlink {relative!r}")
            if entry.is_dir(follow_symlinks=False):
                for part in PurePosixPath(relative).parts:
                    if part.lower() in _FORBIDDEN_DIRECTORY_NAMES:
                        _fail(f"evidence source contains forbidden directory {relative!r}")
                directories.add(relative)
                pending.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                files.add(_canonical_relative(relative, label="evidence source entry"))
            else:
                _fail(f"evidence source contains non-regular entry {relative!r}")
    return files, directories


def _load_manifest(manifest_path: Path, *, expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        _fail("release manifest must be a non-symlink regular file")
    payload, _ = _read_regular_nofollow(manifest_path, label="release manifest")
    actual_hash = _sha256(payload)
    if actual_hash != expected_sha256:
        _fail(f"release manifest SHA-256 differs from pinned v0 ({actual_hash})")
    if _contains_credential(payload):
        _fail("release manifest contains a credential marker")
    value = _load_strict_json(payload, label="release manifest")
    if not isinstance(value, dict):
        _fail("release manifest must contain an object")
    if value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        _fail("release manifest schema differs from the immutable v0 contract")
    if (
        value.get("project") != EXPECTED_PROJECT
        or value.get("local_manifest_authoritative") is not True
    ):
        _fail("release manifest identity or authority marker changed")
    return value, payload


def _validate_manifest_file_record(value: Any, *, relative: str) -> tuple[int, str]:
    if not isinstance(value, dict) or set(value) != {"bytes", "sha256"}:
        _fail(f"manifest record for {relative!r} has unexpected fields")
    size = value["bytes"]
    digest = value["sha256"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        _fail(f"manifest record for {relative!r} has invalid byte count")
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        _fail(f"manifest record for {relative!r} has invalid SHA-256")
    return size, digest


def _collect_sources(
    *, repository_root: Path, manifest: dict[str, Any]
) -> tuple[Path, dict[str, Any], list[_SourceFile]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or "evidence" not in artifacts:
        _fail("release manifest omits the evidence artifact")
    artifact = artifacts["evidence"]
    if not isinstance(artifact, dict) or set(artifact) != {
        "file_count",
        "files",
        "name",
        "source_root",
        "total_bytes",
        "type",
    }:
        _fail("evidence artifact manifest has unexpected fields")
    if artifact["name"] != EXPECTED_ARTIFACT_NAME or artifact["type"] != EXPECTED_ARTIFACT_TYPE:
        _fail("evidence artifact identity differs from public v0")
    source_relative = _canonical_relative(
        artifact["source_root"], label="evidence source_root", allow_directory=True
    )
    source_path = PurePosixPath(source_relative)
    if (
        source_path.name in _ALLOWED_EXTENSIONLESS
        or source_path.suffix.lower() in _ALLOWED_TEXT_SUFFIXES
    ):
        _fail("evidence source_root unexpectedly names a file")
    root_resolved = repository_root.resolve()
    unresolved_source_root = root_resolved / source_relative
    if unresolved_source_root.is_symlink():
        _fail("evidence source_root must not be a symlink")
    source_root = unresolved_source_root.resolve()
    try:
        source_root.relative_to(root_resolved)
    except ValueError:
        _fail("evidence source_root escapes the repository")
    if source_root.is_symlink() or not source_root.is_dir():
        _fail("evidence source_root is not a non-symlink directory")

    declared = artifact["files"]
    if not isinstance(declared, dict) or not declared:
        _fail("evidence artifact file allowlist is empty or invalid")
    declared_paths: set[str] = set()
    records: dict[str, tuple[int, str]] = {}
    for raw_relative, record in declared.items():
        relative = _canonical_relative(raw_relative, label="manifest evidence path")
        if relative == RECEIPT_NAME:
            _fail("source allowlist collides with the distribution receipt")
        if relative in declared_paths:
            _fail(f"duplicate evidence path {relative!r}")
        declared_paths.add(relative)
        records[relative] = _validate_manifest_file_record(record, relative=relative)

    actual_paths, _ = _walk_exact_tree(source_root)
    if actual_paths != declared_paths:
        missing = sorted(declared_paths - actual_paths)
        extra = sorted(actual_paths - declared_paths)
        _fail(f"evidence tree differs from allowlist; missing={missing!r}, extra={extra!r}")
    if artifact["file_count"] != len(declared_paths):
        _fail("evidence artifact file_count differs from the allowlist")

    sources: list[_SourceFile] = []
    total_bytes = 0
    for relative in sorted(declared_paths):
        payload, mode = _read_regular_nofollow(
            source_root / relative, label=f"evidence file {relative!r}"
        )
        expected_bytes, expected_hash = records[relative]
        actual_hash = _sha256(payload)
        if len(payload) != expected_bytes or actual_hash != expected_hash:
            _fail(
                f"evidence file drifted: {relative!r} expected={expected_hash} actual={actual_hash}"
            )
        if b"\x00" in payload:
            _fail(f"evidence file {relative!r} contains NUL bytes")
        try:
            payload.decode("utf-8")
        except UnicodeError as error:
            raise RedactedEvidenceError(f"evidence file {relative!r} is not UTF-8") from error
        if _contains_credential(payload):
            _fail(f"evidence file {relative!r} contains a credential marker")
        sources.append(_SourceFile(relative, payload, actual_hash, mode))
        total_bytes += len(payload)
    if artifact["total_bytes"] != total_bytes:
        _fail("evidence artifact total_bytes differs from the exact source tree")
    return source_root, artifact, sources


def _collect_public_v0_additions(
    *,
    repository_root: Path,
    evidence_root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    manifest_payload: bytes,
) -> list[_SourceFile]:
    """Collect the five files added to the staging tree by the frozen upload contract."""

    contract = manifest.get("int8_retention")
    expected_fields = {
        "disposition",
        "file_count",
        "files",
        "gate",
        "source_root",
        "total_bytes",
    }
    if not isinstance(contract, dict) or set(contract) != expected_fields:
        _fail("release manifest int8-retention contract has unexpected fields")
    if contract["disposition"] != "retained-release":
        _fail("public v0 does not retain the declared int8 evidence")
    source_relative = _canonical_relative(
        contract["source_root"], label="int8-retention source_root", allow_directory=True
    )
    unresolved_root = repository_root / source_relative
    if unresolved_root.is_symlink():
        _fail("int8-retention source_root must not be a symlink")
    source_root = unresolved_root.resolve()
    try:
        source_root.relative_to(repository_root)
    except ValueError:
        _fail("int8-retention source_root escapes the repository")
    if source_root.is_symlink() or not source_root.is_dir():
        _fail("int8-retention source_root is not a non-symlink directory")
    if (
        source_root == evidence_root
        or source_root in evidence_root.parents
        or evidence_root in source_root.parents
    ):
        _fail("int8-retention and evidence source roots must not overlap")

    declared = contract["files"]
    if not isinstance(declared, dict) or not declared:
        _fail("int8-retention file allowlist is empty or invalid")
    records: dict[str, tuple[int, str]] = {}
    for raw_relative, record in declared.items():
        relative = _canonical_relative(raw_relative, label="int8-retention manifest path")
        records[relative] = _validate_manifest_file_record(record, relative=relative)
    actual_paths, _ = _walk_exact_tree(source_root)
    if actual_paths != set(records):
        missing = sorted(set(records) - actual_paths)
        extra = sorted(actual_paths - set(records))
        _fail(f"int8-retention tree differs from allowlist; missing={missing!r}, extra={extra!r}")
    if contract["file_count"] != len(records):
        _fail("int8-retention file_count differs from its allowlist")

    additions: list[_SourceFile] = []
    total_bytes = 0
    for relative in sorted(records):
        payload, mode = _read_regular_nofollow(
            source_root / relative, label=f"int8-retention file {relative!r}"
        )
        expected_bytes, expected_hash = records[relative]
        actual_hash = _sha256(payload)
        if len(payload) != expected_bytes or actual_hash != expected_hash:
            _fail(
                f"int8-retention file drifted: {relative!r} "
                f"expected={expected_hash} actual={actual_hash}"
            )
        if b"\x00" in payload:
            _fail(f"int8-retention file {relative!r} contains NUL bytes")
        try:
            payload.decode("utf-8")
        except UnicodeError as error:
            raise RedactedEvidenceError(f"int8-retention file {relative!r} is not UTF-8") from error
        if _contains_credential(payload):
            _fail(f"int8-retention file {relative!r} contains a credential marker")
        output_relative = _canonical_relative(
            f"int8-retention/{relative}", label="public v0 int8-retention path"
        )
        additions.append(_SourceFile(output_relative, payload, actual_hash, mode))
        total_bytes += len(payload)
    if contract["total_bytes"] != total_bytes:
        _fail("int8-retention total_bytes differs from its exact source tree")

    manifest_mode = stat.S_IMODE(manifest_path.stat(follow_symlinks=False).st_mode)
    additions.append(
        _SourceFile(
            RELEASE_MANIFEST_OUTPUT_NAME,
            manifest_payload,
            _sha256(manifest_payload),
            manifest_mode,
        )
    )
    return additions


def _local_path_matches(payload: bytes, *, relative: str) -> list[re.Match[bytes]]:
    matches = list(_LOCAL_PATH_PATTERN.finditer(payload))
    starts = {match.start() for match in matches}
    for prefix in _LOCAL_PREFIX_PATTERN.finditer(payload):
        if prefix.start() not in starts:
            _fail(f"cannot safely delimit a local path in {relative!r} at byte {prefix.start()}")
    for match in matches:
        if match.end() < len(payload) and payload[match.end() : match.end() + 1] == b"/":
            _fail(f"cannot safely delimit a local path in {relative!r} at byte {match.start()}")
        if match.group()[-1:] in b".,;:":
            _fail(f"ambiguous local-path punctuation in {relative!r} at byte {match.start()}")
    return matches


def _resource_matches(payload: bytes) -> list[_Match]:
    matches: list[_Match] = []
    for source, replacement, identity in _PROTECTED_RESOURCE_IDS:
        pattern = re.compile(rb"(?<![A-Za-z0-9])" + re.escape(source) + rb"(?![A-Za-z0-9])")
        matches.extend(
            _Match("protected_resource_id", match.start(), match.end(), replacement, identity)
            for match in pattern.finditer(payload)
        )
    for pattern, replacement, identity in _PROTECTED_RESOURCE_NAMES:
        matches.extend(
            _Match("protected_resource_name", match.start(), match.end(), replacement, identity)
            for match in pattern.finditer(payload)
        )
    return matches


def _all_matches(source: _SourceFile, *, path_replacements: dict[bytes, bytes]) -> list[_Match]:
    matches = [
        _Match(
            "local_absolute_path",
            match.start(),
            match.end(),
            path_replacements[match.group()],
            path_replacements[match.group()].decode("ascii").strip("[]"),
        )
        for match in _local_path_matches(source.payload, relative=source.relative)
    ]
    matches.extend(_resource_matches(source.payload))
    matches.sort(key=lambda match: (match.start, match.end, match.kind))
    for previous, current in itertools.pairwise(matches):
        if previous.end > current.start:
            _fail(f"overlapping redactions in {source.relative!r} at byte {current.start}")
    return matches


def _reject_replacement_collisions(
    sources: list[_SourceFile], *, path_replacements: dict[bytes, bytes]
) -> None:
    # All path sentinels share one reserved prefix, so one scan detects every possible ordinal.
    # Fixed resource sentinels are few; avoid rescanning the 135 MB bundle once per local path.
    fixed_replacements = {replacement for _, replacement, _ in _PROTECTED_RESOURCE_IDS} | {
        replacement for _, replacement, _ in _PROTECTED_RESOURCE_NAMES
    }
    for source in sources:
        if (
            path_replacements
            and b"[REDACTED_LOCAL_PATH_" in source.payload
            or any(replacement in source.payload for replacement in fixed_replacements)
        ):
            _fail(f"source {source.relative!r} already contains a redaction sentinel")


def _apply_matches(payload: bytes, matches: list[_Match]) -> bytes:
    chunks: list[bytes] = []
    cursor = 0
    for match in matches:
        chunks.append(payload[cursor : match.start])
        chunks.append(match.replacement)
        cursor = match.end
    chunks.append(payload[cursor:])
    return b"".join(chunks)


def _validate_transformed_payload(*, relative: str, source: bytes, output: bytes) -> None:
    if _contains_credential(output):
        _fail(f"redacted output {relative!r} contains a credential marker")
    if _LOCAL_PREFIX_PATTERN.search(output) is not None:
        _fail(f"redacted output {relative!r} retains a local absolute path")
    if _resource_matches(output):
        _fail(f"redacted output {relative!r} retains a protected resource identity")
    suffix = PurePosixPath(relative).suffix.lower()
    if suffix == ".json":
        _load_strict_json(source, label=f"source {relative!r}")
        _load_strict_json(output, label=f"output {relative!r}")
    elif suffix == ".jsonl":
        _validate_jsonl(source, label=f"source {relative!r}")
        _validate_jsonl(output, label=f"output {relative!r}")


def _tree_sha256(records: dict[str, dict[str, Any]]) -> str:
    payload = "".join(
        f"{record['output_sha256']}  {record['output_bytes']}  {relative}\n"
        for relative, record in sorted(records.items())
    ).encode("utf-8")
    return _sha256(payload)


def _write_new_file(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as error:
        raise RedactedEvidenceError(f"refusing to overwrite output {path.name!r}") from error
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, mode)
    finally:
        os.close(descriptor)


def _build_redacted_view(
    *,
    repository_root: Path,
    manifest_path: Path,
    destination: Path,
    expected_manifest_sha256: str,
    expected_public_file_count: int | None = None,
    expected_public_total_bytes: int | None = None,
) -> dict[str, Any]:
    """Internal builder with an injectable manifest pin for isolated tests."""

    repository_root = repository_root.resolve()
    if manifest_path.is_symlink():
        _fail("release manifest must not be a symlink")
    manifest_path = manifest_path.resolve()
    try:
        manifest_relative = manifest_path.relative_to(repository_root).as_posix()
    except ValueError:
        _fail("release manifest must remain inside the repository")
    if destination.exists() or destination.is_symlink():
        _fail("destination already exists; refusing to overwrite it")
    parent = destination.parent.resolve()
    if not parent.is_dir() or parent.is_symlink():
        _fail("destination parent must be an existing non-symlink directory")
    destination = parent / destination.name

    manifest, manifest_payload = _load_manifest(
        manifest_path, expected_sha256=expected_manifest_sha256
    )
    source_root, artifact, sources = _collect_sources(
        repository_root=repository_root, manifest=manifest
    )
    sources.extend(
        _collect_public_v0_additions(
            repository_root=repository_root,
            evidence_root=source_root,
            manifest_path=manifest_path,
            manifest=manifest,
            manifest_payload=manifest_payload,
        )
    )
    if len({source.relative for source in sources}) != len(sources):
        _fail("public v0 distribution paths are not unique")
    source_total_bytes = sum(len(source.payload) for source in sources)
    if expected_public_file_count is not None and len(sources) != expected_public_file_count:
        _fail("reconstructed public v0 file count differs from its pinned receipt")
    if (
        expected_public_total_bytes is not None
        and source_total_bytes != expected_public_total_bytes
    ):
        _fail("reconstructed public v0 byte count differs from its pinned receipt")
    destination_resolved = destination.resolve(strict=False)
    if destination_resolved == source_root or source_root in destination_resolved.parents:
        _fail("destination must not overlap the immutable source tree")
    if destination_resolved in source_root.parents:
        _fail("destination cannot contain the immutable source tree")

    distinct_paths = sorted(
        {
            match.group()
            for source in sources
            for match in _local_path_matches(source.payload, relative=source.relative)
        }
    )
    path_replacements = {
        path: f"[REDACTED_LOCAL_PATH_{index:04d}]".encode("ascii")
        for index, path in enumerate(distinct_paths, start=1)
    }
    _reject_replacement_collisions(sources, path_replacements=path_replacements)

    prepared: list[tuple[_SourceFile, bytes, list[_Match]]] = []
    file_records: dict[str, dict[str, Any]] = {}
    summary = {
        "local_absolute_path": 0,
        "protected_resource_id": 0,
        "protected_resource_name": 0,
    }
    for source in sources:
        matches = _all_matches(source, path_replacements=path_replacements)
        output = _apply_matches(source.payload, matches)
        if bool(matches) != (output != source.payload):
            _fail(f"redaction accounting mismatch for {source.relative!r}")
        _validate_transformed_payload(
            relative=source.relative, source=source.payload, output=output
        )
        redactions = []
        for occurrence, match in enumerate(matches, start=1):
            summary[match.kind] += 1
            redactions.append(
                {
                    "kind": match.kind,
                    "occurrence": occurrence,
                    "source_byte_end": match.end,
                    "source_byte_start": match.start,
                    "stable_identity": match.stable_identity,
                    "replacement": match.replacement.decode("ascii"),
                }
            )
        file_records[source.relative] = {
            "changed": output != source.payload,
            "output_bytes": len(output),
            "output_sha256": _sha256(output),
            "redactions": redactions,
            "source_bytes": len(source.payload),
            "source_sha256": source.sha256,
        }
        prepared.append((source, output, matches))

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "classification": {
            "authoritative_evidence": False,
            "distribution_view": "redacted",
            "source_public_artifact_mutated": False,
            "warning": (
                "This is a privacy-redacted distribution view, not authoritative scientific "
                "evidence. Verify claims against the immutable public W&B v0 artifact and its "
                "pinned local release manifest."
            ),
        },
        "source": {
            "artifact": {
                "name": artifact["name"],
                "source_root": artifact["source_root"],
                "staging_file_count": artifact["file_count"],
                "staging_total_bytes": artifact["total_bytes"],
                "type": artifact["type"],
                "version": "v0",
                "public_file_count": len(sources),
                "public_total_bytes": source_total_bytes,
                "wandb_digest": PINNED_V0_WANDB_DIGEST,
            },
            "release_manifest": {
                "bytes": len(manifest_payload),
                "path": manifest_relative,
                "sha256": expected_manifest_sha256,
            },
        },
        "policy": {
            "credential_handling": "reject_entire_build; never redact credentials",
            "local_path_replacements": (
                "stable bundle-local ordinals; original values intentionally omitted"
            ),
            "nonmatched_bytes": "preserved exactly and in order",
            "protected_resource_replacements": (
                "stable nonprovider numeric/string sentinels preserving scalar type and equality"
            ),
            "remote_training_paths": "preserved as scientific provenance",
        },
        "security": {
            "credential_markers_found": 0,
            "network_operations_performed": False,
            "source_manifest_hash_verified": True,
            "source_symlinks_found": 0,
            "source_unlisted_files_found": 0,
            "uploads_performed": False,
        },
        "semantic_preservation": {
            "all_nonredacted_bytes_preserved": True,
            "changed_json_and_jsonl_remain_strictly_parseable": True,
            "scientific_metrics_predictions_and_config_values_not_matching_policy": "unchanged",
        },
        "redactions": {
            "changed_files": sum(record["changed"] for record in file_records.values()),
            "distinct_local_paths": len(distinct_paths),
            "occurrences_by_kind": summary,
            "total_occurrences": sum(summary.values()),
        },
        "files": file_records,
        "output_tree": {
            "derived_file_count": len(file_records),
            "receipt_excluded_from_self_hash": RECEIPT_NAME,
            "sha256": _tree_sha256(file_records),
        },
    }

    # All validation precedes this atomic directory reservation.  Existing destinations are never
    # merged with or replaced.  Exclusive file creation makes a concurrent collision fail closed.
    try:
        destination.mkdir(mode=0o755, parents=False, exist_ok=False)
    except OSError as error:
        raise RedactedEvidenceError("destination appeared before output creation") from error
    for source, output, _ in prepared:
        _write_new_file(destination / source.relative, output, mode=source.mode & 0o666)
    receipt_payload = (
        json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_new_file(destination / RECEIPT_NAME, receipt_payload)
    return receipt


def build_redacted_view(
    *, repository_root: Path, manifest_path: Path, destination: Path
) -> dict[str, Any]:
    """Build the exact pinned public-v0 evidence distribution view locally."""

    return _build_redacted_view(
        repository_root=repository_root,
        manifest_path=manifest_path,
        destination=destination,
        expected_manifest_sha256=PINNED_V0_MANIFEST_SHA256,
        expected_public_file_count=PINNED_V0_PUBLIC_FILE_COUNT,
        expected_public_total_bytes=PINNED_V0_PUBLIC_TOTAL_BYTES,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "experiments/runs/20260803-2303-candidate-v2-public-release-s17/release-manifest.json"
        ),
    )
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = repository_root / manifest_path
    try:
        receipt = build_redacted_view(
            repository_root=repository_root,
            manifest_path=manifest_path,
            destination=args.destination,
        )
    except RedactedEvidenceError as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "authoritative_evidence": False,
                "destination_created": True,
                "output_tree_sha256": receipt["output_tree"]["sha256"],
                "redactions": receipt["redactions"],
                "uploads_performed": False,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
