"""Dry-run validator for the closed redacted W&B evidence publication proposal.

Dry-run never imports W&B.  The historical ``--upload`` and ``--download-verify`` flags are
terminally disabled at every executor boundary because the v1 run failed its prelaunch audit.
Credentials are never accepted as arguments or written to receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

WANDB_VERSION = "0.28.1"
PROJECT = "barunaction-35m"
ARTIFACT_NAME = "barunaction-35m-candidate-v2-evidence"
ARTIFACT_TYPE = "evaluation"
ARTIFACT_VERSION = "v1"
SOURCE_ARTIFACT_VERSION = "v0"
SOURCE_WANDB_DIGEST = "b5e2cb35698e36bc1a97095c6c2d8565"
SOURCE_RELEASE_MANIFEST_SHA256 = "cbb29c4921855031bfeee1c1f5e9ed1a33a932b902a875c21f255fac793c2165"
SOURCE_RECEIPT_NAME = "redaction-receipt.json"
SOURCE_RECEIPT_SCHEMA = "barunaction-wandb-redacted-evidence-view-v1"
SOURCE_RECEIPT_SHA256 = "141203cff6491f9e23a3bbf62e4c6e0ae9c62bc0c4503e14e9b3702db5564a4e"
SOURCE_TREE_SHA256 = "c01de7b30c516bfb7c44625cf46ddd9bf2034ccd171f3af0f841383a94b73827"
SOURCE_DERIVED_FILE_COUNT = 296
SOURCE_UPLOAD_FILE_COUNT = 297
SOURCE_PUBLIC_TOTAL_BYTES = 135_926_068
PUBLICATION_RECEIPT_SCHEMA = "barunaction-wandb-redacted-evidence-publication-v1"
VERIFICATION_RECEIPT_SCHEMA = "barunaction-wandb-redacted-evidence-download-verification-v1"
PUBLICATION_RUN_CLOSED = True

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SLUG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_DIGEST_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+/=-]{0,255}")
_SENSITIVE_CLI_PATTERN = re.compile(
    r"(?:api[_-]?key|apikey|auth[_-]?token|access[_-]?token|password|secret|credential)",
    re.IGNORECASE,
)
_TOKEN_PATTERNS = (
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
_UNREDACTED_PATTERNS = (
    re.compile(rb"/Users/|/private/tmp/|(?<![A-Za-z0-9:/])/tmp/"),
    re.compile(rb"(?i)(?<![A-Za-z0-9])(?:kroda|cruda)(?![A-Za-z0-9])"),
    re.compile(
        rb"(?<![A-Za-z0-9])(?:463058|463697|463719|463904|463912|463936|463964|464346|464367)"
        rb"(?![A-Za-z0-9])"
    ),
)
_SCAN_OVERLAP = 512
_FORBIDDEN_DIRECTORY_NAMES = frozenset(
    {
        ".aws",
        ".cache",
        ".git",
        ".gnupg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".ssh",
        ".venv",
        "__pycache__",
        "credentials",
        "secrets",
        "venv",
        "wandb",
    }
)
_FORBIDDEN_FILE_NAMES = frozenset(
    {
        ".env",
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


class RedactedEvidencePublicationError(RuntimeError):
    """Publication or verification failed closed."""


def _fail(message: str) -> NoReturn:
    raise RedactedEvidencePublicationError(message)


@dataclass(frozen=True)
class _ValidatedView:
    root: Path
    receipt: dict[str, Any]
    receipt_sha256: str
    tree_sha256: str
    derived_files: dict[str, dict[str, Any]]
    upload_files: dict[str, dict[str, Any]]


def _reject_constant(value: str) -> NoReturn:
    _fail(f"JSON contains non-finite constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            _fail(f"JSON contains duplicate key {key!r}")
        value[key] = child
    return value


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except RedactedEvidencePublicationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RedactedEvidencePublicationError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        _fail(f"{label} must contain an object")
    return value


def _exact_fields(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        _fail(
            f"{label} fields differ: missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_value(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        _fail(f"{label} must be lowercase SHA-256")
    return value


def _metadata(value: Any, *, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        _fail(f"{label} has an invalid value")
    return value


def _strict_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(f"{label} must be a normalized POSIX relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        _fail(f"{label} must be a normalized POSIX relative path")
    _reject_sensitive_relative(relative, label=label)
    return value


def _reject_sensitive_relative(relative: PurePosixPath, *, label: str) -> None:
    for part in relative.parts:
        if part.casefold() in _FORBIDDEN_DIRECTORY_NAMES:
            _fail(f"{label} contains a forbidden runtime or credential directory")
    name = relative.name.casefold()
    if (
        name in _FORBIDDEN_FILE_NAMES
        or name.startswith(".env.")
        or name.endswith(_FORBIDDEN_FILE_SUFFIXES)
    ):
        _fail(f"{label} contains a forbidden credential-like filename")


def _contains_secret(payload: bytes) -> bool:
    return any(pattern.search(payload) is not None for pattern in _TOKEN_PATTERNS)


def _contains_unredacted_private_value(payload: bytes) -> bool:
    return any(pattern.search(payload) is not None for pattern in _UNREDACTED_PATTERNS)


def _read_hash_scan(path: Path, *, label: str) -> tuple[bytes, str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RedactedEvidencePublicationError(f"cannot safely open {label}") from error
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    overlap = b""
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                digest.update(chunk)
                size += len(chunk)
                window = overlap + chunk
                if _contains_secret(window):
                    _fail(f"{label} contains a credential marker")
                if _contains_unredacted_private_value(window):
                    _fail(f"{label} contains an unredacted private resource or local path")
                overlap = window[-_SCAN_OVERLAP:]
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or size != before.st_size:
            _fail(f"{label} changed while being read")
    finally:
        os.close(descriptor)
    return b"".join(chunks), digest.hexdigest(), size


def _check_no_symlink_components(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.exists() and current.is_symlink():
            _fail(f"{label} cannot contain a symlink component")


def _repository_root(value: Path) -> Path:
    _check_no_symlink_components(value, label="repository root")
    if value.is_symlink() or not value.is_dir():
        _fail("repository root must be a non-symlink directory")
    return value.resolve(strict=True)


def _external_source(repository_root: Path, value: Path) -> Path:
    absolute = Path(os.path.abspath(value))
    _check_no_symlink_components(absolute, label="redacted source")
    if absolute.is_symlink() or not absolute.is_dir():
        _fail("redacted source must be an existing non-symlink directory")
    resolved = absolute.resolve(strict=True)
    if (
        resolved == repository_root
        or repository_root in resolved.parents
        or resolved in repository_root.parents
    ):
        _fail("redacted source must be outside and non-overlapping with the repository")
    return resolved


def _external_fresh_path(
    repository_root: Path,
    source_root: Path,
    value: Path,
    *,
    label: str,
    other: Path | None = None,
) -> Path:
    absolute = Path(os.path.abspath(value))
    _check_no_symlink_components(absolute, label=label)
    resolved = absolute.resolve(strict=False)
    if absolute.exists() or absolute.is_symlink():
        _fail(f"{label} must be fresh and absent")
    if not absolute.parent.is_dir() or absolute.parent.is_symlink():
        _fail(f"{label} parent must be an existing non-symlink directory")
    if (
        resolved == repository_root
        or repository_root in resolved.parents
        or resolved in repository_root.parents
    ):
        _fail(f"{label} must be outside the repository")
    if (
        resolved == source_root
        or source_root in resolved.parents
        or resolved in source_root.parents
    ):
        _fail(f"{label} must not overlap the redacted source")
    if other is not None and (
        resolved == other or other in resolved.parents or resolved in other.parents
    ):
        _fail(f"{label} must not overlap another runtime destination")
    return resolved


def _new_receipt_path(repository_root: Path, value: Path) -> tuple[Path, str]:
    absolute = Path(os.path.abspath(value))
    _check_no_symlink_components(absolute, label="output receipt")
    if absolute == repository_root or repository_root not in absolute.parents:
        _fail("output receipt must be inside the repository")
    relative = _strict_relative(
        absolute.relative_to(repository_root).as_posix(), label="output receipt"
    )
    if absolute.exists() or absolute.is_symlink():
        _fail("refusing to overwrite output receipt")
    if not absolute.parent.is_dir() or absolute.parent.is_symlink():
        _fail("output receipt parent must be an existing non-symlink directory")
    return absolute, relative


def _existing_checked_in_path(
    repository_root: Path, value: Path, *, label: str
) -> tuple[Path, str]:
    absolute = Path(os.path.abspath(value))
    _check_no_symlink_components(absolute, label=label)
    if absolute == repository_root or repository_root not in absolute.parents:
        _fail(f"{label} must be inside the repository")
    relative = _strict_relative(absolute.relative_to(repository_root).as_posix(), label=label)
    if absolute.is_symlink() or not absolute.is_file():
        _fail(f"{label} must be an existing non-symlink regular file")
    return absolute, relative


def _tree_sha256(files: Mapping[str, Mapping[str, Any]]) -> str:
    payload = "".join(
        f"{record['sha256']}  {record['bytes']}  {relative}\n"
        for relative, record in sorted(files.items())
    ).encode("utf-8")
    return _sha256(payload)


def _receipt_file_map(value: Any, *, expected_count: int) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or len(value) != expected_count:
        _fail("redaction receipt has an unexpected derived file count")
    files: dict[str, dict[str, Any]] = {}
    for raw_relative, raw_record in sorted(value.items()):
        relative = _strict_relative(raw_relative, label="redaction receipt file")
        if relative == SOURCE_RECEIPT_NAME:
            _fail("redaction receipt cannot list itself as a derived file")
        record = _exact_fields(
            raw_record,
            {
                "changed",
                "output_bytes",
                "output_sha256",
                "redactions",
                "source_bytes",
                "source_sha256",
            },
            label=f"redaction receipt file {relative!r}",
        )
        if not isinstance(record["changed"], bool) or not isinstance(record["redactions"], list):
            _fail(f"redaction receipt file {relative!r} has invalid redaction metadata")
        for field in ("output_bytes", "source_bytes"):
            if type(record[field]) is not int or record[field] < 0:
                _fail(f"redaction receipt file {relative!r} has an invalid byte count")
        output_hash = _sha256_value(
            record["output_sha256"], label=f"redaction receipt file {relative!r} output hash"
        )
        _sha256_value(
            record["source_sha256"], label=f"redaction receipt file {relative!r} source hash"
        )
        files[relative] = {"bytes": record["output_bytes"], "sha256": output_hash}
    return files


def _validate_receipt(
    payload: bytes,
    *,
    receipt_sha256: str,
    expected_receipt_sha256: str,
    expected_tree_sha256: str,
    expected_derived_count: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if receipt_sha256 != expected_receipt_sha256:
        _fail("redaction receipt SHA-256 differs from the pinned local view")
    receipt = _exact_fields(
        _strict_json(payload, label="redaction receipt"),
        {
            "classification",
            "files",
            "output_tree",
            "policy",
            "redactions",
            "schema_version",
            "security",
            "semantic_preservation",
            "source",
        },
        label="redaction receipt",
    )
    if receipt["schema_version"] != SOURCE_RECEIPT_SCHEMA:
        _fail("unsupported redaction receipt schema")
    classification = receipt["classification"]
    if (
        not isinstance(classification, Mapping)
        or classification.get("authoritative_evidence") is not False
        or classification.get("distribution_view") != "redacted"
        or classification.get("source_public_artifact_mutated") is not False
    ):
        _fail("source is not explicitly a nonauthoritative redacted distribution view")
    security = receipt["security"]
    if not isinstance(security, Mapping) or security != {
        "credential_markers_found": 0,
        "network_operations_performed": False,
        "source_manifest_hash_verified": True,
        "source_symlinks_found": 0,
        "source_unlisted_files_found": 0,
        "uploads_performed": False,
    }:
        _fail("redaction receipt security assertions differ from the frozen view")
    source = receipt["source"]
    if not isinstance(source, Mapping):
        _fail("redaction receipt source identity is invalid")
    artifact = source.get("artifact")
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("name") != ARTIFACT_NAME
        or artifact.get("type") != ARTIFACT_TYPE
        or artifact.get("version") != SOURCE_ARTIFACT_VERSION
        or artifact.get("wandb_digest") != SOURCE_WANDB_DIGEST
        or artifact.get("public_file_count") != expected_derived_count
        or artifact.get("public_total_bytes") != SOURCE_PUBLIC_TOTAL_BYTES
    ):
        _fail("redaction receipt source artifact identity changed")
    release_manifest = source.get("release_manifest")
    if (
        not isinstance(release_manifest, Mapping)
        or release_manifest.get("sha256") != SOURCE_RELEASE_MANIFEST_SHA256
    ):
        _fail("redaction receipt source release-manifest identity changed")
    output_tree = receipt["output_tree"]
    if (
        not isinstance(output_tree, Mapping)
        or output_tree.get("derived_file_count") != expected_derived_count
        or output_tree.get("receipt_excluded_from_self_hash") != SOURCE_RECEIPT_NAME
        or output_tree.get("sha256") != expected_tree_sha256
    ):
        _fail("redaction receipt output-tree commitment changed")
    files = _receipt_file_map(receipt["files"], expected_count=expected_derived_count)
    if _tree_sha256(files) != expected_tree_sha256:
        _fail("redaction receipt derived tree does not recompute to the pinned hash")
    return dict(receipt), files


def _expected_directories(files: Mapping[str, Any]) -> set[str]:
    directories: set[str] = set()
    for relative in files:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _inspect_exact_tree(
    root: Path, *, expected: Mapping[str, Mapping[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        _fail(f"{label} must be a non-symlink directory")
    allowed_directories = _expected_directories(expected)
    observed: dict[str, Path] = {}

    def walk(directory: Path, relative_parent: PurePosixPath) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise RedactedEvidencePublicationError(f"cannot inspect {label}") from error
        for entry in entries:
            relative = relative_parent / entry.name
            relative_text = relative.as_posix()
            _reject_sensitive_relative(relative, label=label)
            if entry.is_symlink():
                _fail(f"{label} contains a symlink")
            if entry.is_dir(follow_symlinks=False):
                if relative_text not in allowed_directories:
                    _fail(f"{label} contains an unlisted directory")
                walk(Path(entry.path), relative)
            elif entry.is_file(follow_symlinks=False):
                observed[relative_text] = Path(entry.path)
            else:
                _fail(f"{label} contains a non-regular entry")

    walk(root, PurePosixPath())
    if set(observed) != set(expected):
        _fail(
            f"{label} file set differs: missing={sorted(set(expected) - set(observed))!r}, "
            f"extra={sorted(set(observed) - set(expected))!r}"
        )
    verified: dict[str, dict[str, Any]] = {}
    for relative, path in sorted(observed.items()):
        _, digest, size = _read_hash_scan(path, label=f"{label} file {relative!r}")
        record = expected[relative]
        if digest != record["sha256"] or size != record["bytes"]:
            _fail(f"{label} file {relative!r} differs from its receipt")
        verified[relative] = {"bytes": size, "sha256": digest}
    return verified


def _validate_view(
    repository_root: Path,
    source_root: Path,
    *,
    expected_receipt_sha256: str,
    expected_tree_sha256: str,
    expected_derived_count: int,
    expected_upload_count: int,
) -> _ValidatedView:
    root = _external_source(repository_root, source_root)
    receipt_path = root / SOURCE_RECEIPT_NAME
    payload, receipt_hash, receipt_bytes = _read_hash_scan(receipt_path, label="redaction receipt")
    receipt, derived_files = _validate_receipt(
        payload,
        receipt_sha256=receipt_hash,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_tree_sha256=expected_tree_sha256,
        expected_derived_count=expected_derived_count,
    )
    upload_files = dict(derived_files)
    upload_files[SOURCE_RECEIPT_NAME] = {
        "bytes": receipt_bytes,
        "sha256": receipt_hash,
    }
    if len(upload_files) != expected_upload_count:
        _fail("redacted upload file count differs from the pinned contract")
    _inspect_exact_tree(root, expected=upload_files, label="redacted source")
    return _ValidatedView(
        root, receipt, receipt_hash, expected_tree_sha256, derived_files, upload_files
    )


def _artifact_metadata(view: _ValidatedView) -> dict[str, Any]:
    return {
        "authoritative_evidence": False,
        "distribution_view": "redacted",
        "redacted_tree_sha256": view.tree_sha256,
        "redaction_receipt_sha256": view.receipt_sha256,
        "source_artifact_version": SOURCE_ARTIFACT_VERSION,
        "source_release_manifest_sha256": SOURCE_RELEASE_MANIFEST_SHA256,
        "source_wandb_digest": SOURCE_WANDB_DIGEST,
        "upload_file_count": len(view.upload_files),
    }


def _load_wandb(wandb_module: Any | None, *, operation: str) -> Any:
    if wandb_module is None:
        try:
            wandb_module = importlib.import_module("wandb")
        except ImportError as error:
            raise RedactedEvidencePublicationError(
                f"W&B {operation} requires requirements/wandb-release.txt"
            ) from error
    if getattr(wandb_module, "__version__", None) != WANDB_VERSION:
        _fail(f"W&B {operation} requires exactly wandb=={WANDB_VERSION}")
    return wandb_module


def _remote_identity(
    artifact: Any, *, entity: str, expected_digest: str | None = None
) -> dict[str, str]:
    qualified = f"{entity}/{PROJECT}/{ARTIFACT_NAME}:{ARTIFACT_VERSION}"
    if getattr(artifact, "name", "") != ARTIFACT_NAME:
        _fail("W&B returned an unexpected artifact name")
    if getattr(artifact, "type", "") != ARTIFACT_TYPE:
        _fail("W&B returned an unexpected artifact type")
    if getattr(artifact, "version", "") != ARTIFACT_VERSION:
        _fail("W&B artifact version must be exactly v1")
    if getattr(artifact, "qualified_name", "") != qualified:
        _fail("W&B returned an unexpected immutable qualified name")
    if str(getattr(artifact, "state", "")).upper() != "COMMITTED":
        _fail("W&B artifact is not in COMMITTED state")
    digest = _metadata(
        getattr(artifact, "digest", ""), label="W&B artifact digest", pattern=_DIGEST_PATTERN
    )
    if expected_digest is not None and digest != expected_digest:
        _fail("W&B artifact digest differs from the publication receipt")
    return {
        "digest": digest,
        "name": ARTIFACT_NAME,
        "qualified_name": qualified,
        "state": "COMMITTED",
        "type": ARTIFACT_TYPE,
        "version": ARTIFACT_VERSION,
    }


def _require_source_v0_latest(wandb_module: Any, *, entity: str) -> None:
    requested = f"{entity}/{PROJECT}/{ARTIFACT_NAME}:latest"
    expected_immutable = f"{entity}/{PROJECT}/{ARTIFACT_NAME}:{SOURCE_ARTIFACT_VERSION}"
    try:
        api = wandb_module.Api()
        artifact = api.artifact(requested, type=ARTIFACT_TYPE)
    except Exception as error:
        raise RedactedEvidencePublicationError(
            "W&B could not resolve the required source-v0 latest artifact"
        ) from error
    if artifact is None:
        _fail("W&B returned no source artifact for the latest precondition")
    if getattr(artifact, "name", "") != ARTIFACT_NAME:
        _fail("W&B latest precondition resolved an unexpected artifact name")
    if getattr(artifact, "type", "") != ARTIFACT_TYPE:
        _fail("W&B latest precondition resolved an unexpected artifact type")
    if str(getattr(artifact, "state", "")).upper() != "COMMITTED":
        _fail("W&B latest precondition is not in COMMITTED state")
    if getattr(artifact, "version", "") != SOURCE_ARTIFACT_VERSION:
        _fail("W&B latest must still be exactly v0 before publication")
    if getattr(artifact, "digest", "") != SOURCE_WANDB_DIGEST:
        _fail("W&B latest v0 digest differs from the immutable source artifact")
    if getattr(artifact, "qualified_name", "") != expected_immutable:
        _fail("W&B latest precondition did not resolve the immutable v0 qualified name")


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> str:
    serialized = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if _contains_secret(serialized):
        _fail("refusing to write a receipt containing a credential marker")
    try:
        with path.open("xb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise RedactedEvidencePublicationError("refusing to overwrite output receipt") from error
    return _sha256(serialized)


def _upload(
    *,
    wandb_module: Any,
    view: _ValidatedView,
    entity: str,
    repository_root: Path,
    receipt_path: Path,
    receipt_relative: str,
    wandb_dir: Path,
    validation_contract: Mapping[str, Any],
) -> dict[str, Any]:
    del (
        entity,
        receipt_path,
        receipt_relative,
        repository_root,
        validation_contract,
        view,
        wandb_dir,
        wandb_module,
    )
    _fail(
        "publication run 20260805-0250-candidate-v2-redacted-evidence-v1 is closed; "
        "the upload implementation was removed"
    )


def _load_publication_receipt(
    repository_root: Path,
    path: Path,
    *,
    entity: str,
    view: _ValidatedView,
) -> tuple[dict[str, str], str, str]:
    path, relative = _existing_checked_in_path(repository_root, path, label="publication receipt")
    payload, receipt_hash, _ = _read_hash_scan(path, label="publication receipt")
    receipt = _exact_fields(
        _strict_json(payload, label="publication receipt"),
        {
            "artifact",
            "local_view",
            "metadata",
            "package_version",
            "project",
            "run",
            "schema_version",
            "security",
        },
        label="publication receipt",
    )
    if receipt["schema_version"] != PUBLICATION_RECEIPT_SCHEMA:
        _fail("unsupported publication receipt schema")
    if receipt["package_version"] != WANDB_VERSION or receipt["project"] != PROJECT:
        _fail("publication receipt package or project changed")
    run = receipt["run"]
    if not isinstance(run, Mapping) or set(run) != {"entity", "id"} or run["entity"] != entity:
        _fail("publication receipt entity differs from the requested immutable artifact")
    _metadata(run["id"], label="publication receipt run ID", pattern=_SLUG_PATTERN)
    local_view = receipt["local_view"]
    if local_view != {
        "derived_file_count": len(view.derived_files),
        "redaction_receipt_sha256": view.receipt_sha256,
        "redacted_tree_sha256": view.tree_sha256,
        "upload_file_count": len(view.upload_files),
    }:
        _fail("publication receipt is not bound to this redacted local view")
    if receipt["metadata"] != _artifact_metadata(view):
        _fail("publication receipt artifact metadata changed")
    if receipt["security"] != {
        "credential_arguments_accepted": False,
        "download_performed": False,
        "source_mutated": False,
        "upload_performed": True,
    }:
        _fail("publication receipt security assertions changed")
    artifact = _exact_fields(
        receipt["artifact"],
        {"digest", "name", "qualified_name", "state", "type", "version"},
        label="publication receipt artifact",
    )
    identity = _remote_identity(type("ReceiptArtifact", (), dict(artifact))(), entity=entity)
    return identity, relative, receipt_hash


def _download_verify(
    *,
    wandb_module: Any,
    view: _ValidatedView,
    identity: dict[str, str],
    publication_receipt_relative: str,
    publication_receipt_sha256: str,
    entity: str,
    repository_root: Path,
    receipt_path: Path,
    receipt_relative: str,
    wandb_dir: Path,
    download_root: Path,
    validation_contract: Mapping[str, Any],
) -> dict[str, Any]:
    del (
        download_root,
        entity,
        identity,
        publication_receipt_relative,
        publication_receipt_sha256,
        receipt_path,
        receipt_relative,
        repository_root,
        validation_contract,
        view,
        wandb_dir,
        wandb_module,
    )
    _fail(
        "publication run 20260805-0250-candidate-v2-redacted-evidence-v1 is closed; "
        "no remote v1 verification is authorized"
    )


def _execute_with_contract(
    *,
    repository_root: Path,
    source_root: Path,
    receipt_path: Path,
    wandb_dir: Path,
    entity: str,
    upload: bool,
    download_verify: bool,
    publication_receipt_path: Path | None = None,
    download_destination: Path | None = None,
    wandb_module: Any | None = None,
    expected_receipt_sha256: str = SOURCE_RECEIPT_SHA256,
    expected_tree_sha256: str = SOURCE_TREE_SHA256,
    expected_derived_count: int = SOURCE_DERIVED_FILE_COUNT,
    expected_upload_count: int = SOURCE_UPLOAD_FILE_COUNT,
) -> dict[str, Any]:
    """Internal testable executor; the public wrapper always supplies production pins."""

    if upload or download_verify:
        _fail(
            "publication run 20260805-0250-candidate-v2-redacted-evidence-v1 is closed; "
            "remote operations are disabled"
        )
    entity = _metadata(entity, label="W&B entity", pattern=_SLUG_PATTERN)
    repository_root = _repository_root(repository_root)
    view = _validate_view(
        repository_root,
        source_root,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_tree_sha256=expected_tree_sha256,
        expected_derived_count=expected_derived_count,
        expected_upload_count=expected_upload_count,
    )
    receipt_path, receipt_relative = _new_receipt_path(repository_root, receipt_path)
    external_wandb_dir = _external_fresh_path(
        repository_root, view.root, wandb_dir, label="W&B directory"
    )
    validation_contract = {
        "expected_receipt_sha256": expected_receipt_sha256,
        "expected_tree_sha256": expected_tree_sha256,
        "expected_derived_count": expected_derived_count,
        "expected_upload_count": expected_upload_count,
    }
    qualified = f"{entity}/{PROJECT}/{ARTIFACT_NAME}:{ARTIFACT_VERSION}"

    del external_wandb_dir, validation_contract, wandb_module
    if publication_receipt_path is not None or download_destination is not None:
        _fail("remote-only inputs are invalid in dry-run mode")
    return {
        "proposed_artifact_qualified_name": qualified,
        "mode": "dry-run",
        "receipt": receipt_relative,
        "source_receipt_sha256": view.receipt_sha256,
        "upload_file_count": len(view.upload_files),
    }


def execute(
    *,
    repository_root: Path,
    source_root: Path,
    receipt_path: Path,
    wandb_dir: Path,
    entity: str,
    upload: bool,
    download_verify: bool,
    publication_receipt_path: Path | None = None,
    download_destination: Path | None = None,
) -> dict[str, Any]:
    """Validate the pinned local view; every remote operation is terminally disabled."""

    if upload or download_verify:
        _fail(
            "publication run 20260805-0250-candidate-v2-redacted-evidence-v1 is closed; "
            "no W&B remote operation is authorized"
        )

    return _execute_with_contract(
        repository_root=repository_root,
        source_root=source_root,
        receipt_path=receipt_path,
        wandb_dir=wandb_dir,
        entity=entity,
        upload=upload,
        download_verify=download_verify,
        publication_receipt_path=publication_receipt_path,
        download_destination=download_destination,
        expected_receipt_sha256=SOURCE_RECEIPT_SHA256,
        expected_tree_sha256=SOURCE_TREE_SHA256,
        expected_derived_count=SOURCE_DERIVED_FILE_COUNT,
        expected_upload_count=SOURCE_UPLOAD_FILE_COUNT,
    )


def _reject_sensitive_cli(arguments: Sequence[str]) -> None:
    for argument in arguments:
        if _SENSITIVE_CLI_PATTERN.search(argument) is not None or _contains_secret(
            argument.encode("utf-8", errors="ignore")
        ):
            _fail("refusing a credential-like CLI argument")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--wandb-dir", type=Path, required=True)
    parser.add_argument("--entity", required=True)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument(
        "--upload",
        action="store_true",
        help="closed audit path; any invocation fails before importing W&B",
    )
    operation.add_argument(
        "--download-verify",
        action="store_true",
        help="closed audit path; any invocation fails before importing W&B",
    )
    parser.add_argument("--publication-receipt", type=Path)
    parser.add_argument("--download-destination", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        _reject_sensitive_cli(arguments)
        args = build_parser().parse_args(arguments)
        result = execute(
            repository_root=args.repository_root,
            source_root=args.source_root,
            receipt_path=args.receipt,
            wandb_dir=args.wandb_dir,
            entity=args.entity,
            upload=args.upload,
            download_verify=args.download_verify,
            publication_receipt_path=args.publication_receipt,
            download_destination=args.download_destination,
        )
    except RedactedEvidencePublicationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
