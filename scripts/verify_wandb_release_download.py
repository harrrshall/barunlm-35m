"""Independently re-download and verify the immutable BarunAction-35M W&B release.

The default mode validates the checked-in release manifest, upload receipt, destination, and
verification-receipt path without importing W&B or writing anything.  ``--download`` is the only
mode that imports the exactly pinned W&B client and performs remote reads.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

WANDB_VERSION = "0.28.1"
MANIFEST_SCHEMA_VERSION = "barunaction-wandb-release-manifest-v1"
UPLOAD_RECEIPT_SCHEMA_VERSION = "barunaction-wandb-release-receipt-v1"
VERIFICATION_RECEIPT_SCHEMA_VERSION = "barunaction-wandb-download-verification-v1"

ARTIFACT_CONTRACTS = {
    "float": {
        "name": "barunaction-35m-candidate-v2-float",
        "type": "model",
        "required_files": {
            "LICENSE",
            "MODEL_CARD.md",
            "NOTICE",
            "barun_config.json",
            "checkpoint_manifest.json",
            "model.safetensors",
            "tokenizer.json",
        },
    },
    "int8_darwin_arm64_qnnpack": {
        "name": "barunaction-35m-candidate-v2-int8-darwin-arm64-qnnpack",
        "type": "model",
        "required_files": {
            "LICENSE",
            "MODEL_CARD.md",
            "NOTICE",
            "barun_config.json",
            "model.int8.pt",
            "quantization_manifest.json",
            "tokenizer.json",
        },
    },
    "evidence": {
        "name": "barunaction-35m-candidate-v2-evidence",
        "type": "evaluation",
        "required_files": None,
    },
}
INT8_RETENTION_FILES = {
    "artifact-sha256.json",
    "paired-samples.jsonl",
    "protocol.json",
    "result.json",
}

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
VERSION_PATTERN = re.compile(r"v(?:0|[1-9][0-9]*)")
SLUG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
REMOTE_VALUE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+/=-]{0,511}")
SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:api[_-]?key|apikey|auth[_-]?token|access[_-]?token|password|secret|credential)",
    re.IGNORECASE,
)
TOKEN_MARKERS = (
    re.compile(rb"wandb_v1_[A-Za-z0-9_-]{16,}", re.IGNORECASE),
    re.compile(rb"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{16,}", re.IGNORECASE),
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"WANDB_API_KEY\s*[:=]", re.IGNORECASE),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)
TOKEN_SCAN_OVERLAP = 256
FORBIDDEN_DIRECTORY_NAMES = frozenset(
    {
        ".cache",
        ".git",
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
FORBIDDEN_EXACT_FILE_NAMES = frozenset(
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
FORBIDDEN_FILE_SUFFIXES = (".key", ".p12", ".pem", ".pfx")


class ReleaseVerificationError(RuntimeError):
    """A release record or downloaded artifact failed closed."""


def _fail(message: str) -> NoReturn:
    raise ReleaseVerificationError(message)


def _reject_constant(value: str) -> NoReturn:
    _fail(f"release JSON contains non-finite constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            _fail(f"release JSON contains duplicate key {key!r}")
        value[key] = child
    return value


def _contains_token_marker(payload: bytes) -> bool:
    return any(pattern.search(payload) is not None for pattern in TOKEN_MARKERS)


def _exact_fields(value: Any, expected: set[str], *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{path} must be an object")
    actual = set(value)
    if actual != expected:
        _fail(
            f"{path} fields differ: missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )
    return value


def _strict_relative(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(f"{path} must be a nonempty POSIX relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        _fail(f"{path} must be a normalized POSIX relative path")
    _reject_sensitive_relative(relative, path=path)
    return value


def _reject_sensitive_relative(relative: PurePosixPath, *, path: str) -> None:
    for part in relative.parts:
        folded = part.casefold()
        if folded in FORBIDDEN_DIRECTORY_NAMES:
            _fail(f"{path} contains a forbidden cache or credential directory")
    name = relative.name.casefold()
    if (
        name in FORBIDDEN_EXACT_FILE_NAMES
        or name.startswith(".env.")
        or name.endswith(FORBIDDEN_FILE_SUFFIXES)
    ):
        _fail(f"{path} contains a forbidden credential-like filename")


def _checked_in_path(repository_root: Path, value: Path, *, label: str) -> tuple[Path, str]:
    absolute = Path(os.path.abspath(value))
    if absolute == repository_root or repository_root not in absolute.parents:
        _fail(f"{label} must be checked in beneath the repository root")
    relative = _strict_relative(absolute.relative_to(repository_root).as_posix(), path=label)
    current = repository_root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            _fail(f"{label} cannot contain a symlink component")
    if not absolute.is_file():
        _fail(f"{label} must be a regular file")
    return absolute, relative


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes, str]:
    raw = path.read_bytes()
    if _contains_token_marker(raw):
        _fail(f"{label} contains a credential token marker")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ReleaseVerificationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseVerificationError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value, raw, hashlib.sha256(raw).hexdigest()


def _file_map(value: Any, *, path: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        _fail(f"{path} must be a nonempty object")
    normalized: dict[str, dict[str, Any]] = {}
    for name, raw_record in sorted(value.items()):
        relative = _strict_relative(name, path=f"{path} filename")
        record = _exact_fields(raw_record, {"bytes", "sha256"}, path=f"{path}.{relative}")
        size = record["bytes"]
        digest = record["sha256"]
        if type(size) is not int or size < 0:
            _fail(f"{path}.{relative}.bytes must be a nonnegative integer")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            _fail(f"{path}.{relative}.sha256 must be lowercase SHA-256")
        normalized[relative] = {"bytes": size, "sha256": digest}
    return normalized


def _validate_artifact_manifest(
    value: Any, *, kind: str, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    artifact = _exact_fields(
        value,
        {"file_count", "files", "name", "source_root", "total_bytes", "type"},
        path=f"release manifest artifacts.{kind}",
    )
    if artifact["name"] != contract["name"] or artifact["type"] != contract["type"]:
        _fail(f"release manifest artifacts.{kind} changed its fixed identity")
    _strict_relative(artifact["source_root"], path=f"release manifest artifacts.{kind}.source_root")
    files = _file_map(artifact["files"], path=f"release manifest artifacts.{kind}.files")
    required_files = contract["required_files"]
    if required_files is not None and set(files) != required_files:
        _fail(f"release manifest artifacts.{kind} changed its exact file set")
    if artifact["file_count"] != len(files) or artifact["total_bytes"] != sum(
        record["bytes"] for record in files.values()
    ):
        _fail(f"release manifest artifacts.{kind} file totals are inconsistent")
    return dict(artifact), files


def _validate_manifest(
    value: dict[str, Any], *, raw: bytes, sha256: str
) -> tuple[dict[str, dict[str, dict[str, Any]]], str]:
    manifest = _exact_fields(
        value,
        {
            "artifacts",
            "int8_retention",
            "local_manifest_authoritative",
            "project",
            "run",
            "schema_version",
            "security",
            "spec",
            "summary_metrics",
            "wandb_plan",
        },
        path="release manifest",
    )
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        _fail("unsupported release manifest schema")
    if (
        manifest["project"] != "barunaction-35m"
        or manifest["local_manifest_authoritative"] is not True
    ):
        _fail("release manifest is not the authoritative BarunAction-35M record")
    if manifest["security"] != {
        "credential_markers_found": 0,
        "symlinks_found": 0,
        "unlisted_source_entries_found": 0,
    }:
        _fail("release manifest security checks did not all pass")
    plan = manifest["wandb_plan"]
    if not isinstance(plan, Mapping) or plan.get("package_version") != WANDB_VERSION:
        _fail(f"release manifest must pin exactly wandb=={WANDB_VERSION}")
    if plan.get("evidence_includes_local_manifest_as") != "release-manifest.json":
        _fail("release manifest does not bind its evidence upload filename")

    artifacts = _exact_fields(
        manifest["artifacts"], set(ARTIFACT_CONTRACTS), path="release manifest artifacts"
    )
    expected: dict[str, dict[str, dict[str, Any]]] = {}
    source_roots: list[PurePosixPath] = []
    for kind, contract in ARTIFACT_CONTRACTS.items():
        artifact, files = _validate_artifact_manifest(artifacts[kind], kind=kind, contract=contract)
        expected[kind] = files
        source_roots.append(PurePosixPath(artifact["source_root"]))
    if any(
        first == second or first in second.parents or second in first.parents
        for index, first in enumerate(source_roots)
        for second in source_roots[index + 1 :]
    ):
        _fail("release artifact source roots are not distinct")

    retention = _exact_fields(
        manifest["int8_retention"],
        {"disposition", "file_count", "files", "gate", "source_root", "total_bytes"},
        path="release manifest int8_retention",
    )
    if retention["disposition"] != "retained-release":
        _fail("release manifest does not retain the int8 artifact")
    retention_root = PurePosixPath(
        _strict_relative(
            retention["source_root"], path="release manifest int8_retention.source_root"
        )
    )
    if any(
        retention_root == source
        or retention_root in source.parents
        or source in retention_root.parents
        for source in source_roots
    ):
        _fail("int8 retention source root overlaps an artifact source root")
    retention_files = _file_map(retention["files"], path="release manifest int8_retention.files")
    if set(retention_files) != INT8_RETENTION_FILES:
        _fail("release manifest changed the exact int8-retention file set")
    if retention["file_count"] != len(retention_files) or retention["total_bytes"] != sum(
        record["bytes"] for record in retention_files.values()
    ):
        _fail("release manifest int8-retention file totals are inconsistent")
    if not isinstance(retention["gate"], Mapping) or retention["gate"].get("passed") is not True:
        _fail("release manifest int8-retention gate did not pass")

    evidence = expected["evidence"]
    if "release-manifest.json" in evidence or any(
        name == "int8-retention" or name.startswith("int8-retention/") for name in evidence
    ):
        _fail("base evidence file names collide with W&B-added release evidence")
    evidence.update({f"int8-retention/{name}": record for name, record in retention_files.items()})
    evidence["release-manifest.json"] = {"bytes": len(raw), "sha256": sha256}
    return expected, str(manifest["project"])


def _metadata(value: Any, *, path: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        _fail(f"{path} has an invalid value")
    return value


def _validate_upload_receipt(
    value: dict[str, Any],
    *,
    manifest_relative: str,
    manifest_sha256: str,
    project: str,
) -> dict[str, dict[str, str]]:
    receipt = _exact_fields(
        value,
        {"artifacts", "entity", "local_manifest", "project", "run_id", "schema_version"},
        path="W&B upload receipt",
    )
    if receipt["schema_version"] != UPLOAD_RECEIPT_SCHEMA_VERSION:
        _fail("unsupported W&B upload receipt schema")
    entity = _metadata(receipt["entity"], path="W&B upload receipt entity", pattern=SLUG_PATTERN)
    if receipt["project"] != project:
        _fail("W&B upload receipt project differs from the release manifest")
    _metadata(receipt["run_id"], path="W&B upload receipt run_id", pattern=SLUG_PATTERN)
    local_manifest = _exact_fields(
        receipt["local_manifest"], {"path", "sha256"}, path="W&B upload receipt local_manifest"
    )
    if local_manifest["path"] != manifest_relative or local_manifest["sha256"] != manifest_sha256:
        _fail("W&B upload receipt is not hash-bound to this release manifest")

    artifacts = _exact_fields(
        receipt["artifacts"], set(ARTIFACT_CONTRACTS), path="W&B upload receipt artifacts"
    )
    normalized: dict[str, dict[str, str]] = {}
    for kind, contract in ARTIFACT_CONTRACTS.items():
        record = _exact_fields(
            artifacts[kind],
            {"digest", "name", "qualified_name", "version"},
            path=f"W&B upload receipt artifacts.{kind}",
        )
        version = _metadata(
            record["version"],
            path=f"W&B upload receipt artifacts.{kind}.version",
            pattern=VERSION_PATTERN,
        )
        if record["name"] != contract["name"]:
            _fail(f"W&B upload receipt artifacts.{kind} changed its fixed name")
        qualified_name = f"{entity}/{project}/{contract['name']}:{version}"
        if record["qualified_name"] != qualified_name or "latest" in qualified_name.casefold():
            _fail(f"W&B upload receipt artifacts.{kind} is not an immutable qualified vN name")
        digest = _metadata(
            record["digest"],
            path=f"W&B upload receipt artifacts.{kind}.digest",
            pattern=REMOTE_VALUE_PATTERN,
        )
        normalized[kind] = {
            "digest": digest,
            "name": contract["name"],
            "qualified_name": qualified_name,
            "type": contract["type"],
            "version": version,
        }
    return normalized


def _external_fresh_root(repository_root: Path, value: Path) -> Path:
    absolute = Path(os.path.abspath(value))
    current = absolute
    while current != current.parent:
        if current.exists() and current.is_symlink():
            _fail("download root cannot contain a symlink component")
        current = current.parent
    resolved = absolute.resolve(strict=False)
    if resolved == repository_root or repository_root in resolved.parents:
        _fail("download root must be outside the repository")
    if absolute.exists() or absolute.is_symlink():
        _fail("download root must be fresh and absent")
    if not absolute.parent.is_dir() or absolute.parent.is_symlink():
        _fail("download root parent must be an existing non-symlink directory")
    return resolved


def _new_receipt_path(repository_root: Path, value: Path, *, source_roots: Sequence[str]) -> Path:
    absolute = Path(os.path.abspath(value))
    if absolute == repository_root or repository_root not in absolute.parents:
        _fail("verification receipt must be inside the repository")
    relative = _strict_relative(
        absolute.relative_to(repository_root).as_posix(), path="verification receipt"
    )
    current = repository_root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.exists() and current.is_symlink():
            _fail("verification receipt cannot contain a symlink component")
    if absolute.exists() or absolute.is_symlink():
        _fail("refusing to overwrite verification receipt")
    if not absolute.parent.is_dir() or absolute.parent.is_symlink():
        _fail("verification receipt parent must be an existing non-symlink directory")
    output_relative = PurePosixPath(relative)
    for source in source_roots:
        root = PurePosixPath(source)
        if output_relative == root or root in output_relative.parents:
            _fail("verification receipt cannot be inside an artifact source root")
    return absolute


def _hash_and_scan(path: Path, *, label: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    overlap = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
            window = overlap + chunk
            if _contains_token_marker(window):
                _fail(f"{label} contains a credential token marker")
            overlap = window[-TOKEN_SCAN_OVERLAP:]
    return digest.hexdigest(), size


def _inspect_download(
    root: Path, *, expected: Mapping[str, Mapping[str, Any]], kind: str
) -> dict[str, dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        _fail(f"downloaded artifact {kind} did not produce a regular directory")
    expected_directories: set[str] = set()
    for name in expected:
        parent = PurePosixPath(name).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent

    observed_paths: dict[str, Path] = {}

    def walk(directory: Path, relative_parent: PurePosixPath) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise ReleaseVerificationError(f"cannot inspect downloaded artifact {kind}") from error
        for entry in entries:
            relative = relative_parent / entry.name
            relative_text = relative.as_posix()
            _reject_sensitive_relative(relative, path=f"downloaded artifact {kind}")
            if entry.is_symlink():
                _fail(f"downloaded artifact {kind} contains a symlink")
            if entry.is_dir(follow_symlinks=False):
                if relative_text not in expected_directories:
                    _fail(f"downloaded artifact {kind} contains an unlisted directory")
                walk(Path(entry.path), relative)
            elif entry.is_file(follow_symlinks=False):
                observed_paths[relative_text] = Path(entry.path)
            else:
                _fail(f"downloaded artifact {kind} contains a non-regular entry")

    walk(root, PurePosixPath())
    if set(observed_paths) != set(expected):
        _fail(
            f"downloaded artifact {kind} file set differs: "
            f"missing={sorted(set(expected) - set(observed_paths))!r}, "
            f"extra={sorted(set(observed_paths) - set(expected))!r}"
        )
    verified: dict[str, dict[str, Any]] = {}
    for relative, path in sorted(observed_paths.items()):
        digest, size = _hash_and_scan(path, label=f"downloaded artifact {kind} file {relative!r}")
        expected_record = expected[relative]
        if digest != expected_record["sha256"] or size != expected_record["bytes"]:
            _fail(f"downloaded artifact {kind} file {relative!r} differs from the manifest")
        verified[relative] = {"bytes": size, "sha256": digest}
    return verified


def _recheck_source(path: Path, expected_sha256: str, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} changed during verification")
    digest, _ = _hash_and_scan(path, label=label)
    if digest != expected_sha256:
        _fail(f"{label} changed during verification")


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> str:
    serialized = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with path.open("xb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise ReleaseVerificationError("refusing to overwrite verification receipt") from error
    return hashlib.sha256(serialized).hexdigest()


def execute(
    *,
    repository_root: Path,
    manifest_path: Path,
    wandb_receipt_path: Path,
    download_root: Path,
    verification_receipt_path: Path,
    download: bool,
    wandb_module: Any | None = None,
) -> dict[str, Any]:
    """Validate the records, and optionally perform an immutable three-artifact re-download."""

    if repository_root.is_symlink() or not repository_root.is_dir():
        _fail("repository root must be a non-symlink directory")
    repository_root = repository_root.resolve(strict=True)
    manifest_path, manifest_relative = _checked_in_path(
        repository_root, manifest_path, label="release manifest"
    )
    wandb_receipt_path, receipt_relative = _checked_in_path(
        repository_root, wandb_receipt_path, label="W&B upload receipt"
    )
    if manifest_path == wandb_receipt_path:
        _fail("release manifest and W&B upload receipt must be distinct")
    manifest, manifest_raw, manifest_sha256 = _load_json(manifest_path, label="release manifest")
    expected_files, project = _validate_manifest(manifest, raw=manifest_raw, sha256=manifest_sha256)
    upload_receipt, _, upload_receipt_sha256 = _load_json(
        wandb_receipt_path, label="W&B upload receipt"
    )
    artifact_identities = _validate_upload_receipt(
        upload_receipt,
        manifest_relative=manifest_relative,
        manifest_sha256=manifest_sha256,
        project=project,
    )
    external_root = _external_fresh_root(repository_root, download_root)
    source_roots = [artifact["source_root"] for artifact in manifest["artifacts"].values()] + [
        manifest["int8_retention"]["source_root"]
    ]
    verification_receipt_path = _new_receipt_path(
        repository_root, verification_receipt_path, source_roots=source_roots
    )
    if verification_receipt_path in {manifest_path, wandb_receipt_path}:
        _fail("verification receipt must be distinct from its source records")

    base_result = {
        "artifact_qualified_names": [
            artifact_identities[kind]["qualified_name"] for kind in ARTIFACT_CONTRACTS
        ],
        "download_root": str(external_root),
        "manifest_sha256": manifest_sha256,
        "upload_receipt_sha256": upload_receipt_sha256,
    }
    if not download:
        return {"mode": "dry-run", **base_result}

    if wandb_module is None:
        try:
            wandb_module = importlib.import_module("wandb")
        except ImportError as error:
            raise ReleaseVerificationError(
                "W&B download requires requirements/wandb-release.txt"
            ) from error
    if getattr(wandb_module, "__version__", None) != WANDB_VERSION:
        _fail(f"W&B download requires exactly wandb=={WANDB_VERSION}")
    try:
        api = wandb_module.Api()
    except Exception as error:
        raise ReleaseVerificationError("W&B API initialization failed") from error

    try:
        external_root.mkdir(mode=0o700)
    except (FileExistsError, OSError) as error:
        raise ReleaseVerificationError("could not create the fresh download root") from error

    verified_artifacts: dict[str, dict[str, Any]] = {}
    for kind, identity in artifact_identities.items():
        destination = external_root / kind
        if destination.exists() or destination.is_symlink():
            _fail(f"download destination for {kind} was not absent")
        try:
            artifact = api.artifact(identity["qualified_name"], type=identity["type"])
        except Exception as error:
            raise ReleaseVerificationError(
                f"W&B could not resolve immutable artifact {kind}"
            ) from error
        if artifact is None:
            _fail(f"W&B returned no immutable artifact for {kind}")
        remote_digest = getattr(artifact, "digest", None)
        if remote_digest != identity["digest"]:
            _fail(f"W&B artifact digest differs from the upload receipt for {kind}")
        remote_qualified_name = getattr(artifact, "qualified_name", None)
        if (
            remote_qualified_name is not None
            and remote_qualified_name != identity["qualified_name"]
        ):
            _fail(f"W&B resolved a different qualified artifact for {kind}")
        try:
            returned_root = artifact.download(root=str(destination))
        except Exception as error:
            raise ReleaseVerificationError(f"W&B artifact download failed for {kind}") from error
        if not isinstance(returned_root, (str, os.PathLike)):
            _fail(f"W&B artifact download returned no directory for {kind}")
        returned_path = Path(os.path.abspath(returned_root)).resolve(strict=False)
        if returned_path != destination.resolve(strict=False):
            _fail(f"W&B artifact {kind} downloaded outside its dedicated destination")
        if getattr(artifact, "digest", None) != identity["digest"]:
            _fail(f"W&B artifact digest changed during download for {kind}")
        files = _inspect_download(destination, expected=expected_files[kind], kind=kind)
        verified_artifacts[kind] = {
            "digest": identity["digest"],
            "download_directory": kind,
            "file_count": len(files),
            "files": files,
            "qualified_name": identity["qualified_name"],
            "total_bytes": sum(record["bytes"] for record in files.values()),
            "type": identity["type"],
            "version": identity["version"],
        }

    _recheck_source(manifest_path, manifest_sha256, label="release manifest")
    _recheck_source(wandb_receipt_path, upload_receipt_sha256, label="W&B upload receipt")
    verification_receipt = {
        "artifacts": verified_artifacts,
        "download_root": str(external_root),
        "package_version": WANDB_VERSION,
        "schema_version": VERIFICATION_RECEIPT_SCHEMA_VERSION,
        "security": {
            "credential_markers_found": 0,
            "extra_files_found": 0,
            "symlinks_found": 0,
        },
        "sources": {
            "release_manifest": {"path": manifest_relative, "sha256": manifest_sha256},
            "wandb_upload_receipt": {
                "path": receipt_relative,
                "sha256": upload_receipt_sha256,
            },
        },
    }
    verification_sha256 = _write_new_json(verification_receipt_path, verification_receipt)
    return {
        "mode": "download",
        **base_result,
        "verification_receipt": verification_receipt_path.relative_to(repository_root).as_posix(),
        "verification_receipt_sha256": verification_sha256,
    }


def _reject_sensitive_cli(arguments: Sequence[str]) -> None:
    for argument in arguments:
        if SENSITIVE_KEY_PATTERN.search(argument) is not None or _contains_token_marker(
            argument.encode("utf-8", errors="ignore")
        ):
            _fail("refusing a credential-like CLI argument")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wandb-receipt", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--verification-receipt", type=Path, required=True)
    parser.add_argument(
        "--download",
        action="store_true",
        help=f"explicitly import wandb=={WANDB_VERSION} and re-download all three artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        _reject_sensitive_cli(arguments)
        args = build_parser().parse_args(arguments)
        result = execute(
            repository_root=args.repository_root,
            manifest_path=args.manifest,
            wandb_receipt_path=args.wandb_receipt,
            download_root=args.download_root,
            verification_receipt_path=args.verification_receipt,
            download=args.download,
        )
    except ReleaseVerificationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
