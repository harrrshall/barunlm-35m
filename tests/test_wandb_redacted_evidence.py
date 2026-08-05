from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_builder() -> ModuleType:
    path = ROOT / "scripts" / "build_wandb_redacted_evidence.py"
    spec = importlib.util.spec_from_file_location("barun_wandb_redacted_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = _load_builder()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(
    tmp_path: Path, *, replacement: dict[str, bytes] | None = None
) -> tuple[Path, Path, Path, dict[str, bytes], str]:
    repository = tmp_path / "repo"
    source = repository / "staging/evidence"
    source.mkdir(parents=True)
    files = {
        "metrics.json": (
            json.dumps(
                {
                    "checkpoint_dir": "/Users/alice/work/checkpoint",
                    "gradient_norm": 0.43784472346305847,
                    "machine_id": 463058,
                    "machine_name": "Kroda",
                    "protected_kroda_463058_mutated": False,
                    "qwen_project_instance": 463689,
                    "schema_valid": 754,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        "predictions.jsonl": (
            json.dumps(
                {
                    "ast_exact": True,
                    "id": "sample-a463058b",
                    "prediction": '{"decision":"ABSTAIN"}',
                    "run_dir": "/tmp/barun-run/predictions.jsonl",
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        "config.json": (
            json.dumps(
                {
                    "editable_install": "file:///private/tmp/barun-stage/work",
                    "protected_unknown": 463697,
                    "protected_recreated_opt": 464346,
                    "protected_recreated_opt128": 464367,
                    "remote_artifact_root": "/home/barun-artifacts/run-1",
                    "workstation_stage": "/private/tmp/barun-stage/work",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        "notes.md": (
            b"# BarunAction-35M evidence\n\n"
            b"Kroda (also dictated as cruda), ID 463058, remained untouched. "
            b"Local receipt: `/Users/alice/work/receipt.json`.\n"
        ),
    }
    if replacement:
        files.update(replacement)
    records: dict[str, dict[str, Any]] = {}
    for relative, payload in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        records[relative] = {"bytes": len(payload), "sha256": _sha256(payload)}
    retention_root = repository / "retention"
    retention_root.mkdir()
    retention_payload = b'{"local_receipt":"/Users/alice/work/int8/result.json","rows":756}\n'
    (retention_root / "result.json").write_bytes(retention_payload)
    manifest = {
        "artifacts": {
            "evidence": {
                "file_count": len(files),
                "files": records,
                "name": "barunaction-35m-candidate-v2-evidence",
                "source_root": "staging/evidence",
                "total_bytes": sum(map(len, files.values())),
                "type": "evaluation",
            }
        },
        "int8_retention": {
            "disposition": "retained-release",
            "file_count": 1,
            "files": {
                "result.json": {
                    "bytes": len(retention_payload),
                    "sha256": _sha256(retention_payload),
                }
            },
            "gate": {"passed": True},
            "source_root": "retention",
            "total_bytes": len(retention_payload),
        },
        "local_manifest_authoritative": True,
        "project": "barunaction-35m",
        "schema_version": "barunaction-wandb-release-manifest-v1",
    }
    manifest_payload = (
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    manifest_path = repository / "release-manifest.json"
    manifest_path.write_bytes(manifest_payload)
    return repository, manifest_path, source, files, _sha256(manifest_payload)


def _build(
    *, repository: Path, manifest: Path, destination: Path, manifest_hash: str
) -> dict[str, Any]:
    return BUILDER._build_redacted_view(
        repository_root=repository,
        manifest_path=manifest,
        destination=destination,
        expected_manifest_sha256=manifest_hash,
    )


def test_builds_complete_local_distribution_view_without_changing_science(
    tmp_path: Path,
) -> None:
    repository, manifest, source, originals, manifest_hash = _fixture(tmp_path)
    source_hashes_before = {
        relative: _sha256((source / relative).read_bytes()) for relative in originals
    }
    destination = tmp_path / "redacted"

    receipt = _build(
        repository=repository,
        manifest=manifest,
        destination=destination,
        manifest_hash=manifest_hash,
    )

    assert receipt["schema_version"] == "barunaction-wandb-redacted-evidence-view-v1"
    assert receipt["classification"]["distribution_view"] == "redacted"
    assert receipt["classification"]["authoritative_evidence"] is False
    assert receipt["security"]["uploads_performed"] is False
    assert receipt["security"]["network_operations_performed"] is False
    expected_outputs = set(originals) | {
        "int8-retention/result.json",
        "release-manifest.json",
    }
    assert receipt["output_tree"]["derived_file_count"] == len(expected_outputs)
    assert set(receipt["files"]) == expected_outputs
    assert {path.name for path in destination.iterdir()} >= {
        "metrics.json",
        "predictions.jsonl",
        "config.json",
        "notes.md",
        "int8-retention",
        "release-manifest.json",
        "redaction-receipt.json",
    }
    assert json.loads((destination / "redaction-receipt.json").read_text()) == receipt

    metrics = json.loads((destination / "metrics.json").read_text())
    assert metrics["gradient_norm"] == 0.43784472346305847
    assert metrics["schema_valid"] == 754
    assert metrics["qwen_project_instance"] == 463689
    assert metrics["machine_id"] == 900000001
    assert metrics["machine_name"] == "PROTECTED_RESOURCE_01"
    assert metrics["protected_PROTECTED_RESOURCE_01_900000001_mutated"] is False

    prediction = json.loads((destination / "predictions.jsonl").read_text())
    assert prediction["ast_exact"] is True
    assert prediction["id"] == "sample-a463058b"
    assert prediction["prediction"] == '{"decision":"ABSTAIN"}'
    assert prediction["run_dir"].startswith("[REDACTED_LOCAL_PATH_")

    config = json.loads((destination / "config.json").read_text())
    assert config["remote_artifact_root"] == "/home/barun-artifacts/run-1"
    assert config["protected_unknown"] == 900000002
    assert config["protected_recreated_opt"] == 900000008
    assert config["protected_recreated_opt128"] == 900000009
    assert config["workstation_stage"].startswith("[REDACTED_LOCAL_PATH_")
    assert config["editable_install"].startswith("file://[REDACTED_LOCAL_PATH_")
    retention = json.loads((destination / "int8-retention/result.json").read_text())
    assert retention["rows"] == 756
    assert retention["local_receipt"].startswith("[REDACTED_LOCAL_PATH_")
    combined_output = b"".join(
        (destination / relative).read_bytes() for relative in sorted(originals)
    )
    assert b"/Users/" not in combined_output
    assert b"/private/tmp/" not in combined_output
    assert b"/tmp/" not in combined_output
    assert b"Kroda" not in combined_output
    assert b"cruda" not in combined_output
    assert b"464346" not in combined_output
    assert b"464367" not in combined_output

    assert (
        manifest.read_bytes()
        == (
            json.dumps(json.loads(manifest.read_text()), allow_nan=False, indent=2, sort_keys=True)
            + "\n"
        ).encode()
    )
    assert {
        relative: _sha256((source / relative).read_bytes()) for relative in originals
    } == source_hashes_before
    for relative, record in receipt["files"].items():
        if relative in source_hashes_before:
            assert record["source_sha256"] == source_hashes_before[relative]
        assert len(record["source_sha256"]) == 64
        assert record["output_sha256"] == _sha256((destination / relative).read_bytes())


@pytest.mark.parametrize("failure", ["extra", "drift", "credential", "sentinel", "unsafe_path"])
def test_fails_before_creating_destination_for_untrusted_source(
    tmp_path: Path, failure: str
) -> None:
    replacement = None
    if failure == "credential":
        marker = b"wandb" + b"_v1_" + b"A" * 32
        replacement = {"notes.md": b"credential=" + marker + b"\n"}
    elif failure == "sentinel":
        replacement = {"notes.md": b"preexisting PROTECTED_RESOURCE_01 sentinel\n"}
    elif failure == "unsafe_path":
        replacement = {"notes.md": b"unsupported URL https://example.test/tmp/private/run.json\n"}
    repository, manifest, source, _, manifest_hash = _fixture(tmp_path, replacement=replacement)
    if failure == "extra":
        (source / "unlisted.json").write_text("{}\n")
    elif failure == "drift":
        (source / "metrics.json").write_text("{}\n")
    destination = tmp_path / "redacted"

    with pytest.raises(BUILDER.RedactedEvidenceError):
        _build(
            repository=repository,
            manifest=manifest,
            destination=destination,
            manifest_hash=manifest_hash,
        )
    assert not destination.exists()


def test_rejects_symlink_and_existing_destination(tmp_path: Path) -> None:
    repository, manifest, source, _, manifest_hash = _fixture(tmp_path)
    link = source / "notes.md"
    link.unlink()
    link.symlink_to(source / "metrics.json")
    with pytest.raises(BUILDER.RedactedEvidenceError, match="symlink"):
        _build(
            repository=repository,
            manifest=manifest,
            destination=tmp_path / "redacted-symlink",
            manifest_hash=manifest_hash,
        )

    repository, manifest, _, _, manifest_hash = _fixture(tmp_path / "second")
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("do not overwrite\n")
    with pytest.raises(BUILDER.RedactedEvidenceError, match="already exists"):
        _build(
            repository=repository,
            manifest=manifest,
            destination=destination,
            manifest_hash=manifest_hash,
        )
    assert sentinel.read_text() == "do not overwrite\n"


def test_cli_has_no_network_upload_or_remote_execution_surface() -> None:
    source_path = ROOT / "scripts" / "build_wandb_redacted_evidence.py"
    tree = ast.parse(source_path.read_text())
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports.isdisjoint(
        {"boto3", "httpx", "paramiko", "requests", "socket", "subprocess", "urllib", "wandb"}
    )
    options = {action.dest for action in BUILDER._parser()._actions}
    assert options == {"help", "repository_root", "manifest", "destination"}
