from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_logger() -> ModuleType:
    path = ROOT / "scripts" / "log_candidate_release_wandb.py"
    spec = importlib.util.spec_from_file_location("barun_wandb_release", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_files(root: Path, files: dict[str, bytes]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        hashes[relative] = sha256(path)
    return hashes


def release_tree(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    repository = tmp_path / "repo"
    repository.mkdir(parents=True)
    float_root = repository / "release/float"
    float_hashes = write_files(
        float_root,
        {
            "LICENSE": b"Apache License 2.0\n",
            "MODEL_CARD.md": b"# BarunAction-35M float test artifact\n",
            "NOTICE": b"Synthetic Mobile Actions and PRESTO attribution fixture.\n",
            "barun_config.json": b"{}\n",
            "model.safetensors": b"synthetic float weights\n",
            "tokenizer.json": b'{"version":"test"}\n',
        },
    )
    float_payload_hashes = {
        name: float_hashes[name]
        for name in ("barun_config.json", "model.safetensors", "tokenizer.json")
    }
    checkpoint_manifest = {
        "arm_id": "batch63",
        "file_sha256": float_payload_hashes,
        "schema_version": "barun-release-checkpoint-v1",
        "source_checkpoint": "/remote/checkpoint/step-00000126",
    }
    checkpoint_path = float_root / "checkpoint_manifest.json"
    checkpoint_path.write_text(json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n")
    float_hashes["checkpoint_manifest.json"] = sha256(checkpoint_path)
    candidate = {
        "arm_id": "batch63",
        "candidate_id": "candidate-v2",
        "checkpoint_manifest_sha256": float_hashes["checkpoint_manifest.json"],
        "checkpoint_relative_path": "release/float",
        "file_sha256": float_payload_hashes,
        "run_id": "20260803-1845-mob-batch63-s17",
        "schema_version": "barunaction-candidate-provenance-v1",
        "selection_run_id": "20260803-1845-mobile-followup-retry-s17",
        "step": 126,
    }
    candidate_path = repository / "configs/barunaction/candidate-v2.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")

    int8_root = repository / "release/int8"
    int8_hashes = write_files(
        int8_root,
        {
            "LICENSE": b"Apache License 2.0\n",
            "MODEL_CARD.md": b"# BarunAction-35M int8 test artifact\n",
            "NOTICE": b"Synthetic Mobile Actions and PRESTO attribution fixture.\n",
            "barun_config.json": b"{}\n",
            "model.int8.pt": b"synthetic int8 weights\n",
            "tokenizer.json": b'{"version":"test"}\n',
        },
    )
    int8_payload_hashes = {
        name: int8_hashes[name] for name in ("barun_config.json", "model.int8.pt", "tokenizer.json")
    }
    quantization_manifest = {
        "algorithm": {"qengine": "qnnpack"},
        "architecture": "barunlm.model.BarunLM",
        "artifact_sha256": int8_payload_hashes,
        "float_linear_modules": ["lm_head"],
        "parameter_counts": {"total_unique_parameters": 35_072_768},
        "quantized_modules": ["layers.0.attn.q_proj"],
        "runtime": {
            "platform_machine": "arm64",
            "platform_system": "Darwin",
            "torch_version": "2.13.0",
        },
        "schema_version": "barun-cpu-dynamic-int8-v1",
        "size_bytes": {"quantized_payload": 23, "source_payload": 24},
        "source_checkpoint_sha256": candidate["file_sha256"],
    }
    quantization_path = int8_root / "quantization_manifest.json"
    quantization_path.write_text(json.dumps(quantization_manifest, indent=2, sort_keys=True) + "\n")
    int8_hashes["quantization_manifest.json"] = sha256(quantization_path)
    evidence_hashes = write_files(
        repository / "release/evidence",
        {
            "LICENSE": b"Apache License 2.0\n",
            "NOTICE": b"Synthetic Mobile Actions and PRESTO attribution fixture.\n",
            "mobile/result.json": b'{"ast_exact":0.7962962962962963}\n',
            "presto/result.json": b'{"ast_exact":0.7432810750279956}\n',
        },
    )
    retention_root = repository / "release/int8-retention"
    protocol_bytes = (
        ROOT / "configs/barunaction/candidate-v2-arm64-int8-retention-v1.json"
    ).read_bytes()
    paired_rows = []
    for index in range(756):
        float_exact = index < 602
        int8_exact = index < 587
        transition = (
            "retained_correct"
            if float_exact and int8_exact
            else "regressed"
            if float_exact
            else "fixed"
            if int8_exact
            else "retained_incorrect"
        )
        paired_rows.append(
            {
                "float_ast_exact": float_exact,
                "int8_ast_exact": int8_exact,
                "sample_id": f"mobile-actions-{index:05d}-fixture",
                "scenario": "fixture_action",
                "schema_version": "barunaction-arm64-int8-retention-paired-v1",
                "transition": transition,
            }
        )
    paired_bytes = (
        "\n".join(
            json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":"))
            for row in paired_rows
        )
        + "\n"
    ).encode()
    retention_result = {
        "schema_version": "barunaction-arm64-int8-retention-result-v1",
        "protocol": {
            "protocol_id": "candidate-v2-arm64-int8-retention-v1",
            "path": "/immutable/receipt/protocol.json",
            "sha256": "7229572bea461a7899b09c0310a77b9b4d8af74211d1b9ac8b2b5f77dc6b67cb",
        },
        "candidate": {
            "candidate_id": "candidate-v2",
            "float_checkpoint_file_sha256": float_payload_hashes,
        },
        "int8_checkpoint": {
            "directory": "/immutable/int8-checkpoint",
            "format_version": "barun-cpu-dynamic-int8-v1",
            "manifest_sha256": int8_hashes["quantization_manifest.json"],
            "artifact_sha256": int8_payload_hashes,
            "source_checkpoint_file_sha256": float_payload_hashes,
            "qengine": "qnnpack",
        },
        "population": {
            "rows": 756,
            "official_evaluation_rows_opaque_unparsed": 961,
            "official_evaluation_artifacts_accessed": [],
        },
        "generation": {"checkpoint_format": "int8"},
        "evidence": {"paired_samples": "/immutable/receipt/paired-samples.jsonl"},
        "gate": {
            "metric": "ast_exact_match",
            "passed": True,
            "status": "pass",
            "source_float": {"numerator": 602, "denominator": 756, "value": 602 / 756},
            "int8": {"numerator": 587, "denominator": 756, "value": 587 / 756},
            "float_to_int8": {
                "correct_rows_lost": 15,
                "absolute_accuracy_change": -15 / 756,
            },
            "thresholds": {
                "minimum_int8_numerator": 587,
                "maximum_float_to_int8_correct_rows_lost": 15,
            },
            "component_gates": {
                "minimum_int8_accuracy": True,
                "maximum_float_to_int8_loss": True,
                "complete_aligned_samples": True,
                "explicit_int8_no_fallback": True,
            },
            "paired_transition_counts": {
                "retained_correct": 587,
                "fixed": 0,
                "regressed": 15,
                "retained_incorrect": 154,
            },
            "decision": "retain_candidate_v2_int8_artifact",
        },
        "interpretation": {"scope": "post_selection_arm64_int8_retention_diagnostic"},
    }
    retention_hashes = write_files(
        retention_root,
        {
            "paired-samples.jsonl": paired_bytes,
            "protocol.json": protocol_bytes,
            "result.json": (
                json.dumps(retention_result, allow_nan=False, indent=2, sort_keys=True) + "\n"
            ).encode(),
        },
    )
    retention_artifact_manifest = {
        "file_sha256": retention_hashes,
        "schema_version": "barunaction-arm64-int8-retention-artifacts-v1",
    }
    retention_artifact_path = retention_root / "artifact-sha256.json"
    retention_artifact_path.write_text(
        json.dumps(retention_artifact_manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    retention_hashes["artifact-sha256.json"] = sha256(retention_artifact_path)
    payload: dict[str, Any] = {
        "schema_version": "barunaction-wandb-release-spec-v1",
        "project": "barunaction-35m",
        "run": {
            "candidate_id": "candidate-v2",
            "group": "candidate-v2-release",
            "job_type": "release-evidence",
            "name": "barunaction-candidate-v2-release-s17",
            "release_id": "candidate-v2-20260803",
            "source_runs": {
                "continual_recovery": "20260803-2310-continual-recovery-retry-s17",
                "interpolation_rescue": "20260803-2240-interpolation-rescue-s17",
                "matched_baseline": "20260803-1955-mobile-smollm2-matched-retry-s17",
                "matched_qwen_baseline": "20260803-2300-mobile-qwen-matched-s17",
                "mobile_candidate_v2": "20260803-1845-mobile-followup-retry-s17",
                "mobile_int8_retention": "20260803-2218-candidate-v2-arm64-int8-retention-s17",
                "parallel_rescue_selection": "20260803-2203-parallel-rescue-selection-s17",
                "presto_mobile_regression": "20260803-2045-presto-mobile-regression-s17",
                "presto_stage": "20260803-1920-presto-stage-s17",
            },
            "tags": ["barunaction-35m", "candidate-v2", "release-evidence"],
        },
        "artifacts": {
            "float": {
                "name": "barunaction-35m-candidate-v2-float",
                "type": "model",
                "source_root": "release/float",
                "file_sha256": float_hashes,
            },
            "int8_darwin_arm64_qnnpack": {
                "name": "barunaction-35m-candidate-v2-int8-darwin-arm64-qnnpack",
                "type": "model",
                "source_root": "release/int8",
                "file_sha256": int8_hashes,
            },
            "evidence": {
                "name": "barunaction-35m-candidate-v2-evidence",
                "type": "evaluation",
                "source_root": "release/evidence",
                "file_sha256": evidence_hashes,
            },
        },
        "int8_retention": {
            "source_root": "release/int8-retention",
            "file_sha256": retention_hashes,
        },
        "summary_metrics": {
            "continual_recovery/gate_passed": False,
            "continual_recovery/mobile_ast_exact_match": 642 / 756,
            "continual_recovery/presto_abstention_f1": 0.98,
            "continual_recovery/presto_ast_exact_match": 9862 / 14288,
            "continual_recovery/presto_false_call_on_gate": 0.01,
            "continual_recovery/presto_schema_valid": 0.999,
            "interpolation_rescue/alpha_025/eligible": True,
            "interpolation_rescue/alpha_025/mobile_ast_exact_match": 0.80,
            "interpolation_rescue/alpha_025/presto_ast_exact_match": 0.70,
            "interpolation_rescue/alpha_050/eligible": True,
            "interpolation_rescue/alpha_050/mobile_ast_exact_match": 0.81,
            "interpolation_rescue/alpha_050/presto_ast_exact_match": 0.71,
            "interpolation_rescue/alpha_075/eligible": False,
            "interpolation_rescue/alpha_075/mobile_ast_exact_match": 0.75,
            "interpolation_rescue/alpha_075/presto_ast_exact_match": 0.72,
            "interpolation_rescue/gate_passed": True,
            "interpolation_rescue/selection_exists": True,
            "matched_baseline/ast_exact_match": 0.72,
            "matched_baseline/gate_passed": False,
            "matched_baseline/parameters": 135_000_000,
            "matched_baseline/schema_valid": 0.99,
            "matched_qwen_baseline/barunaction_ast_exact_match": 602 / 756,
            "matched_qwen_baseline/parameters": 494_032_768,
            "matched_qwen_baseline/qwen_ast_exact_match": 663 / 756,
            "matched_qwen_baseline/qwen_outperformed_barunaction": True,
            "matched_qwen_baseline/qwen_parse_valid": 755 / 756,
            "matched_qwen_baseline/qwen_schema_valid": 754 / 756,
            "mobile_candidate_v2/ast_exact_match": 0.7962962962962963,
            "mobile_candidate_v2/schema_valid": 0.9986772486772487,
            "mobile_int8_retention/ast_exact_match": 587 / 756,
            "mobile_int8_retention/correct_rows_lost": 15,
            "mobile_int8_retention/gate_passed": True,
            "mobile_int8_retention/source_float_ast_exact_match": 602 / 756,
            "parallel_rescue_selection/eligible_trials": 2,
            "parallel_rescue_selection/gate_passed": True,
            "presto_mobile_regression/ast_exact_match": 0.79,
            "presto_mobile_regression/gate_passed": True,
            "presto_mobile_regression/schema_valid": 0.99,
            "presto_stage/abstention_f1": 0.9795653584171261,
            "presto_stage/ast_exact_match": 0.7432810750279956,
            "presto_stage/false_call_on_gate": 0.008551936196790622,
            "presto_stage/gate_passed": True,
            "presto_stage/schema_valid": 0.9997900335946248,
        },
    }
    spec_path = repository / "release/spec.json"
    spec_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return repository, spec_path, tmp_path / "external-wandb", payload


def rewrite_spec(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_dry_run_writes_deterministic_manifest_without_importing_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_logger()
    repository, spec_path, wandb_dir, _ = release_tree(tmp_path)
    manifest_path = repository / "release-records/wandb-manifest.json"
    monkeypatch.delenv("WANDB_DIR", raising=False)

    def forbidden_import(name: str) -> None:
        raise AssertionError(f"dry run imported {name}")

    monkeypatch.setattr(module.importlib, "import_module", forbidden_import)
    first = module.build_release_manifest(
        spec_path=spec_path,
        repository_root=repository,
        wandb_dir=wandb_dir,
    )
    second = module.build_release_manifest(
        spec_path=spec_path,
        repository_root=repository,
        wandb_dir=wandb_dir,
    )
    assert first == second

    result = module.execute(
        spec_path=spec_path,
        manifest_path=manifest_path,
        repository_root=repository,
        wandb_dir=wandb_dir,
        upload=False,
        receipt_path=None,
    )

    manifest = json.loads(manifest_path.read_text())
    assert result["mode"] == "dry-run"
    assert result["manifest_sha256"] == sha256(manifest_path)
    assert manifest == first
    assert manifest["local_manifest_authoritative"] is True
    assert manifest["security"] == {
        "credential_markers_found": 0,
        "symlinks_found": 0,
        "unlisted_source_entries_found": 0,
    }
    assert [manifest["artifacts"][kind]["name"] for kind in module.ARTIFACT_CONTRACTS] == [
        "barunaction-35m-candidate-v2-float",
        "barunaction-35m-candidate-v2-int8-darwin-arm64-qnnpack",
        "barunaction-35m-candidate-v2-evidence",
    ]
    assert "WANDB_DIR" not in module.os.environ


def test_artifact_roots_reject_unlisted_files_symlinks_and_credentials(tmp_path: Path) -> None:
    module = load_logger()
    repository, spec_path, wandb_dir, payload = release_tree(tmp_path)

    (repository / "release/float/unlisted.txt").write_text("not allowlisted\n")
    with pytest.raises(module.ReleaseLoggingError, match="unlisted"):
        module.build_release_manifest(
            spec_path=spec_path,
            repository_root=repository,
            wandb_dir=wandb_dir,
        )
    (repository / "release/float/unlisted.txt").unlink()

    symlink = repository / "release/evidence/link.json"
    try:
        symlink.symlink_to(repository / "release/evidence/mobile/result.json")
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(module.ReleaseLoggingError, match="symlink"):
        module.build_release_manifest(
            spec_path=spec_path,
            repository_root=repository,
            wandb_dir=wandb_dir,
        )
    symlink.unlink()

    credential = repository / "release/evidence/.env.local"
    credential.write_text("placeholder\n")
    with pytest.raises(module.ReleaseLoggingError, match="credential"):
        module.build_release_manifest(
            spec_path=spec_path,
            repository_root=repository,
            wandb_dir=wandb_dir,
        )
    credential.unlink()

    cache_file = repository / "release/evidence/.cache/hidden.json"
    cache_file.parent.mkdir()
    cache_file.write_text("{}\n")
    with pytest.raises(module.ReleaseLoggingError, match="cache"):
        module.build_release_manifest(
            spec_path=spec_path,
            repository_root=repository,
            wandb_dir=wandb_dir,
        )

    assert payload["artifacts"]["float"]["file_sha256"]


def test_evidence_allows_legal_notices_but_rejects_arbitrary_binary_files(
    tmp_path: Path,
) -> None:
    module = load_logger()
    repository, spec_path, wandb_dir, payload = release_tree(tmp_path)

    # LICENSE and NOTICE are deliberately part of the evidence fixture, so a
    # complete manifest build proves that both exact no-suffix names are allowed.
    module.build_release_manifest(
        spec_path=spec_path,
        repository_root=repository,
        wandb_dir=wandb_dir,
    )

    binary = repository / "release/evidence/weights.bin"
    binary.write_bytes(b"not release evidence\n")
    payload["artifacts"]["evidence"]["file_sha256"]["weights.bin"] = sha256(binary)
    rewrite_spec(spec_path, payload)
    with pytest.raises(module.ReleaseLoggingError, match="non-evidence file extension"):
        module.build_release_manifest(
            spec_path=spec_path,
            repository_root=repository,
            wandb_dir=wandb_dir,
        )


def test_token_markers_and_sensitive_spec_fields_fail_without_echoing_values(
    tmp_path: Path,
) -> None:
    module = load_logger()
    repository, spec_path, wandb_dir, payload = release_tree(tmp_path)
    evidence = repository / "release/evidence/mobile/result.json"
    evidence.write_text('{"value":"wandb_v1_' + "A" * 32 + '"}\n')
    payload["artifacts"]["evidence"]["file_sha256"]["mobile/result.json"] = sha256(evidence)
    rewrite_spec(spec_path, payload)
    with pytest.raises(module.ReleaseLoggingError, match="credential token marker") as caught:
        module.build_release_manifest(
            spec_path=spec_path,
            repository_root=repository,
            wandb_dir=wandb_dir,
        )
    assert "A" * 32 not in str(caught.value)

    evidence.write_text('{"value":"safe"}\n')
    payload["artifacts"]["evidence"]["file_sha256"]["mobile/result.json"] = sha256(evidence)
    payload["api_key"] = "do-not-echo-this-value"
    rewrite_spec(spec_path, payload)
    with pytest.raises(module.ReleaseLoggingError, match="credential-like field") as caught:
        module.build_release_manifest(
            spec_path=spec_path,
            repository_root=repository,
            wandb_dir=wandb_dir,
        )
    assert "do-not-echo-this-value" not in str(caught.value)


def test_exact_artifact_names_files_metrics_and_external_wandb_dir(tmp_path: Path) -> None:
    module = load_logger()
    repository, spec_path, wandb_dir, payload = release_tree(tmp_path)

    payload["artifacts"]["float"]["name"] = "arbitrary-model-name"
    rewrite_spec(spec_path, payload)
    with pytest.raises(module.ReleaseLoggingError, match="fixed release contract"):
        module.build_release_manifest(
            spec_path=spec_path,
            repository_root=repository,
            wandb_dir=wandb_dir,
        )

    repository, spec_path, wandb_dir, payload = release_tree(tmp_path / "second")
    del payload["artifacts"]["int8_darwin_arm64_qnnpack"]["file_sha256"]["model.int8.pt"]
    (repository / "release/int8/model.int8.pt").unlink()
    rewrite_spec(spec_path, payload)
    with pytest.raises(module.ReleaseLoggingError, match="exact required file set"):
        module.build_release_manifest(
            spec_path=spec_path,
            repository_root=repository,
            wandb_dir=wandb_dir,
        )

    repository, spec_path, _, payload = release_tree(tmp_path / "third")
    payload["summary_metrics"]["invented/score"] = 0.5
    rewrite_spec(spec_path, payload)
    with pytest.raises(module.ReleaseLoggingError, match="unknown names"):
        module.build_release_manifest(
            spec_path=spec_path,
            repository_root=repository,
            wandb_dir=tmp_path / "third-wandb",
        )

    repository, spec_path, _, _ = release_tree(tmp_path / "fourth")
    with pytest.raises(module.ReleaseLoggingError, match="outside the repository"):
        module.build_release_manifest(
            spec_path=spec_path,
            repository_root=repository,
            wandb_dir=repository / "wandb",
        )

    external = tmp_path / "external-real"
    external.mkdir()
    link = tmp_path / "external-link"
    try:
        link.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(module.ReleaseLoggingError, match="symlink component"):
        module.build_release_manifest(
            spec_path=spec_path,
            repository_root=repository,
            wandb_dir=link / "wandb",
        )

    repository, spec_path, wandb_dir, payload = release_tree(tmp_path / "fifth")
    del payload["summary_metrics"]["continual_recovery/gate_passed"]
    rewrite_spec(spec_path, payload)
    with pytest.raises(module.ReleaseLoggingError, match="omits required completed evidence"):
        module.build_release_manifest(
            spec_path=spec_path,
            repository_root=repository,
            wandb_dir=wandb_dir,
        )


def test_float_and_int8_artifacts_are_bound_to_canonical_candidate_identity(
    tmp_path: Path,
) -> None:
    module = load_logger()
    repository, spec_path, wandb_dir, payload = release_tree(tmp_path)
    float_root = repository / "release/float"
    model_path = float_root / "model.safetensors"
    model_path.write_bytes(b"different but self-consistent float bytes\n")
    checkpoint_path = float_root / "checkpoint_manifest.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["file_sha256"]["model.safetensors"] = sha256(model_path)
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
    payload["artifacts"]["float"]["file_sha256"]["model.safetensors"] = sha256(model_path)
    payload["artifacts"]["float"]["file_sha256"]["checkpoint_manifest.json"] = sha256(
        checkpoint_path
    )
    rewrite_spec(spec_path, payload)
    with pytest.raises(module.ReleaseLoggingError, match="canonical candidate-v2"):
        module.build_release_manifest(
            spec_path=spec_path,
            repository_root=repository,
            wandb_dir=wandb_dir,
        )

    repository, spec_path, wandb_dir, payload = release_tree(tmp_path / "int8")
    quantization_path = repository / "release/int8/quantization_manifest.json"
    quantization = json.loads(quantization_path.read_text())
    quantization["runtime"]["platform_system"] = "Linux"
    quantization_path.write_text(json.dumps(quantization, indent=2, sort_keys=True) + "\n")
    payload["artifacts"]["int8_darwin_arm64_qnnpack"]["file_sha256"][
        "quantization_manifest.json"
    ] = sha256(quantization_path)
    rewrite_spec(spec_path, payload)
    with pytest.raises(module.ReleaseLoggingError, match="Darwin ARM64"):
        module.build_release_manifest(
            spec_path=spec_path,
            repository_root=repository,
            wandb_dir=wandb_dir,
        )


def test_int8_release_requires_a_hash_bound_passing_retention_receipt(tmp_path: Path) -> None:
    module = load_logger()
    repository, spec_path, wandb_dir, payload = release_tree(tmp_path)
    retention_root = repository / "release/int8-retention"
    result_path = retention_root / "result.json"
    result = json.loads(result_path.read_text())
    result["gate"]["passed"] = False
    result["gate"]["status"] = "fail"
    result["gate"]["decision"] = "reject_candidate_v2_int8_artifact"
    result_path.write_text(json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n")
    artifact_manifest_path = retention_root / "artifact-sha256.json"
    artifact_manifest = json.loads(artifact_manifest_path.read_text())
    artifact_manifest["file_sha256"]["result.json"] = sha256(result_path)
    artifact_manifest_path.write_text(
        json.dumps(artifact_manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    payload["int8_retention"]["file_sha256"]["result.json"] = sha256(result_path)
    payload["int8_retention"]["file_sha256"]["artifact-sha256.json"] = sha256(
        artifact_manifest_path
    )
    rewrite_spec(spec_path, payload)

    with pytest.raises(module.ReleaseLoggingError, match="retention gate did not pass"):
        module.build_release_manifest(
            spec_path=spec_path,
            repository_root=repository,
            wandb_dir=wandb_dir,
        )


class FakeArtifact:
    def __init__(self, *, name: str, type: str, metadata: dict[str, Any]) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[tuple[str, str]] = []
        self.digest = ""
        self.qualified_name = ""
        self.version = ""

    def add_file(self, path: str, *, name: str) -> None:
        self.files.append((path, name))

    def wait(self) -> FakeArtifact:
        return self


class FakeRun:
    def __init__(self) -> None:
        self.entity = "test-entity"
        self.id = "offline-test-run"
        self.project = "barunaction-35m"
        self.summary: dict[str, Any] = {}
        self.artifacts: list[FakeArtifact] = []
        self.finish_codes: list[int] = []

    def log_artifact(self, artifact: FakeArtifact) -> FakeArtifact:
        version = f"v{len(self.artifacts)}"
        artifact.digest = f"digest-{len(self.artifacts)}"
        artifact.qualified_name = f"{self.entity}/{self.project}/{artifact.name}:{version}"
        artifact.version = version
        self.artifacts.append(artifact)
        return artifact

    def finish(self, *, exit_code: int) -> None:
        self.finish_codes.append(exit_code)


def test_upload_is_explicit_pinned_and_disables_code_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_logger()
    repository, spec_path, wandb_dir, _ = release_tree(tmp_path)
    manifest_path = repository / "release-records/wandb-manifest.json"
    receipt_path = repository / "release-records/wandb-receipt.json"
    fake_run = FakeRun()
    init_calls: list[dict[str, Any]] = []

    class FakeSettings:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    def fake_init(**kwargs: Any) -> FakeRun:
        init_calls.append(kwargs)
        return fake_run

    fake_wandb = SimpleNamespace(
        __version__="0.28.1",
        Artifact=FakeArtifact,
        Settings=FakeSettings,
        init=fake_init,
    )
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: fake_wandb if name == "wandb" else pytest.fail(name),
    )
    monkeypatch.delenv("WANDB_DIR", raising=False)
    monkeypatch.delenv("WANDB_DISABLE_CODE", raising=False)

    result = module.execute(
        spec_path=spec_path,
        manifest_path=manifest_path,
        repository_root=repository,
        wandb_dir=wandb_dir,
        upload=True,
        receipt_path=receipt_path,
    )

    assert result["mode"] == "upload"
    assert result["receipt_sha256"] == sha256(receipt_path)
    assert len(init_calls) == 1
    init = init_calls[0]
    assert init["mode"] == "online"
    assert init["save_code"] is False
    assert init["settings"].kwargs == {"disable_git": True}
    assert init["dir"] == str(wandb_dir.resolve())
    assert module.os.environ["WANDB_DIR"] == str(wandb_dir.resolve())
    assert module.os.environ["WANDB_DISABLE_CODE"] == "true"
    assert [artifact.name for artifact in fake_run.artifacts] == [
        "barunaction-35m-candidate-v2-float",
        "barunaction-35m-candidate-v2-int8-darwin-arm64-qnnpack",
        "barunaction-35m-candidate-v2-evidence",
    ]
    assert fake_run.artifacts[-1].files[-1][1] == "release-manifest.json"
    assert {
        name for _, name in fake_run.artifacts[-1].files if name.startswith("int8-retention/")
    } == {
        "int8-retention/artifact-sha256.json",
        "int8-retention/paired-samples.jsonl",
        "int8-retention/protocol.json",
        "int8-retention/result.json",
    }
    int8_metadata = fake_run.artifacts[1].metadata
    assert int8_metadata["release_disposition"] == "retained-release"
    assert int8_metadata["retention_gate_passed"] is True
    assert fake_run.finish_codes == [0]
    assert fake_run.summary == json.loads(manifest_path.read_text())["summary_metrics"]
    receipt = json.loads(receipt_path.read_text())
    assert receipt["entity"] == "test-entity"
    assert receipt["project"] == "barunaction-35m"
    assert receipt["run_id"] == "offline-test-run"
    assert {kind: record["version"] for kind, record in receipt["artifacts"].items()} == {
        "evidence": "v2",
        "float": "v0",
        "int8_darwin_arm64_qnnpack": "v1",
    }
    assert all(":v" in record["qualified_name"] for record in receipt["artifacts"].values())
    assert all("latest" not in record["qualified_name"] for record in receipt["artifacts"].values())


def test_upload_rechecks_files_after_add_and_rejects_post_manifest_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_logger()
    repository, spec_path, wandb_dir, _ = release_tree(tmp_path)
    manifest_path = repository / "release-records/wandb-manifest.json"
    receipt_path = repository / "release-records/wandb-receipt.json"
    fake_run = FakeRun()

    class MutatingArtifact(FakeArtifact):
        def add_file(self, path: str, *, name: str) -> None:
            super().add_file(path, name=name)
            if name == "model.safetensors":
                Path(path).write_bytes(b"mutated after manifest creation\n")

    class FakeSettings:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    fake_wandb = SimpleNamespace(
        __version__="0.28.1",
        Artifact=MutatingArtifact,
        Settings=FakeSettings,
        init=lambda **kwargs: fake_run,
    )
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_wandb)

    with pytest.raises(module.ReleaseLoggingError, match="local manifest remains authoritative"):
        module.execute(
            spec_path=spec_path,
            manifest_path=manifest_path,
            repository_root=repository,
            wandb_dir=wandb_dir,
            upload=True,
            receipt_path=receipt_path,
        )

    assert manifest_path.is_file()
    assert not receipt_path.exists()
    assert fake_run.artifacts == []
    assert fake_run.finish_codes == [1]


def test_local_records_are_immutable_and_upload_requires_receipt(tmp_path: Path) -> None:
    module = load_logger()
    repository, spec_path, wandb_dir, _ = release_tree(tmp_path)
    manifest_path = repository / "release-records/wandb-manifest.json"
    module.execute(
        spec_path=spec_path,
        manifest_path=manifest_path,
        repository_root=repository,
        wandb_dir=wandb_dir,
        upload=False,
        receipt_path=None,
    )
    with pytest.raises(module.ReleaseLoggingError, match="refusing to overwrite"):
        module.execute(
            spec_path=spec_path,
            manifest_path=manifest_path,
            repository_root=repository,
            wandb_dir=wandb_dir,
            upload=False,
            receipt_path=None,
        )
    with pytest.raises(module.ReleaseLoggingError, match="requires --receipt"):
        module.execute(
            spec_path=spec_path,
            manifest_path=repository / "release-records/second.json",
            repository_root=repository,
            wandb_dir=wandb_dir,
            upload=True,
            receipt_path=None,
        )

    repository, spec_path, wandb_dir, _ = release_tree(tmp_path / "retention-output")
    with pytest.raises(module.ReleaseLoggingError, match="artifact source root"):
        module.execute(
            spec_path=spec_path,
            manifest_path=repository / "release/int8-retention/manifest.json",
            repository_root=repository,
            wandb_dir=wandb_dir,
            upload=False,
            receipt_path=None,
        )


def test_logger_has_no_login_or_api_key_cli_surface() -> None:
    module = load_logger()
    source = (ROOT / "scripts/log_candidate_release_wandb.py").read_text()
    tree = ast.parse(source)
    assert not [
        node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "login"
    ]
    assert not [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            (isinstance(node, ast.Import) and any(alias.name == "wandb" for alias in node.names))
            or (isinstance(node, ast.ImportFrom) and node.module == "wandb")
        )
    ]
    option_strings = {
        option for action in module.build_parser()._actions for option in action.option_strings
    }
    assert not any(
        "key" in option or "token" in option or "password" in option for option in option_strings
    )
    assert "wandb==0.28.1" in (ROOT / "requirements/wandb-release.txt").read_text()


def test_sensitive_cli_argument_is_rejected_without_echo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_logger()
    marker = "wandb_v1_" + "Z" * 32
    assert module.main(["--api-key", marker]) == 2
    captured = capsys.readouterr()
    assert "credential-like CLI argument" in captured.err
    assert marker not in captured.err
