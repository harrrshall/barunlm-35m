from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_builder():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_mobile_temporal_launch_provenance.py"
    )
    return _load_module("mobile_temporal_launch_builder", path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(root: Path, *, excluded_exact_paths: list[str]) -> dict[str, object]:
    records = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded_exact_paths:
            continue
        records.append((relative, _sha256(path), path.stat().st_size))
    digest = hashlib.sha256()
    for relative, file_sha256, _ in records:
        digest.update(f"{file_sha256}  {relative}\n".encode())
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(records),
        "content_bytes": sum(size for _, _, size in records),
        "files": [
            {"path": relative, "sha256": file_sha256, "bytes": size}
            for relative, file_sha256, size in records
        ],
    }


def _git(source: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _fixture(tmp_path: Path):
    builder = _load_builder()
    builder._REQUIRED_IMPLEMENTATION_PATHS = frozenset(
        {"implementation.py", builder._REQUIREMENTS_RELATIVE_PATH}
    )
    source = tmp_path / "source"
    stage = tmp_path / "stage"
    source.mkdir()
    implementation = source / "implementation.py"
    implementation.write_text("# frozen\n", encoding="utf-8")
    screening = source / "experiments/run/materialized-screening-v1/view-audit.json"
    screening.parent.mkdir(parents=True)
    screening.write_text('{"screening":true}\n', encoding="utf-8")
    (screening.parent / "shadow-audit.json").write_text('{"shadow":true}\n', encoding="utf-8")
    full = source / "experiments/run/materialized-full-refit-v1/full-view-audit.json"
    full.parent.mkdir(parents=True)
    full.write_text('{"full":true}\n', encoding="utf-8")
    requirements = source / builder._REQUIREMENTS_RELATIVE_PATH
    requirements.parent.mkdir(parents=True)
    requirements.write_text(
        "# dependency-only\n" + "\n".join(builder._REQUIRED_REQUIREMENT_LINES) + "\n",
        encoding="utf-8",
    )
    config = {
        "schema_version": "test-config-v1",
        "run_id": "20260803-2353-mobile-temporal-counterfactual-s17",
        "status": "frozen_before_model_or_cuda_access",
        "implementation_contract": {
            "files": {
                "implementation.py": _sha256(implementation),
                builder._REQUIREMENTS_RELATIVE_PATH: _sha256(requirements),
            }
        },
        "materialization": {"view_audit_sha256": _sha256(screening)},
        "conditional_full_refit": {"audit_sha256": _sha256(full)},
        "retry_policy": {
            "retry_after_any_held_out_signal": "forbidden",
            "retry_execution_policy": "fresh-only",
        },
        "compute": {
            "provider": "JarvisLabs",
            "template": "axolotl",
            "python_implementation": "CPython",
            "python_version": "3.11.10",
            "hardware": "one newly created NVIDIA H200",
            "gpu": "H200",
            "num_gpus": 1,
            "region": "IN2",
            "is_spot": False,
            "maximum_gpu_job_minutes": 30,
        },
    }
    config_path = source / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test")
    _git(source, "remote", "add", "origin", "https://github.com/harrrshall/barunlm-35m.git")
    _git(source, "checkout", "-q", "-b", "agent/test-mbcf")
    _git(source, "add", ".")
    _git(source, "commit", "-q", "-m", "frozen test source")
    shutil.copytree(source, stage, ignore=shutil.ignore_patterns(".git"))
    terminal = stage / "terminal.jsonl"
    terminal.write_text('{"reused":true}\n', encoding="utf-8")
    shutil.copy2(stage / builder._REQUIREMENTS_RELATIVE_PATH, stage / requirements.name)
    screening_names = {
        "construction": "construction-train.jsonl",
        "selection": "selection.jsonl",
    }
    for name in screening_names.values():
        (stage / screening.relative_to(source).parent / name).write_text(
            '{"raw":"screening"}\n', encoding="utf-8"
        )
    full_names = {
        "audit": "full-view-audit.json",
        "receipts": "full-transform-receipts.jsonl",
        "train": "full-counterfactual-train.jsonl",
    }
    for name in (full_names["receipts"], full_names["train"]):
        (stage / full.relative_to(source).parent / name).write_text(
            '{"raw":"full"}\n', encoding="utf-8"
        )
    runner = SimpleNamespace(
        CONFIG_SCHEMA_VERSION="test-config-v1",
        PINNED_REUSED_756_SHA256=_sha256(terminal),
        PINNED_REUSED_756_ROWS=756,
        ATTEMPT_PREREGISTRATION_RELATIVE_PATH="attempt-preregistration.json",
        SOURCE_SNAPSHOT_RELATIVE_PATH="source-snapshot.json",
        SOURCE_SNAPSHOT_SCHEMA_VERSION="test-source-snapshot-v1",
        ATTEMPT_PREREGISTRATION_SCHEMA_VERSION="test-attempt-v1",
        ATTEMPT_STATUS="frozen-before-create",
        KNOWN_PROTECTED_JARVIS_IDS=frozenset({463058, 463689}),
        EXPECTED_JARVIS_TEMPLATE="axolotl",
        EXPECTED_PYTHON_IMPLEMENTATION="CPython",
        EXPECTED_PYTHON_VERSION="3.11.10",
        SOURCE_TREE_ALGORITHM="test-tree-v1",
        SOURCE_TREE_EXCLUDED_DIRECTORY_NAMES=(".git", "__pycache__"),
        SOURCE_TREE_EXCLUDED_DIRECTORY_SUFFIXES=(".egg-info",),
        SOURCE_TREE_EXCLUDED_FILE_NAMES=(".DS_Store",),
        SOURCE_TREE_EXCLUDED_FILE_SUFFIXES=(".pyc",),
        EXPECTED_ARTIFACT_FILENAMES=screening_names,
        EXPECTED_FULL_REFIT_FILENAMES=full_names,
        _validate_config=lambda _payload: None,
        _preflight=lambda **_kwargs: ({"passed": True}, {"passed": True}),
        _recompute_source_tree=_tree,
    )
    safe_run = SimpleNamespace(
        MACHINE_ID_PLACEHOLDER=builder.MACHINE_ID_PLACEHOLDER,
        PREEXISTING_IDS_PLACEHOLDER=builder.PREEXISTING_IDS_PLACEHOLDER,
    )
    return builder, runner, safe_run, source, stage


def _build(builder, runner, safe_run, source: Path, stage: Path):
    return builder.build_launch_provenance(
        stage_root=stage,
        config_relative_path="config.json",
        screening_audit_relative_path=("experiments/run/materialized-screening-v1/view-audit.json"),
        full_audit_relative_path=(
            "experiments/run/materialized-full-refit-v1/full-view-audit.json"
        ),
        terminal_relative_path="terminal.jsonl",
        source_repository_root=source,
        runner=runner,
        safe_run=safe_run,
    )


def test_builds_idempotent_git_bound_attempt_with_safe_run_placeholders(tmp_path: Path) -> None:
    builder, runner, safe_run, source, stage = _fixture(tmp_path)

    first = _build(builder, runner, safe_run, source, stage)
    second = _build(builder, runner, safe_run, source, stage)

    assert first == second
    attempt = json.loads((stage / "attempt-preregistration.json").read_text())
    assert attempt["attempt_ordinal"] == 1
    assert attempt["prior_attempts"] == []
    assert attempt["compute"] == {
        "provider": "JarvisLabs",
        "template": "axolotl",
        "python_implementation": "CPython",
        "python_version": "3.11.10",
        "gpu": "H200",
        "num_gpus": 1,
        "region": "IN2",
        "is_spot": False,
        "max_gpu_job_minutes": 30,
    }
    assert attempt["prelaunch_inventory"] == {
        "captured_before_project_instance_creation": True,
        "fresh_project_instance": True,
        "project_machine_id": builder.MACHINE_ID_PLACEHOLDER,
        "protected_machine_ids": builder.PREEXISTING_IDS_PLACEHOLDER,
    }
    snapshot = json.loads((stage / "source-snapshot.json").read_text())
    assert snapshot["git"]["repository"].endswith("harrrshall/barunlm-35m.git")
    assert len(snapshot["git"]["commit"]) == 40
    assert snapshot["git"]["branch"] == "agent/test-mbcf"
    assert snapshot["content_tree"]["excluded_exact_paths"] == [
        "attempt-preregistration.json",
        "source-snapshot.json",
        "terminal.jsonl",
    ]
    assert first["inventory_binding"] == "pending_safe_run_after_fresh_instance_creation"
    assert first["schema_version"] == "barun-mobile-temporal-launch-provenance-build-v2"


def test_rejects_dependency_or_source_drift_and_unallowlisted_stage_file(tmp_path: Path) -> None:
    builder, runner, safe_run, source, stage = _fixture(tmp_path)
    _build(builder, runner, safe_run, source, stage)
    (stage / "implementation.py").write_text("# changed\n", encoding="utf-8")
    with pytest.raises(builder.LaunchProvenanceError, match="implementation hash differs"):
        _build(builder, runner, safe_run, source, stage)

    (stage / "implementation.py").write_text("# frozen\n", encoding="utf-8")
    requirements = stage / builder._REQUIREMENTS_RELATIVE_PATH
    requirements.write_text("-e .\n", encoding="utf-8")
    with pytest.raises(builder.LaunchProvenanceError, match="implementation hash differs"):
        _build(builder, runner, safe_run, source, stage)

    shutil.copy2(source / builder._REQUIREMENTS_RELATIVE_PATH, requirements)
    (stage / "unlisted-source.txt").write_text("new tree member\n", encoding="utf-8")
    with pytest.raises(builder.LaunchProvenanceError, match="outside the committed tree"):
        _build(builder, runner, safe_run, source, stage)


def test_requires_screening_audits_to_be_committed_at_claimed_head(tmp_path: Path) -> None:
    builder, runner, safe_run, source, stage = _fixture(tmp_path)
    shadow_audit = stage / "experiments/run/materialized-screening-v1/shadow-audit.json"
    shadow_audit.unlink()

    with pytest.raises(builder.LaunchProvenanceError, match="stage differs from claimed git"):
        _build(builder, runner, safe_run, source, stage)


def test_detects_secret_content_and_matches_real_safe_run_placeholders(tmp_path: Path) -> None:
    builder = _load_builder()
    secret = tmp_path / "secret.txt"
    secret.write_text("wandb_v1_" + "A7" * 16 + "\n", encoding="utf-8")
    with pytest.raises(builder.LaunchProvenanceError, match="credential-like content"):
        builder._scan_stage_for_secrets(tmp_path)

    safe_run_path = Path(__file__).resolve().parents[1] / "infra" / "jarvis" / "safe_run.py"
    safe_run = _load_module("real_safe_run_for_builder_test", safe_run_path)
    assert builder.MACHINE_ID_PLACEHOLDER == safe_run.MACHINE_ID_PLACEHOLDER
    assert builder.PREEXISTING_IDS_PLACEHOLDER == safe_run.PREEXISTING_IDS_PLACEHOLDER


def test_real_safe_run_binds_only_inventory_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder, runner, fake_safe_run, source, stage = _fixture(tmp_path)
    _build(builder, runner, fake_safe_run, source, stage)
    attempt_path = stage / "attempt-preregistration.json"
    before = json.loads(attempt_path.read_text())
    safe_run_path = Path(__file__).resolve().parents[1] / "infra" / "jarvis" / "safe_run.py"
    safe_run = _load_module("real_safe_run_binding_test", safe_run_path)
    protected = tmp_path / "protected.json"
    protected.write_text(
        json.dumps(
            {"resources": [{"machine_id": 463058}, {"machine_id": 463689}]},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(safe_run.PROTECTED_RESOURCES_ENV, str(protected))
    record = {
        "machine_id": 999_001,
        "instance_name": "barun-unit-test-s0",
        "created_by_safe_run": True,
        "preexisting_resources": [{"machine_id": 463719, "name": "protected"}],
        "events": [],
    }
    safe_run.bind_attempt_inventory(
        target=stage,
        relative_path=Path("attempt-preregistration.json"),
        record_path=tmp_path / "record.json",
        record=record,
        machine_id=999_001,
    )
    after = json.loads(attempt_path.read_text())
    assert after["prelaunch_inventory"] == {
        "captured_before_project_instance_creation": True,
        "fresh_project_instance": True,
        "project_machine_id": 999_001,
        "protected_machine_ids": [463058, 463689, 463719],
    }
    assert {key: value for key, value in after.items() if key != "prelaunch_inventory"} == {
        key: value for key, value in before.items() if key != "prelaunch_inventory"
    }
    assert record["attempt_inventory_binding"]["bound_before_attached_run"] is True


def test_requires_complete_critical_implementation_contract() -> None:
    builder = _load_builder()
    assert {
        "infra/jarvis/safe_run.py",
        "requirements/mobile-temporal-counterfactual.txt",
        "scripts/build_mobile_temporal_launch_provenance.py",
        "scripts/run_mobile_temporal_counterfactual.py",
        "src/barunlm/evaluation/mobile_temporal_counterfactual.py",
        "src/barunlm/training/mobile_temporal_experiment.py",
        "src/barunlm/training/trainer.py",
    } <= builder._REQUIRED_IMPLEMENTATION_PATHS


def test_loading_real_staged_runner_never_creates_bytecode_cache(tmp_path: Path) -> None:
    builder = _load_builder()
    source_runner = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_mobile_temporal_counterfactual.py"
    )
    staged_runner = tmp_path / "scripts" / source_runner.name
    staged_runner.parent.mkdir(parents=True)
    shutil.copy2(source_runner, staged_runner)

    module = builder._load_staged_runner(tmp_path)

    assert Path(module.REPOSITORY_ROOT) == tmp_path
    assert not (tmp_path / "scripts" / "__pycache__").exists()
