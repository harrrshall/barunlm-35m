from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _load_builder():
    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "build_mobile_planir_launch_provenance.py"
    )
    spec = importlib.util.spec_from_file_location("mobile_planir_launch_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_entrypoint():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_mobile_planir_screen.py"
    spec = importlib.util.spec_from_file_location("mobile_planir_remote_entrypoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(source: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _scientific_config(builder, hashes: dict[str, str]) -> dict[str, Any]:
    config_path = Path(builder.__file__).resolve().parents[1] / builder.CONFIG_RELATIVE_PATH
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["data"]["artifact_sha256"] = hashes
    return payload


def _materialized_bytes(builder) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for name in sorted(set(builder.MATERIALIZED_FILENAMES) - {"audit"}):
        payloads[name] = (
            json.dumps(
                {"schema_version": "test-row-v1", "artifact": name},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
    audit = {
        "schema_version": "test-screen-audit-v1",
        "forbidden_population_rows_read": 0,
        "outputs": {
            name: {
                "filename": builder.MATERIALIZED_FILENAMES[name],
                "sha256": hashes[name],
                "rows": 1,
            }
            for name in sorted(payloads)
        },
    }
    payloads["audit"] = (
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    return payloads


def _fixture(tmp_path: Path):
    builder = _load_builder()
    source = tmp_path / "source"
    stage = tmp_path / "stage"
    source.mkdir(parents=True)
    stage.mkdir(parents=True)

    materialized = _materialized_bytes(builder)
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in materialized.items()}
    config = _scientific_config(builder, hashes)
    source_config = source / builder.CONFIG_RELATIVE_PATH
    _write_json(source_config, config)
    requirements = source / builder.REQUIREMENTS_RELATIVE_PATH
    requirements.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        Path(__file__).resolve().parents[1] / builder.REQUIREMENTS_RELATIVE_PATH,
        requirements,
    )
    code = source / "src/test_backend.py"
    code.parent.mkdir(parents=True)
    code.write_text("# exact committed backend\n", encoding="utf-8")
    safe_run = source / builder.SAFE_RUN_RELATIVE_PATH
    safe_run.parent.mkdir(parents=True)
    safe_run.write_text(
        'MACHINE_ID_PLACEHOLDER = "__SAFE_RUN_MACHINE_ID__"\n'
        'PREEXISTING_IDS_PLACEHOLDER = "__SAFE_RUN_PREEXISTING_IDS__"\n',
        encoding="utf-8",
    )

    _git(source, "init", "-q")
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test")
    _git(source, "remote", "add", "origin", builder.CANONICAL_REPOSITORY)
    _git(source, "checkout", "-q", "-b", "agent/test-planir-provenance")
    _git(source, "add", ".")
    _git(source, "commit", "-q", "-m", "frozen fixture")

    commit_bound = tuple(
        sorted(
            {
                builder.CONFIG_RELATIVE_PATH,
                builder.REQUIREMENTS_RELATIVE_PATH,
                "src/test_backend.py",
            }
        )
    )
    for relative in commit_bound:
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    for name, content in materialized.items():
        destination = stage / builder.MATERIALIZED_RELATIVE_PATHS[name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    allowlist = tuple(sorted({*commit_bound, *builder.MATERIALIZED_RELATIVE_PATHS.values()}))
    return builder, source, stage, allowlist, commit_bound


def _build(builder, source, stage, allowlist, commit_bound, **kwargs):
    return builder.build_launch_provenance(
        stage_root=stage,
        source_repository_root=source,
        _allowlist=allowlist,
        _commit_bound_paths=commit_bound,
        _expected_config_sha256=_sha256(source / builder.CONFIG_RELATIVE_PATH),
        **kwargs,
    )


def _bind_attempt(builder, stage: Path, *, machine_id: int = 999_001) -> str:
    path = stage / builder.ATTEMPT_RELATIVE_PATH
    attempt = json.loads(path.read_text(encoding="utf-8"))
    protected = sorted({*attempt["durable_protected_machine_ids"], 463999})
    attempt["prelaunch_inventory"] = {
        "captured_before_project_instance_creation": True,
        "fresh_project_instance": True,
        "project_machine_id": machine_id,
        "protected_machine_ids": protected,
    }
    _write_json(path, attempt)
    return _sha256(path)


def test_builds_exact_no_overwrite_unbound_attempt_and_snapshot(tmp_path: Path) -> None:
    builder, source, stage, allowlist, commit_bound = _fixture(tmp_path)

    receipt = _build(builder, source, stage, allowlist, commit_bound)

    snapshot = json.loads((stage / builder.SOURCE_SNAPSHOT_RELATIVE_PATH).read_text())
    attempt = json.loads((stage / builder.ATTEMPT_RELATIVE_PATH).read_text())
    assert snapshot["scientific_tree"]["allowlist"] == list(allowlist)
    assert snapshot["scientific_tree"]["commit_bound_paths"] == list(commit_bound)
    assert snapshot["scientific_tree"]["excluded_exact_paths"] == [
        "attempt-preregistration.json",
        "source-snapshot.json",
    ]
    assert snapshot["scientific_tree"]["file_count"] == len(allowlist)
    assert snapshot["scientific_tree"]["content_bytes"] == sum(
        (stage / relative).stat().st_size for relative in allowlist
    )
    assert len(snapshot["scientific_tree"]["files"]) == len(allowlist)
    assert snapshot["git"]["clean_checkout"] is True
    assert len(snapshot["git"]["commit"]) == 40
    assert attempt["compute"] == builder.COMPUTE_CONTRACT
    assert attempt["prelaunch_inventory"] == {
        "captured_before_project_instance_creation": True,
        "fresh_project_instance": True,
        "project_machine_id": builder.MACHINE_ID_PLACEHOLDER,
        "protected_machine_ids": builder.PREEXISTING_IDS_PLACEHOLDER,
    }
    assert attempt["local_controller"]["path"] == builder.SAFE_RUN_RELATIVE_PATH
    assert attempt["local_controller"]["uploaded_to_stage"] is False
    assert attempt["attempt_ordinal"] == 1
    assert attempt["attempt_id"] == f"{builder.RUN_ID}-attempt-1"
    assert attempt["retry_lock"] == {
        "automatic_retry": "forbidden",
        "every_new_attempt": "forbidden",
        "resume_or_reuse_instance": "forbidden",
        "retry_authorized": False,
        "same_attempt_retry": "forbidden",
    }
    assert not (stage / builder.SAFE_RUN_RELATIVE_PATH).exists()
    assert receipt["inventory_binding"].startswith("pending_safe_run")
    with pytest.raises(builder.LaunchProvenanceError, match="overwrite|already exists"):
        _build(builder, source, stage, allowlist, commit_bound)


def test_bound_validator_allows_only_root_venv_and_frozen_caches(tmp_path: Path) -> None:
    builder, source, stage, allowlist, commit_bound = _fixture(tmp_path)
    build = _build(builder, source, stage, allowlist, commit_bound)
    attempt_sha256 = _bind_attempt(builder, stage)
    (stage / ".venv/bin").mkdir(parents=True)
    os.symlink("/usr/bin/python3", stage / ".venv/bin/python")
    (stage / "src/__pycache__").mkdir()
    (stage / "src/__pycache__/test_backend.cpython-311.pyc").write_bytes(b"runtime")
    (stage / ".pytest_cache").mkdir()
    (stage / ".pytest_cache/CACHEDIR.TAG").write_text("runtime", encoding="utf-8")
    (stage / ".ruff_cache").mkdir()
    (stage / ".ruff_cache/state").write_text("runtime", encoding="utf-8")

    validation = builder.validate_launch_provenance(
        stage_root=stage,
        expected_source_snapshot_sha256=build["source_snapshot"]["sha256"],
        expected_attempt_sha256=attempt_sha256,
        _allowlist=allowlist,
        _commit_bound_paths=commit_bound,
        _expected_config_sha256=_sha256(source / builder.CONFIG_RELATIVE_PATH),
    )

    assert validation["inventory_bound"] is True
    assert validation["scientific_tree"] == build["scientific_tree"]
    (stage / "unexpected.txt").write_text("not allowlisted\n", encoding="utf-8")
    with pytest.raises(builder.LaunchProvenanceError, match="outside the stage allowlist"):
        builder.validate_launch_provenance(
            stage_root=stage,
            expected_source_snapshot_sha256=build["source_snapshot"]["sha256"],
            expected_attempt_sha256=attempt_sha256,
            _allowlist=allowlist,
            _commit_bound_paths=commit_bound,
            _expected_config_sha256=_sha256(source / builder.CONFIG_RELATIVE_PATH),
        )


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("extra", "outside the stage allowlist"),
        ("symlink", "symlink is forbidden"),
        ("heldout", "forbidden held-out population"),
        ("wandb", "forbidden runtime/credential directory"),
        ("nested_venv", "only the provider root .venv"),
        ("root_venv", "must not exist when the snapshot is built"),
        ("pycache", "runtime cache directory is forbidden before snapshot"),
        ("pytest_cache", "runtime cache directory is forbidden before snapshot"),
        ("pyc", "runtime cache file is forbidden before snapshot"),
        ("ds_store", "runtime cache file is forbidden before snapshot"),
    ],
)
def test_builder_rejects_extra_symlink_heldout_and_runtime_paths(
    tmp_path: Path, kind: str, message: str
) -> None:
    builder, source, stage, allowlist, commit_bound = _fixture(tmp_path)
    if kind == "extra":
        (stage / "unexpected.txt").write_text("extra\n", encoding="utf-8")
    elif kind == "symlink":
        os.symlink(stage / allowlist[0], stage / "linked-file")
    elif kind == "heldout":
        (stage / "official-mobile-961.jsonl").write_text("{}\n", encoding="utf-8")
    elif kind == "wandb":
        (stage / "wandb").mkdir()
    elif kind == "nested_venv":
        (stage / "src/.venv").mkdir()
    elif kind == "root_venv":
        (stage / ".venv").mkdir()
    elif kind == "pycache":
        (stage / "src/__pycache__").mkdir()
    elif kind == "pytest_cache":
        (stage / ".pytest_cache").mkdir()
    elif kind == "pyc":
        (stage / "orphan.pyc").write_bytes(b"cache")
    else:
        (stage / ".DS_Store").write_bytes(b"finder")

    with pytest.raises(builder.LaunchProvenanceError, match=message):
        _build(builder, source, stage, allowlist, commit_bound)


def test_builder_rejects_secret_content_without_echoing_it(tmp_path: Path) -> None:
    builder, source, stage, allowlist, commit_bound = _fixture(tmp_path)
    secret = "wandb" + "_v1_" + "A7" * 16
    backend = stage / "src/test_backend.py"
    backend.write_text(f"TOKEN = {secret!r}\n", encoding="utf-8")

    with pytest.raises(builder.LaunchProvenanceError, match="credential-like content") as caught:
        _build(builder, source, stage, allowlist, commit_bound)

    assert secret not in str(caught.value)


def test_builder_rejects_staged_commit_drift_and_dirty_source(tmp_path: Path) -> None:
    builder, source, stage, allowlist, commit_bound = _fixture(tmp_path)
    (stage / "src/test_backend.py").write_text("# drift\n", encoding="utf-8")
    with pytest.raises(builder.LaunchProvenanceError, match="differs from the claimed Git commit"):
        _build(builder, source, stage, allowlist, commit_bound)

    shutil.copy2(source / "src/test_backend.py", stage / "src/test_backend.py")
    (source / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(builder.LaunchProvenanceError, match="completely clean checkout"):
        _build(builder, source, stage, allowlist, commit_bound)


def test_builder_rejects_missing_transitive_project_import(tmp_path: Path) -> None:
    builder, source, stage, allowlist, commit_bound = _fixture(tmp_path)
    (stage / "src/test_backend.py").write_text(
        "import barunlm.not_in_the_stage\n",
        encoding="utf-8",
    )

    with pytest.raises(builder.LaunchProvenanceError, match="statically imported module"):
        _build(builder, source, stage, allowlist, commit_bound)


def test_builder_rejects_artifact_drift_and_nonzero_forbidden_access(tmp_path: Path) -> None:
    builder, source, stage, allowlist, commit_bound = _fixture(tmp_path)
    (stage / builder.MATERIALIZED_RELATIVE_PATHS["train_a"]).write_bytes(b"changed\n")
    with pytest.raises(builder.LaunchProvenanceError, match="differs from the scientific config"):
        _build(builder, source, stage, allowlist, commit_bound)

    builder, source, stage, allowlist, commit_bound = _fixture(tmp_path / "second")
    config_path = stage / builder.CONFIG_RELATIVE_PATH
    config = json.loads(config_path.read_text())
    config["data"]["forbidden_populations"]["official_mobile_961_rows_read"] = 1
    _write_json(config_path, config)
    with pytest.raises(builder.LaunchProvenanceError, match="zero access"):
        builder._validate_config(config)
    with pytest.raises(builder.LaunchProvenanceError, match="exact preregistered SHA-256"):
        _build(builder, source, stage, allowlist, commit_bound)


@pytest.mark.parametrize(
    "retry_kwargs",
    [
        {"attempt_ordinal": 2},
        {"prior_zero_signal_receipts": [Path("prior.json")]},
    ],
)
def test_every_new_attempt_and_prior_receipt_are_forbidden(
    tmp_path: Path, retry_kwargs: dict[str, object]
) -> None:
    builder, source, stage, allowlist, commit_bound = _fixture(tmp_path)
    with pytest.raises(builder.LaunchProvenanceError, match="every new attempt is forbidden"):
        _build(
            builder,
            source,
            stage,
            allowlist,
            commit_bound,
            **retry_kwargs,
        )
    assert not (stage / builder.SOURCE_SNAPSHOT_RELATIVE_PATH).exists()
    assert not (stage / builder.ATTEMPT_RELATIVE_PATH).exists()


def test_validator_rejects_tree_snapshot_attempt_and_inventory_tampering(tmp_path: Path) -> None:
    builder, source, stage, allowlist, commit_bound = _fixture(tmp_path)
    build = _build(builder, source, stage, allowlist, commit_bound)
    attempt_path = stage / builder.ATTEMPT_RELATIVE_PATH
    attempt = json.loads(attempt_path.read_text())
    attempt["prelaunch_inventory"] = {
        "captured_before_project_instance_creation": True,
        "fresh_project_instance": True,
        "project_machine_id": 463058,
        "protected_machine_ids": [463058, 463689, 463802],
    }
    _write_json(attempt_path, attempt)
    with pytest.raises(builder.LaunchProvenanceError, match="bound safe_run inventory"):
        builder.validate_launch_provenance(
            stage_root=stage,
            expected_source_snapshot_sha256=build["source_snapshot"]["sha256"],
            expected_attempt_sha256=_sha256(attempt_path),
            _allowlist=allowlist,
            _commit_bound_paths=commit_bound,
            _expected_config_sha256=_sha256(source / builder.CONFIG_RELATIVE_PATH),
        )

    attempt["prelaunch_inventory"] = {
        "captured_before_project_instance_creation": True,
        "fresh_project_instance": True,
        "project_machine_id": 999_001,
        "protected_machine_ids": attempt["durable_protected_machine_ids"],
    }
    attempt["retry_lock"]["same_attempt_retry"] = "allowed"
    _write_json(attempt_path, attempt)
    with pytest.raises(builder.LaunchProvenanceError, match="retry_lock changed"):
        builder.validate_launch_provenance(
            stage_root=stage,
            expected_source_snapshot_sha256=build["source_snapshot"]["sha256"],
            expected_attempt_sha256=_sha256(attempt_path),
            _allowlist=allowlist,
            _commit_bound_paths=commit_bound,
            _expected_config_sha256=_sha256(source / builder.CONFIG_RELATIVE_PATH),
        )


def test_default_allowlist_is_exact_narrow_and_cli_cannot_broaden_it() -> None:
    builder = _load_builder()
    assert builder.DEFAULT_ALLOWLIST == tuple(sorted(set(builder.DEFAULT_ALLOWLIST)))
    assert builder.CONFIG_RELATIVE_PATH in builder.DEFAULT_ALLOWLIST
    assert builder.ENTRYPOINT_RELATIVE_PATH in builder.DEFAULT_ALLOWLIST
    assert builder.ENTRYPOINT_RELATIVE_PATH in builder.DEFAULT_COMMIT_BOUND_PATHS
    assert "src/barunlm/training/mobile_planir_experiment.py" in builder.DEFAULT_ALLOWLIST
    assert set(builder.REMOTE_TEST_PATHS) <= set(builder.DEFAULT_ALLOWLIST)
    assert set(builder.MATERIALIZED_RELATIVE_PATHS.values()) <= set(builder.DEFAULT_ALLOWLIST)
    assert builder.SAFE_RUN_RELATIVE_PATH not in builder.DEFAULT_ALLOWLIST
    assert "scripts/build_mobile_temporal_launch_provenance.py" not in builder.DEFAULT_ALLOWLIST
    assert "scripts/run_mobile_temporal_counterfactual.py" not in builder.DEFAULT_ALLOWLIST

    parser = builder.build_parser()
    base = [
        "build",
        "--stage-root",
        "stage",
        "--source-repository-root",
        "source",
    ]
    for forbidden_arguments in (
        ["--allow-file", "extra.txt"],
        ["--attempt-ordinal", "2"],
        ["--prior-zero-signal-receipt", "prior.json"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args([*base, *forbidden_arguments])


def test_production_default_allowlist_is_complete_and_transitively_closed() -> None:
    builder = _load_builder()
    root = Path(__file__).resolve().parents[1]
    missing = [
        relative for relative in builder.DEFAULT_ALLOWLIST if not (root / relative).is_file()
    ]
    assert missing == []
    files = {relative: root / relative for relative in builder.DEFAULT_ALLOWLIST}

    builder._validate_python_import_closure(
        files=files,
        allowlist=builder.DEFAULT_ALLOWLIST,
    )
    builder._scan_files_for_secrets(files)
    config = builder._strict_json_object(
        root / builder.CONFIG_RELATIVE_PATH,
        label="production scientific config",
    )
    assert _sha256(root / builder.CONFIG_RELATIVE_PATH) == builder.EXPECTED_CONFIG_SHA256
    assert (
        _sha256(root / builder.REQUIREMENTS_RELATIVE_PATH) == builder.EXPECTED_REQUIREMENTS_SHA256
    )
    _run_id, config_hashes, _protected = builder._validate_config(config)
    builder._validate_requirements(root / builder.REQUIREMENTS_RELATIVE_PATH)
    builder._validate_materialized_artifacts(root=root, config_hashes=config_hashes)
    assert set(builder.DEFAULT_COMMIT_BOUND_PATHS) == set(builder.DEFAULT_ALLOWLIST) - set(
        builder.MATERIALIZED_RELATIVE_PATHS.values()
    )


def test_entrypoint_is_commit_bound_and_imports_without_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    root = Path(__file__).resolve().parents[1]
    entrypoint = root / builder.ENTRYPOINT_RELATIVE_PATH
    assert builder.ENTRYPOINT_RELATIVE_PATH in builder.DEFAULT_COMMIT_BOUND_PATHS
    assert entrypoint.is_file() and not entrypoint.is_symlink()

    imported: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        imported.append(name)
        if name.split(".", 1)[0] in {"barunlm", "safetensors", "tokenizers", "torch"}:
            raise AssertionError(f"heavy import during stdlib preflight: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "test-restore")
    monkeypatch.setenv("WANDB_MODE", "test-restore")
    monkeypatch.setenv("WANDB_DISABLED", "test-restore")
    spec = importlib.util.spec_from_file_location("planir_entrypoint_no_torch", entrypoint)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
    assert module.REMOTE_TEST_PATHS == builder.REMOTE_TEST_PATHS
    assert module.SNAPSHOT_RELATIVE_PATH == builder.SOURCE_SNAPSHOT_RELATIVE_PATH
    assert module.ATTEMPT_RELATIVE_PATH == builder.ATTEMPT_RELATIVE_PATH


def test_safe_run_compute_contract_matches_storage_and_runtime_binding() -> None:
    builder = _load_builder()
    root = Path(__file__).resolve().parents[1]
    safe_run_path = root / builder.SAFE_RUN_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("planir_safe_run_compute_contract", safe_run_path)
    assert spec is not None and spec.loader is not None
    safe_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(safe_run)
    args = SimpleNamespace(
        template="axolotl",
        python_implementation="CPython",
        python_version="3.11.10",
        gpu="H200",
        num_gpus=1,
        region="IN2",
        spot=False,
        max_runtime_minutes=45,
        storage=40,
    )

    assert safe_run.requested_attempt_compute(args) == builder.COMPUTE_CONTRACT


def _essential_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    entrypoint = _load_entrypoint()
    root = tmp_path / "stage"
    output = tmp_path / "output"
    root.mkdir()
    output.mkdir()

    snapshot = root / entrypoint.SNAPSHOT_RELATIVE_PATH
    attempt = root / entrypoint.ATTEMPT_RELATIVE_PATH
    config = root / entrypoint.CONFIG_RELATIVE_PATH
    requirements = root / "requirements/mobile-planir-screen.txt"
    _write_json(snapshot, {"snapshot": "test"})
    _write_json(attempt, {"attempt": "test"})
    _write_json(config, {"config": "test"})
    requirements.parent.mkdir(parents=True)
    requirements.write_text("requirements\n", encoding="utf-8")
    monkeypatch.setattr(entrypoint, "EXPECTED_CONFIG_SHA256", _sha256(config))
    monkeypatch.setattr(entrypoint, "EXPECTED_REQUIREMENTS_SHA256", _sha256(requirements))

    materialized = root / entrypoint.MATERIALIZATION_RELATIVE_PATH
    materialized.mkdir(parents=True)
    materialized_hashes: dict[str, str] = {}
    for name in entrypoint.MATERIALIZED_FILENAMES:
        path = materialized / name
        path.write_text(f"{name}\n", encoding="utf-8")
        materialized_hashes[name] = _sha256(path)
    monkeypatch.setattr(entrypoint, "EXPECTED_MATERIALIZED_SHA256", materialized_hashes)

    preflight = {
        "source_snapshot_sha256": _sha256(snapshot),
        "attempt_preregistration_sha256": _sha256(attempt),
    }
    _write_json(output / "preflight-receipt.json", preflight)
    _write_json(output / "pre-cuda-validation.json", {"status": "passed"})
    (output / "pre-cuda-validation.log").write_text("passed\n", encoding="utf-8")
    _write_json(output / "torch-runtime-receipt.json", {"status": "matched"})
    return entrypoint, root, output, materialized_hashes


def test_essential_bundle_copies_exact_replay_data_and_recomputes_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint, root, output, materialized_hashes = _essential_fixture(tmp_path, monkeypatch)
    _write_json(output / "execution/evaluation/scores.json", {"score": 1})
    _write_json(
        output / "execution/training/a/checkpoints/step-00000073/checkpoint_manifest.json",
        {"checkpoint": "test"},
    )

    essential = entrypoint._build_essential_bundle(
        root=root,
        output_root=output,
        status="unit-test",
        error=None,
    )
    manifest = json.loads((essential / "artifact-manifest.json").read_text(encoding="utf-8"))
    observed = {
        path.relative_to(essential).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(essential.rglob("*"))
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    assert manifest["files"] == observed
    assert manifest["weight_like_files_in_bundle"] == []
    assert manifest["screening_weights_in_bundle"] is False
    assert manifest["materialized_construction_artifacts_in_bundle"] == 9
    for name, expected in materialized_hashes.items():
        assert _sha256(essential / "data/materialized-v1" / name) == expected


@pytest.mark.parametrize("mutation", ["bytes", "extra"])
def test_essential_bundle_rejects_materialized_mutation_or_extra_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    entrypoint, root, output, _hashes = _essential_fixture(tmp_path, monkeypatch)
    materialized = root / entrypoint.MATERIALIZATION_RELATIVE_PATH
    if mutation == "bytes":
        (materialized / "train-a.jsonl").write_text("changed\n", encoding="utf-8")
    else:
        (materialized / "extra.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        entrypoint.MobilePlanIRControllerError,
        match="materialized evidence (?:changed|inventory changed)",
    ):
        entrypoint._build_essential_bundle(
            root=root,
            output_root=output,
            status="unit-test",
            error=None,
        )


def test_essential_bundle_requires_controller_provenance_even_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint, root, output, _hashes = _essential_fixture(tmp_path, monkeypatch)
    (root / entrypoint.SNAPSHOT_RELATIVE_PATH).unlink()

    with pytest.raises(entrypoint.MobilePlanIRControllerError, match="mandatory controller"):
        entrypoint._build_essential_bundle(
            root=root,
            output_root=output,
            status="failed-closed",
            error=RuntimeError("test failure"),
        )


def test_essential_bundle_rejects_any_selected_weight_like_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint, root, output, _hashes = _essential_fixture(tmp_path, monkeypatch)
    weight = output / "execution/evaluation/model.safetensors"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"not-a-real-weight")
    monkeypatch.setattr(entrypoint, "_selected_execution_files", lambda _root: [weight])

    with pytest.raises(entrypoint.MobilePlanIRControllerError, match="model/optimizer weights"):
        entrypoint._build_essential_bundle(
            root=root,
            output_root=output,
            status="unit-test",
            error=None,
        )
