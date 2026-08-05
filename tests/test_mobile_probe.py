from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_probe_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_barun_run_mobile_probe", ROOT / "scripts" / "run_mobile_probe.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load run_mobile_probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mobile_probe = _load_probe_module()


def test_cublas_configuration_precedes_torch_import_in_fresh_process() -> None:
    code = """
import json
import os
import runpy
import sys

assert "torch" not in sys.modules
probe = runpy.run_path("scripts/run_mobile_probe.py", run_name="_barun_probe_test")
preflight = probe["_cuda_determinism_preflight"]()

print(json.dumps({
    "config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    "set_before_torch": probe["_CUBLAS_CONFIG_SET_BEFORE_TORCH_IMPORT"],
    "cuda_initialized": probe["torch"].cuda.is_initialized(),
    "sdpa_backends": preflight["sdpa_backends"],
}))
"""
    environment = dict(os.environ)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    evidence = json.loads(result.stdout)
    assert evidence["config"] == ":4096:8"
    assert evidence["set_before_torch"] is True
    assert evidence["cuda_initialized"] is False
    assert evidence["sdpa_backends"] == {
        "flash": False,
        "memory_efficient": False,
        "math": True,
        "cudnn": False,
    }


def test_determinism_preflight_fails_if_configuration_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    with pytest.raises(RuntimeError, match="changed after import"):
        mobile_probe._cuda_determinism_preflight()


def test_environment_records_determinism_preflight() -> None:
    evidence = mobile_probe._cuda_determinism_preflight()
    environment = mobile_probe._environment(evidence)

    assert environment["determinism_preflight"] == {
        "cublas_workspace_config": ":4096:8",
        "configured_before_torch_import": True,
        "cuda_initialized_before_preflight": False,
        "sdpa_backends": {
            "flash": False,
            "memory_efficient": False,
            "math": True,
            "cudnn": False,
        },
    }


def test_math_sdpa_preflight_fails_if_backend_state_cannot_be_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mobile_probe.torch.backends.cuda,
        "flash_sdp_enabled",
        lambda: True,
    )

    with pytest.raises(RuntimeError, match="failed to force flash"):
        mobile_probe._force_math_sdpa()


def test_official_eval_firewall_requires_opaque_unparsed_evidence() -> None:
    prepared = SimpleNamespace(hashes={"train": "a" * 64, "dev": "b" * 64})
    audit = {
        "official_eval": {
            "opaque_unparsed": True,
            "prompts_parsed": False,
            "labels_parsed": False,
            "materialized": False,
            "lengths_computed": False,
            "overlaps_computed": False,
            "summaries_computed": False,
            "targets_parsed": False,
            "tool_schemas_parsed": False,
        }
    }

    assert mobile_probe._official_eval_firewall(prepared, audit) == audit["official_eval"]


def test_official_eval_firewall_rejects_label_processing_or_hashes() -> None:
    parsed_audit = {
        "official_eval": {
            "opaque_unparsed": False,
            "prompts_parsed": True,
            "labels_parsed": True,
            "materialized": False,
            "lengths_computed": False,
            "overlaps_computed": False,
            "summaries_computed": False,
            "targets_parsed": False,
            "tool_schemas_parsed": False,
        }
    }
    with pytest.raises(RuntimeError, match="remained opaque"):
        mobile_probe._official_eval_firewall(
            SimpleNamespace(hashes={"train": "a" * 64, "dev": "b" * 64}),
            parsed_audit,
        )

    opaque_audit = {
        "official_eval": {
            "opaque_unparsed": True,
            "prompts_parsed": False,
            "labels_parsed": False,
            "materialized": False,
            "lengths_computed": False,
            "overlaps_computed": False,
            "summaries_computed": False,
            "targets_parsed": False,
            "tool_schemas_parsed": False,
        }
    }
    with pytest.raises(RuntimeError, match="content hash"):
        mobile_probe._official_eval_firewall(
            SimpleNamespace(hashes={"train": "a" * 64, "dev": "b" * 64, "final_eval": "c" * 64}),
            opaque_audit,
        )


def test_stage_prepared_data_exports_only_verified_train_dev_and_audit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    train = source / "train-private-name.jsonl"
    dev = source / "dev-private-name.jsonl"
    audit = source / "audit-private-name.json"
    raw_combined = source / "dataset.jsonl"
    train.write_text('{"id":"train"}\n', encoding="utf-8")
    dev.write_text('{"id":"dev"}\n', encoding="utf-8")
    audit.write_text('{"official_eval":{"opaque_unparsed":true}}\n', encoding="utf-8")
    raw_combined.write_text("must-not-export\n", encoding="utf-8")
    prepared = SimpleNamespace(
        train_manifest=train,
        dev_manifest=dev,
        audit_path=audit,
        hashes={
            "train": mobile_probe.sha256_file(train),
            "dev": mobile_probe.sha256_file(dev),
            "audit": mobile_probe.sha256_file(audit),
        },
    )
    exported = tmp_path / "exported"

    mobile_probe._stage_prepared_data(prepared, exported)

    assert {path.name for path in exported.iterdir()} == {
        "train.jsonl",
        "dev.jsonl",
        "audit.json",
    }
    assert not (exported / raw_combined.name).exists()


def test_probe_cli_requires_positive_jarvis_machine_id() -> None:
    parser = mobile_probe.build_parser()
    args = parser.parse_args(
        [
            "--run-id",
            "20260803-1723-mobile-probe-s17",
            "--jarvis-machine-id",
            "463554",
        ]
    )
    assert args.jarvis_machine_id == 463554

    with pytest.raises(SystemExit):
        parser.parse_args(["--run-id", "20260803-1723-mobile-probe-s17"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--run-id",
                "20260803-1723-mobile-probe-s17",
                "--jarvis-machine-id",
                "0",
            ]
        )


def test_training_config_records_machine_and_exported_data_paths(tmp_path: Path) -> None:
    args = mobile_probe.build_parser().parse_args(
        [
            "--run-id",
            "20260803-1723-mobile-probe-s17",
            "--jarvis-machine-id",
            "463554",
            "--artifact-root",
            str(tmp_path),
        ]
    )
    export = tmp_path / args.run_id / "export"
    prepared_data = export / "data"
    prepared = SimpleNamespace(hashes={"train": "a" * 64, "dev": "b" * 64})

    config = mobile_probe._training_config(
        args=args,
        workspace=export,
        prepared=prepared,
        prepared_data=prepared_data,
    )

    assert config["execution"]["jarvis_resource_id"] == "463554"
    assert config["output_root"] == str(export / "training")
    assert config["optimization"]["save_every_steps"] == 50
    assert config["data"]["train_manifest"] == str(prepared_data / "train.jsonl")
    assert config["data"]["dev_manifest"] == str(prepared_data / "dev.jsonl")


def test_main_refuses_existing_run_without_modifying_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "20260803-1723-mobile-probe-s17"
    export = tmp_path / run_id / "export"
    export.mkdir(parents=True)
    sentinel = export / "sentinel.txt"
    sentinel.write_text("preserve me\n", encoding="utf-8")
    before = {path.relative_to(export): path.read_bytes() for path in export.rglob("*")}
    monkeypatch.setattr(
        mobile_probe,
        "_cuda_determinism_preflight",
        lambda: (_ for _ in ()).throw(AssertionError("preflight must not run for a duplicate ID")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mobile_probe.py",
            "--run-id",
            run_id,
            "--jarvis-machine-id",
            "463554",
            "--artifact-root",
            str(tmp_path),
        ],
    )

    with pytest.raises(mobile_probe.ExistingRunError, match="refusing to overwrite"):
        mobile_probe.main()

    after = {path.relative_to(export): path.read_bytes() for path in export.rglob("*")}
    assert after == before
