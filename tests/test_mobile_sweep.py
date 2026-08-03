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


def _load_sweep_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_barun_run_mobile_sweep", ROOT / "scripts" / "run_mobile_sweep.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load run_mobile_sweep.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mobile_sweep = _load_sweep_module()


def _row(example_id: str, call_names: list[str]) -> dict[str, object]:
    return {
        "schema_version": "barun-sft-example-v1",
        "id": example_id,
        "prompt": f"prompt {example_id}",
        "target": '{"calls":[],"decision":"CALL","mode":"SINGLE"}',
        "metadata": {
            "call_names": call_names,
            "derived_split": "train",
            "source_split": "train",
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_sweep_sets_determinism_before_torch_in_fresh_process() -> None:
    code = """
import json
import os
import runpy
import sys

assert "torch" not in sys.modules
sweep = runpy.run_path("scripts/run_mobile_sweep.py", run_name="_sweep_test")
evidence = sweep["_cuda_determinism_preflight"]()
print(json.dumps({
    "config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    "preimport": sweep["_CUBLAS_CONFIG_SET_BEFORE_TORCH_IMPORT"],
    "cuda_initialized": sweep["torch"].cuda.is_initialized(),
    "backends": evidence["sdpa_backends"],
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
    assert evidence == {
        "config": ":4096:8",
        "preimport": True,
        "cuda_initialized": False,
        "backends": {
            "flash": False,
            "memory_efficient": False,
            "math": True,
            "cudnn": False,
        },
    }


def test_frozen_preregistration_has_exactly_two_remaining_arms() -> None:
    preregistration = mobile_sweep._load_preregistration()

    assert preregistration["dev_trial_budget"] == {
        "total": 3,
        "consumed_by_reference": 1,
        "remaining_arms": 2,
    }
    batch63, hardmix70 = preregistration["arms"]
    assert batch63["arm_id"] == "batch63"
    assert batch63["train_view"] == {"kind": "uniform"}
    assert batch63["optimization"]["batch_size"] == 63
    assert batch63["optimization"]["warmup_steps"] == 12
    assert batch63["optimization"]["learning_rate"] == 1e-4
    assert hardmix70["arm_id"] == "hardmix70"
    assert hardmix70["train_view"]["hard_fraction"] == 0.7
    assert hardmix70["optimization"]["batch_size"] == 127
    assert hardmix70["optimization"]["warmup_steps"] == 6
    assert hardmix70["optimization"]["learning_rate"] == 1e-4
    assert all(arm["optimization"]["seed"] == 17 for arm in (batch63, hardmix70))
    assert all(arm["optimization"]["epochs"] == 1 for arm in (batch63, hardmix70))


def test_hard_mix_is_fixed_size_deterministic_and_never_reads_dev_or_eval(
    tmp_path: Path,
) -> None:
    rows = [
        _row("calendar", ["create_calendar_event"]),
        _row("map", ["show_map"]),
        _row("multi", ["send_email", "create_contact"]),
        _row("hard-4", ["create_calendar_event", "show_map"]),
        _row("easy-1", ["send_email"]),
        _row("easy-2", ["create_contact"]),
        _row("easy-3", ["open_wifi_settings"]),
        _row("easy-4", ["turn_on_flashlight"]),
        _row("easy-5", ["turn_off_flashlight"]),
        _row("easy-6", ["send_email"]),
    ]
    source = tmp_path / "train.jsonl"
    _write_jsonl(source, rows)
    source_hash = mobile_sweep.sha256_file(source)

    outputs = []
    for suffix in ("a", "b"):
        destination = tmp_path / f"hardmix-{suffix}.jsonl"
        output_hash, audit = mobile_sweep._materialize_hard_mix(
            source=source,
            source_sha256=source_hash,
            destination=destination,
            audit_path=tmp_path / f"audit-{suffix}.json",
            hard_fraction=0.7,
            seed=17,
        )
        materialized = [json.loads(line) for line in destination.read_text().splitlines()]
        outputs.append(destination.read_bytes())
        assert output_hash == mobile_sweep.sha256_file(destination)
        assert len(materialized) == len(rows)
        assert len({row["id"] for row in materialized}) == len(rows)
        assert (
            sum(row["metadata"]["sweep_curriculum"]["bucket"] == "hard" for row in materialized)
            == 7
        )
        assert all(row["metadata"]["source_split"] == "train" for row in materialized)
        assert all(row["metadata"]["derived_split"] == "train" for row in materialized)
        assert audit["source_rows"] == audit["output_rows"] == 10
        assert audit["official_eval_rows_read"] == 0
        assert audit["development_rows_read"] == 0
    assert outputs[0] == outputs[1]


def test_training_config_pins_machine_final_only_schedule_and_base(tmp_path: Path) -> None:
    preregistration = mobile_sweep._load_preregistration()
    args = mobile_sweep.build_parser().parse_args(
        [
            "--run-id",
            "20260803-1900-mobile-sweep-s17",
            "--jarvis-machine-id",
            "463999",
        ]
    )
    arm = preregistration["arms"][0]
    config = mobile_sweep._training_config(
        args=args,
        arm=arm,
        arm_dir=tmp_path / "arm",
        train_manifest=tmp_path / "train.jsonl",
        train_sha256="a" * 64,
        dev_manifest=tmp_path / "dev.jsonl",
        dev_sha256="b" * 64,
    )

    assert config["run_id"] == "20260803-1900-mob-batch63-s17"
    assert config["base_checkpoint"]["revision"] == mobile_sweep.BASE_REVISION
    assert config["base_checkpoint"]["expected_sha256"] == mobile_sweep.BASE_HASHES
    assert config["execution"]["jarvis_resource_id"] == "463999"
    assert config["optimization"]["eval_every_steps"] == 1_000_000
    assert config["optimization"]["save_every_steps"] == 50


def test_selection_uses_only_ast_exact_with_fixed_margin_and_tie_rule() -> None:
    preregistration = mobile_sweep._load_preregistration()
    promoted = mobile_sweep._select_by_preregistered_metric(
        preregistration,
        {
            "batch63": {"numerator": 605, "denominator": 756, "value": 605 / 756},
            "hardmix70": {"numerator": 590, "denominator": 756, "value": 590 / 756},
        },
    )
    assert promoted["selected"] == "batch63"
    assert promoted["decision"] == "promote_followup_arm"
    assert promoted["non_selection_metrics_consulted"] == []

    below_margin = mobile_sweep._select_by_preregistered_metric(
        preregistration,
        {
            "batch63": {"numerator": 583, "denominator": 756, "value": 583 / 756},
            "hardmix70": {"numerator": 580, "denominator": 756, "value": 580 / 756},
        },
    )
    assert below_margin["selected"] == "reference"
    assert below_margin["decision"] == "inconclusive_below_practical_margin_keep_reference"

    tied = mobile_sweep._select_by_preregistered_metric(
        preregistration,
        {
            "batch63": {"numerator": 600, "denominator": 756, "value": 600 / 756},
            "hardmix70": {"numerator": 600, "denominator": 756, "value": 600 / 756},
        },
    )
    assert tied["selected"] == "reference"
    assert tied["decision"] == "inconclusive_tie_keep_reference"


def test_sweep_cli_requires_positive_machine_id() -> None:
    parser = mobile_sweep.build_parser()
    args = parser.parse_args(
        [
            "--run-id",
            "20260803-1900-mobile-sweep-s17",
            "--jarvis-machine-id",
            "463999",
        ]
    )
    assert args.jarvis_machine_id == 463999
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-id", "20260803-1900-mobile-sweep-s17"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--run-id",
                "20260803-1900-mobile-sweep-s17",
                "--jarvis-machine-id",
                "0",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--run-id",
                "../../not-a-run",
                "--jarvis-machine-id",
                "463999",
            ]
        )


def test_duplicate_sweep_refuses_before_cuda_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "20260803-1900-mobile-sweep-s17"
    run_root = tmp_path / run_id
    run_root.mkdir()
    sentinel = run_root / "sentinel.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    called = False

    def forbidden_preflight() -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("duplicate sweep must not call CUDA preflight")

    monkeypatch.setattr(mobile_sweep, "_cuda_determinism_preflight", forbidden_preflight)
    args = SimpleNamespace(artifact_root=tmp_path, run_id=run_id)

    with pytest.raises(mobile_sweep.ExistingSweepError, match="refusing to overwrite"):
        mobile_sweep.run(args)

    assert called is False
    assert list(run_root.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


def test_failed_repository_tests_reach_progressive_essential_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    export = tmp_path / "export"
    essential = export / "essential"
    essential.mkdir(parents=True)
    failed = SimpleNamespace(returncode=1, stdout="one failed\n", stderr="traceback\n")
    monkeypatch.setattr(mobile_sweep.subprocess, "run", lambda *args, **kwargs: failed)

    with pytest.raises(RuntimeError, match="repository tests failed"):
        mobile_sweep._run_tests(export, essential=essential)

    expected = "one failed\ntraceback\n"
    assert (export / "repository-tests.log").read_text(encoding="utf-8") == expected
    assert (essential / "repository-tests.log").read_text(encoding="utf-8") == expected
    captured = capsys.readouterr()
    assert captured.err == expected


def test_essential_arm_bundle_has_inference_checkpoint_and_no_optimizer(
    tmp_path: Path,
) -> None:
    arm_dir = tmp_path / "arm"
    checkpoint = (
        arm_dir / "training" / "20260803-1900-mob-batch63-s17" / "checkpoints" / "step-00000126"
    )
    checkpoint.mkdir(parents=True)
    for name in ("model.safetensors", "barun_config.json", "tokenizer.json"):
        (checkpoint / name).write_bytes(f"content:{name}".encode())
    (checkpoint / "optimizer.pt").write_bytes(b"must stay remote")
    training_run = checkpoint.parent.parent
    for name in ("metrics.jsonl", "run_manifest.json", "summary.json", "best_checkpoint.json"):
        (training_run / name).write_text(f'{{"artifact":"{name}"}}\n', encoding="utf-8")
    mobile_sweep._write_json(arm_dir / "training-config.json", {"arm_id": "batch63"})
    final_eval = arm_dir / "final-eval"
    scores = final_eval / "scores"
    scores.mkdir(parents=True)
    (final_eval / "predictions.jsonl").write_text('{"id":"x"}\n', encoding="utf-8")
    (final_eval / "predictions.jsonl.manifest.json").write_text("{}\n", encoding="utf-8")
    (scores / "aggregate.json").write_text("{}\n", encoding="utf-8")
    (scores / "sample_scores.jsonl").write_text('{"id":"x"}\n', encoding="utf-8")
    mobile_sweep._write_json(arm_dir / "result.json", {"arm_id": "batch63"})
    essential = tmp_path / "essential"
    essential.mkdir()

    mobile_sweep._export_arm_essential(
        arm_id="batch63",
        arm_dir=arm_dir,
        final_checkpoint=checkpoint,
        essential=essential,
    )

    bundled = essential / "arms" / "batch63"
    assert (bundled / "checkpoint" / "model.safetensors").is_file()
    assert (bundled / "final-eval" / "predictions.jsonl").is_file()
    assert (bundled / "final-eval" / "scores" / "sample_scores.jsonl").is_file()
    assert (bundled / "result.json").is_file()
    assert (bundled / "training-config.json").is_file()
    assert (bundled / "training" / "metrics.jsonl").is_file()
    assert (bundled / "training" / "run_manifest.json").is_file()
    assert (bundled / "training" / "summary.json").is_file()
    assert (bundled / "training" / "best_checkpoint.json").is_file()
    assert (bundled / "bundle-manifest.json").is_file()
    assert not list(bundled.rglob("optimizer.pt"))
    assert not list((bundled / "training").rglob("checkpoint*"))
    manifest = json.loads(
        (bundled / "checkpoint" / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["file_sha256"]) == {
        "model.safetensors",
        "barun_config.json",
        "tokenizer.json",
    }


def test_shared_essential_includes_test_log_and_verified_adapter_audit(
    tmp_path: Path,
) -> None:
    export = tmp_path / "export"
    essential = export / "essential"
    data = export / "data"
    essential.mkdir(parents=True)
    data.mkdir()
    (export / "repository-tests.log").write_text("all tests passed\n", encoding="utf-8")
    audit = data / "audit.json"
    audit.write_text(
        '{"official_eval":{"opaque_unparsed":true,"prompts_parsed":false}}\n',
        encoding="utf-8",
    )

    mobile_sweep._stage_shared_essential(
        export=export,
        essential=essential,
        audit_path=audit,
        audit_sha256=mobile_sweep.sha256_file(audit),
    )

    assert (essential / "repository-tests.log").read_text() == "all tests passed\n"
    assert (essential / "data" / "audit.json").read_bytes() == audit.read_bytes()


def test_official_eval_firewall_accepts_only_train_dev_audit_hashes() -> None:
    expected = {
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
    prepared = SimpleNamespace(hashes={"train": "a", "dev": "b", "audit": "c"})
    assert mobile_sweep._official_eval_firewall(prepared, {"official_eval": expected}) == expected

    prepared_with_eval = SimpleNamespace(
        hashes={"train": "a", "dev": "b", "audit": "c", "final_eval": "d"}
    )
    with pytest.raises(RuntimeError, match="train/dev/audit"):
        mobile_sweep._official_eval_firewall(prepared_with_eval, {"official_eval": expected})
