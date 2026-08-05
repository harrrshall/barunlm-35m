from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from barunlm.evaluation.presto import (
    PRESTO_PHENOMENON_TAXONOMY_VERSION,
    PRESTO_USER_REVISION_ALIAS_SET_VERSION,
    PRESTO_USER_REVISION_RAW_LABELS_V2,
    parse_presto_action,
    phenomenon_group,
    score_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_stage_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_barun_run_presto_stage", ROOT / "scripts" / "run_presto_stage.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load run_presto_stage.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


presto_stage = _load_stage_module()


def _prompt(*, contextual: bool) -> str:
    context = (
        {"contacts": ["Asha"], "lists": [], "notes": []}
        if contextual
        else {"contacts": [], "lists": [], "notes": []}
    )
    dialogue = [{"assistant": "Which note?", "user": "Find a note"}] if contextual else []
    return (
        "<bos><system>\nACTION_IR_V1\nPRESTO_CONTEXT_V1\n"
        f"CONTEXT {json.dumps(context, sort_keys=True, separators=(',', ':'))}\n"
        f"DIALOGUE {json.dumps(dialogue, sort_keys=True, separators=(',', ':'))}\n"
        "<user>\nrequest\n<assistant>\n"
    )


def _target(decision: str, tool: str | None = None) -> str:
    if decision == "ABSTAIN":
        return '{"decision":"ABSTAIN"}'
    return json.dumps(
        {
            "calls": [
                {
                    "args": {
                        "slots": [
                            {"name": "name", "value": {"text": "Asha"}},
                            {
                                "name": "detail",
                                "value": {
                                    "node": "personal_contact",
                                    "slots": [
                                        {
                                            "name": "person",
                                            "value": {"symbol": "infer_from_context"},
                                        }
                                    ],
                                },
                            },
                        ]
                    },
                    "tool": tool,
                }
            ],
            "decision": decision,
            "mode": "SINGLE",
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _manifest(
    sample_id: str,
    *,
    target: str,
    decision: str,
    root: str,
    phenomenon: str,
    contextual: bool,
    split: str = "dev",
) -> dict[str, object]:
    return {
        "schema_version": "barun-sft-example-v1",
        "id": sample_id,
        "prompt": _prompt(contextual=contextual),
        "target": target,
        "metadata": {
            "derived_split": split,
            "source_split": split,
            "policy_decision": decision,
            "root_intent": root,
            "linguistic_phenomenon": phenomenon,
        },
    }


def _prediction(sample_id: str, raw: str | None) -> dict[str, object]:
    return {
        "id": sample_id,
        "prediction_raw": raw,
        "truncated": False,
        "generation_failure": None,
        "prompt_tokens": 10,
        "generated_tokens": 10,
    }


def _train_row(
    sample_id: str, *, decision: str, phenomenon: str, contextual: bool
) -> dict[str, object]:
    tool = "create_note" if decision == "CONFIRM" else "get_note"
    root = "Create_note" if decision == "CONFIRM" else "Get_note"
    if decision == "ABSTAIN":
        tool = None
        root = "Other"
    return _manifest(
        sample_id,
        target=_target(decision, tool),
        decision=decision,
        root=root,
        phenomenon=phenomenon,
        contextual=contextual,
        split="train",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_presto_scorer_validates_recursive_slots_and_reports_buckets() -> None:
    call = _target("CALL", "get_note")
    confirm = _target("CONFIRM", "create_note")
    abstain = _target("ABSTAIN")
    manifests = [
        _manifest(
            "simple",
            target=call,
            decision="CALL",
            root="Get_note",
            phenomenon="none",
            contextual=False,
        ),
        _manifest(
            "revision",
            target=confirm,
            decision="CONFIRM",
            root="Create_note",
            phenomenon="within-turn-correction",
            contextual=True,
        ),
        _manifest(
            "oos",
            target=abstain,
            decision="ABSTAIN",
            root="Other",
            phenomenon="disfluency",
            contextual=False,
        ),
    ]
    predictions = [
        _prediction("simple", call),
        _prediction("revision", confirm),
        _prediction("oos", abstain),
    ]

    aggregate, samples = score_rows(manifests, predictions)

    assert parse_presto_action(call).canonical_json() == call
    assert aggregate["ast_exact_match"] == {"numerator": 3, "denominator": 3, "value": 1.0}
    assert aggregate["abstention"]["f1"] == 1.0
    assert aggregate["confirmation_exact"]["value"] == 1.0
    assert aggregate["headline_phenomenon_buckets"]["revision"]["count"] == 1
    assert aggregate["headline_phenomenon_buckets"]["disfluency"]["count"] == 1
    assert aggregate["contextual"]["count"] == 1
    assert len(samples) == 3


def test_presto_primary_taxonomy_maps_full_user_revision_family() -> None:
    assert PRESTO_PHENOMENON_TAXONOMY_VERSION == "barun-presto-phenomenon-taxonomy-v2"
    assert PRESTO_USER_REVISION_ALIAS_SET_VERSION == "barun-presto-user-revision-raw-aliases-v2"
    assert PRESTO_USER_REVISION_RAW_LABELS_V2 == {
        "cancel-action",
        "correct-action",
        "correct-argument",
        "within-turn-correction",
    }
    assert all(
        phenomenon_group(label) == "revision" for label in PRESTO_USER_REVISION_RAW_LABELS_V2
    )
    assert phenomenon_group("WITHIN TURN CORRECTION") == "revision"
    assert phenomenon_group("correct_argument") == "revision"
    assert phenomenon_group("code-mixing") == "other"


@pytest.mark.parametrize("phenomenon", sorted(PRESTO_USER_REVISION_RAW_LABELS_V2))
def test_focus_categories_include_each_raw_revision_tag(phenomenon: str) -> None:
    row = _train_row(
        f"revision-{phenomenon}",
        decision="CALL",
        phenomenon=phenomenon,
        contextual=False,
    )

    assert presto_stage._focus_categories(row) == ("revision",)


def test_presto_scorer_marks_wrong_call_policy_invalid_and_unsafe() -> None:
    gold = _target("CONFIRM", "create_note")
    unsafe = _target("CALL", "create_note")
    manifest = _manifest(
        "unsafe",
        target=gold,
        decision="CONFIRM",
        root="Create_note",
        phenomenon="revision",
        contextual=True,
    )

    aggregate, samples = score_rows([manifest], [_prediction("unsafe", unsafe)])

    assert samples[0]["parse_valid"] is True
    assert samples[0]["schema_valid"] is False
    assert samples[0]["prediction_error_code"] == "presto_policy_decision"
    assert samples[0]["false_call_on_gate"] is True
    assert aggregate["false_call_on_gate"]["value"] == 1.0


def test_presto_scorer_counts_parse_valid_unknown_tool_call_as_false_call() -> None:
    gold = _target("ABSTAIN")
    unknown_tool_call = (
        '{"calls":[{"args":{"slots":[]},"tool":"unknown_tool"}],"decision":"CALL","mode":"SINGLE"}'
    )
    manifest = _manifest(
        "unknown-call",
        target=gold,
        decision="ABSTAIN",
        root="Other",
        phenomenon="none",
        contextual=False,
    )

    aggregate, samples = score_rows([manifest], [_prediction("unknown-call", unknown_tool_call)])

    assert samples[0]["parse_valid"] is True
    assert samples[0]["schema_valid"] is False
    assert samples[0]["prediction_error_code"] == "unknown_tool"
    assert samples[0]["false_call_on_gate"] is True
    assert aggregate["false_call_on_gate"] == {
        "numerator": 1,
        "denominator": 1,
        "value": 1.0,
    }


def test_frozen_recipe_has_one_full_parameter_recipe_and_exact_proposed_input() -> None:
    recipe = presto_stage._load_recipe()

    assert recipe["optimization"]["epochs"] == 1
    assert recipe["optimization"]["learning_rate"] == 5e-5
    assert recipe["train_view"]["kind"] == ("dev_deduplicated_train_plus_stratified_focus_replay")
    assert recipe["train_view"]["categories"] == list(presto_stage.EXPECTED_CATEGORIES)
    assert recipe["train_view"]["development_rows_read_for_focus_selection"] == 0
    assert recipe["train_view"]["development_rows_read_for_exact_duplicate_exclusion"] == 14288
    assert recipe["train_view"]["official_test_rows_read"] == 0
    assert recipe["proposed_input_checkpoint"] == {
        "local_dir": (
            "experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/"
            "arms/batch63/checkpoint"
        ),
        "model_sha256": "fdb95ccf58a095e0d321be998924318b35ee59a334f6dd97d8726d2cf80021d3",
        "config_sha256": "9b3a1d71baa95a198744d250f9629231738d942570b8685c44307fd83dd33565",
        "tokenizer_sha256": ("70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6"),
    }


def test_focus_view_is_train_only_deterministic_and_stratified(tmp_path: Path) -> None:
    rows = [
        _train_row("abstain", decision="ABSTAIN", phenomenon="none", contextual=False),
        _train_row("confirm", decision="CONFIRM", phenomenon="none", contextual=False),
        _train_row(
            "revision",
            decision="CALL",
            phenomenon="within-turn-correction",
            contextual=False,
        ),
        _train_row("disfluency", decision="CALL", phenomenon="disfluency", contextual=False),
        _train_row("contextual", decision="CALL", phenomenon="none", contextual=True),
    ]
    source = tmp_path / "train.jsonl"
    _write_jsonl(source, rows)
    source_hash = presto_stage.sha256_file(source)
    dev = tmp_path / "dev.jsonl"
    dev_rows = [
        _manifest(
            "dev-only",
            target=_target("ABSTAIN"),
            decision="ABSTAIN",
            root="Other",
            phenomenon="none",
            contextual=True,
        )
    ]
    _write_jsonl(dev, dev_rows)
    dev_hash = presto_stage.sha256_file(dev)
    outputs: list[bytes] = []
    for suffix in ("a", "b"):
        destination = tmp_path / f"focus-{suffix}.jsonl"
        output_hash, audit = presto_stage._materialize_focus_view(
            source=source,
            source_sha256=source_hash,
            dev_source=dev,
            dev_sha256=dev_hash,
            destination=destination,
            audit_path=tmp_path / f"focus-{suffix}-audit.json",
            seed=17,
            maximum_per_category=1,
        )
        output_rows = [json.loads(line) for line in destination.read_text().splitlines()]
        outputs.append(destination.read_bytes())
        assert output_hash == presto_stage.sha256_file(destination)
        assert len(output_rows) == 10
        assert len({row["id"] for row in output_rows}) == 10
        assert audit["replay_counts"] == {
            category: 1 for category in presto_stage.EXPECTED_CATEGORIES
        }
        assert audit["development_rows_read_for_exact_duplicate_exclusion"] == 1
        assert audit["development_rows_read_for_focus_selection"] == 0
        assert audit["official_test_rows_read"] == 0
        assert all(row["metadata"]["source_split"] == "train" for row in output_rows)
        assert all(row["metadata"]["derived_split"] == "train" for row in output_rows)
    assert outputs[0] == outputs[1]


def test_focus_view_excludes_exact_train_development_content(tmp_path: Path) -> None:
    duplicate = _train_row(
        "train-duplicate", decision="CONFIRM", phenomenon="revision", contextual=True
    )
    retained = _train_row("train-retained", decision="ABSTAIN", phenomenon="none", contextual=False)
    source = tmp_path / "train.jsonl"
    _write_jsonl(source, [duplicate, retained])
    dev_duplicate = dict(duplicate)
    dev_duplicate["id"] = "dev-duplicate"
    dev_duplicate["metadata"] = {
        **duplicate["metadata"],
        "source_split": "dev",
        "derived_split": "dev",
    }
    dev = tmp_path / "dev.jsonl"
    _write_jsonl(dev, [dev_duplicate])

    destination = tmp_path / "focus.jsonl"
    _, audit = presto_stage._materialize_focus_view(
        source=source,
        source_sha256=presto_stage.sha256_file(source),
        dev_source=dev,
        dev_sha256=presto_stage.sha256_file(dev),
        destination=destination,
        audit_path=tmp_path / "audit.json",
        seed=17,
        maximum_per_category=1,
    )

    output = [json.loads(line) for line in destination.read_text().splitlines()]
    assert audit["exact_prompt_target_train_rows_excluded"] == 1
    assert audit["excluded_train_ids"] == ["train-duplicate"]
    assert audit["rows_after_exact_dev_exclusion"] == 1
    assert all(row["id"] != "train-duplicate" for row in output)


def test_official_test_firewall_requires_zero_member_opens() -> None:
    expected = {
        "archive_payload_hashing_only": True,
        "bytes_read_from_sensitive_members": 0,
        "combined_dataset_member_opened": False,
        "labels_parsed": False,
        "materialized": False,
        "member_opened": False,
        "opaque_unparsed": True,
        "prompts_parsed": False,
        "rows": 194118,
        "sensitive_members_opened": [],
        "targets_parsed": False,
        "test_partition_members_opened": False,
    }
    audit = {
        "official_test": expected,
        "source": {
            "members": {
                "presto_dataset.jsonl": {
                    "runtime_open_count": 0,
                    "runtime_bytes_read": 0,
                },
                "presto_test.jsonl": {
                    "runtime_open_count": 0,
                    "runtime_bytes_read": 0,
                },
            }
        },
    }
    prepared = SimpleNamespace(
        hashes={"train": "a" * 64, "dev": "b" * 64, "audit": "c" * 64},
        official_test_rows=194118,
    )

    evidence = presto_stage._official_test_firewall(prepared, audit)
    assert evidence["official_test_rows_opaque_unparsed"] == 194118

    audit["source"]["members"]["presto_test.jsonl"]["runtime_open_count"] = 1
    with pytest.raises(RuntimeError, match="sealed PRESTO member was read"):
        presto_stage._official_test_firewall(prepared, audit)


def test_training_config_uses_exact_local_input_and_jarvis_id(tmp_path: Path) -> None:
    args = presto_stage.build_parser().parse_args(
        [
            "--run-id",
            "20260803-2000-presto-stage-s17",
            "--jarvis-machine-id",
            "464001",
            "--input-checkpoint-dir",
            str(tmp_path / "input"),
            "--input-model-sha256",
            "a" * 64,
            "--input-config-sha256",
            "b" * 64,
            "--input-tokenizer-sha256",
            "c" * 64,
        ]
    )
    hashes = presto_stage._checkpoint_hashes(args)
    config = presto_stage._training_config(
        args=args,
        recipe=presto_stage._load_recipe(),
        input_hashes=hashes,
        train_manifest=tmp_path / "train.jsonl",
        train_sha256="d" * 64,
        dev_manifest=tmp_path / "dev.jsonl",
        dev_sha256="e" * 64,
        export=tmp_path / "export",
    )

    assert config["base_checkpoint"] == {
        "source": "local",
        "local_dir": str((tmp_path / "input").resolve()),
        "expected_sha256": hashes,
    }
    assert config["execution"]["jarvis_resource_id"] == "464001"
    assert config["optimization"] == presto_stage._load_recipe()["optimization"]


def test_cli_requires_exact_checkpoint_hashes_and_positive_jarvis_id(tmp_path: Path) -> None:
    parser = presto_stage.build_parser()
    required = [
        "--run-id",
        "20260803-2000-presto-stage-s17",
        "--jarvis-machine-id",
        "464001",
        "--input-checkpoint-dir",
        str(tmp_path),
        "--input-model-sha256",
        "a" * 64,
        "--input-config-sha256",
        "b" * 64,
        "--input-tokenizer-sha256",
        "c" * 64,
    ]
    assert parser.parse_args(required).jarvis_machine_id == 464001
    with pytest.raises(SystemExit):
        parser.parse_args(required[:-2])
    invalid = list(required)
    invalid[3] = "0"
    with pytest.raises(SystemExit):
        parser.parse_args(invalid)


def test_run_root_is_claimed_before_cuda_and_duplicate_is_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "20260803-2000-presto-stage-s17"

    def inspect_then_fail() -> dict[str, object]:
        assert (tmp_path / run_id / "export" / "essential").is_dir()
        raise RuntimeError("stop after atomic claim")

    monkeypatch.setattr(presto_stage, "_cuda_determinism_preflight", inspect_then_fail)
    with pytest.raises(RuntimeError, match="atomic claim"):
        presto_stage.run(SimpleNamespace(artifact_root=tmp_path, run_id=run_id))

    called = False

    def forbidden() -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(presto_stage, "_cuda_determinism_preflight", forbidden)
    with pytest.raises(presto_stage.ExistingPrestoRunError, match="refusing to overwrite"):
        presto_stage.run(SimpleNamespace(artifact_root=tmp_path, run_id=run_id))
    assert called is False


def test_compact_checkpoint_export_never_copies_optimizer(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in ("model.safetensors", "barun_config.json", "tokenizer.json"):
        (source / name).write_bytes(name.encode())
    (source / "optimizer.pt").write_bytes(b"large optimizer")
    destination = tmp_path / "essential" / "checkpoint"

    hashes = presto_stage._export_checkpoint(
        source,
        destination,
        run_id="20260803-2000-presto-stage-s17",
        input_hashes={
            "model.safetensors": "a" * 64,
            "barun_config.json": "b" * 64,
            "tokenizer.json": "c" * 64,
        },
    )

    assert set(hashes) == {"model.safetensors", "barun_config.json", "tokenizer.json"}
    assert not list(destination.rglob("optimizer.pt"))


def test_determinism_configuration_precedes_torch_import_in_fresh_process() -> None:
    code = """
import json
import os
import runpy
import sys

assert "torch" not in sys.modules
stage = runpy.run_path("scripts/run_presto_stage.py", run_name="_presto_stage_test")
evidence = stage["_cuda_determinism_preflight"]()
print(json.dumps({
    "config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    "preimport": stage["_CUBLAS_CONFIG_SET_BEFORE_TORCH_IMPORT"],
    "cuda_initialized": stage["torch"].cuda.is_initialized(),
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
