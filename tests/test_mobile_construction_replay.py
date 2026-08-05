from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import barunlm.datasets.mobile_construction_replay as replay

ROOT = Path(__file__).resolve().parents[1]
CONSTRUCTION = (
    ROOT
    / "experiments/runs/20260803-2353-mobile-temporal-counterfactual-s17"
    / "materialized-screening-v1/construction-train.jsonl"
)
EXCLUSION = (
    ROOT
    / "experiments/runs/20260804-0545-mobile-planir-construction-screen-s17"
    / "materialized-v1/exclusion.jsonl"
)
REQUIRES_REAL = pytest.mark.skipif(
    not CONSTRUCTION.is_file() or not EXCLUSION.is_file(),
    reason="the pinned construction replay inputs are absent from this checkout",
)

DIRECT_TARGET = (
    '{"calls":[{"args":{},"tool":"turn_on_flashlight"}],"decision":"CALL","mode":"SINGLE"}'
)


def test_replay_imports_shared_action_schema_without_importing_planir_module() -> None:
    source = ROOT / "src/barunlm/datasets/mobile_construction_replay.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "barunlm.evaluation.mobile_action_schemas" in imported
    assert not any("planir" in module for module in imported)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _line(payload: object) -> bytes:
    return (_canonical(payload) + "\n").encode("utf-8")


def _make_row(index: int, *, target: str = DIRECT_TARGET) -> dict[str, Any]:
    example_id = f"mobile-actions-synthetic-{index:05d}"
    prompt = f"ACTION_IR_V1\n<user>\nSynthetic request {index}\n<assistant>\n"
    metadata = {
        "adapter_schema_version": "barun-mobile-actions-adapter-v2",
        "cluster_id": f"synthetic-component-{index}",
        "dataset": "google/mobile-actions",
        "derived_split": "train",
        "family_id": f"synthetic-family-{index}",
        "prompt_contract_sha256": replay._PROMPT_CONTRACT_SHA256,
        "prompt_contract_version": "barun-action-prompt-v1",
        "prompt_sha256": _sha256(prompt.encode()),
        "source_revision": replay._SOURCE_REVISION,
        "source_split": "train",
        "target_sha256": _sha256(target.encode()),
    }
    return {
        "id": example_id,
        "metadata": metadata,
        "prompt": prompt,
        "schema_version": "barun-sft-example-v1",
        "target": target,
    }


def _fixture() -> tuple[bytes, bytes, replay._ReplayContract, list[dict[str, Any]]]:
    rows = [_make_row(index) for index in range(1, 4)]
    source_lines = [_line(row) for row in rows]
    source = b"".join(source_lines)
    excluded = rows[1]
    excluded_line = source_lines[1][:-1]
    excluded_record = {
        "code": "temporal_reference_mismatch",
        "component_id": excluded["metadata"]["cluster_id"],
        "excluded_from_arms": ["A", "B", "C"],
        "family_id": excluded["metadata"]["family_id"],
        "id": excluded["id"],
        "population": "screen",
        "schema_version": "barun-mobile-planir-screen-exclusion-v1",
        "singleton_component": True,
        "source_content_sha256": _sha256(excluded_line),
        "source_prompt_sha256": excluded["metadata"]["prompt_sha256"],
        "source_target_sha256": excluded["metadata"]["target_sha256"],
        "split_version": "barun-mobile-construction-component-screen-split-v1",
    }
    exclusion = _line(excluded_record)
    accepted = [rows[0], rows[2]]
    output = source_lines[0] + source_lines[2]
    source_ids = [str(row["id"]) for row in rows]
    output_ids = [str(row["id"]) for row in accepted]
    target_commitment = "".join(
        f"{index:06d}\t{row['id']}\t{row['metadata']['target_sha256']}\n"
        for index, row in enumerate(accepted, 1)
    ).encode()
    contract = replay._ReplayContract(
        source_rows=len(rows),
        source_sha256=_sha256(source),
        source_membership_sha256=replay._membership_sha256(source_ids),
        source_order_sha256=replay._ordered_sha256(source_ids),
        exclusion_ledger_rows=1,
        exclusion_ledger_sha256=_sha256(exclusion),
        excluded_id=str(excluded["id"]),
        excluded_code="temporal_reference_mismatch",
        excluded_content_sha256=_sha256(excluded_line),
        excluded_prompt_sha256=str(excluded["metadata"]["prompt_sha256"]),
        excluded_target_sha256=str(excluded["metadata"]["target_sha256"]),
        excluded_component_id=str(excluded["metadata"]["cluster_id"]),
        excluded_family_id=str(excluded["metadata"]["family_id"]),
        output_rows=len(accepted),
        output_sha256=_sha256(output),
        output_membership_sha256=replay._membership_sha256(output_ids),
        output_order_sha256=replay._ordered_sha256(output_ids),
        direct_target_commitment_sha256=_sha256(target_commitment),
    )
    return source, exclusion, contract, rows


@REQUIRES_REAL
def test_real_replay_is_exact_training_only_and_nonauthorizing() -> None:
    result = replay.materialize_direct_action_replay(CONSTRUCTION, EXCLUSION)

    assert result.manifest_sha256 == replay.OUTPUT_MANIFEST_SHA256
    assert result.receipt_sha256 == (
        "3df25123e1761c54fa0e0b01ca501fd1913ec65abece38683c2b3706cb7a15f5"
    )
    assert len(result.manifest_bytes.splitlines()) == replay.OUTPUT_ROWS == 5_744
    assert replay.EXCLUDED_ID.encode() not in result.manifest_bytes
    receipt = result.receipt_dict()
    assert receipt["training_only"] is True
    assert receipt["fresh_evaluation"] is False
    assert receipt["selection_eligible"] is False
    assert receipt["authorizes_model_access"] is False
    assert receipt["source"] == {
        "adapter_schema_version": "barun-mobile-actions-adapter-v2",
        "dataset": "google/mobile-actions",
        "membership_sha256": replay.SOURCE_MEMBERSHIP_SHA256,
        "ordered_membership_sha256": replay.SOURCE_ORDER_SHA256,
        "revision": replay._SOURCE_REVISION,
        "rows": replay.SOURCE_ROWS,
        "sha256": replay.SOURCE_MANIFEST_SHA256,
    }
    assert receipt["output"]["sha256"] == replay.OUTPUT_MANIFEST_SHA256
    assert receipt["output"]["membership_sha256"] == replay.OUTPUT_MEMBERSHIP_SHA256
    assert (
        receipt["output"]["direct_target_commitment_sha256"]
        == replay.DIRECT_TARGET_COMMITMENT_SHA256
    )
    assert receipt["output"]["direct_action_ir_targets"] is True
    assert receipt["output"]["target_bytes_preserved"] is True
    assert receipt["forbidden_v3_populations"]["selection_rows_in_output"] == 0
    assert receipt["forbidden_v3_populations"]["confirmation_rows_in_output"] == 0


def test_small_valid_replay_preserves_exact_row_bytes() -> None:
    source, exclusion, contract, rows = _fixture()
    result = replay._materialize_snapshots(source, exclusion, contract=contract)

    assert result.manifest_bytes == _line(rows[0]) + _line(rows[2])
    assert result.manifest_sha256 == contract.output_sha256
    assert result.receipt_dict()["output"]["rows"] == 2


def test_snapshots_and_decoded_receipts_are_detached_from_caller_mutation() -> None:
    source, exclusion, contract, _rows = _fixture()
    mutable_source = bytearray(source)
    mutable_exclusion = bytearray(exclusion)
    result = replay._materialize_snapshots(
        mutable_source,
        mutable_exclusion,
        contract=contract,
    )
    frozen_manifest = result.manifest_bytes
    frozen_receipt = result.receipt_bytes

    mutable_source[:] = b"destroyed"
    mutable_exclusion[:] = b"destroyed"
    decoded = result.receipt_dict()
    decoded["output"]["rows"] = 999
    decoded["training_only"] = False

    assert result.manifest_bytes == frozen_manifest
    assert result.receipt_bytes == frozen_receipt
    assert result.receipt_dict()["output"]["rows"] == 2
    assert result.receipt_dict()["training_only"] is True


@pytest.mark.parametrize("which", ["source", "exclusion"])
def test_any_unbound_input_byte_fails_before_materialization(which: str) -> None:
    source, exclusion, contract, _rows = _fixture()
    if which == "source":
        source = source[:-1] + b" "
        message = "source SHA-256"
    else:
        exclusion = exclusion[:-1] + b" "
        message = "exclusion ledger SHA-256"
    with pytest.raises(replay.DirectActionReplayError, match=message):
        replay._materialize_snapshots(source, exclusion, contract=contract)


def test_duplicate_source_id_fails_even_when_mutated_bytes_are_rebound() -> None:
    _source, exclusion, contract, rows = _fixture()
    rows[2]["id"] = rows[0]["id"]
    mutated = _line(rows[0]) + _line(rows[1]) + _line(rows[2])
    rebound = replace(contract, source_sha256=_sha256(mutated))

    with pytest.raises(replay.DirectActionReplayError, match="duplicate example IDs"):
        replay._materialize_snapshots(mutated, exclusion, contract=rebound)


def test_source_order_change_fails_independently_of_membership() -> None:
    _source, exclusion, contract, rows = _fixture()
    reordered = _line(rows[2]) + _line(rows[1]) + _line(rows[0])
    rebound = replace(contract, source_sha256=_sha256(reordered))

    with pytest.raises(replay.DirectActionReplayError, match="source ordering changed"):
        replay._materialize_snapshots(reordered, exclusion, contract=rebound)


def test_extra_source_row_fails_membership_after_count_and_bytes_are_rebound() -> None:
    source, exclusion, contract, _rows = _fixture()
    extra = _make_row(4)
    mutated = source + _line(extra)
    rebound = replace(
        contract,
        source_sha256=_sha256(mutated),
        source_rows=4,
    )

    with pytest.raises(replay.DirectActionReplayError, match="source membership changed"):
        replay._materialize_snapshots(mutated, exclusion, contract=rebound)


@pytest.mark.parametrize(
    ("derived_split", "role"),
    [("dev", None), ("confirmation", None), ("train", "selection")],
)
def test_v3_selection_or_confirmation_membership_is_rejected(
    derived_split: str,
    role: str | None,
) -> None:
    _source, exclusion, contract, rows = _fixture()
    rows[0]["metadata"]["derived_split"] = derived_split
    if role is not None:
        rows[0]["metadata"]["role"] = role
    mutated = _line(rows[0]) + _line(rows[1]) + _line(rows[2])
    rebound = replace(contract, source_sha256=_sha256(mutated))

    with pytest.raises(replay.DirectActionReplayError, match="forbidden|must equal 'train'"):
        replay._materialize_snapshots(mutated, exclusion, contract=rebound)


def test_planir_placeholder_target_is_rejected_even_when_hash_metadata_matches() -> None:
    _source, exclusion, contract, rows = _fixture()
    target = (
        '{"calls":[{"args":{"datetime":"@A:D00:T00","title":"x"},'
        '"tool":"create_calendar_event"}],"decision":"CALL","mode":"SINGLE"}'
    )
    rows[0]["target"] = target
    rows[0]["metadata"]["target_sha256"] = _sha256(target.encode())
    mutated = _line(rows[0]) + _line(rows[1]) + _line(rows[2])
    rebound = replace(contract, source_sha256=_sha256(mutated))

    with pytest.raises(replay.DirectActionReplayError, match="PlanIR placeholder"):
        replay._materialize_snapshots(mutated, exclusion, contract=rebound)


@pytest.mark.parametrize("field", ["mobile_planir_screen", "mobile_temporal_view"])
def test_planir_or_mbcf_metadata_is_rejected(field: str) -> None:
    _source, exclusion, contract, rows = _fixture()
    rows[0]["metadata"][field] = {"schema_version": "forbidden"}
    mutated = _line(rows[0]) + _line(rows[1]) + _line(rows[2])
    rebound = replace(contract, source_sha256=_sha256(mutated))

    with pytest.raises(replay.DirectActionReplayError, match="forbidden metadata field"):
        replay._materialize_snapshots(mutated, exclusion, contract=rebound)


def test_mbcf_derived_id_is_rejected_without_trusting_metadata() -> None:
    _source, exclusion, contract, rows = _fixture()
    rows[0]["id"] += "--mbcf-v1-deadbeef"
    mutated = _line(rows[0]) + _line(rows[1]) + _line(rows[2])
    rebound = replace(contract, source_sha256=_sha256(mutated))

    with pytest.raises(replay.DirectActionReplayError, match="MBCF derivative"):
        replay._materialize_snapshots(mutated, exclusion, contract=rebound)


def test_noncanonical_or_non_action_ir_target_is_rejected() -> None:
    _source, exclusion, contract, rows = _fixture()
    target = '{"decision":"ABSTAIN","schema_version":"grounded-plan-ir-v1"}'
    rows[0]["target"] = target
    rows[0]["metadata"]["target_sha256"] = _sha256(target.encode())
    mutated = _line(rows[0]) + _line(rows[1]) + _line(rows[2])
    rebound = replace(contract, source_sha256=_sha256(mutated))

    with pytest.raises(replay.DirectActionReplayError, match="not direct Action IR"):
        replay._materialize_snapshots(mutated, exclusion, contract=rebound)


def test_exclusion_content_cannot_be_rebound_by_hash_alone() -> None:
    source, exclusion, contract, _rows = _fixture()
    record = json.loads(exclusion)
    record["code"] = "different_reason"
    mutated = _line(record)
    rebound = replace(contract, exclusion_ledger_sha256=_sha256(mutated))

    with pytest.raises(replay.DirectActionReplayError, match="ledger content changed"):
        replay._materialize_snapshots(source, mutated, contract=rebound)


def test_public_materializer_reads_each_path_once(monkeypatch: pytest.MonkeyPatch) -> None:
    source, exclusion, contract, _rows = _fixture()
    observed: list[Path] = []

    def read_once(path: Path) -> bytes:
        observed.append(path)
        return source if len(observed) == 1 else exclusion

    monkeypatch.setattr(Path, "read_bytes", read_once)
    monkeypatch.setattr(replay, "_PINNED_CONTRACT", contract)
    result = replay.materialize_direct_action_replay("source.jsonl", "exclusion.jsonl")

    assert observed == [Path("source.jsonl"), Path("exclusion.jsonl")]
    assert result.manifest_sha256 == contract.output_sha256
