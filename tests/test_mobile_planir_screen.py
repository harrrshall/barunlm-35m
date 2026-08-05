from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from barunlm.datasets.mobile_planir_screen import (
    COMPONENT_ASSIGNMENT_SHA256,
    EXPECTED_OUTPUT_SHA256,
    EXPECTED_REJECTION_COMPONENT,
    EXPECTED_REJECTION_FAMILY,
    OUTPUT_FILENAMES,
    SCREEN_MANIFEST_ROW_VERSION,
    SCREEN_MEMBERSHIP_SHA256,
    SCREEN_ROWS,
    SCREEN_SOURCE_MEMBERSHIP_SHA256,
    SCREEN_SOURCE_ROWS,
    SOURCE_MEMBERSHIP_SHA256,
    SPLIT_VERSION,
    TRAIN_MEMBERSHIP_SHA256,
    TRAIN_ROWS,
    MobilePlanIRScreenError,
    materialize_planir_screen,
    plan_construction_screen,
)
from barunlm.datasets.mobile_planir_v2 import (
    CONSTRUCTION_MANIFEST_SHA256,
    EXPECTED_REJECTION_ID,
)
from barunlm.training.data import SFTExample, load_manifest

ROOT = Path(__file__).resolve().parents[1]
CONSTRUCTION_MANIFEST = (
    ROOT
    / "experiments/runs/20260803-2353-mobile-temporal-counterfactual-s17"
    / "materialized-screening-v1/construction-train.jsonl"
)
requires_construction = pytest.mark.skipif(
    not CONSTRUCTION_MANIFEST.is_file(),
    reason="hash-pinned construction manifest is not materialized",
)


def _synthetic(
    index: int,
    *,
    signature: tuple[str, ...],
    component: str | None = None,
    family: str | None = None,
) -> SFTExample:
    example_id = f"synthetic-{index:02d}"
    metadata = {
        "call_names": list(signature),
        "cluster_id": component or f"component-{index:02d}",
        "family_id": family or f"family-{index:02d}",
    }
    payload = {
        "schema_version": "barun-sft-example-v1",
        "id": example_id,
        "prompt": f"prompt-{index}",
        "target": f"target-{index}",
        "metadata": metadata,
    }
    content_sha256 = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return SFTExample(
        example_id=example_id,
        prompt=payload["prompt"],
        target=payload["target"],
        metadata=metadata,
        content_sha256=content_sha256,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _membership(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def test_pure_split_is_permutation_invariant_and_uses_canonical_signature_quotas() -> None:
    rows = tuple(
        _synthetic(
            index,
            signature=("create_calendar_event",) if index < 5 else ("send_email",),
        )
        for index in range(10)
    )
    forward = plan_construction_screen(rows)
    reverse = plan_construction_screen(tuple(reversed(rows)))
    assert forward.component_roles == reverse.component_roles
    assert (
        forward.signature_quotas
        == reverse.signature_quotas
        == {
            '["create_calendar_event"]': 1,
            '["send_email"]': 1,
        }
    )
    assert sum(item.population == "screen" for item in forward.assignments) == 2


def test_split_fails_closed_when_one_component_cannot_satisfy_exact_quota() -> None:
    rows = tuple(
        _synthetic(
            index,
            signature=("send_email",),
            component="one-component",
        )
        for index in range(5)
    )
    with pytest.raises(MobilePlanIRScreenError, match="infeasible"):
        plan_construction_screen(rows)


def test_split_fails_closed_when_a_family_spans_components() -> None:
    rows = tuple(
        _synthetic(
            index,
            signature=("show_map",),
            family="crossing-family" if index < 2 else None,
        )
        for index in range(5)
    )
    with pytest.raises(MobilePlanIRScreenError, match="families span components"):
        plan_construction_screen(rows)


@requires_construction
def test_pinned_plan_has_exact_memberships_and_singleton_screen_rejection() -> None:
    examples = load_manifest(
        CONSTRUCTION_MANIFEST,
        expected_sha256=CONSTRUCTION_MANIFEST_SHA256,
        expected_derived_split="train",
    )
    assert _membership([example.example_id for example in examples]) == SOURCE_MEMBERSHIP_SHA256
    plan = plan_construction_screen(examples, enforce_pinned=True)
    train = [item for item in plan.assignments if item.population == "train"]
    screen = [item for item in plan.assignments if item.population == "screen"]
    assert (len(train), len(screen)) == (TRAIN_ROWS, SCREEN_SOURCE_ROWS) == (4_596, 1_149)
    assert _membership([item.example_id for item in train]) == TRAIN_MEMBERSHIP_SHA256
    assert _membership([item.example_id for item in screen]) == SCREEN_SOURCE_MEMBERSHIP_SHA256
    assignment_bytes = b"".join(
        component.encode() + b"\t" + plan.component_roles[component].encode() + b"\n"
        for component in sorted(plan.component_roles)
    )
    assert hashlib.sha256(assignment_bytes).hexdigest() == COMPONENT_ASSIGNMENT_SHA256
    rejection = next(item for item in screen if item.example_id == EXPECTED_REJECTION_ID)
    assert rejection.component_id == EXPECTED_REJECTION_COMPONENT
    assert rejection.family_id == EXPECTED_REJECTION_FAMILY
    assert sum(item.component_id == rejection.component_id for item in plan.assignments) == 1


@requires_construction
def test_pinned_materialization_is_atomic_matched_and_hash_exact(tmp_path: Path) -> None:
    output_dir = tmp_path / "construction-screen-v1"
    artifacts = materialize_planir_screen(
        source_manifest=CONSTRUCTION_MANIFEST,
        output_dir=output_dir,
    )
    assert artifacts.sha256 == EXPECTED_OUTPUT_SHA256
    assert (artifacts.train_rows, artifacts.screen_rows) == (TRAIN_ROWS, SCREEN_ROWS)
    assert {path.name for path in output_dir.iterdir()} == set(OUTPUT_FILENAMES.values())
    for name, expected in EXPECTED_OUTPUT_SHA256.items():
        assert hashlib.sha256(artifacts.paths[name].read_bytes()).hexdigest() == expected

    source = load_manifest(
        CONSTRUCTION_MANIFEST,
        expected_sha256=CONSTRUCTION_MANIFEST_SHA256,
        expected_derived_split="train",
    )
    source_by_id = {example.example_id: example for example in source}
    populations: dict[str, list[str]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    for population, expected_rows, split in (
        ("train", TRAIN_ROWS, "train"),
        ("screen", SCREEN_ROWS, "dev"),
    ):
        for arm in ("a", "b", "c"):
            name = f"{population}_{arm}"
            loaded = load_manifest(
                artifacts.paths[name],
                expected_sha256=EXPECTED_OUTPUT_SHA256[name],
                expected_derived_split=split,
            )
            assert len(loaded) == expected_rows
            ids = [example.example_id for example in loaded]
            populations.setdefault(population, ids)
            assert ids == populations[population]
            records[name] = _read_jsonl(artifacts.paths[name])

    assert _membership(populations["train"]) == TRAIN_MEMBERSHIP_SHA256
    assert _membership(populations["screen"]) == SCREEN_MEMBERSHIP_SHA256
    assert EXPECTED_REJECTION_ID not in populations["train"]
    assert EXPECTED_REJECTION_ID not in populations["screen"]
    assert set(populations["train"]).isdisjoint(populations["screen"])
    source_order = [example.example_id for example in source]
    assert populations["train"] == [item for item in source_order if item in populations["train"]]
    assert populations["screen"] == [item for item in source_order if item in populations["screen"]]

    for population in ("train", "screen"):
        by_arm = {
            arm: {row["id"]: row for row in records[f"{population}_{arm}"]}
            for arm in ("a", "b", "c")
        }
        for example_id in populations[population]:
            rows = [by_arm[arm][example_id] for arm in ("a", "b", "c")]
            nested = [row["metadata"]["mobile_planir_screen"] for row in rows]
            assert {item["schema_version"] for item in nested} == {SCREEN_MANIFEST_ROW_VERSION}
            assert {item["split_version"] for item in nested} == {SPLIT_VERSION}
            assert {item["population"] for item in nested} == {population}
            assert [item["arm"] for item in nested] == ["A", "B", "C"]
            assert {item["source_id"] for item in nested} == {example_id}
            assert {item["source_target_sha256"] for item in nested} == {
                source_by_id[example_id].metadata["target_sha256"]
            }
            assert (
                nested[1]["prompt_evidence"]["table_sha256"]
                == nested[2]["prompt_evidence"]["table_sha256"]
            )
            assert (
                nested[1]["prompt_evidence"]["request_sha256"]
                == nested[2]["prompt_evidence"]["request_sha256"]
            )
            if population == "screen":
                assert {row["target"] for row in rows} == {source_by_id[example_id].target}
            else:
                assert rows[0]["target"] == rows[1]["target"] == source_by_id[example_id].target

    assignment = _read_jsonl(artifacts.paths["assignment"])
    exclusion = _read_jsonl(artifacts.paths["exclusion"])
    audit = json.loads(artifacts.paths["audit"].read_text(encoding="utf-8"))
    assert len(assignment) == 5_745
    assert exclusion == [audit["exclusion"]]
    assert exclusion[0]["id"] == EXPECTED_REJECTION_ID
    assert exclusion[0]["excluded_from_arms"] == ["A", "B", "C"]
    assert audit["post_exclusion"]["train_rows_per_arm"] == TRAIN_ROWS
    assert audit["post_exclusion"]["screen_rows_per_arm"] == SCREEN_ROWS
    assert audit["overlap"] == {
        "component_id": 0,
        "exact_prompt_target": 0,
        "example_id": 0,
        "family_id": 0,
        "source_content_sha256": 0,
    }
    with pytest.raises(MobilePlanIRScreenError, match="refusing to overwrite"):
        materialize_planir_screen(
            source_manifest=CONSTRUCTION_MANIFEST,
            output_dir=output_dir,
        )
