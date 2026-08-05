"""Deterministic construction-only A/B/C screen materialization for PlanIR v2.

This module accepts only the hash-pinned 5,745-row Mobile Actions construction
manifest.  It has no API for development, selection, confirmation, reused, or
official-evaluation populations.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from barunlm.datasets.mobile_planir_v2 import (
    ACCEPTED_MEMBERSHIP_SHA256,
    ACCEPTED_ROWS,
    CONSTRUCTION_MANIFEST_SHA256,
    CONSTRUCTION_ROWS,
    EXPECTED_REJECTION_CODE,
    EXPECTED_REJECTION_ID,
    Arm,
    MaterializedArmExample,
    materialize_construction_arm,
)
from barunlm.training.data import SFTExample, load_manifest

SPLIT_VERSION = "barun-mobile-construction-component-screen-split-v1"
SPLIT_SCHEMA_VERSION = "barun-mobile-planir-construction-screen-v1"
SCREEN_MANIFEST_ROW_VERSION = "barun-mobile-planir-screen-row-v1"
ASSIGNMENT_SCHEMA_VERSION = "barun-mobile-planir-screen-assignment-v1"
EXCLUSION_SCHEMA_VERSION = "barun-mobile-planir-screen-exclusion-v1"
AUDIT_SCHEMA_VERSION = "barun-mobile-planir-screen-audit-v1"

SOURCE_MEMBERSHIP_SHA256 = "b70f6dfe79cfe49792b6e6f5d414fd509c889d18101f75cd8df6d4e4f146c588"
SOURCE_COMPONENTS = 3_722
SOURCE_FAMILIES = 3_931
TRAIN_ROWS = 4_596
SCREEN_SOURCE_ROWS = 1_149
SCREEN_ROWS = 1_148
TRAIN_COMPONENTS = 2_887
SCREEN_SOURCE_COMPONENTS = 835
SCREEN_COMPONENTS = 834
TRAIN_FAMILIES = 3_079
SCREEN_SOURCE_FAMILIES = 852
SCREEN_FAMILIES = 851

TRAIN_MEMBERSHIP_SHA256 = "7bb1b7dc6a6914d3c68a5d07541d6851fcc545a083a6f0f9c99e7a48a89378fd"
SCREEN_SOURCE_MEMBERSHIP_SHA256 = "307f13c2673cc9a159a200ccedc50e6302d37fd106d9cb0d9c01aa8ee40f15a0"
SCREEN_MEMBERSHIP_SHA256 = "ec922c1f18c55535319f6bf0e1e7dd68461456e5cb2a3472404ce2d874a28642"
TRAIN_COMPONENT_MEMBERSHIP_SHA256 = (
    "c167e1ee3b0ecbcd6386ba63bb6dc7d80072decd84e81ccee387c38225dffd23"
)
SCREEN_SOURCE_COMPONENT_MEMBERSHIP_SHA256 = (
    "f8b4bf18a3a2d4b5af5832ca0ee1edcf1a4edd9f3f15ce6df324f08fc4b58a45"
)
TRAIN_FAMILY_MEMBERSHIP_SHA256 = "797b3266c1d36792d2e914c858f9df619b362108ee3204a1b1c1a4c0e5e010e4"
SCREEN_SOURCE_FAMILY_MEMBERSHIP_SHA256 = (
    "1715d5b870f70e048db2054319075e3ba299a862097c0632f04f1b1756cb2453"
)
COMPONENT_ASSIGNMENT_SHA256 = "9ade076fca8028e09c803b6eb2439e232011de7de8dbd0aef3c8395715ca1856"

EXPECTED_REJECTION_COMPONENT = (
    "ma-component-v1-2e7e3504d097fdde9776b24ca88096887d8242c6b1d63a1d434f66339838a853"
)
EXPECTED_REJECTION_FAMILY = (
    "ma-family-v1-3bcd0729a5d858980eef576453cafe64d52a44a8fb72b5df64b1a7d726a3fa09"
)

OUTPUT_FILENAMES = {
    "train_a": "train-a.jsonl",
    "train_b": "train-b.jsonl",
    "train_c": "train-c.jsonl",
    "screen_a": "screen-a.jsonl",
    "screen_b": "screen-b.jsonl",
    "screen_c": "screen-c.jsonl",
    "assignment": "assignment.jsonl",
    "exclusion": "exclusion.jsonl",
    "audit": "audit.json",
}

EXPECTED_OUTPUT_SHA256: Mapping[str, str] = {
    "assignment": "13eaad3ba5cf9da6dfad3dc5a73300599bf79ad29697c7d576807fed00ea6c3f",
    "audit": "646523ed299bb4b742c25c594aa08f4ecf79e3f2445db84fff9a6b6eead7d946",
    "exclusion": "7593b556b9c096308ca1a9e9d9b89627ebf19710da930bd4dd5794c58879616a",
    "screen_a": "08fd5e09bfda66b4b0cb31e7ef19bbb3a71a52e13f4fbe7c3b1dd3ea1cf402e7",
    "screen_b": "81ab11b96a8330d19ebd41c6fcac5680f86a7652e721b495c1d24fc9f7a6c50d",
    "screen_c": "c5c16ba5a785118dfdf8ca4665369ff375592c46ee8d066adbad9dcb0eff04ef",
    "train_a": "702b6c0abd4c12d128b4e089cd8b2f7c57058fa7e50c5641978d225fb7529317",
    "train_b": "30ccad10845196116c5205445b4b3e3aa2eb108434469e5fed3e27ae9d6fe3d2",
    "train_c": "9aa0987b2b43af55e8dcbb1d6f921349282878ac58fe92d3ddadc953f97e51f5",
}

Population = Literal["train", "screen"]


class MobilePlanIRScreenError(ValueError):
    """The construction-only screen contract was violated."""


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    example_id: str
    component_id: str
    family_id: str
    population: Population


@dataclass(frozen=True, slots=True)
class ConstructionScreenPlan:
    assignments: tuple[SplitAssignment, ...]
    signature_quotas: Mapping[str, int]
    component_roles: Mapping[str, Population]


@dataclass(frozen=True, slots=True)
class ConstructionScreenArtifacts:
    output_dir: Path
    paths: Mapping[str, Path]
    sha256: Mapping[str, str]
    train_rows: int
    screen_rows: int


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return (text + "\n").encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _prompt_target_identity(example: SFTExample) -> str:
    return _sha256_bytes(
        _canonical_json_bytes({"prompt": example.prompt, "target": example.target})[:-1]
    )


def _membership_sha256(values: Sequence[str] | set[str]) -> str:
    items = list(values)
    if len(items) != len(set(items)):
        raise MobilePlanIRScreenError("membership contains duplicate values")
    return _sha256_bytes(b"".join(item.encode("utf-8") + b"\n" for item in sorted(items)))


def _ordered_call_signature(example: SFTExample) -> bytes:
    call_names = example.metadata.get("call_names")
    if (
        not isinstance(call_names, list)
        or not call_names
        or any(not isinstance(item, str) or not item for item in call_names)
    ):
        raise MobilePlanIRScreenError(
            f"example {example.example_id!r} has invalid metadata.call_names"
        )
    return json.dumps(
        call_names,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _grouping_fields(example: SFTExample) -> tuple[str, str]:
    component = example.metadata.get("cluster_id")
    family = example.metadata.get("family_id")
    if not isinstance(component, str) or not component:
        raise MobilePlanIRScreenError(f"example {example.example_id!r} lacks cluster_id")
    if not isinstance(family, str) or not family:
        raise MobilePlanIRScreenError(f"example {example.example_id!r} lacks family_id")
    return component, family


def _component_assignment_hash(component_roles: Mapping[str, Population]) -> str:
    payload = b"".join(
        component.encode("utf-8") + b"\t" + component_roles[component].encode("ascii") + b"\n"
        for component in sorted(component_roles)
    )
    return _sha256_bytes(payload)


def plan_construction_screen(
    examples: Sequence[SFTExample],
    *,
    enforce_pinned: bool = False,
) -> ConstructionScreenPlan:
    """Assign whole components using the frozen ordered-signature quota algorithm."""

    if not examples or len(examples) < 5 or len(examples) % 5:
        raise MobilePlanIRScreenError("v1 split requires a nonempty row count divisible by five")
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise MobilePlanIRScreenError("construction examples contain duplicate IDs")

    component_indexes: defaultdict[str, list[int]] = defaultdict(list)
    family_components: defaultdict[str, set[str]] = defaultdict(set)
    signatures: list[bytes] = []
    for index, example in enumerate(examples):
        component, family = _grouping_fields(example)
        component_indexes[component].append(index)
        family_components[family].add(component)
        signatures.append(_ordered_call_signature(example))
    crossing = sorted(
        family for family, components in family_components.items() if len(components) != 1
    )
    if crossing:
        raise MobilePlanIRScreenError(f"families span components: {crossing[:3]}")

    signature_counts = Counter(signatures)
    quotas = Counter({signature: count // 5 for signature, count in signature_counts.items()})
    extra = len(examples) // 5 - sum(quotas.values())
    remainder_order = sorted(
        signature_counts,
        key=lambda signature: (-(signature_counts[signature] % 5), signature),
    )
    for signature in remainder_order[:extra]:
        quotas[signature] += 1

    version = SPLIT_VERSION.encode("utf-8")
    ranked_components = sorted(
        component_indexes,
        key=lambda component: (
            hashlib.sha256(version + b":" + component.encode("utf-8")).digest(),
            component.encode("utf-8"),
        ),
    )
    remaining = quotas.copy()
    screen_components: set[str] = set()
    for component in ranked_components:
        component_counts = Counter(signatures[index] for index in component_indexes[component])
        if all(count <= remaining[signature] for signature, count in component_counts.items()):
            screen_components.add(component)
            remaining.subtract(component_counts)
            if not any(remaining.values()):
                break
    residual = {signature.decode("utf-8"): count for signature, count in remaining.items() if count}
    if residual:
        raise MobilePlanIRScreenError(f"component-safe quota allocation is infeasible: {residual}")

    component_roles: dict[str, Population] = {
        component: "screen" if component in screen_components else "train"
        for component in component_indexes
    }
    assignments = tuple(
        SplitAssignment(
            example_id=example.example_id,
            component_id=_grouping_fields(example)[0],
            family_id=_grouping_fields(example)[1],
            population=component_roles[_grouping_fields(example)[0]],
        )
        for example in examples
    )
    plan = ConstructionScreenPlan(
        assignments=assignments,
        signature_quotas={
            signature.decode("utf-8"): quotas[signature] for signature in sorted(quotas)
        },
        component_roles=component_roles,
    )
    if enforce_pinned:
        _validate_pinned_plan(examples, plan)
    return plan


def _validate_pinned_plan(examples: Sequence[SFTExample], plan: ConstructionScreenPlan) -> None:
    if len(examples) != CONSTRUCTION_ROWS:
        raise MobilePlanIRScreenError("pinned construction row count changed")
    train_ids = [item.example_id for item in plan.assignments if item.population == "train"]
    screen_ids = [item.example_id for item in plan.assignments if item.population == "screen"]
    train_components = {
        item.component_id for item in plan.assignments if item.population == "train"
    }
    screen_components = {
        item.component_id for item in plan.assignments if item.population == "screen"
    }
    train_families = {item.family_id for item in plan.assignments if item.population == "train"}
    screen_families = {item.family_id for item in plan.assignments if item.population == "screen"}
    observed = (
        len(train_ids),
        len(screen_ids),
        len(train_components),
        len(screen_components),
        len(train_families),
        len(screen_families),
        _membership_sha256(train_ids),
        _membership_sha256(screen_ids),
        _membership_sha256(train_components),
        _membership_sha256(screen_components),
        _membership_sha256(train_families),
        _membership_sha256(screen_families),
        _component_assignment_hash(plan.component_roles),
    )
    expected = (
        TRAIN_ROWS,
        SCREEN_SOURCE_ROWS,
        TRAIN_COMPONENTS,
        SCREEN_SOURCE_COMPONENTS,
        TRAIN_FAMILIES,
        SCREEN_SOURCE_FAMILIES,
        TRAIN_MEMBERSHIP_SHA256,
        SCREEN_SOURCE_MEMBERSHIP_SHA256,
        TRAIN_COMPONENT_MEMBERSHIP_SHA256,
        SCREEN_SOURCE_COMPONENT_MEMBERSHIP_SHA256,
        TRAIN_FAMILY_MEMBERSHIP_SHA256,
        SCREEN_SOURCE_FAMILY_MEMBERSHIP_SHA256,
        COMPONENT_ASSIGNMENT_SHA256,
    )
    if observed != expected:
        raise MobilePlanIRScreenError("pinned construction split changed")


def _metadata_record(
    source: SFTExample,
    materialized: MaterializedArmExample,
    *,
    population: Population,
    target: str,
) -> dict[str, Any]:
    metadata = dict(source.metadata)
    source_metadata_sha256 = _sha256_bytes(_canonical_json_bytes(source.metadata)[:-1])
    output_prompt_sha256 = _sha256_bytes(materialized.prompt.encode("utf-8"))
    output_target_sha256 = _sha256_bytes(target.encode("utf-8"))
    metadata["derived_split"] = "train" if population == "train" else "dev"
    metadata["prompt_sha256"] = output_prompt_sha256
    metadata["target_sha256"] = output_target_sha256
    metadata["mobile_planir_screen"] = {
        "schema_version": SCREEN_MANIFEST_ROW_VERSION,
        "split_version": SPLIT_VERSION,
        "arm": materialized.arm,
        "population": population,
        "source_id": source.example_id,
        "source_content_sha256": source.content_sha256,
        "source_metadata_sha256": source_metadata_sha256,
        "source_prompt_sha256": str(source.metadata["prompt_sha256"]),
        "source_target_sha256": str(source.metadata["target_sha256"]),
        "output_prompt_sha256": output_prompt_sha256,
        "output_target_sha256": output_target_sha256,
        "prompt_evidence": asdict(materialized.prompt_evidence),
    }
    return metadata


def _arm_record(
    source: SFTExample,
    materialized: MaterializedArmExample,
    *,
    population: Population,
) -> dict[str, Any]:
    target = materialized.target if population == "train" else source.target
    return {
        "schema_version": "barun-sft-example-v1",
        "id": source.example_id,
        "prompt": materialized.prompt,
        "target": target,
        "metadata": _metadata_record(
            source,
            materialized,
            population=population,
            target=target,
        ),
    }


def _validate_matched_arms(arms: Mapping[Arm, Mapping[str, MaterializedArmExample]]) -> None:
    memberships = {arm: tuple(sorted(rows)) for arm, rows in arms.items()}
    if len(set(memberships.values())) != 1:
        raise MobilePlanIRScreenError("A/B/C accepted memberships differ")
    ids = memberships["A"]
    if len(ids) != ACCEPTED_ROWS or _membership_sha256(list(ids)) != ACCEPTED_MEMBERSHIP_SHA256:
        raise MobilePlanIRScreenError("A/B/C accepted membership changed")
    for example_id in ids:
        a = arms["A"][example_id]
        b = arms["B"][example_id]
        c = arms["C"][example_id]
        if a.target != b.target:
            raise MobilePlanIRScreenError(f"example {example_id!r} A/B gold targets differ")
        evidence = (b.prompt_evidence, c.prompt_evidence)
        if (
            evidence[0].renderer_version != evidence[1].renderer_version
            or evidence[0].request_sha256 != evidence[1].request_sha256
            or evidence[0].now != evidence[1].now
            or evidence[0].table_sha256 != evidence[1].table_sha256
        ):
            raise MobilePlanIRScreenError(f"example {example_id!r} B/C prompt evidence differs")
        if b.prompt.replace("ACTION_IR_V1", "PLAN_IR_V2", 1) != c.prompt:
            raise MobilePlanIRScreenError(f"example {example_id!r} B/C prompts are not matched")


def _build_artifact_bytes(source_manifest: str | Path) -> dict[str, bytes]:
    examples = tuple(
        load_manifest(
            source_manifest,
            expected_sha256=CONSTRUCTION_MANIFEST_SHA256,
            expected_derived_split="train",
        )
    )
    if len(examples) != CONSTRUCTION_ROWS:
        raise MobilePlanIRScreenError("construction row count changed")
    if _membership_sha256([example.example_id for example in examples]) != SOURCE_MEMBERSHIP_SHA256:
        raise MobilePlanIRScreenError("construction membership changed")
    plan = plan_construction_screen(examples, enforce_pinned=True)
    assignment_by_id = {item.example_id: item for item in plan.assignments}
    source_by_id = {example.example_id: example for example in examples}

    arms: dict[Arm, dict[str, MaterializedArmExample]] = {}
    for arm in ("A", "B", "C"):
        arms[arm] = {
            row.example_id: row for row in materialize_construction_arm(source_manifest, arm)
        }
    _validate_matched_arms(arms)

    rejection = source_by_id.get(EXPECTED_REJECTION_ID)
    if rejection is None:
        raise MobilePlanIRScreenError("expected construction rejection is missing")
    rejection_assignment = assignment_by_id[EXPECTED_REJECTION_ID]
    rejection_component_rows = [
        item for item in plan.assignments if item.component_id == rejection_assignment.component_id
    ]
    if (
        rejection_assignment.population != "screen"
        or rejection_assignment.component_id != EXPECTED_REJECTION_COMPONENT
        or rejection_assignment.family_id != EXPECTED_REJECTION_FAMILY
        or len(rejection_component_rows) != 1
        or any(EXPECTED_REJECTION_ID in rows for rows in arms.values())
    ):
        raise MobilePlanIRScreenError("expected singleton screen rejection changed")

    train_ids = [
        example.example_id
        for example in examples
        if assignment_by_id[example.example_id].population == "train"
    ]
    screen_ids = [
        example.example_id
        for example in examples
        if assignment_by_id[example.example_id].population == "screen"
        and example.example_id != EXPECTED_REJECTION_ID
    ]
    if (
        len(train_ids) != TRAIN_ROWS
        or len(screen_ids) != SCREEN_ROWS
        or _membership_sha256(train_ids) != TRAIN_MEMBERSHIP_SHA256
        or _membership_sha256(screen_ids) != SCREEN_MEMBERSHIP_SHA256
    ):
        raise MobilePlanIRScreenError("post-exclusion population membership changed")

    output: dict[str, bytes] = {}
    for arm in ("A", "B", "C"):
        for population, ids in (("train", train_ids), ("screen", screen_ids)):
            records = [
                _arm_record(
                    source_by_id[example_id],
                    arms[arm][example_id],
                    population=population,
                )
                for example_id in ids
            ]
            output[f"{population}_{arm.lower()}"] = _jsonl_bytes(records)

    assignment_rows = [
        {
            "schema_version": ASSIGNMENT_SCHEMA_VERSION,
            "id": item.example_id,
            "component_id": item.component_id,
            "family_id": item.family_id,
            "population": item.population,
            "excluded": item.example_id == EXPECTED_REJECTION_ID,
            "split_version": SPLIT_VERSION,
        }
        for item in plan.assignments
    ]
    output["assignment"] = _jsonl_bytes(assignment_rows)
    exclusion_row = {
        "schema_version": EXCLUSION_SCHEMA_VERSION,
        "id": EXPECTED_REJECTION_ID,
        "code": EXPECTED_REJECTION_CODE,
        "component_id": EXPECTED_REJECTION_COMPONENT,
        "family_id": EXPECTED_REJECTION_FAMILY,
        "population": "screen",
        "singleton_component": True,
        "excluded_from_arms": ["A", "B", "C"],
        "source_content_sha256": rejection.content_sha256,
        "source_prompt_sha256": str(rejection.metadata["prompt_sha256"]),
        "source_target_sha256": str(rejection.metadata["target_sha256"]),
        "split_version": SPLIT_VERSION,
    }
    output["exclusion"] = _jsonl_bytes([exclusion_row])

    train_assignments = [item for item in plan.assignments if item.population == "train"]
    screen_assignments = [
        item
        for item in plan.assignments
        if item.population == "screen" and item.example_id != EXPECTED_REJECTION_ID
    ]
    overlap = {
        "example_id": len({item.example_id for item in train_assignments} & set(screen_ids)),
        "component_id": len(
            {item.component_id for item in train_assignments}
            & {item.component_id for item in screen_assignments}
        ),
        "family_id": len(
            {item.family_id for item in train_assignments}
            & {item.family_id for item in screen_assignments}
        ),
        "source_content_sha256": len(
            {source_by_id[item.example_id].content_sha256 for item in train_assignments}
            & {source_by_id[item.example_id].content_sha256 for item in screen_assignments}
        ),
        "exact_prompt_target": len(
            {_prompt_target_identity(source_by_id[item.example_id]) for item in train_assignments}
            & {
                _prompt_target_identity(source_by_id[item.example_id])
                for item in screen_assignments
            }
        ),
    }
    if any(overlap.values()):
        raise MobilePlanIRScreenError(f"train/screen overlap changed: {overlap}")

    output_hashes = {name: _sha256_bytes(content) for name, content in output.items()}
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "use": "construction-internal-mechanism-screen-only",
        "source": {
            "sha256": CONSTRUCTION_MANIFEST_SHA256,
            "rows": CONSTRUCTION_ROWS,
            "membership_sha256": SOURCE_MEMBERSHIP_SHA256,
            "components": SOURCE_COMPONENTS,
            "families": SOURCE_FAMILIES,
        },
        "split": {
            "version": SPLIT_VERSION,
            "screen_fraction": [1, 5],
            "component_assignment_sha256": COMPONENT_ASSIGNMENT_SHA256,
            "signature_quotas": dict(plan.signature_quotas),
            "pre_exclusion": {
                "train_rows": TRAIN_ROWS,
                "screen_rows": SCREEN_SOURCE_ROWS,
                "train_membership_sha256": TRAIN_MEMBERSHIP_SHA256,
                "screen_membership_sha256": SCREEN_SOURCE_MEMBERSHIP_SHA256,
            },
        },
        "exclusion": exclusion_row,
        "post_exclusion": {
            "train_rows_per_arm": TRAIN_ROWS,
            "screen_rows_per_arm": SCREEN_ROWS,
            "train_membership_sha256": TRAIN_MEMBERSHIP_SHA256,
            "screen_membership_sha256": SCREEN_MEMBERSHIP_SHA256,
            "train_components": TRAIN_COMPONENTS,
            "screen_components": SCREEN_COMPONENTS,
            "train_families": TRAIN_FAMILIES,
            "screen_families": SCREEN_FAMILIES,
        },
        "matched_arms": {
            "membership_equal": True,
            "screen_gold_equal": True,
            "a_b_train_gold_equal": True,
            "b_c_reference_table_equal": True,
            "b_c_prompts_differ_only_by_contract": True,
        },
        "overlap": overlap,
        "outputs": {
            name: {
                "filename": OUTPUT_FILENAMES[name],
                "sha256": digest,
                "rows": (
                    TRAIN_ROWS
                    if name.startswith("train_")
                    else SCREEN_ROWS
                    if name.startswith("screen_")
                    else CONSTRUCTION_ROWS
                    if name == "assignment"
                    else 1
                ),
            }
            for name, digest in sorted(output_hashes.items())
        },
        "forbidden_population_rows_read": 0,
    }
    output["audit"] = _canonical_json_bytes(audit, pretty=True)
    return output


def materialize_planir_screen(
    *,
    source_manifest: str | Path,
    output_dir: str | Path,
) -> ConstructionScreenArtifacts:
    """Materialize all construction-only A/B/C fit and screen artifacts once."""

    destination = Path(output_dir)
    if destination.exists():
        raise MobilePlanIRScreenError(f"refusing to overwrite {destination}")
    content = _build_artifact_bytes(source_manifest)
    actual_hashes = {name: _sha256_bytes(value) for name, value in content.items()}
    if dict(EXPECTED_OUTPUT_SHA256) != actual_hashes:
        raise MobilePlanIRScreenError(f"materialized output hashes changed: {actual_hashes!r}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for name, value in content.items():
            path = staging / OUTPUT_FILENAMES[name]
            with path.open("xb") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if destination.exists():
            raise MobilePlanIRScreenError(f"refusing to overwrite {destination}")
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    paths = {name: destination / filename for name, filename in OUTPUT_FILENAMES.items()}
    return ConstructionScreenArtifacts(
        output_dir=destination,
        paths=paths,
        sha256=actual_hashes,
        train_rows=TRAIN_ROWS,
        screen_rows=SCREEN_ROWS,
    )


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "COMPONENT_ASSIGNMENT_SHA256",
    "EXPECTED_OUTPUT_SHA256",
    "EXPECTED_REJECTION_COMPONENT",
    "EXPECTED_REJECTION_FAMILY",
    "OUTPUT_FILENAMES",
    "SCREEN_MANIFEST_ROW_VERSION",
    "SCREEN_MEMBERSHIP_SHA256",
    "SCREEN_ROWS",
    "SCREEN_SOURCE_MEMBERSHIP_SHA256",
    "SCREEN_SOURCE_ROWS",
    "SOURCE_MEMBERSHIP_SHA256",
    "SPLIT_SCHEMA_VERSION",
    "SPLIT_VERSION",
    "TRAIN_MEMBERSHIP_SHA256",
    "TRAIN_ROWS",
    "ConstructionScreenArtifacts",
    "ConstructionScreenPlan",
    "MobilePlanIRScreenError",
    "SplitAssignment",
    "materialize_planir_screen",
    "plan_construction_screen",
]
