"""Construction-only oracle and matched prompt encoder for PlanIR v2.

Only the pinned former-v3 construction-training manifest is accepted here.  The
reference table is built before the label is decoded; labels are used solely to
choose a supervised placeholder and prove an exact compiler round trip.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from tokenizers import Tokenizer

from barunlm.evaluation.grounded_planir_v2 import (
    ACTION_PROMPT_CONTRACT,
    PROMPT_CONTRACT,
    ClockCandidate,
    DateCandidate,
    PromptEvidence,
    ReferenceTable,
    SpanRef,
    build_reference_table,
    compile_mobile_plan,
    make_prompt_evidence,
    parse_construction_prompt,
    render_construction_prompt,
)
from barunlm.training.data import SFTExample, load_manifest, sha256_file

CONSTRUCTION_USE = "training-and-feasibility-only"
CONSTRUCTION_MANIFEST_SHA256 = "800a3ba0a7f0215e5cf95c77c32e47abed48b2f07f375954b0e0b08296a33d10"
CONSTRUCTION_ROWS = 5_745
ACCEPTED_ROWS = 5_744
ACCEPTED_MEMBERSHIP_SHA256 = "27f9acb8905db132420ddb5d11a20bed138f2e2927aec41672f6852b3895452d"
EXPECTED_DATE_OPERATOR_COUNTS = {"A": 1_333, "M": 357, "R": 138, "W": 241}
EXPECTED_CLOCK_REFERENCES = 2_070
EXPECTED_REJECTION_ID = "mobile-actions-04894-6bd7643b1bc95697"
EXPECTED_REJECTION_CODE = "temporal_reference_mismatch"
PINNED_TOKENIZER_SHA256 = "70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6"

Arm = Literal["A", "B", "C"]


class MobilePlanIRV2Error(ValueError):
    """The construction-only v2 oracle contract was violated."""


@dataclass(frozen=True, slots=True)
class ParsedConstructionExample:
    request: str
    now: str
    system_body: str
    table: ReferenceTable


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    prompt: str
    evidence: PromptEvidence


@dataclass(frozen=True, slots=True)
class MaterializedArmExample:
    example_id: str
    arm: Arm
    prompt: str
    target: str
    prompt_evidence: PromptEvidence


@dataclass(frozen=True, slots=True)
class OracleDerivation:
    example_id: str
    accepted: bool
    code: str
    target: str | None
    compiled_action_ir: str | None
    selected_opcode: str | None
    selected_date_ref: SpanRef | None
    selected_time_ref: SpanRef | None
    date_candidates: int
    clock_candidates: int


@dataclass(frozen=True, slots=True)
class OracleAudit:
    use: str
    source_rows: int
    accepted_rows: int
    rejected_rows: int
    accepted_membership_sha256: str
    date_operator_counts: Mapping[str, int]
    clock_references: int
    rejections: tuple[OracleDerivation, ...]


@dataclass(frozen=True, slots=True)
class Distribution:
    total: int
    mean: float
    median: float
    p90: int
    maximum: int


@dataclass(frozen=True, slots=True)
class ArmTokenAudit:
    prompt: Distribution
    target: Distribution
    sequence: Distribution


@dataclass(frozen=True, slots=True)
class TokenAudit:
    use: str
    manifest_sha256: str
    tokenizer_sha256: str
    compared_rows: int
    rejected_rows: int
    accepted_membership_sha256: str
    rejection_ids: tuple[str, ...]
    max_sequence_length: int
    arm_a: ArmTokenAudit
    arm_b: ArmTokenAudit
    arm_c: ArmTokenAudit
    table_nonempty_rows: int
    date_reference_instances: int
    time_reference_instances: int
    c_targets_shorter_than_a: int
    c_targets_equal_to_a: int
    c_targets_longer_than_a: int
    over_limit_a: int
    over_limit_b: int
    over_limit_c: int


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_construction_metadata(example: SFTExample) -> None:
    """Defense in depth; the pinned bulk-manifest hash remains authoritative."""

    required = {
        "adapter_schema_version": "barun-mobile-actions-adapter-v2",
        "dataset": "google/mobile-actions",
        "derived_split": "train",
        "prompt_contract_version": "barun-action-prompt-v1",
        "source_split": "train",
    }
    for key, expected in required.items():
        if example.metadata.get(key) != expected:
            raise MobilePlanIRV2Error(
                f"example {example.example_id!r} metadata.{key} must equal {expected!r}"
            )
    forbidden_roles = {
        "confirmation",
        "development",
        "dev",
        "eval",
        "evaluation",
        "final",
        "official",
        "reused",
        "selection",
        "test",
    }
    for key in ("data_role", "evaluation_role", "population_role", "role", "split_role"):
        value = example.metadata.get(key)
        if isinstance(value, str) and value.casefold() in forbidden_roles:
            raise MobilePlanIRV2Error(
                f"example {example.example_id!r} metadata.{key} is forbidden construction input"
            )


def _parse_input(example: SFTExample) -> ParsedConstructionExample:
    _validate_construction_metadata(example)
    try:
        prompt = parse_construction_prompt(example.prompt)
    except Exception as exc:
        if isinstance(exc, MobilePlanIRV2Error):
            raise
        raise MobilePlanIRV2Error(
            f"example {example.example_id!r} violates the canonical prompt contract: {exc}"
        ) from exc
    if prompt.prompt_contract != ACTION_PROMPT_CONTRACT or prompt.rendered_table is not None:
        raise MobilePlanIRV2Error(
            f"example {example.example_id!r} must be an unaugmented ACTION_IR_V1 prompt"
        )
    # This call is intentionally before target decoding.
    table = build_reference_table(prompt.request, prompt.now)
    return ParsedConstructionExample(
        request=prompt.request,
        now=prompt.now,
        system_body=prompt.system_body,
        table=table,
    )


def _decode_target(example: SFTExample) -> dict[str, Any]:
    try:
        target = json.loads(example.target)
    except json.JSONDecodeError as exc:
        raise MobilePlanIRV2Error(f"example {example.example_id!r} target is invalid JSON") from exc
    if not isinstance(target, dict) or _canonical_json(target) != example.target:
        raise MobilePlanIRV2Error(f"example {example.example_id!r} target is not canonical JSON")
    return target


def _candidate_key(candidate: DateCandidate | ClockCandidate) -> tuple[object, ...]:
    priority = {"A": 0, "M": 1, "R": 2, "W1": 3, "W2": 4}
    return (
        priority.get(getattr(candidate, "opcode", None), 0),
        len(candidate.ref.quote),
        candidate.ref.start,
        candidate.ref.quote,
        candidate.ref.occurrence,
    )


def _reference_id(items: tuple[SpanRef, ...], selected: SpanRef, prefix: str) -> str:
    index = items.index(selected)
    if index >= 100:
        raise MobilePlanIRV2Error("reference ID exceeds the frozen two-digit grammar")
    return f"{prefix}{index:02d}"


def _derive_oracle(example: SFTExample) -> OracleDerivation:
    """Create one label-supervised target after target-free table construction."""

    parsed = _parse_input(example)
    target = _decode_target(example)
    calendar_calls = [
        call
        for call in target.get("calls", [])
        if isinstance(call, dict) and call.get("tool") == "create_calendar_event"
    ]
    if len(calendar_calls) > 1:
        raise MobilePlanIRV2Error("construction contract permits at most one calendar call")

    selected_date: DateCandidate | None = None
    selected_clock: ClockCandidate | None = None
    if calendar_calls:
        raw_datetime = calendar_calls[0].get("args", {}).get("datetime")
        try:
            gold = datetime.fromisoformat(raw_datetime)
        except (TypeError, ValueError) as exc:
            raise MobilePlanIRV2Error("calendar gold datetime must be ISO local datetime") from exc
        selected_date = next(
            (
                candidate
                for candidate in sorted(parsed.table.candidates.dates, key=_candidate_key)
                if candidate.resolved == gold.date()
            ),
            None,
        )
        selected_clock = next(
            (
                candidate
                for candidate in sorted(parsed.table.candidates.clocks, key=_candidate_key)
                if candidate.resolved == gold.time()
            ),
            None,
        )
        if selected_date is None or selected_clock is None:
            return OracleDerivation(
                example_id=example.example_id,
                accepted=False,
                code=EXPECTED_REJECTION_CODE,
                target=None,
                compiled_action_ir=None,
                selected_opcode=selected_date.opcode if selected_date else None,
                selected_date_ref=selected_date.ref if selected_date else None,
                selected_time_ref=selected_clock.ref if selected_clock else None,
                date_candidates=len(parsed.table.candidates.dates),
                clock_candidates=len(parsed.table.candidates.clocks),
            )
        date_id = _reference_id(parsed.table.date_refs, selected_date.ref, "D")
        time_id = _reference_id(parsed.table.time_refs, selected_clock.ref, "T")
        calendar_calls[0]["args"]["datetime"] = f"@{selected_date.opcode}:{date_id}:{time_id}"

    plan_target = _canonical_json(target)
    rendered_prompt = _render_prompt(example, "C", parsed=parsed)
    compiled = compile_mobile_plan(
        plan_target,
        parsed.request,
        parsed.now,
        prompt_contract=PROMPT_CONTRACT,
        rendered_prompt=rendered_prompt.prompt,
        prompt_evidence=rendered_prompt.evidence,
        rendered_table=parsed.table.render(),
        table_sha256=parsed.table.sha256(),
    )
    if not compiled.ok or compiled.action_ir != example.target:
        return OracleDerivation(
            example_id=example.example_id,
            accepted=False,
            code="compiler_rejected_oracle" if not compiled.ok else "compiled_target_mismatch",
            target=plan_target,
            compiled_action_ir=compiled.action_ir,
            selected_opcode=selected_date.opcode if selected_date else None,
            selected_date_ref=selected_date.ref if selected_date else None,
            selected_time_ref=selected_clock.ref if selected_clock else None,
            date_candidates=len(parsed.table.candidates.dates),
            clock_candidates=len(parsed.table.candidates.clocks),
        )
    return OracleDerivation(
        example_id=example.example_id,
        accepted=True,
        code="accepted",
        target=plan_target,
        compiled_action_ir=compiled.action_ir,
        selected_opcode=selected_date.opcode if selected_date else None,
        selected_date_ref=selected_date.ref if selected_date else None,
        selected_time_ref=selected_clock.ref if selected_clock else None,
        date_candidates=len(parsed.table.candidates.dates),
        clock_candidates=len(parsed.table.candidates.clocks),
    )


def _render_prompt(
    example: SFTExample,
    arm: Arm,
    *,
    parsed: ParsedConstructionExample | None = None,
) -> RenderedPrompt:
    """Render matched A/B/C prompts without consulting the target."""

    if arm not in {"A", "B", "C"}:
        raise MobilePlanIRV2Error(f"unknown arm {arm!r}")
    parsed = parsed or _parse_input(example)
    prompt_contract = PROMPT_CONTRACT if arm == "C" else ACTION_PROMPT_CONTRACT
    rendered_table = parsed.table.render() if arm in {"B", "C"} else None
    prompt = render_construction_prompt(
        prompt_contract=prompt_contract,
        system_body=parsed.system_body,
        request=parsed.request,
        rendered_table=rendered_table,
    )
    reparsed = parse_construction_prompt(prompt)
    if (
        reparsed.prompt_contract != prompt_contract
        or reparsed.request != parsed.request
        or reparsed.now != parsed.now
        or reparsed.rendered_table != rendered_table
    ):
        raise MobilePlanIRV2Error("rendered prompt failed its structural round trip")
    evidence = make_prompt_evidence(
        prompt,
        prompt_contract,
        parsed.table,
        parsed.request,
        parsed.now,
    )
    return RenderedPrompt(prompt=prompt, evidence=evidence)


def _render_target(example: SFTExample, arm: Arm) -> str:
    if arm in {"A", "B"}:
        return example.target
    if arm != "C":
        raise MobilePlanIRV2Error(f"unknown arm {arm!r}")
    oracle = _derive_oracle(example)
    if not oracle.accepted or oracle.target is None:
        raise MobilePlanIRV2Error(f"example {example.example_id!r} has no accepted v2 oracle")
    return oracle.target


def _load_pinned_construction_manifest(path: str | Path) -> tuple[SFTExample, ...]:
    examples = load_manifest(
        path,
        expected_sha256=CONSTRUCTION_MANIFEST_SHA256,
        expected_derived_split="train",
    )
    if len(examples) != CONSTRUCTION_ROWS:
        raise MobilePlanIRV2Error("construction row count changed")
    for example in examples:
        _validate_construction_metadata(example)
    return tuple(examples)


def materialize_construction_arm(
    manifest_path: str | Path,
    arm: Arm,
) -> tuple[MaterializedArmExample, ...]:
    """Materialize one matched arm only from the hash-pinned training manifest."""

    rows: list[MaterializedArmExample] = []
    rejected: list[tuple[str, str]] = []
    for example in _load_pinned_construction_manifest(manifest_path):
        oracle = _derive_oracle(example)
        if not oracle.accepted:
            rejected.append((oracle.example_id, oracle.code))
            continue
        rendered = _render_prompt(example, arm)
        target = example.target if arm in {"A", "B"} else oracle.target
        assert target is not None
        rows.append(
            MaterializedArmExample(
                example_id=example.example_id,
                arm=arm,
                prompt=rendered.prompt,
                target=target,
                prompt_evidence=rendered.evidence,
            )
        )
    membership = _membership_sha256([row.example_id for row in rows])
    if len(rows) != ACCEPTED_ROWS or membership != ACCEPTED_MEMBERSHIP_SHA256:
        raise MobilePlanIRV2Error("materialized accepted construction membership changed")
    if rejected != [(EXPECTED_REJECTION_ID, EXPECTED_REJECTION_CODE)]:
        raise MobilePlanIRV2Error("materialized construction rejection evidence changed")
    return tuple(rows)


def _membership_sha256(example_ids: Sequence[str]) -> str:
    if len(example_ids) != len(set(example_ids)):
        raise MobilePlanIRV2Error("accepted oracle membership contains duplicate IDs")
    return hashlib.sha256(("\n".join(sorted(example_ids)) + "\n").encode()).hexdigest()


def audit_construction_oracles(path: str | Path) -> OracleAudit:
    examples = _load_pinned_construction_manifest(path)
    derivations = tuple(_derive_oracle(example) for example in examples)
    accepted = tuple(item for item in derivations if item.accepted)
    rejected = tuple(item for item in derivations if not item.accepted)
    membership = _membership_sha256([item.example_id for item in accepted])
    counts = Counter(
        item.selected_opcode[0] for item in accepted if item.selected_opcode is not None
    )
    clocks = sum(item.selected_time_ref is not None for item in derivations)
    if len(accepted) != ACCEPTED_ROWS or membership != ACCEPTED_MEMBERSHIP_SHA256:
        raise MobilePlanIRV2Error("accepted construction membership changed")
    if dict(counts) != EXPECTED_DATE_OPERATOR_COUNTS:
        raise MobilePlanIRV2Error(f"date operator counts changed: {dict(counts)!r}")
    if clocks != EXPECTED_CLOCK_REFERENCES:
        raise MobilePlanIRV2Error(f"clock count changed: {clocks}")
    if [(item.example_id, item.code) for item in rejected] != [
        (EXPECTED_REJECTION_ID, EXPECTED_REJECTION_CODE)
    ]:
        raise MobilePlanIRV2Error("construction rejection evidence changed")
    return OracleAudit(
        use=CONSTRUCTION_USE,
        source_rows=len(examples),
        accepted_rows=len(accepted),
        rejected_rows=len(rejected),
        accepted_membership_sha256=membership,
        date_operator_counts=dict(counts),
        clock_references=clocks,
        rejections=rejected,
    )


def _distribution(values: Sequence[int]) -> Distribution:
    ordered = sorted(values)
    if not ordered:
        raise MobilePlanIRV2Error("cannot summarize an empty population")
    p90_index = max(0, (9 * len(ordered) + 9) // 10 - 1)
    return Distribution(
        total=sum(ordered),
        mean=statistics.fmean(ordered),
        median=float(statistics.median(ordered)),
        p90=ordered[p90_index],
        maximum=ordered[-1],
    )


def audit_construction_tokens(
    manifest_path: str | Path,
    tokenizer_path: str | Path,
    *,
    max_sequence_length: int = 2_048,
) -> TokenAudit:
    if (
        not isinstance(max_sequence_length, int)
        or isinstance(max_sequence_length, bool)
        or max_sequence_length <= 0
    ):
        raise MobilePlanIRV2Error("max_sequence_length must be a positive integer")
    if sha256_file(tokenizer_path) != PINNED_TOKENIZER_SHA256:
        raise MobilePlanIRV2Error("pinned tokenizer hash changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    examples = _load_pinned_construction_manifest(manifest_path)
    lengths = {arm: {"prompt": [], "target": [], "sequence": []} for arm in ("A", "B", "C")}
    rejected = 0
    nonempty = 0
    date_refs = 0
    time_refs = 0
    target_deltas: list[int] = []
    accepted_ids: list[str] = []
    rejection_ids: list[str] = []
    for example in examples:
        oracle = _derive_oracle(example)
        if not oracle.accepted:
            rejected += 1
            rejection_ids.append(oracle.example_id)
            continue
        accepted_ids.append(oracle.example_id)
        parsed = _parse_input(example)
        nonempty += parsed.table.render() is not None
        date_refs += len(parsed.table.date_refs)
        time_refs += len(parsed.table.time_refs)
        for arm in ("A", "B", "C"):
            prompt = _render_prompt(example, arm, parsed=parsed).prompt
            target = example.target if arm != "C" else oracle.target
            assert target is not None
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            target_tokens = len(tokenizer.encode(target, add_special_tokens=False).ids)
            lengths[arm]["prompt"].append(prompt_tokens)
            lengths[arm]["target"].append(target_tokens)
            lengths[arm]["sequence"].append(prompt_tokens + target_tokens + 1)
        if lengths["B"]["prompt"][-1] != lengths["C"]["prompt"][-1]:
            raise MobilePlanIRV2Error(
                f"example {example.example_id!r} B/C prompt token budgets differ"
            )
        target_deltas.append(lengths["C"]["target"][-1] - lengths["A"]["target"][-1])
    membership = _membership_sha256(accepted_ids)
    if (
        rejected != 1
        or len(target_deltas) != ACCEPTED_ROWS
        or membership != ACCEPTED_MEMBERSHIP_SHA256
        or rejection_ids != [EXPECTED_REJECTION_ID]
    ):
        raise MobilePlanIRV2Error("token audit oracle membership changed")

    def arm_audit(arm: Arm) -> ArmTokenAudit:
        return ArmTokenAudit(
            prompt=_distribution(lengths[arm]["prompt"]),
            target=_distribution(lengths[arm]["target"]),
            sequence=_distribution(lengths[arm]["sequence"]),
        )

    a = arm_audit("A")
    b = arm_audit("B")
    c = arm_audit("C")
    return TokenAudit(
        use=CONSTRUCTION_USE,
        manifest_sha256=CONSTRUCTION_MANIFEST_SHA256,
        tokenizer_sha256=PINNED_TOKENIZER_SHA256,
        compared_rows=ACCEPTED_ROWS,
        rejected_rows=rejected,
        accepted_membership_sha256=membership,
        rejection_ids=tuple(rejection_ids),
        max_sequence_length=max_sequence_length,
        arm_a=a,
        arm_b=b,
        arm_c=c,
        table_nonempty_rows=nonempty,
        date_reference_instances=date_refs,
        time_reference_instances=time_refs,
        c_targets_shorter_than_a=sum(delta < 0 for delta in target_deltas),
        c_targets_equal_to_a=sum(delta == 0 for delta in target_deltas),
        c_targets_longer_than_a=sum(delta > 0 for delta in target_deltas),
        over_limit_a=sum(value > max_sequence_length for value in lengths["A"]["sequence"]),
        over_limit_b=sum(value > max_sequence_length for value in lengths["B"]["sequence"]),
        over_limit_c=sum(value > max_sequence_length for value in lengths["C"]["sequence"]),
    )


__all__ = [
    "ACCEPTED_MEMBERSHIP_SHA256",
    "ACCEPTED_ROWS",
    "CONSTRUCTION_MANIFEST_SHA256",
    "CONSTRUCTION_ROWS",
    "CONSTRUCTION_USE",
    "EXPECTED_CLOCK_REFERENCES",
    "EXPECTED_DATE_OPERATOR_COUNTS",
    "EXPECTED_REJECTION_CODE",
    "EXPECTED_REJECTION_ID",
    "PINNED_TOKENIZER_SHA256",
    "ArmTokenAudit",
    "Distribution",
    "MaterializedArmExample",
    "MobilePlanIRV2Error",
    "OracleAudit",
    "OracleDerivation",
    "TokenAudit",
    "audit_construction_oracles",
    "audit_construction_tokens",
    "materialize_construction_arm",
]
