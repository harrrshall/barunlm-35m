from __future__ import annotations

import hashlib
import inspect
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import barunlm.datasets.mobile_temporal_counterfactual as temporal
from barunlm.datasets.mobile_temporal_counterfactual import (
    EXPECTED_CONSTRUCTION_ELIGIBLE_BY_CLASS,
    EXPECTED_CONSTRUCTION_ELIGIBLE_MEMBERSHIP_SHA256,
    EXPECTED_CONSTRUCTION_ELIGIBLE_ROWS,
    EXPECTED_FULL_ELIGIBLE_BY_CLASS,
    EXPECTED_FULL_ELIGIBLE_MEMBERSHIP_SHA256,
    EXPECTED_FULL_ELIGIBLE_ROWS,
    EXPECTED_SHADOW_MEMBERSHIP_SHA256,
    EXPECTED_SHADOW_ROWS,
    PINNED_MOBILE_TRAIN_MEMBERSHIP_SHA256,
    PINNED_MOBILE_TRAIN_ROWS,
    PINNED_MOBILE_TRAIN_SHA256,
    TEMPORAL_SPLIT_VERSION,
    MobileTemporalError,
    TemporalTokenizerIdentity,
    build_month_boundary_variant,
    classify_counterfactual_source,
    load_pinned_mobile_train_manifest,
    materialize_full_counterfactual_view,
    materialize_temporal_views,
    membership_sha256,
    plan_temporal_views,
    shadow_role,
)
from barunlm.training.data import SFTExample, sha256_file

REAL_TRAIN_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "experiments/runs/20260803-1810-mobile-blind-s17/remote/data/train.jsonl"
)


class _CharacterTokenizer:
    def encode(self, sequence: str, add_special_tokens: bool = False) -> SimpleNamespace:
        assert not add_special_tokens
        return SimpleNamespace(ids=list(sequence.encode("utf-8")))


TOKENIZER = _CharacterTokenizer()
TOKENIZER_IDENTITY = TemporalTokenizerIdentity(
    identifier="test/byte-tokenizer",
    revision="test-revision",
    sha256="a" * 64,
)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _example(
    example_id: str,
    *,
    now: str = "2026-08-03T16:30:00",
    user: str,
    calendar_datetime: str | None,
    extra_calls: tuple[dict[str, Any], ...] = (),
    cluster_id: str | None = None,
    family_id: str | None = None,
    second_calendar_datetime: str | None = None,
) -> SFTExample:
    calls: list[dict[str, Any]] = []
    if calendar_datetime is not None:
        calls.append(
            {
                "args": {"datetime": calendar_datetime, "title": "Team Sync"},
                "tool": "create_calendar_event",
            }
        )
    calls.extend(extra_calls)
    if second_calendar_datetime is not None:
        calls.append(
            {
                "args": {"datetime": second_calendar_datetime, "title": "Review"},
                "tool": "create_calendar_event",
            }
        )
    if not calls:
        calls.append(
            {
                "args": {"query": "coffee"},
                "tool": "search_maps",
            }
        )
    mode = "SINGLE" if len(calls) == 1 else "SERIAL"
    target = _canonical({"calls": calls, "decision": "CALL", "mode": mode})
    prompt = (
        "<bos><system>\n"
        "ACTION_IR_V1\n"
        f"NOW {now}\n"
        "TOOLS\n"
        "create_calendar_event(datetime:string!, title:string!)\n"
        "send_email(subject:string!, to:string!)\n"
        "search_maps(query:string!)\n"
        "<user>\n"
        f"{user}\n"
        "<assistant>\n"
    )
    call_names = [str(call["tool"]) for call in calls]
    metadata = {
        "source_split": "train",
        "derived_split": "train",
        "cluster_id": cluster_id or f"cluster-{example_id}",
        "family_id": family_id or f"family-{example_id}",
        "call_names": call_names,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
        "token_lengths": {
            "prompt": len(prompt.encode()),
            "target": len(target.encode()),
            "total_with_eos": len(prompt.encode()) + len(target.encode()) + 1,
        },
    }
    content_sha256 = hashlib.sha256(
        _canonical(
            {
                "schema_version": "barun-sft-example-v1",
                "id": example_id,
                "prompt": prompt,
                "target": target,
                "metadata": metadata,
            }
        ).encode()
    ).hexdigest()
    return SFTExample(
        example_id=example_id,
        prompt=prompt,
        target=target,
        metadata=metadata,
        content_sha256=content_sha256,
    )


def _cluster_for(role: str, marker: str) -> str:
    for index in range(10_000):
        candidate = f"cluster-{marker}-{index}"
        if shadow_role(candidate) == role:
            return candidate
    raise AssertionError(f"could not find cluster for {role}")


def _nested_values(value: object, key: str) -> list[object]:
    found: list[object] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key == key:
                found.append(child)
            found.extend(_nested_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_nested_values(child, key))
    return found


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_shadow_role_matches_frozen_unsigned_first_eight_byte_policy() -> None:
    roles_by_fold = {
        **{fold: "construction_train" for fold in range(14)},
        **{fold: "selection" for fold in range(14, 17)},
        **{fold: "confirmation" for fold in range(17, 20)},
    }
    for index in range(100):
        cluster_id = f"independent-component-{index}"
        digest = hashlib.sha256(f"{TEMPORAL_SPLIT_VERSION}:{cluster_id}".encode()).digest()
        fold = int.from_bytes(digest[:8], byteorder="big") % 20
        assert shadow_role(cluster_id) == roles_by_fold[fold]


def test_plan_keeps_components_and_families_disjoint() -> None:
    construction_cluster = _cluster_for("construction_train", "construction")
    selection_cluster = _cluster_for("selection", "selection")
    confirmation_cluster = _cluster_for("confirmation", "confirmation")
    rows = (
        _example(
            "construction-a",
            user="Schedule Team Sync on August 21 at 9 AM.",
            calendar_datetime="2026-08-21T09:00:00",
            cluster_id=construction_cluster,
            family_id="construction-family",
        ),
        _example(
            "construction-b",
            user="Schedule Review on August 22 at 10 AM.",
            calendar_datetime="2026-08-22T10:00:00",
            cluster_id=construction_cluster,
            family_id="construction-family",
        ),
        _example(
            "selection-a",
            user="Schedule Team Sync on August 23 at 9 AM.",
            calendar_datetime="2026-08-23T09:00:00",
            cluster_id=selection_cluster,
            family_id="selection-family",
        ),
        _example(
            "confirmation-a",
            user="Schedule Team Sync on August 24 at 9 AM.",
            calendar_datetime="2026-08-24T09:00:00",
            cluster_id=confirmation_cluster,
            family_id="confirmation-family",
        ),
    )
    plan = plan_temporal_views(rows)
    assert tuple(row.example_id for row in plan.roles["construction_train"]) == (
        "construction-a",
        "construction-b",
    )
    assert tuple(row.example_id for row in plan.roles["selection"]) == ("selection-a",)
    assert tuple(row.example_id for row in plan.roles["confirmation"]) == ("confirmation-a",)

    crossing_family = _example(
        "selection-crossing-family",
        user="Schedule Team Sync on August 25 at 9 AM.",
        calendar_datetime="2026-08-25T09:00:00",
        cluster_id=selection_cluster,
        family_id="construction-family",
    )
    with pytest.raises(MobileTemporalError, match="families cross shadow roles"):
        plan_temporal_views((*rows[:2], crossing_family, rows[3]))


def test_absolute_transform_changes_only_now_and_keeps_target_bytes() -> None:
    email_call = {
        "args": {"subject": "Agenda", "to": "asha@example.com"},
        "tool": "send_email",
    }
    source = _example(
        "absolute",
        user="Schedule Team Sync on August 21 at 9 AM, then email Asha.",
        calendar_datetime="2026-08-21T09:00:00",
        extra_calls=(email_call,),
    )
    decision = classify_counterfactual_source(source)
    assert decision.eligible
    assert decision.kind == "absolute_explicit"

    variant = build_month_boundary_variant(source, tokenizer=TOKENIZER)
    assert variant is not None
    record = variant.record
    assert record["target"] == source.target
    assert record["prompt"] == source.prompt.replace(
        "NOW 2026-08-03T16:30:00",
        "NOW 2026-07-31T16:30:00",
    )
    assert _nested_values(record["metadata"], "source_id") == [source.example_id]
    view = record["metadata"]["mobile_temporal_view"]
    assert view["original_datetime"] == view["counterfactual_datetime"]
    assert view["shift_days"] is None
    assert (
        record["metadata"]["prompt_sha256"]
        == hashlib.sha256(str(record["prompt"]).encode()).hexdigest()
    )
    assert (
        record["metadata"]["target_sha256"]
        == hashlib.sha256(str(record["target"]).encode()).hexdigest()
    )


def test_relative_transform_preserves_delta_weekday_time_user_and_other_calls() -> None:
    email_call = {
        "args": {"subject": "Agenda", "to": "asha@example.com"},
        "tool": "send_email",
    }
    source = _example(
        "relative",
        user="Schedule Team Sync next Tuesday at 9 AM, then email Asha.",
        calendar_datetime="2026-08-11T09:00:00",
        extra_calls=(email_call,),
    )
    decision = classify_counterfactual_source(source)
    assert decision.eligible
    assert decision.kind == "pure_relative"

    first = build_month_boundary_variant(source, tokenizer=TOKENIZER)
    second = build_month_boundary_variant(source, tokenizer=TOKENIZER)
    assert first is not None and second is not None
    assert first.record == second.record
    assert first.receipt == second.receipt

    source_target = json.loads(source.target)
    variant_target = json.loads(str(first.record["target"]))
    source_datetime = datetime.fromisoformat(source_target["calls"][0]["args"]["datetime"])
    variant_datetime = datetime.fromisoformat(variant_target["calls"][0]["args"]["datetime"])
    source_now = datetime.fromisoformat("2026-08-03T16:30:00")
    variant_now = datetime.fromisoformat(
        first.record["metadata"]["mobile_temporal_view"]["counterfactual_now"]
    )
    shift_days = first.record["metadata"]["mobile_temporal_view"]["shift_days"]
    assert shift_days != 0
    assert shift_days % 7 == 0
    assert variant_datetime - variant_now == source_datetime - source_now
    assert variant_datetime.weekday() == source_datetime.weekday()
    assert variant_datetime.time() == source_datetime.time()
    assert (variant_datetime.year, variant_datetime.month) != (
        variant_now.year,
        variant_now.month,
    )
    assert variant_target["calls"][1] == source_target["calls"][1]
    assert "<user>\n" + source.prompt.split("<user>\n", 1)[1] in str(first.record["prompt"])


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (
            _example(
                "mixed",
                user="Schedule Team Sync tomorrow, August 4, at 9 AM.",
                calendar_datetime="2026-08-04T09:00:00",
            ),
            "mixed_ambiguous_or_unsupported",
        ),
        (
            _example(
                "same-day",
                user="Schedule Team Sync today at 6 PM.",
                calendar_datetime="2026-08-03T18:00:00",
            ),
            "same_day_relative_cannot_cross",
        ),
        (
            _example(
                "numeric",
                user="Schedule Team Sync on 08/21 at 9 AM.",
                calendar_datetime="2026-08-21T09:00:00",
            ),
            "mixed_ambiguous_or_unsupported",
        ),
        (
            _example(
                "multiple",
                user="Schedule Team Sync and Review next Tuesday.",
                calendar_datetime="2026-08-11T09:00:00",
                second_calendar_datetime="2026-08-11T10:00:00",
            ),
            "multiple_calendar_calls",
        ),
    ],
)
def test_ambiguous_or_unverifiable_forms_are_rejected(
    source: SFTExample,
    reason: str,
) -> None:
    decision = classify_counterfactual_source(source)
    assert not decision.eligible
    assert decision.kind is None
    assert decision.reason == reason
    assert build_month_boundary_variant(source, tokenizer=TOKENIZER) is None


def test_materialization_is_byte_deterministic_and_records_zero_heldout_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction_cluster = _cluster_for("construction_train", "materialize-construction")
    construction_other_cluster = _cluster_for("construction_train", "materialize-other")
    selection_cluster = _cluster_for("selection", "materialize-selection")
    confirmation_cluster = _cluster_for("confirmation", "materialize-confirmation")
    rows = (
        _example(
            "construction-eligible",
            user="Schedule Team Sync on August 21 at 9 AM.",
            calendar_datetime="2026-08-21T09:00:00",
            cluster_id=construction_cluster,
            family_id="materialize-family-construction",
        ),
        _example(
            "construction-no-calendar",
            user="Find coffee nearby.",
            calendar_datetime=None,
            cluster_id=construction_other_cluster,
            family_id="materialize-family-other",
        ),
        _example(
            "selection",
            user="Schedule Team Sync on August 22 at 9 AM.",
            calendar_datetime="2026-08-22T09:00:00",
            cluster_id=selection_cluster,
            family_id="materialize-family-selection",
        ),
        _example(
            "confirmation",
            user="Schedule Team Sync on August 23 at 9 AM.",
            calendar_datetime="2026-08-23T09:00:00",
            cluster_id=confirmation_cluster,
            family_id="materialize-family-confirmation",
        ),
    )
    monkeypatch.setattr(temporal, "load_pinned_mobile_train_manifest", lambda _path: rows)
    source = tmp_path / "opaque-train.jsonl"
    first = materialize_temporal_views(
        source_train_path=source,
        tokenizer=TOKENIZER,
        tokenizer_identity=TOKENIZER_IDENTITY,
        output_dir=tmp_path / "first",
        minimum_variants=1,
        enforce_pinned_shadow=False,
    )
    second = materialize_temporal_views(
        source_train_path=source,
        tokenizer=TOKENIZER,
        tokenizer_identity=TOKENIZER_IDENTITY,
        output_dir=tmp_path / "second",
        minimum_variants=1,
        enforce_pinned_shadow=False,
    )
    assert first.counts == {
        "construction": 2,
        "selection": 1,
        "confirmation": 1,
        "variants": 1,
        "repeat": 3,
        "counterfactual": 3,
    }
    assert first.hashes == second.hashes
    assert {path.name for path in first.output_dir.iterdir()} == {
        "construction-train.jsonl",
        "selection.jsonl",
        "confirmation.jsonl",
        "repeat-train.jsonl",
        "counterfactual-train.jsonl",
        "transform-receipts.jsonl",
        "shadow-audit.json",
        "view-audit.json",
    }
    for first_path in sorted(first.output_dir.iterdir()):
        second_path = second.output_dir / first_path.name
        assert first_path.read_bytes() == second_path.read_bytes()

    selection = _read_jsonl(first.selection_manifest)
    confirmation = _read_jsonl(first.confirmation_manifest)
    assert selection[0]["metadata"]["derived_split"] == "dev"
    assert confirmation[0]["metadata"]["derived_split"] == "confirmation"
    counterfactual = _read_jsonl(first.counterfactual_manifest)
    added = [row for row in counterfactual if "mobile_temporal_view" in row["metadata"]]
    assert len(added) == 1
    assert _nested_values(added[0]["metadata"], "source_id") == ["construction-eligible"]

    shadow_audit = json.loads(first.shadow_audit_path.read_text(encoding="utf-8"))
    view_audit = json.loads(first.view_audit_path.read_text(encoding="utf-8"))
    assert shadow_audit["development_rows_read"] == 0
    assert shadow_audit["official_evaluation_rows_read"] == 0
    assert view_audit["leakage"] == {
        "selection_source_rows_used_for_variants": 0,
        "confirmation_source_rows_used_for_variants": 0,
        "development_rows_read": 0,
        "official_evaluation_rows_read": 0,
    }
    assert view_audit["matched_control"]["same_source_ids"] is True
    assert view_audit["matched_control"]["same_added_rows"] is True


def test_public_materializer_cannot_accept_dev_or_official_eval_inputs() -> None:
    screening_parameters = inspect.signature(materialize_temporal_views).parameters
    assert set(screening_parameters) == {
        "source_train_path",
        "tokenizer",
        "tokenizer_identity",
        "output_dir",
        "max_seq_len",
        "minimum_variants",
        "minimum_cross_month_fraction",
        "enforce_pinned_shadow",
    }
    full_parameters = inspect.signature(materialize_full_counterfactual_view).parameters
    assert set(full_parameters) == {
        "source_train_path",
        "tokenizer",
        "tokenizer_identity",
        "output_dir",
        "max_seq_len",
        "minimum_variants",
        "minimum_cross_month_fraction",
    }
    heldout_names = {"dev", "development", "eval", "evaluation", "test"}
    assert not heldout_names & set(screening_parameters)
    assert not heldout_names & set(full_parameters)


def test_full_view_materializer_is_callable_and_audits_zero_heldout_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = (
        _example(
            "full-eligible",
            user="Schedule Team Sync on August 21 at 9 AM.",
            calendar_datetime="2026-08-21T09:00:00",
        ),
        _example(
            "full-no-calendar",
            user="Find coffee nearby.",
            calendar_datetime=None,
        ),
    )
    monkeypatch.setattr(temporal, "load_pinned_mobile_train_manifest", lambda _path: rows)
    monkeypatch.setattr(temporal, "EXPECTED_FULL_ELIGIBLE_ROWS", 1)
    monkeypatch.setattr(
        temporal,
        "EXPECTED_FULL_ELIGIBLE_BY_CLASS",
        {"absolute_explicit": 1},
    )
    monkeypatch.setattr(
        temporal,
        "EXPECTED_FULL_ELIGIBLE_MEMBERSHIP_SHA256",
        membership_sha256(["full-eligible"]),
    )
    monkeypatch.setattr(
        temporal,
        "PINNED_MOBILE_TOKENIZER_IDENTIFIER",
        TOKENIZER_IDENTITY.identifier,
    )
    monkeypatch.setattr(
        temporal,
        "PINNED_MOBILE_TOKENIZER_REVISION",
        TOKENIZER_IDENTITY.revision,
    )
    monkeypatch.setattr(
        temporal,
        "PINNED_MOBILE_TOKENIZER_SHA256",
        TOKENIZER_IDENTITY.sha256,
    )

    artifacts = materialize_full_counterfactual_view(
        source_train_path=tmp_path / "opaque-train.jsonl",
        tokenizer=TOKENIZER,
        tokenizer_identity=TOKENIZER_IDENTITY,
        output_dir=tmp_path / "full",
        minimum_variants=1,
    )
    assert artifacts.source_rows == 2
    assert artifacts.variant_rows == 1
    assert len(_read_jsonl(artifacts.train_manifest)) == 3
    assert len(_read_jsonl(artifacts.receipts_path)) == 1
    assert artifacts.hashes["train"] == sha256_file(artifacts.train_manifest)
    assert artifacts.hashes["receipts"] == sha256_file(artifacts.receipts_path)
    assert artifacts.hashes["audit"] == sha256_file(artifacts.audit_path)
    audit = json.loads(artifacts.audit_path.read_text(encoding="utf-8"))
    assert audit["development_rows_read"] == 0
    assert audit["official_evaluation_rows_read"] == 0


@pytest.mark.skipif(
    not REAL_TRAIN_MANIFEST.is_file(),
    reason="the immutable 17 MiB Mobile train artifact is intentionally not in Git",
)
def test_real_pinned_train_manifest_split_and_eligibility_counts() -> None:
    assert sha256_file(REAL_TRAIN_MANIFEST) == PINNED_MOBILE_TRAIN_SHA256
    examples = load_pinned_mobile_train_manifest(REAL_TRAIN_MANIFEST)
    assert len(examples) == PINNED_MOBILE_TRAIN_ROWS == 7_937
    assert (
        membership_sha256(example.example_id for example in examples)
        == PINNED_MOBILE_TRAIN_MEMBERSHIP_SHA256
    )

    plan = plan_temporal_views(examples)
    assert {
        role: int(summary["rows"]) for role, summary in plan.role_summaries.items()
    } == EXPECTED_SHADOW_ROWS
    assert {
        role: str(summary["membership_sha256"]) for role, summary in plan.role_summaries.items()
    } == EXPECTED_SHADOW_MEMBERSHIP_SHA256
    assert plan.role_summaries["construction_train"]["calendar"] == {
        "rows": 2_070,
        "same_month": 1_686,
        "cross_month": 384,
    }
    assert len(plan.eligible) == EXPECTED_CONSTRUCTION_ELIGIBLE_ROWS == 1_093
    assert {
        kind: sum(item.kind == kind for item in plan.eligible)
        for kind in EXPECTED_CONSTRUCTION_ELIGIBLE_BY_CLASS
    } == EXPECTED_CONSTRUCTION_ELIGIBLE_BY_CLASS
    assert (
        membership_sha256(item.parsed.example.example_id for item in plan.eligible)
        == EXPECTED_CONSTRUCTION_ELIGIBLE_MEMBERSHIP_SHA256
    )

    full_decisions = [classify_counterfactual_source(example) for example in examples]
    full_eligible = [decision for decision in full_decisions if decision.eligible]
    assert len(full_eligible) == EXPECTED_FULL_ELIGIBLE_ROWS == 1_546
    assert {
        kind: sum(item.kind == kind for item in full_eligible)
        for kind in EXPECTED_FULL_ELIGIBLE_BY_CLASS
    } == EXPECTED_FULL_ELIGIBLE_BY_CLASS
    assert (
        membership_sha256(item.parsed.example.example_id for item in full_eligible)
        == EXPECTED_FULL_ELIGIBLE_MEMBERSHIP_SHA256
    )
