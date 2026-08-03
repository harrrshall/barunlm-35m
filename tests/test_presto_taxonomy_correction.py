from __future__ import annotations

import json

import pytest

from barunlm.evaluation.presto import (
    PRESTO_PHENOMENON_TAXONOMY_VERSION,
    PRESTO_SCORER_VERSION,
    PRESTO_SCORER_VERSION_V1,
    PRESTO_USER_REVISION_ALIAS_SET_VERSION,
    PRESTO_USER_REVISION_RAW_LABELS_V2,
    phenomenon_group,
)
from barunlm.evaluation.presto_taxonomy_correction import (
    LEGACY_PROTOCOL_PATH,
    LEGACY_PROTOCOL_SHA256,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    PrestoTaxonomyCorrectionError,
    _corrected_metrics,
    _gate,
    _sha256_file,
)
from barunlm.presto_bundle_audit import _original_phenomenon_group


def _row(
    sample_id: str,
    phenomenon: str,
    original_group: str,
    *,
    exact: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": PRESTO_SCORER_VERSION_V1,
        "sample_id": sample_id,
        "phenomenon": phenomenon,
        "phenomenon_group": original_group,
        "ast_exact": exact,
        "schema_valid": True,
        "decision_correct": True,
        "gold_decision": "CALL",
        "false_call_on_gate": False,
        "truncated": False,
        "generation_failure": None,
    }


def test_protocol_is_pinned_and_explicitly_post_hoc() -> None:
    assert _sha256_file(PROTOCOL_PATH) == PROTOCOL_SHA256
    assert _sha256_file(LEGACY_PROTOCOL_PATH) == LEGACY_PROTOCOL_SHA256
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["schema_version"] == "barun-presto-taxonomy-posthoc-protocol-v2"
    assert protocol["status"] == "post_hoc_after_original_result"
    assert protocol["supersedes"]["sha256"] == LEGACY_PROTOCOL_SHA256
    assert protocol["correction"]["alias_set_version"] == (PRESTO_USER_REVISION_ALIAS_SET_VERSION)
    assert protocol["correction"]["raw_labels"] == sorted(PRESTO_USER_REVISION_RAW_LABELS_V2)
    assert protocol["procedure"]["rewrite_original_bundle_or_result"] is False
    assert protocol["procedure"]["rerun_training"] is False
    assert protocol["procedure"]["read_official_test"] is False


def test_taxonomy_v2_alias_is_complete_and_legacy_v1_stays_reproducible() -> None:
    assert PRESTO_SCORER_VERSION_V1 == "barun-presto-action-ir-score-v1"
    assert PRESTO_SCORER_VERSION == "barun-presto-action-ir-score-v2"
    assert PRESTO_PHENOMENON_TAXONOMY_VERSION == "barun-presto-phenomenon-taxonomy-v2"
    for label in PRESTO_USER_REVISION_RAW_LABELS_V2:
        assert _original_phenomenon_group(label) == "other"
        assert phenomenon_group(label) == "revision"


def test_posthoc_reclassification_changes_full_revision_family_and_recomputes_gate() -> None:
    rows = [
        _row("simple", "", "no_phenomenon"),
        *[
            _row(f"revision-{label}", label, "other")
            for label in sorted(PRESTO_USER_REVISION_RAW_LABELS_V2)
        ],
        _row("disfluency", "disfluency", "disfluency"),
        _row("code", "code-mixing", "other", exact=False),
    ]
    original = {
        "schema_version": PRESTO_SCORER_VERSION_V1,
        "sample_count": 7,
        "ast_exact_match": {"numerator": 6, "denominator": 7, "value": 6 / 7},
        "schema_valid": {"numerator": 7, "denominator": 7, "value": 1.0},
        "abstention": {"f1": 1.0},
        "false_call_on_gate": {"numerator": 0, "denominator": 2, "value": 0.0},
        "headline_phenomenon_buckets": {},
    }
    corrected, changed, raw_counts, subgroups = _corrected_metrics(rows, original)

    assert changed == 4
    assert raw_counts["within-turn-correction"] == 1
    assert corrected["schema_version"] == PRESTO_SCORER_VERSION
    assert corrected["headline_phenomenon_buckets"]["revision"]["count"] == 4
    assert corrected["headline_phenomenon_buckets"]["other"]["count"] == 1
    assert set(subgroups) == PRESTO_USER_REVISION_RAW_LABELS_V2
    assert all(metrics["count"] == 1 for metrics in subgroups.values())
    thresholds = {
        "derived_ast_exact_at_least": 0.7,
        "schema_valid_at_least": 0.95,
        "abstention_f1_at_least": 0.8,
        "false_call_on_gate_at_most": 0.01,
        "maximum_no_phenomenon_to_revision_or_disfluency_gap": 0.1,
        "require_represented_gap_buckets": True,
    }
    gate = _gate(corrected, thresholds)
    assert gate["passed"] is True
    assert gate["observed"]["no_phenomenon_minus_hard_bucket_gap"] == {
        "revision": 0.0,
        "disfluency": 0.0,
    }


@pytest.mark.parametrize(
    ("raw_label", "original_group"),
    [("correct_action", "other"), ("code-mixing", "revision")],
)
def test_posthoc_protocol_rejects_any_unapproved_taxonomy_change(
    raw_label: str, original_group: str
) -> None:
    rows = [_row("wrong", raw_label, original_group)]
    with pytest.raises(PrestoTaxonomyCorrectionError, match="unapproved taxonomy change"):
        _corrected_metrics(rows, {"schema_version": PRESTO_SCORER_VERSION_V1})
