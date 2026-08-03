"""Post-hoc, non-retroactive audit for PRESTO's user-revision taxonomy."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from barunlm.presto_bundle_audit import (
    PrestoBundleAuditError,
    validate_presto_bundle,
)

from .presto import (
    PRESTO_PHENOMENON_TAXONOMY_VERSION,
    PRESTO_SCORER_VERSION,
    PRESTO_SCORER_VERSION_V1,
    PRESTO_USER_REVISION_ALIAS_SET_VERSION,
    PRESTO_USER_REVISION_RAW_LABELS_V2,
    phenomenon_group,
)

LEGACY_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "presto_taxonomy_posthoc_v1.json"
)
LEGACY_PROTOCOL_SHA256 = "64a02c519b259c76f3dcee00fef74c0221342797185773602cca72a7154e89d0"
PROTOCOL_PATH = Path(__file__).resolve().parents[3] / "configs" / "presto_taxonomy_posthoc_v2.json"
PROTOCOL_SHA256 = "71a15ca2ad31914e21f5a9d75ea8ee68370085434d12ef6feb961da6670bb7ca"
CORRECTION_SCHEMA_VERSION = "barun-presto-taxonomy-posthoc-result-v2"


class PrestoTaxonomyCorrectionError(RuntimeError):
    """The immutable evidence cannot support the bounded post-hoc correction."""


@dataclass(frozen=True, slots=True)
class PrestoTaxonomyCorrectionResult:
    schema_version: str
    audit_passed: bool
    status: str
    correction_id: str
    protocol_sha256: str
    source_run_id: str
    source_artifact_manifest_sha256: str
    source_result_sha256: str
    source_output_model_sha256: str
    original_scorer_version: str
    corrected_scorer_version: str
    corrected_taxonomy_version: str
    revision_alias_set_version: str
    revision_raw_labels: Sequence[str]
    rows_reclassified: Mapping[str, int]
    raw_phenomenon_counts: Mapping[str, int]
    original_focus_revision: Mapping[str, int]
    original_gate: Mapping[str, Any]
    corrected_base: Mapping[str, Any]
    corrected_post_sft: Mapping[str, Any]
    revision_raw_tag_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]]
    corrected_gate: Mapping[str, Any]
    training_intervention_corrected: bool
    original_result_rewritten: bool
    claim_limits: Sequence[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fail(message: str) -> None:
    raise PrestoTaxonomyCorrectionError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    _fail(f"JSON contains non-finite constant {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PrestoTaxonomyCorrectionError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        _fail(f"JSON artifact is not an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    _fail(f"blank JSONL row at {path}:{line_number}")
                value = json.loads(
                    line,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
                if not isinstance(value, dict):
                    _fail(f"non-object JSONL row at {path}:{line_number}")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PrestoTaxonomyCorrectionError(f"invalid JSONL artifact {path}: {error}") from error
    return rows


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        _fail(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _rate(numerator: int, denominator: int) -> dict[str, int | float]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else 0.0,
    }


def _bucket(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "count": count,
        "ast_exact_match": _rate(sum(row.get("ast_exact") is True for row in rows), count),
        "schema_valid": _rate(sum(row.get("schema_valid") is True for row in rows), count),
        "decision_accuracy": _rate(sum(row.get("decision_correct") is True for row in rows), count),
        "truncation": _rate(sum(row.get("truncated") is True for row in rows), count),
        "generation_failure": _rate(
            sum(row.get("generation_failure") is not None for row in rows), count
        ),
    }


def _revision_raw_tag_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for label in sorted(PRESTO_USER_REVISION_RAW_LABELS_V2):
        subgroup = [row for row in rows if row.get("phenomenon") == label]
        gate_rows: list[Mapping[str, Any]] = []
        gold_decisions: Counter[str] = Counter()
        for index, row in enumerate(subgroup):
            gold_decision = row.get("gold_decision")
            if gold_decision not in {"ABSTAIN", "CALL", "CONFIRM"}:
                _fail(f"{label} sample {index} has invalid gold_decision")
            false_call = row.get("false_call_on_gate")
            if type(false_call) is not bool:
                _fail(f"{label} sample {index} has invalid false_call_on_gate")
            gold_decisions[str(gold_decision)] += 1
            if gold_decision in {"ABSTAIN", "CONFIRM"}:
                gate_rows.append(row)
        metrics = _bucket(subgroup)
        metrics["false_call_on_gate"] = _rate(
            sum(row["false_call_on_gate"] is True for row in gate_rows),
            len(gate_rows),
        )
        metrics["gold_decision_counts"] = dict(sorted(gold_decisions.items()))
        result[label] = metrics
    return result


def _corrected_metrics(
    rows: Sequence[Mapping[str, Any]],
    original: Mapping[str, Any],
) -> tuple[dict[str, Any], int, dict[str, int], dict[str, Mapping[str, Any]]]:
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    raw_counts: dict[str, int] = defaultdict(int)
    changed = 0
    for index, row in enumerate(rows):
        _expect(
            row.get("schema_version"),
            PRESTO_SCORER_VERSION_V1,
            f"sample {index} original scorer",
        )
        raw = row.get("phenomenon")
        original_group = row.get("phenomenon_group")
        if not isinstance(raw, str) or not isinstance(original_group, str):
            _fail(f"sample {index} lacks phenomenon taxonomy evidence")
        corrected_group = phenomenon_group(raw)
        raw_counts[raw] += 1
        if raw in PRESTO_USER_REVISION_RAW_LABELS_V2:
            _expect(original_group, "other", f"sample {index} original group")
            _expect(corrected_group, "revision", f"sample {index} corrected group")
            changed += 1
        elif corrected_group != original_group:
            _fail(f"unapproved taxonomy change for raw label {raw!r}")
        by_group[corrected_group].append(row)
    corrected = dict(original)
    corrected["schema_version"] = PRESTO_SCORER_VERSION
    corrected["headline_phenomenon_buckets"] = {
        name: _bucket(bucket) for name, bucket in sorted(by_group.items())
    }
    return (
        corrected,
        changed,
        dict(sorted(raw_counts.items())),
        _revision_raw_tag_metrics(rows),
    )


def _gate(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    exact = _number(_mapping(metrics.get("ast_exact_match"), "exact").get("value"), "exact")
    schema = _number(_mapping(metrics.get("schema_valid"), "schema").get("value"), "schema")
    abstention = _number(
        _mapping(metrics.get("abstention"), "abstention").get("f1"), "abstention F1"
    )
    false_call = _number(
        _mapping(metrics.get("false_call_on_gate"), "false call").get("value"),
        "false call",
    )
    buckets = _mapping(metrics.get("headline_phenomenon_buckets"), "phenomenon buckets")
    simple = _mapping(buckets.get("no_phenomenon"), "no-phenomenon bucket")
    revision = _mapping(buckets.get("revision"), "revision bucket")
    disfluency = _mapping(buckets.get("disfluency"), "disfluency bucket")
    represented = all(
        _integer(bucket.get("count"), f"{name} count", minimum=1) > 0
        for name, bucket in (
            ("no_phenomenon", simple),
            ("revision", revision),
            ("disfluency", disfluency),
        )
    )
    simple_exact = _number(
        _mapping(simple.get("ast_exact_match"), "simple exact").get("value"),
        "simple exact",
    )
    gaps = {
        "revision": simple_exact
        - _number(
            _mapping(revision.get("ast_exact_match"), "revision exact").get("value"),
            "revision exact",
        ),
        "disfluency": simple_exact
        - _number(
            _mapping(disfluency.get("ast_exact_match"), "disfluency exact").get("value"),
            "disfluency exact",
        ),
    }
    maximum_gap = _number(
        thresholds.get("maximum_no_phenomenon_to_revision_or_disfluency_gap"),
        "maximum bucket gap",
    )
    checks = {
        "derived_ast_exact_at_least_threshold": exact
        >= _number(thresholds.get("derived_ast_exact_at_least"), "exact threshold"),
        "schema_valid_at_least_threshold": schema
        >= _number(thresholds.get("schema_valid_at_least"), "schema threshold"),
        "abstention_f1_at_least_threshold": abstention
        >= _number(thresholds.get("abstention_f1_at_least"), "abstention threshold"),
        "false_call_on_gate_at_most_threshold": false_call
        <= _number(thresholds.get("false_call_on_gate_at_most"), "false-call threshold"),
        "gap_buckets_represented": represented,
        "revision_disfluency_gap_at_most_threshold": represented
        and all(gap <= maximum_gap for gap in gaps.values()),
    }
    return {
        "thresholds": dict(thresholds),
        "observed": {
            "derived_ast_exact": exact,
            "schema_valid": schema,
            "abstention_f1": abstention,
            "false_call_on_gate": false_call,
            "no_phenomenon_minus_hard_bucket_gap": gaps,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def audit_presto_taxonomy_correction(
    bundle_root: str | Path,
    *,
    preregistration_path: str | Path,
    jarvis_record_path: str | Path,
    protocol_path: str | Path = PROTOCOL_PATH,
) -> PrestoTaxonomyCorrectionResult:
    """Audit and reclassify existing samples without changing original evidence."""

    root = Path(bundle_root).resolve(strict=True)
    protocol_file = Path(protocol_path).resolve(strict=True)
    protocol_sha = _sha256_file(protocol_file)
    _expect(protocol_sha, PROTOCOL_SHA256, "post-hoc protocol SHA-256")
    protocol = _load_json(protocol_file)
    _expect(
        protocol.get("schema_version"),
        "barun-presto-taxonomy-posthoc-protocol-v2",
        "protocol schema",
    )
    _expect(protocol.get("status"), "post_hoc_after_original_result", "protocol status")
    supersedes = _mapping(protocol.get("supersedes"), "superseded protocol")
    _expect(
        supersedes.get("path"),
        "configs/presto_taxonomy_posthoc_v1.json",
        "superseded protocol path",
    )
    _expect(
        supersedes.get("sha256"),
        LEGACY_PROTOCOL_SHA256,
        "superseded protocol hash",
    )
    _expect(
        _sha256_file(LEGACY_PROTOCOL_PATH),
        LEGACY_PROTOCOL_SHA256,
        "preserved legacy protocol hash",
    )

    try:
        original = validate_presto_bundle(
            root,
            preregistration_path=preregistration_path,
            jarvis_record_path=jarvis_record_path,
        )
    except PrestoBundleAuditError as error:
        raise PrestoTaxonomyCorrectionError(
            f"original immutable bundle failed its legacy-v1 audit: {error}"
        ) from error
    source = _mapping(protocol.get("source"), "protocol source")
    _expect(original.run_id, source.get("run_id"), "source run_id")
    _expect(
        original.artifact_manifest_sha256,
        source.get("artifact_manifest_sha256"),
        "source artifact manifest",
    )
    _expect(_sha256_file(root / "result.json"), source.get("result_sha256"), "source result hash")
    _expect(
        original.output_checkpoint_sha256["model.safetensors"],
        source.get("output_model_sha256"),
        "source output model",
    )
    _expect(original.recomputed_gate["passed"], False, "original gate decision")
    _expect(
        original.recomputed_gate["checks"]["gap_buckets_represented"],
        False,
        "original missing-bucket check",
    )

    correction = _mapping(protocol.get("correction"), "protocol correction")
    expected_raw_labels = sorted(PRESTO_USER_REVISION_RAW_LABELS_V2)
    _expect(
        correction.get("alias_set_version"),
        PRESTO_USER_REVISION_ALIAS_SET_VERSION,
        "revision alias-set version",
    )
    _expect(correction.get("raw_labels"), expected_raw_labels, "raw revision aliases")
    _expect(
        correction.get("normalized_labels"),
        sorted(label.replace("-", "_") for label in PRESTO_USER_REVISION_RAW_LABELS_V2),
        "normalized revision aliases",
    )
    _expect(correction.get("original_group"), "other", "original alias group")
    _expect(correction.get("corrected_group"), "revision", "corrected alias group")
    _expect(
        correction.get("original_scorer_version"),
        PRESTO_SCORER_VERSION_V1,
        "original scorer version",
    )
    _expect(
        correction.get("corrected_scorer_version"),
        PRESTO_SCORER_VERSION,
        "corrected scorer version",
    )
    _expect(
        correction.get("corrected_taxonomy_version"),
        PRESTO_PHENOMENON_TAXONOMY_VERSION,
        "corrected taxonomy version",
    )

    stage_evidence: dict[
        str,
        tuple[
            dict[str, Any],
            int,
            dict[str, int],
            dict[str, Mapping[str, Any]],
        ],
    ] = {}
    for stage, expected_hash, original_metrics in (
        ("base", source.get("base_sample_scores_sha256"), original.recomputed_base),
        ("post", source.get("post_sample_scores_sha256"), original.recomputed_post_sft),
    ):
        path = root / f"{stage}-eval" / "scores" / "sample_scores.jsonl"
        _expect(_sha256_file(path), expected_hash, f"{stage} sample evidence hash")
        stage_evidence[stage] = _corrected_metrics(_load_jsonl(path), original_metrics)

    corrected_base, base_changed, base_raw_counts, base_subgroups = stage_evidence["base"]
    corrected_post, post_changed, post_raw_counts, post_subgroups = stage_evidence["post"]
    _expect(post_raw_counts, base_raw_counts, "base/post raw phenomenon membership")
    observed = _mapping(protocol.get("observed_before_protocol"), "observed evidence")
    observed_raw_counts = _mapping(
        observed.get("revision_raw_label_rows"), "observed revision raw-label rows"
    )
    expected_revision_counts = {
        label: _integer(observed_raw_counts.get(label), f"{label} rows", minimum=1)
        for label in expected_raw_labels
    }
    _expect(set(observed_raw_counts), set(expected_raw_labels), "observed revision raw labels")
    _expect(
        {label: base_raw_counts.get(label, 0) for label in expected_raw_labels},
        expected_revision_counts,
        "sample revision raw-label counts",
    )
    expected_changed = sum(expected_revision_counts.values())
    _expect(
        observed.get("revision_family_rows"),
        expected_changed,
        "observed revision-family rows",
    )
    _expect(base_changed, expected_changed, "base reclassified rows")
    _expect(post_changed, expected_changed, "post reclassified rows")
    _expect(
        corrected_post.get("sample_count"),
        observed.get("english_development_rows"),
        "development row count",
    )

    focus = _load_json(root / "data" / "train-focus-audit.json")
    focus_eligible = _mapping(focus.get("eligible_counts"), "focus eligible counts")
    focus_replay = _mapping(focus.get("replay_counts"), "focus replay counts")
    _expect(
        focus_eligible.get("revision"),
        observed.get("focus_revision_eligible_rows"),
        "original focus revision eligibility",
    )
    _expect(
        focus_replay.get("revision"),
        observed.get("focus_revision_replay_rows"),
        "original focus revision replay",
    )
    result = _load_json(root / "result.json")
    thresholds = _mapping(
        _mapping(result.get("gate"), "original gate").get("thresholds"), "thresholds"
    )
    corrected_gate = _gate(corrected_post, thresholds)
    limits = protocol.get("claim_limits")
    if (
        not isinstance(limits, list)
        or not limits
        or any(not isinstance(item, str) or not item for item in limits)
    ):
        _fail("protocol claim limits must be a non-empty string array")
    return PrestoTaxonomyCorrectionResult(
        schema_version=CORRECTION_SCHEMA_VERSION,
        audit_passed=True,
        status=(
            "posthoc_taxonomy_gate_passes_but_original_gate_remains_failed"
            if corrected_gate["passed"]
            else "posthoc_taxonomy_gate_still_fails"
        ),
        correction_id=str(protocol["correction_id"]),
        protocol_sha256=protocol_sha,
        source_run_id=original.run_id,
        source_artifact_manifest_sha256=original.artifact_manifest_sha256,
        source_result_sha256=str(source["result_sha256"]),
        source_output_model_sha256=str(source["output_model_sha256"]),
        original_scorer_version=PRESTO_SCORER_VERSION_V1,
        corrected_scorer_version=PRESTO_SCORER_VERSION,
        corrected_taxonomy_version=PRESTO_PHENOMENON_TAXONOMY_VERSION,
        revision_alias_set_version=PRESTO_USER_REVISION_ALIAS_SET_VERSION,
        revision_raw_labels=tuple(expected_raw_labels),
        rows_reclassified={"base": base_changed, "post_sft": post_changed},
        raw_phenomenon_counts=post_raw_counts,
        original_focus_revision={
            "eligible_rows": int(focus_eligible["revision"]),
            "replay_rows": int(focus_replay["revision"]),
        },
        original_gate=original.recomputed_gate,
        corrected_base=corrected_base,
        corrected_post_sft=corrected_post,
        revision_raw_tag_metrics={
            "base": base_subgroups,
            "post_sft": post_subgroups,
        },
        corrected_gate=corrected_gate,
        training_intervention_corrected=False,
        original_result_rewritten=False,
        claim_limits=tuple(limits),
    )


__all__ = [
    "CORRECTION_SCHEMA_VERSION",
    "LEGACY_PROTOCOL_PATH",
    "LEGACY_PROTOCOL_SHA256",
    "PROTOCOL_PATH",
    "PROTOCOL_SHA256",
    "PrestoTaxonomyCorrectionError",
    "PrestoTaxonomyCorrectionResult",
    "audit_presto_taxonomy_correction",
]
