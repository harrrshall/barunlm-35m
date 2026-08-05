"""Leakage-controlled month-boundary counterfactual views for Mobile Actions.

This module operates only on the already-prepared, hash-pinned 7,937-row Mobile
Actions training manifest.  Its API intentionally has no development or official
evaluation input.  The original adapter remains responsible for keeping the 961
official rows opaque.

The treatment changes only information whose semantics can be verified without a
teacher model:

* for an explicit date with no relative marker, only ``NOW`` is moved to the last
  day of the preceding month and the target remains byte-identical;
* for a pure relative date, ``NOW`` and the single calendar datetime are shifted by
  the same multiple of seven days until they cross a month boundary.

Mixed or ambiguous temporal forms are rejected instead of being rewritten.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

from barunlm.evaluation.action_ir import canonical_json_value, decode_json_object
from barunlm.training.data import SFTExample, load_manifest

TEMPORAL_SPLIT_VERSION = "barun-mobile-temporal-shadow-split-v1"
TEMPORAL_VIEW_VERSION = "barun-mobile-temporal-view-v1"
TEMPORAL_TRANSFORM_VERSION = "barun-mobile-month-boundary-counterfactual-v1"
TEMPORAL_RECEIPT_VERSION = "barun-mobile-temporal-transform-receipt-v1"
TEMPORAL_SHADOW_AUDIT_VERSION = "barun-mobile-temporal-shadow-audit-v1"
TEMPORAL_VIEW_AUDIT_VERSION = "barun-mobile-temporal-view-audit-v1"
TEMPORAL_FULL_VIEW_AUDIT_VERSION = "barun-mobile-temporal-full-view-audit-v1"

PINNED_MOBILE_TRAIN_ROWS = 7_937
PINNED_MOBILE_TRAIN_SHA256 = "131473ccb5bfb51cac0439b42159e72ec4c598025e50364a52b122b056c2e84e"
PINNED_MOBILE_TRAIN_MEMBERSHIP_SHA256 = (
    "4cdfc3649c21a9f0d3dc4d8d9cffcae3cb5c6abc4414a9fb09636e9220afe96f"
)
PINNED_MOBILE_TOKENIZER_IDENTIFIER = "harrrshall/BarunLM-35M"
PINNED_MOBILE_TOKENIZER_REVISION = "ef3e483a9fd7d906ecf2a7929babeffaf82d1d16"
PINNED_MOBILE_TOKENIZER_SHA256 = "70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6"

SHADOW_FOLDS = 20
CONSTRUCTION_FOLDS = frozenset(range(14))
SELECTION_FOLDS = frozenset(range(14, 17))
CONFIRMATION_FOLDS = frozenset(range(17, 20))

EXPECTED_SHADOW_ROWS = {
    "construction_train": 5_745,
    "selection": 1_024,
    "confirmation": 1_168,
}
EXPECTED_SHADOW_MEMBERSHIP_SHA256 = {
    "construction_train": ("b70f6dfe79cfe49792b6e6f5d414fd509c889d18101f75cd8df6d4e4f146c588"),
    "selection": "b4191540878ec28c6819e08e0b390fbfd57eab836d3361727ad89862a46cf927",
    "confirmation": "519c22724873ee579d0f59be886cfc297088113bd1a297521b9ca280a79c35b9",
}
EXPECTED_CONSTRUCTION_ELIGIBLE_ROWS = 1_093
EXPECTED_CONSTRUCTION_ELIGIBLE_BY_CLASS = {
    "absolute_explicit": 748,
    "pure_relative": 345,
}
EXPECTED_CONSTRUCTION_ELIGIBLE_MEMBERSHIP_SHA256 = (
    "3fcfd233e6c9e8cc7c552bb48a741ee075c90c43e23ac72c6f700a463f456a2a"
)
EXPECTED_FULL_ELIGIBLE_ROWS = 1_546
EXPECTED_FULL_ELIGIBLE_BY_CLASS = {
    "absolute_explicit": 1_039,
    "pure_relative": 507,
}
EXPECTED_FULL_ELIGIBLE_MEMBERSHIP_SHA256 = (
    "05d9fd21753e8a3e8740a900c6a0fca9435e80b27f7d43297eddd9aff3e1f32a"
)

DEFAULT_MAX_SEQ_LEN = 2_048
DEFAULT_MINIMUM_VARIANTS = 1_000
DEFAULT_MINIMUM_CROSS_MONTH_FRACTION = 0.45
TRANSFORM_YEAR_MIN = 2024
TRANSFORM_YEAR_MAX = 2026
RELATIVE_SHIFT_LIMIT_DAYS = 364

ShadowRole = Literal["construction_train", "selection", "confirmation"]
EligibilityKind = Literal["absolute_explicit", "pure_relative"]

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_NOW_LINE_RE = re.compile(r"(?m)^NOW (?P<now>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})$")
_USER_BLOCK_RE = re.compile(r"\n<user>\n(?P<user>.*?)\n<assistant>\n", re.DOTALL)
_NUMERIC_DATE_RE = re.compile(r"\b\d{1,4}[-/]\d{1,2}(?:[-/]\d{1,4})?\b")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T[^\s]+)?\b", re.IGNORECASE)

_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
_MONTH_FORMS = tuple(dict.fromkeys(_MONTHS + tuple(month[:3] for month in _MONTHS)))
_ANY_MONTH_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(month) for month in _MONTH_FORMS) + r")\b",
    re.IGNORECASE,
)
_WEEKDAY_PATTERN = "(?:" + "|".join(_WEEKDAYS) + ")"
_NUMBER_WORD_PATTERN = r"(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
_RELATIVE_MARKER_RE = re.compile(
    rf"\b(?:today|tonight|tomorrow|day\s+after\s+tomorrow|"
    rf"(?:this(?:\s+coming)?|next|coming|upcoming|following)\s+{_WEEKDAY_PATTERN}|"
    rf"(?:this|next|coming|following)\s+week|"
    rf"in\s+{_NUMBER_WORD_PATTERN}\s+days?|"
    rf"{_NUMBER_WORD_PATTERN}\s+days?\s+from\s+now)\b",
    re.IGNORECASE,
)


class MobileTemporalError(ValueError):
    """A temporal view input or derived artifact violates its frozen contract."""


class TokenizerLike(Protocol):
    """Minimal tokenizer interface used by the view materializer."""

    def encode(self, sequence: str, add_special_tokens: bool = False) -> Any: ...


@dataclass(frozen=True, slots=True)
class TemporalTokenizerIdentity:
    identifier: str
    sha256: str
    revision: str | None = None

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("tokenizer identifier cannot be empty")
        if _SHA256_RE.fullmatch(self.sha256) is None:
            raise ValueError("tokenizer sha256 must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class ParsedTemporalExample:
    example: SFTExample
    now: datetime
    user_text: str
    target: dict[str, Any]
    calendar_call_indexes: tuple[int, ...]
    calendar_datetime: datetime | None


@dataclass(frozen=True, slots=True)
class TemporalEligibility:
    eligible: bool
    reason: str
    kind: EligibilityKind | None
    parsed: ParsedTemporalExample


@dataclass(frozen=True, slots=True)
class CounterfactualVariant:
    source_id: str
    kind: EligibilityKind
    record: Mapping[str, Any]
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TemporalPlan:
    roles: Mapping[ShadowRole, tuple[SFTExample, ...]]
    eligible: tuple[TemporalEligibility, ...]
    eligibility_counts: Mapping[str, int]
    role_summaries: Mapping[ShadowRole, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class TemporalViewArtifacts:
    output_dir: Path
    construction_manifest: Path
    selection_manifest: Path
    confirmation_manifest: Path
    repeat_manifest: Path
    counterfactual_manifest: Path
    receipts_path: Path
    shadow_audit_path: Path
    view_audit_path: Path
    hashes: Mapping[str, str]
    counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class FullTemporalViewArtifacts:
    output_dir: Path
    train_manifest: Path
    receipts_path: Path
    audit_path: Path
    hashes: Mapping[str, str]
    source_rows: int
    variant_rows: int


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise MobileTemporalError(f"value is not strict JSON: {error}") from error


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def membership_sha256(example_ids: Iterable[str]) -> str:
    """Hash a set-like manifest membership using the repository convention."""

    ordered = sorted(example_ids)
    if len(ordered) != len(set(ordered)):
        raise MobileTemporalError("membership contains duplicate example IDs")
    return _sha256_text("\n".join(ordered) + "\n")


def _content_identity(example: SFTExample | Mapping[str, Any]) -> str:
    if isinstance(example, SFTExample):
        prompt = example.prompt
        target = example.target
    else:
        prompt = example.get("prompt")
        target = example.get("target")
        if not isinstance(prompt, str) or not isinstance(target, str):
            raise MobileTemporalError("manifest record lacks string prompt or target")
    return _sha256_text(_canonical_json({"prompt": prompt, "target": target}))


def _strict_datetime(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise MobileTemporalError(f"{label} is not a valid ISO datetime: {value!r}") from error
    if parsed.tzinfo is not None:
        raise MobileTemporalError(f"{label} must remain timezone-naive")
    if parsed.isoformat(timespec="seconds") != value:
        raise MobileTemporalError(f"{label} must use YYYY-MM-DDTHH:MM:SS exactly")
    return parsed


def shadow_role(cluster_id: str) -> ShadowRole:
    """Assign one existing connected component to the frozen 20-fold shadow split."""

    if not isinstance(cluster_id, str) or not cluster_id:
        raise MobileTemporalError("cluster_id must be a non-empty string")
    digest = hashlib.sha256(f"{TEMPORAL_SPLIT_VERSION}:{cluster_id}".encode()).digest()
    fold = int.from_bytes(digest[:8], byteorder="big") % SHADOW_FOLDS
    if fold in CONSTRUCTION_FOLDS:
        return "construction_train"
    if fold in SELECTION_FOLDS:
        return "selection"
    if fold in CONFIRMATION_FOLDS:
        return "confirmation"
    raise AssertionError("20-fold shadow policy left a fold unassigned")


def load_pinned_mobile_train_manifest(path: str | Path) -> tuple[SFTExample, ...]:
    """Load exactly the known 7,937-row training artifact and nothing else."""

    examples = load_manifest(
        path,
        expected_sha256=PINNED_MOBILE_TRAIN_SHA256,
        expected_derived_split="train",
    )
    if len(examples) != PINNED_MOBILE_TRAIN_ROWS:
        raise MobileTemporalError(
            f"pinned Mobile train row count changed: {len(examples)} != {PINNED_MOBILE_TRAIN_ROWS}"
        )
    actual_membership = membership_sha256(example.example_id for example in examples)
    if actual_membership != PINNED_MOBILE_TRAIN_MEMBERSHIP_SHA256:
        raise MobileTemporalError(
            "pinned Mobile train membership changed: "
            f"{actual_membership} != {PINNED_MOBILE_TRAIN_MEMBERSHIP_SHA256}"
        )
    return tuple(examples)


def parse_temporal_example(example: SFTExample) -> ParsedTemporalExample:
    """Parse the frozen prompt and canonical target without repairing either."""

    now_matches = list(_NOW_LINE_RE.finditer(example.prompt))
    if len(now_matches) != 1:
        raise MobileTemporalError(
            f"example {example.example_id!r}: prompt must contain exactly one NOW line"
        )
    user_matches = list(_USER_BLOCK_RE.finditer(example.prompt))
    if len(user_matches) != 1:
        raise MobileTemporalError(
            f"example {example.example_id!r}: prompt must contain exactly one user block"
        )
    now = _strict_datetime(now_matches[0].group("now"), label="NOW")
    decoded = decode_json_object(example.target)
    if canonical_json_value(decoded) != example.target:
        raise MobileTemporalError(
            f"example {example.example_id!r}: target is not canonical strict JSON"
        )
    target = json.loads(example.target)
    if set(target) != {"calls", "decision", "mode"} or target["decision"] != "CALL":
        raise MobileTemporalError(
            f"example {example.example_id!r}: expected canonical CALL Action IR"
        )
    calls = target.get("calls")
    if not isinstance(calls, list) or not calls:
        raise MobileTemporalError(f"example {example.example_id!r}: target calls must be non-empty")

    calendar_indexes: list[int] = []
    calendar_datetime: datetime | None = None
    for index, call in enumerate(calls):
        if not isinstance(call, dict) or set(call) != {"args", "tool"}:
            raise MobileTemporalError(
                f"example {example.example_id!r}: call {index} is not canonical"
            )
        if call.get("tool") != "create_calendar_event":
            continue
        arguments = call.get("args")
        if not isinstance(arguments, dict):
            raise MobileTemporalError(
                f"example {example.example_id!r}: calendar args must be an object"
            )
        raw_datetime = arguments.get("datetime")
        if not isinstance(raw_datetime, str):
            raise MobileTemporalError(
                f"example {example.example_id!r}: calendar datetime must be a string"
            )
        parsed_datetime = _strict_datetime(raw_datetime, label="calendar datetime")
        calendar_indexes.append(index)
        if calendar_datetime is None:
            calendar_datetime = parsed_datetime

    return ParsedTemporalExample(
        example=example,
        now=now,
        user_text=user_matches[0].group("user"),
        target=target,
        calendar_call_indexes=tuple(calendar_indexes),
        calendar_datetime=calendar_datetime,
    )


def _explicit_target_date(user_text: str, target: datetime) -> bool:
    month_forms = (_MONTHS[target.month - 1], _MONTHS[target.month - 1][:3])
    month_pattern = "(?:" + "|".join(re.escape(item) for item in month_forms) + ")"
    day_pattern = rf"{target.day}(?:st|nd|rd|th)?"
    return (
        re.search(
            rf"\b{month_pattern}\s+{day_pattern}\b|"
            rf"\b{day_pattern}\s+(?:of\s+)?{month_pattern}\b",
            user_text,
            re.IGNORECASE,
        )
        is not None
    )


def _relative_anchor(user_text: str, target: datetime) -> bool:
    if _RELATIVE_MARKER_RE.search(user_text) is not None:
        return True
    return re.search(rf"\b{_WEEKDAYS[target.weekday()]}\b", user_text, re.IGNORECASE) is not None


def classify_counterfactual_source(example: SFTExample) -> TemporalEligibility:
    """Return the conservative v1 eligibility decision for one source example."""

    parsed = parse_temporal_example(example)
    if not parsed.calendar_call_indexes:
        return TemporalEligibility(False, "no_calendar_call", None, parsed)
    if len(parsed.calendar_call_indexes) != 1:
        return TemporalEligibility(False, "multiple_calendar_calls", None, parsed)
    assert parsed.calendar_datetime is not None
    target = parsed.calendar_datetime
    if (target.year, target.month) != (parsed.now.year, parsed.now.month):
        return TemporalEligibility(False, "already_cross_month", None, parsed)

    explicit = _explicit_target_date(parsed.user_text, target)
    has_relative_marker = _RELATIVE_MARKER_RE.search(parsed.user_text) is not None
    if explicit and not has_relative_marker:
        return TemporalEligibility(True, "eligible", "absolute_explicit", parsed)

    has_explicit_date_surface = (
        _ANY_MONTH_RE.search(parsed.user_text) is not None
        or _NUMERIC_DATE_RE.search(parsed.user_text) is not None
        or _ISO_DATE_RE.search(parsed.user_text) is not None
    )
    if target.date() == parsed.now.date():
        return TemporalEligibility(False, "same_day_relative_cannot_cross", None, parsed)
    if _relative_anchor(parsed.user_text, target) and not has_explicit_date_surface:
        return TemporalEligibility(True, "eligible", "pure_relative", parsed)
    return TemporalEligibility(False, "mixed_ambiguous_or_unsupported", None, parsed)


def _replace_now(prompt: str, new_now: datetime) -> str:
    matches = list(_NOW_LINE_RE.finditer(prompt))
    if len(matches) != 1:
        raise MobileTemporalError("cannot replace NOW because its prompt occurrence is ambiguous")
    match = matches[0]
    replacement = f"NOW {new_now.isoformat(timespec='seconds')}"
    return prompt[: match.start()] + replacement + prompt[match.end() :]


def _encode_ids(tokenizer: TokenizerLike, text: str) -> tuple[int, ...]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = getattr(encoded, "ids", encoded)
    if not isinstance(ids, (list, tuple)) or any(not isinstance(item, int) for item in ids):
        raise MobileTemporalError("tokenizer.encode must return integer IDs or an object with .ids")
    if not ids:
        raise MobileTemporalError("tokenizer encoded text to zero tokens")
    return tuple(ids)


def _updated_metadata(
    metadata: Mapping[str, Any],
    *,
    prompt: str,
    target: str,
    tokenizer: TokenizerLike,
    max_seq_len: int,
    view: Mapping[str, Any],
) -> dict[str, Any]:
    if "mobile_temporal_view" in metadata:
        raise MobileTemporalError("source metadata already contains mobile_temporal_view")
    output = dict(metadata)
    output["derived_split"] = "train"
    output["prompt_sha256"] = _sha256_text(prompt)
    output["target_sha256"] = _sha256_text(target)
    prompt_tokens = len(_encode_ids(tokenizer, prompt))
    target_tokens = len(_encode_ids(tokenizer, target))
    total_tokens = prompt_tokens + target_tokens + 1
    if total_tokens > max_seq_len:
        raise MobileTemporalError(
            f"derived example has {total_tokens} tokens, above max_seq_len={max_seq_len}"
        )
    output["token_lengths"] = {
        "prompt": prompt_tokens,
        "target": target_tokens,
        "total_with_eos": total_tokens,
    }
    output["mobile_temporal_view"] = dict(view)
    return output


def _manifest_record(
    example: SFTExample,
    *,
    example_id: str | None = None,
    prompt: str | None = None,
    target: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "barun-sft-example-v1",
        "id": example.example_id if example_id is None else example_id,
        "prompt": example.prompt if prompt is None else prompt,
        "target": example.target if target is None else target,
        "metadata": dict(example.metadata if metadata is None else metadata),
    }


def _relative_shift(eligibility: TemporalEligibility) -> int:
    parsed = eligibility.parsed
    assert eligibility.kind == "pure_relative"
    assert parsed.calendar_datetime is not None
    candidates: list[tuple[str, int]] = []
    for shift_days in range(
        -RELATIVE_SHIFT_LIMIT_DAYS,
        RELATIVE_SHIFT_LIMIT_DAYS + 1,
        7,
    ):
        if shift_days == 0:
            continue
        shifted_now = parsed.now + timedelta(days=shift_days)
        shifted_target = parsed.calendar_datetime + timedelta(days=shift_days)
        if not (
            TRANSFORM_YEAR_MIN <= shifted_now.year <= TRANSFORM_YEAR_MAX
            and TRANSFORM_YEAR_MIN <= shifted_target.year <= TRANSFORM_YEAR_MAX
        ):
            continue
        if (shifted_now.year, shifted_now.month) == (
            shifted_target.year,
            shifted_target.month,
        ):
            continue
        rank = _sha256_text(
            f"{TEMPORAL_TRANSFORM_VERSION}:{parsed.example.example_id}:{shift_days}"
        )
        candidates.append((rank, shift_days))
    if not candidates:
        raise MobileTemporalError(
            f"example {parsed.example.example_id!r}: no valid relative month-boundary shift"
        )
    return min(candidates)[1]


def build_month_boundary_variant(
    example: SFTExample,
    *,
    tokenizer: TokenizerLike,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> CounterfactualVariant | None:
    """Build one verified counterfactual, or return ``None`` when ineligible."""

    eligibility = classify_counterfactual_source(example)
    if not eligibility.eligible:
        return None
    assert eligibility.kind is not None
    parsed = eligibility.parsed
    assert parsed.calendar_datetime is not None
    original_now = parsed.now
    original_datetime = parsed.calendar_datetime

    if eligibility.kind == "absolute_explicit":
        previous_year = (
            original_datetime.year if original_datetime.month > 1 else original_datetime.year - 1
        )
        previous_month = original_datetime.month - 1 if original_datetime.month > 1 else 12
        previous_last_day = calendar.monthrange(previous_year, previous_month)[1]
        new_now = original_now.replace(
            year=previous_year,
            month=previous_month,
            day=previous_last_day,
        )
        new_datetime = original_datetime
        shift_days: int | None = None
        target = example.target
    else:
        shift_days = _relative_shift(eligibility)
        new_now = original_now + timedelta(days=shift_days)
        new_datetime = original_datetime + timedelta(days=shift_days)
        target_payload = json.loads(example.target)
        calendar_index = parsed.calendar_call_indexes[0]
        target_payload["calls"][calendar_index]["args"]["datetime"] = new_datetime.isoformat(
            timespec="seconds"
        )
        target = _canonical_json(target_payload)

    if (new_now.year, new_now.month) == (new_datetime.year, new_datetime.month):
        raise MobileTemporalError("counterfactual failed to cross a month boundary")
    if eligibility.kind == "absolute_explicit" and target != example.target:
        raise AssertionError("absolute transform changed its target")
    if eligibility.kind == "pure_relative":
        if new_datetime - new_now != original_datetime - original_now:
            raise MobileTemporalError("relative transform changed target-minus-NOW timedelta")
        if new_datetime.weekday() != original_datetime.weekday():
            raise MobileTemporalError("relative transform changed target weekday")
        if new_datetime.time() != original_datetime.time():
            raise MobileTemporalError("relative transform changed target time")

    prompt = _replace_now(example.prompt, new_now)
    transform_payload = {
        "kind": eligibility.kind,
        "new_datetime": new_datetime.isoformat(timespec="seconds"),
        "new_now": new_now.isoformat(timespec="seconds"),
        "shift_days": shift_days,
        "source_id": example.example_id,
        "transform_version": TEMPORAL_TRANSFORM_VERSION,
    }
    transform_sha256 = _sha256_text(_canonical_json(transform_payload))
    variant_id = f"{example.example_id}--mbcf-v1-{transform_sha256[:12]}"
    view = {
        "schema_version": TEMPORAL_VIEW_VERSION,
        "kind": "month_boundary_counterfactual",
        "source_id": example.example_id,
        "source_content_sha256": example.content_sha256,
        "transform_sha256": transform_sha256,
        "transform_version": TEMPORAL_TRANSFORM_VERSION,
        "temporal_class": eligibility.kind,
        "original_now": original_now.isoformat(timespec="seconds"),
        "counterfactual_now": new_now.isoformat(timespec="seconds"),
        "original_datetime": original_datetime.isoformat(timespec="seconds"),
        "counterfactual_datetime": new_datetime.isoformat(timespec="seconds"),
        "shift_days": shift_days,
    }
    metadata = _updated_metadata(
        example.metadata,
        prompt=prompt,
        target=target,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        view=view,
    )
    record = _manifest_record(
        example,
        example_id=variant_id,
        prompt=prompt,
        target=target,
        metadata=metadata,
    )
    receipt = {
        "schema_version": TEMPORAL_RECEIPT_VERSION,
        "source_id": example.example_id,
        "variant_id": variant_id,
        "temporal_class": eligibility.kind,
        "source_prompt_sha256": _sha256_text(example.prompt),
        "source_target_sha256": _sha256_text(example.target),
        "variant_prompt_sha256": _sha256_text(prompt),
        "variant_target_sha256": _sha256_text(target),
        "original_now": original_now.isoformat(timespec="seconds"),
        "counterfactual_now": new_now.isoformat(timespec="seconds"),
        "original_datetime": original_datetime.isoformat(timespec="seconds"),
        "counterfactual_datetime": new_datetime.isoformat(timespec="seconds"),
        "shift_days": shift_days,
        "invariants": {
            "cross_month": True,
            "source_user_text_unchanged": True,
            "non_calendar_calls_unchanged": True,
            "target_timedelta_preserved_for_relative": eligibility.kind == "pure_relative",
            "target_byte_identical_for_absolute": eligibility.kind == "absolute_explicit",
        },
    }
    return CounterfactualVariant(
        source_id=example.example_id,
        kind=eligibility.kind,
        record=record,
        receipt=receipt,
    )


def _repeat_record(
    eligibility: TemporalEligibility,
    *,
    tokenizer: TokenizerLike,
    max_seq_len: int,
) -> dict[str, Any]:
    example = eligibility.parsed.example
    repeat_id = f"{example.example_id}--temporal-repeat-v1"
    metadata = _updated_metadata(
        example.metadata,
        prompt=example.prompt,
        target=example.target,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        view={
            "schema_version": TEMPORAL_VIEW_VERSION,
            "kind": "repeat_control",
            "source_id": example.example_id,
            "source_content_sha256": example.content_sha256,
            "temporal_class": eligibility.kind,
        },
    )
    return _manifest_record(example, example_id=repeat_id, metadata=metadata)


def _shadow_record(example: SFTExample, role: ShadowRole) -> dict[str, Any]:
    if role == "construction_train":
        return _manifest_record(example)
    if "mobile_temporal_shadow" in example.metadata:
        raise MobileTemporalError("source metadata already contains mobile_temporal_shadow")
    metadata = dict(example.metadata)
    metadata["derived_split"] = "dev" if role == "selection" else "confirmation"
    metadata["mobile_temporal_shadow"] = {
        "schema_version": TEMPORAL_VIEW_VERSION,
        "role": role,
        "source_id": example.example_id,
    }
    return _manifest_record(example, metadata=metadata)


def _calendar_summary(examples: Sequence[SFTExample]) -> dict[str, int]:
    same_month = 0
    cross_month = 0
    calendar_rows = 0
    for example in examples:
        parsed = parse_temporal_example(example)
        if not parsed.calendar_call_indexes:
            continue
        if len(parsed.calendar_call_indexes) != 1 or parsed.calendar_datetime is None:
            raise MobileTemporalError(
                f"example {example.example_id!r}: expected at most one calendar call"
            )
        calendar_rows += 1
        if (parsed.now.year, parsed.now.month) == (
            parsed.calendar_datetime.year,
            parsed.calendar_datetime.month,
        ):
            same_month += 1
        else:
            cross_month += 1
    return {
        "rows": calendar_rows,
        "same_month": same_month,
        "cross_month": cross_month,
    }


def _role_summary(examples: Sequence[SFTExample]) -> dict[str, Any]:
    scenarios = Counter()
    call_counts = Counter()
    clusters: set[str] = set()
    families: set[str] = set()
    for example in examples:
        metadata = example.metadata
        call_names = metadata.get("call_names")
        cluster_id = metadata.get("cluster_id")
        family_id = metadata.get("family_id")
        if not isinstance(call_names, list) or not call_names:
            raise MobileTemporalError(
                f"example {example.example_id!r}: metadata.call_names is invalid"
            )
        if not isinstance(cluster_id, str) or not isinstance(family_id, str):
            raise MobileTemporalError(
                f"example {example.example_id!r}: cluster/family metadata is invalid"
            )
        scenarios[str(call_names[0])] += 1
        call_counts[str(len(call_names))] += 1
        clusters.add(cluster_id)
        families.add(family_id)
    return {
        "rows": len(examples),
        "components": len(clusters),
        "families": len(families),
        "membership_sha256": membership_sha256(row.example_id for row in examples),
        "scenario_counts": dict(sorted(scenarios.items())),
        "call_count_counts": dict(sorted(call_counts.items())),
        "calendar": _calendar_summary(examples),
    }


def _verify_shadow_disjointness(roles: Mapping[ShadowRole, Sequence[SFTExample]]) -> None:
    indexes: dict[str, dict[ShadowRole, set[str]]] = {
        "id": {},
        "cluster": {},
        "family": {},
        "content": {},
    }
    for role, rows in roles.items():
        indexes["id"][role] = {row.example_id for row in rows}
        indexes["cluster"][role] = {str(row.metadata.get("cluster_id")) for row in rows}
        indexes["family"][role] = {str(row.metadata.get("family_id")) for row in rows}
        indexes["content"][role] = {_content_identity(row) for row in rows}
    role_names = tuple(roles)
    for left_index, left in enumerate(role_names):
        for right in role_names[left_index + 1 :]:
            for name, role_index in indexes.items():
                overlap = role_index[left] & role_index[right]
                if overlap:
                    raise MobileTemporalError(
                        f"shadow {name} overlap between {left} and {right}: {sorted(overlap)[:3]}"
                    )


def plan_temporal_views(examples: Sequence[SFTExample]) -> TemporalPlan:
    """Compute the frozen split and treatment eligibility without writing artifacts."""

    if not examples:
        raise MobileTemporalError("cannot plan temporal views from an empty manifest")
    grouped: dict[ShadowRole, list[SFTExample]] = {
        "construction_train": [],
        "selection": [],
        "confirmation": [],
    }
    cluster_roles: dict[str, ShadowRole] = {}
    family_roles: defaultdict[str, set[ShadowRole]] = defaultdict(set)
    for example in examples:
        source_split = example.metadata.get("source_split")
        derived_split = example.metadata.get("derived_split")
        cluster_id = example.metadata.get("cluster_id")
        family_id = example.metadata.get("family_id")
        if source_split != "train" or derived_split != "train":
            raise MobileTemporalError(
                f"example {example.example_id!r}: temporal source must be train/train"
            )
        if not isinstance(cluster_id, str) or not cluster_id:
            raise MobileTemporalError(f"example {example.example_id!r}: missing cluster_id")
        if not isinstance(family_id, str) or not family_id:
            raise MobileTemporalError(f"example {example.example_id!r}: missing family_id")
        role = shadow_role(cluster_id)
        previous = cluster_roles.setdefault(cluster_id, role)
        if previous != role:  # pragma: no cover - deterministic function invariant
            raise AssertionError("one cluster received two shadow roles")
        family_roles[family_id].add(role)
        grouped[role].append(example)
    crossing_families = sorted(name for name, roles in family_roles.items() if len(roles) != 1)
    if crossing_families:
        raise MobileTemporalError(f"families cross shadow roles: {crossing_families[:3]}")
    frozen_roles = {role: tuple(rows) for role, rows in grouped.items()}
    _verify_shadow_disjointness(frozen_roles)

    eligibility: list[TemporalEligibility] = []
    counts = Counter()
    for example in frozen_roles["construction_train"]:
        decision = classify_counterfactual_source(example)
        counts[decision.kind if decision.eligible else decision.reason] += 1
        if decision.eligible:
            eligibility.append(decision)
    eligibility.sort(key=lambda item: item.parsed.example.example_id)
    return TemporalPlan(
        roles=frozen_roles,
        eligible=tuple(eligibility),
        eligibility_counts=dict(sorted(counts.items())),
        role_summaries={role: _role_summary(rows) for role, rows in frozen_roles.items()},
    )


def _write_bytes(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise MobileTemporalError(f"refusing to overwrite artifact {path}") from error
    return hashlib.sha256(content).hexdigest()


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return ("".join(_canonical_json(row) + "\n" for row in rows)).encode("utf-8")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _validate_expected_shadow(plan: TemporalPlan) -> None:
    for role in ("construction_train", "selection", "confirmation"):
        summary = plan.role_summaries[role]
        if summary["rows"] != EXPECTED_SHADOW_ROWS[role]:
            raise MobileTemporalError(
                f"{role} row count changed: {summary['rows']} != {EXPECTED_SHADOW_ROWS[role]}"
            )
        expected_membership = EXPECTED_SHADOW_MEMBERSHIP_SHA256[role]
        if summary["membership_sha256"] != expected_membership:
            raise MobileTemporalError(
                f"{role} membership changed: {summary['membership_sha256']} != "
                f"{expected_membership}"
            )
    eligible_ids = [item.parsed.example.example_id for item in plan.eligible]
    eligible_by_class = Counter(item.kind for item in plan.eligible)
    actual_membership = membership_sha256(eligible_ids)
    if len(eligible_ids) != EXPECTED_CONSTRUCTION_ELIGIBLE_ROWS:
        raise MobileTemporalError(
            "construction eligibility count changed: "
            f"{len(eligible_ids)} != {EXPECTED_CONSTRUCTION_ELIGIBLE_ROWS}"
        )
    if dict(eligible_by_class) != EXPECTED_CONSTRUCTION_ELIGIBLE_BY_CLASS:
        raise MobileTemporalError(
            "construction eligibility classes changed: "
            f"{dict(eligible_by_class)} != {EXPECTED_CONSTRUCTION_ELIGIBLE_BY_CLASS}"
        )
    if actual_membership != EXPECTED_CONSTRUCTION_ELIGIBLE_MEMBERSHIP_SHA256:
        raise MobileTemporalError(
            "construction eligibility membership changed: "
            f"{actual_membership} != {EXPECTED_CONSTRUCTION_ELIGIBLE_MEMBERSHIP_SHA256}"
        )


def _validate_pinned_tokenizer_contract(
    identity: TemporalTokenizerIdentity,
    *,
    max_seq_len: int,
) -> None:
    expected = (
        PINNED_MOBILE_TOKENIZER_IDENTIFIER,
        PINNED_MOBILE_TOKENIZER_REVISION,
        PINNED_MOBILE_TOKENIZER_SHA256,
    )
    actual = (identity.identifier, identity.revision, identity.sha256)
    if actual != expected:
        raise MobileTemporalError(
            "tokenizer identity differs from the tokenizer used for the pinned source manifest: "
            f"{actual!r} != {expected!r}"
        )
    if max_seq_len != DEFAULT_MAX_SEQ_LEN:
        raise MobileTemporalError(
            "pinned Mobile temporal views require max_seq_len="
            f"{DEFAULT_MAX_SEQ_LEN}, got {max_seq_len}"
        )


def _token_totals(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    prompt = 0
    target = 0
    total = 0
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            raise MobileTemporalError("record metadata is not an object")
        lengths = metadata.get("token_lengths")
        if not isinstance(lengths, Mapping):
            raise MobileTemporalError("record lacks token_lengths metadata")
        prompt += int(lengths["prompt"])
        target += int(lengths["target"])
        total += int(lengths["total_with_eos"])
    return {"prompt": prompt, "target": target, "total_with_eos": total}


def materialize_temporal_views(
    *,
    source_train_path: str | Path,
    tokenizer: TokenizerLike,
    tokenizer_identity: TemporalTokenizerIdentity,
    output_dir: str | Path,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    minimum_variants: int = DEFAULT_MINIMUM_VARIANTS,
    minimum_cross_month_fraction: float = DEFAULT_MINIMUM_CROSS_MONTH_FRACTION,
    enforce_pinned_shadow: bool = True,
) -> TemporalViewArtifacts:
    """Materialize immutable standard, repeat-control, and treatment views."""

    if max_seq_len < 2:
        raise ValueError("max_seq_len must be at least 2")
    if minimum_variants < 0:
        raise ValueError("minimum_variants cannot be negative")
    if not 0 <= minimum_cross_month_fraction <= 1:
        raise ValueError("minimum_cross_month_fraction must be in [0, 1]")
    destination = Path(output_dir)
    if destination.exists():
        raise MobileTemporalError(f"refusing to reuse output directory {destination}")

    examples = load_pinned_mobile_train_manifest(source_train_path)
    plan = plan_temporal_views(examples)
    if enforce_pinned_shadow:
        _validate_pinned_tokenizer_contract(tokenizer_identity, max_seq_len=max_seq_len)
        _validate_expected_shadow(plan)
    if len(plan.eligible) < minimum_variants:
        raise MobileTemporalError(
            f"safe variants {len(plan.eligible)} are below minimum {minimum_variants}"
        )

    construction_records = [
        _shadow_record(example, "construction_train")
        for example in plan.roles["construction_train"]
    ]
    selection_records = [
        _shadow_record(example, "selection") for example in plan.roles["selection"]
    ]
    confirmation_records = [
        _shadow_record(example, "confirmation") for example in plan.roles["confirmation"]
    ]

    variants: list[CounterfactualVariant] = []
    repeats: list[dict[str, Any]] = []
    for eligibility in plan.eligible:
        variant = build_month_boundary_variant(
            eligibility.parsed.example,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        if variant is None:  # pragma: no cover - plan and builder share classifier
            raise AssertionError("planned eligible source became ineligible")
        variants.append(variant)
        repeats.append(
            _repeat_record(
                eligibility,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
            )
        )
    repeat_source_ids = [eligibility.parsed.example.example_id for eligibility in plan.eligible]
    variant_source_ids = [variant.source_id for variant in variants]
    if repeat_source_ids != variant_source_ids:
        raise MobileTemporalError("repeat and treatment source IDs differ")

    repeat_records = construction_records + repeats
    counterfactual_records = construction_records + [dict(variant.record) for variant in variants]
    for label, records in (
        ("construction", construction_records),
        ("selection", selection_records),
        ("confirmation", confirmation_records),
        ("repeat", repeat_records),
        ("counterfactual", counterfactual_records),
    ):
        ids = [str(record["id"]) for record in records]
        if len(ids) != len(set(ids)):
            raise MobileTemporalError(f"{label} manifest contains duplicate IDs")

    construction_calendar = plan.role_summaries["construction_train"]["calendar"]
    post_cross = int(construction_calendar["cross_month"]) + len(variants)
    post_calendar = int(construction_calendar["rows"]) + len(variants)
    cross_fraction = post_cross / post_calendar
    if cross_fraction < minimum_cross_month_fraction:
        raise MobileTemporalError(
            f"cross-month fraction {cross_fraction:.12f} is below minimum "
            f"{minimum_cross_month_fraction:.12f}"
        )

    output_paths = {
        "construction": destination / "construction-train.jsonl",
        "selection": destination / "selection.jsonl",
        "confirmation": destination / "confirmation.jsonl",
        "repeat": destination / "repeat-train.jsonl",
        "counterfactual": destination / "counterfactual-train.jsonl",
        "receipts": destination / "transform-receipts.jsonl",
    }
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise MobileTemporalError(f"refusing to reuse output directory {destination}") from error
    hashes = {
        "construction": _write_bytes(
            output_paths["construction"], _jsonl_bytes(construction_records)
        ),
        "selection": _write_bytes(output_paths["selection"], _jsonl_bytes(selection_records)),
        "confirmation": _write_bytes(
            output_paths["confirmation"], _jsonl_bytes(confirmation_records)
        ),
        "repeat": _write_bytes(output_paths["repeat"], _jsonl_bytes(repeat_records)),
        "counterfactual": _write_bytes(
            output_paths["counterfactual"], _jsonl_bytes(counterfactual_records)
        ),
        "receipts": _write_bytes(
            output_paths["receipts"],
            _jsonl_bytes([dict(variant.receipt) for variant in variants]),
        ),
    }

    shadow_audit = {
        "schema_version": TEMPORAL_SHADOW_AUDIT_VERSION,
        "source": {
            "filename": Path(source_train_path).name,
            "sha256": PINNED_MOBILE_TRAIN_SHA256,
            "rows": len(examples),
            "membership_sha256": PINNED_MOBILE_TRAIN_MEMBERSHIP_SHA256,
        },
        "split_policy": {
            "version": TEMPORAL_SPLIT_VERSION,
            "folds": SHADOW_FOLDS,
            "construction_folds": sorted(CONSTRUCTION_FOLDS),
            "selection_folds": sorted(SELECTION_FOLDS),
            "confirmation_folds": sorted(CONFIRMATION_FOLDS),
            "hash_input": "{version}:{cluster_id}",
            "fold_integer": "unsigned big-endian first eight SHA-256 bytes",
        },
        "roles": {
            role: {**dict(plan.role_summaries[role]), "manifest_sha256": hashes[name]}
            for role, name in (
                ("construction_train", "construction"),
                ("selection", "selection"),
                ("confirmation", "confirmation"),
            )
        },
        "overlap": {
            "example_id": 0,
            "cluster_id": 0,
            "family_id": 0,
            "exact_prompt_target": 0,
        },
        "development_rows_read": 0,
        "official_evaluation_rows_read": 0,
    }
    shadow_audit_path = destination / "shadow-audit.json"
    hashes["shadow_audit"] = _write_bytes(shadow_audit_path, _json_bytes(shadow_audit))

    repeat_added = repeats
    counterfactual_added = [dict(variant.record) for variant in variants]
    repeat_target_tokens = _token_totals(repeat_added)["target"]
    counterfactual_target_tokens = _token_totals(counterfactual_added)["target"]
    target_token_difference_fraction = (
        abs(counterfactual_target_tokens - repeat_target_tokens) / repeat_target_tokens
        if repeat_target_tokens
        else 0.0
    )
    view_audit = {
        "schema_version": TEMPORAL_VIEW_AUDIT_VERSION,
        "transform_version": TEMPORAL_TRANSFORM_VERSION,
        "tokenizer": {
            "identifier": tokenizer_identity.identifier,
            "revision": tokenizer_identity.revision,
            "sha256": tokenizer_identity.sha256,
            "max_seq_len": max_seq_len,
        },
        "eligibility": {
            "counts": dict(plan.eligibility_counts),
            "safe_variant_rows": len(variants),
            "safe_source_membership_sha256": membership_sha256(variant_source_ids),
            "minimum_required": minimum_variants,
        },
        "views": {
            "standard": {
                "filename": output_paths["construction"].name,
                "sha256": hashes["construction"],
                "rows": len(construction_records),
                "optimizer_steps_at_batch_63": (len(construction_records) + 62) // 63,
            },
            "repeat": {
                "filename": output_paths["repeat"].name,
                "sha256": hashes["repeat"],
                "rows": len(repeat_records),
                "added_rows": len(repeats),
                "optimizer_steps_at_batch_63": (len(repeat_records) + 62) // 63,
            },
            "counterfactual": {
                "filename": output_paths["counterfactual"].name,
                "sha256": hashes["counterfactual"],
                "rows": len(counterfactual_records),
                "added_rows": len(variants),
                "optimizer_steps_at_batch_63": (len(counterfactual_records) + 62) // 63,
            },
        },
        "matched_control": {
            "same_source_ids": repeat_source_ids == variant_source_ids,
            "source_membership_sha256": membership_sha256(repeat_source_ids),
            "same_added_rows": len(repeats) == len(variants),
            "same_optimizer_steps": len(repeat_records) == len(counterfactual_records),
            "repeat_added_target_tokens": repeat_target_tokens,
            "counterfactual_added_target_tokens": counterfactual_target_tokens,
            "target_token_difference_fraction": target_token_difference_fraction,
        },
        "calendar_presentations": {
            "original_same_month": construction_calendar["same_month"],
            "original_cross_month": construction_calendar["cross_month"],
            "added_cross_month": len(variants),
            "post_cross_month": post_cross,
            "post_total": post_calendar,
            "post_cross_month_fraction": cross_fraction,
            "minimum_fraction": minimum_cross_month_fraction,
        },
        "receipts": {
            "filename": output_paths["receipts"].name,
            "sha256": hashes["receipts"],
            "rows": len(variants),
        },
        "leakage": {
            "selection_source_rows_used_for_variants": 0,
            "confirmation_source_rows_used_for_variants": 0,
            "development_rows_read": 0,
            "official_evaluation_rows_read": 0,
        },
    }
    view_audit_path = destination / "view-audit.json"
    hashes["view_audit"] = _write_bytes(view_audit_path, _json_bytes(view_audit))
    counts = {
        "construction": len(construction_records),
        "selection": len(selection_records),
        "confirmation": len(confirmation_records),
        "variants": len(variants),
        "repeat": len(repeat_records),
        "counterfactual": len(counterfactual_records),
    }
    return TemporalViewArtifacts(
        output_dir=destination,
        construction_manifest=output_paths["construction"],
        selection_manifest=output_paths["selection"],
        confirmation_manifest=output_paths["confirmation"],
        repeat_manifest=output_paths["repeat"],
        counterfactual_manifest=output_paths["counterfactual"],
        receipts_path=output_paths["receipts"],
        shadow_audit_path=shadow_audit_path,
        view_audit_path=view_audit_path,
        hashes=dict(hashes),
        counts=counts,
    )


def materialize_full_counterfactual_view(
    *,
    source_train_path: str | Path,
    tokenizer: TokenizerLike,
    tokenizer_identity: TemporalTokenizerIdentity,
    output_dir: str | Path,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    minimum_variants: int = 1_400,
    minimum_cross_month_fraction: float = DEFAULT_MINIMUM_CROSS_MONTH_FRACTION,
) -> FullTemporalViewArtifacts:
    """Materialize the fixed-seed final-fit view after recipe confirmation."""

    if max_seq_len < 2:
        raise ValueError("max_seq_len must be at least 2")
    if minimum_variants < 0:
        raise ValueError("minimum_variants cannot be negative")
    if not 0 <= minimum_cross_month_fraction <= 1:
        raise ValueError("minimum_cross_month_fraction must be in [0, 1]")
    destination = Path(output_dir)
    if destination.exists():
        raise MobileTemporalError(f"refusing to reuse output directory {destination}")
    _validate_pinned_tokenizer_contract(tokenizer_identity, max_seq_len=max_seq_len)
    examples = load_pinned_mobile_train_manifest(source_train_path)
    decisions = [classify_counterfactual_source(example) for example in examples]
    eligible = sorted(
        (decision for decision in decisions if decision.eligible),
        key=lambda item: item.parsed.example.example_id,
    )
    eligible_by_class = Counter(decision.kind for decision in eligible)
    eligible_membership = membership_sha256(
        decision.parsed.example.example_id for decision in eligible
    )
    if len(eligible) != EXPECTED_FULL_ELIGIBLE_ROWS:
        raise MobileTemporalError(
            f"full-view eligibility count changed: {len(eligible)} != {EXPECTED_FULL_ELIGIBLE_ROWS}"
        )
    if dict(eligible_by_class) != EXPECTED_FULL_ELIGIBLE_BY_CLASS:
        raise MobileTemporalError(
            "full-view eligibility classes changed: "
            f"{dict(eligible_by_class)} != {EXPECTED_FULL_ELIGIBLE_BY_CLASS}"
        )
    if eligible_membership != EXPECTED_FULL_ELIGIBLE_MEMBERSHIP_SHA256:
        raise MobileTemporalError(
            "full-view eligibility membership changed: "
            f"{eligible_membership} != {EXPECTED_FULL_ELIGIBLE_MEMBERSHIP_SHA256}"
        )
    if len(eligible) < minimum_variants:
        raise MobileTemporalError(
            f"full-view variants {len(eligible)} are below minimum {minimum_variants}"
        )
    variants: list[CounterfactualVariant] = []
    for decision in eligible:
        variant = build_month_boundary_variant(
            decision.parsed.example,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        if variant is None:  # pragma: no cover - shared classifier invariant
            raise AssertionError("eligible full-view row became ineligible")
        variants.append(variant)

    source_records = [_manifest_record(example) for example in examples]
    train_records = source_records + [dict(variant.record) for variant in variants]
    ids = [str(record["id"]) for record in train_records]
    if len(ids) != len(set(ids)):
        raise MobileTemporalError("full counterfactual view contains duplicate IDs")
    calendar = _calendar_summary(examples)
    post_cross = calendar["cross_month"] + len(variants)
    post_total = calendar["rows"] + len(variants)
    cross_fraction = post_cross / post_total
    if cross_fraction < minimum_cross_month_fraction:
        raise MobileTemporalError(
            f"full cross-month fraction {cross_fraction:.12f} is below minimum "
            f"{minimum_cross_month_fraction:.12f}"
        )

    train_path = destination / "full-counterfactual-train.jsonl"
    receipts_path = destination / "full-transform-receipts.jsonl"
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise MobileTemporalError(f"refusing to reuse output directory {destination}") from error
    train_sha256 = _write_bytes(train_path, _jsonl_bytes(train_records))
    receipts_sha256 = _write_bytes(
        receipts_path,
        _jsonl_bytes([dict(variant.receipt) for variant in variants]),
    )
    audit = {
        "schema_version": TEMPORAL_FULL_VIEW_AUDIT_VERSION,
        "transform_version": TEMPORAL_TRANSFORM_VERSION,
        "source": {
            "filename": Path(source_train_path).name,
            "sha256": PINNED_MOBILE_TRAIN_SHA256,
            "rows": len(examples),
            "membership_sha256": PINNED_MOBILE_TRAIN_MEMBERSHIP_SHA256,
        },
        "tokenizer": {
            "identifier": tokenizer_identity.identifier,
            "revision": tokenizer_identity.revision,
            "sha256": tokenizer_identity.sha256,
            "max_seq_len": max_seq_len,
        },
        "variants": {
            "rows": len(variants),
            "by_class": dict(sorted((str(key), value) for key, value in eligible_by_class.items())),
            "source_membership_sha256": eligible_membership,
            "minimum_required": minimum_variants,
        },
        "calendar_presentations": {
            "original_same_month": calendar["same_month"],
            "original_cross_month": calendar["cross_month"],
            "added_cross_month": len(variants),
            "post_cross_month": post_cross,
            "post_total": post_total,
            "post_cross_month_fraction": cross_fraction,
            "minimum_fraction": minimum_cross_month_fraction,
        },
        "output": {
            "filename": train_path.name,
            "sha256": train_sha256,
            "rows": len(train_records),
            "optimizer_steps_at_batch_63": (len(train_records) + 62) // 63,
        },
        "receipts": {
            "filename": receipts_path.name,
            "sha256": receipts_sha256,
            "rows": len(variants),
        },
        "development_rows_read": 0,
        "official_evaluation_rows_read": 0,
    }
    audit_path = destination / "full-view-audit.json"
    audit_sha256 = _write_bytes(audit_path, _json_bytes(audit))
    return FullTemporalViewArtifacts(
        output_dir=destination,
        train_manifest=train_path,
        receipts_path=receipts_path,
        audit_path=audit_path,
        hashes={
            "train": train_sha256,
            "receipts": receipts_sha256,
            "audit": audit_sha256,
        },
        source_rows=len(examples),
        variant_rows=len(variants),
    )


__all__ = [
    "CONFIRMATION_FOLDS",
    "CONSTRUCTION_FOLDS",
    "DEFAULT_MAX_SEQ_LEN",
    "DEFAULT_MINIMUM_CROSS_MONTH_FRACTION",
    "DEFAULT_MINIMUM_VARIANTS",
    "EXPECTED_CONSTRUCTION_ELIGIBLE_BY_CLASS",
    "EXPECTED_CONSTRUCTION_ELIGIBLE_MEMBERSHIP_SHA256",
    "EXPECTED_CONSTRUCTION_ELIGIBLE_ROWS",
    "EXPECTED_FULL_ELIGIBLE_BY_CLASS",
    "EXPECTED_FULL_ELIGIBLE_MEMBERSHIP_SHA256",
    "EXPECTED_FULL_ELIGIBLE_ROWS",
    "EXPECTED_SHADOW_MEMBERSHIP_SHA256",
    "EXPECTED_SHADOW_ROWS",
    "PINNED_MOBILE_TOKENIZER_IDENTIFIER",
    "PINNED_MOBILE_TOKENIZER_REVISION",
    "PINNED_MOBILE_TOKENIZER_SHA256",
    "PINNED_MOBILE_TRAIN_MEMBERSHIP_SHA256",
    "PINNED_MOBILE_TRAIN_ROWS",
    "PINNED_MOBILE_TRAIN_SHA256",
    "SELECTION_FOLDS",
    "SHADOW_FOLDS",
    "TEMPORAL_SPLIT_VERSION",
    "TEMPORAL_TRANSFORM_VERSION",
    "TEMPORAL_VIEW_VERSION",
    "CounterfactualVariant",
    "FullTemporalViewArtifacts",
    "MobileTemporalError",
    "ParsedTemporalExample",
    "TemporalEligibility",
    "TemporalPlan",
    "TemporalTokenizerIdentity",
    "TemporalViewArtifacts",
    "build_month_boundary_variant",
    "classify_counterfactual_source",
    "load_pinned_mobile_train_manifest",
    "materialize_full_counterfactual_view",
    "materialize_temporal_views",
    "membership_sha256",
    "parse_temporal_example",
    "plan_temporal_views",
    "shadow_role",
]
