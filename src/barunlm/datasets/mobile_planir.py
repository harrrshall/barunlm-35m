"""Construction-only Grounded PlanIR oracle for the frozen Mobile Actions view.

This module is deliberately limited to the former v3 *construction-training*
population.  It does not expose selection, confirmation, reused-development, or
official-evaluation inputs.  Runtime grounding candidates are derived from the
request and ``NOW`` only; the gold Action IR is consulted later solely to choose a
training oracle and to prove that the deterministic compiler reproduces the target.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal

from tokenizers import Tokenizer

from barunlm.evaluation.action_ir import ActionIRError, parse_action_ir
from barunlm.evaluation.grounded_planir import MOBILE_TOOL_SCHEMAS, compile_mobile_plan
from barunlm.evaluation.grounded_planir_v2 import (
    ACTION_PROMPT_CONTRACT,
    parse_construction_prompt,
)
from barunlm.training.data import SFTExample, load_manifest, sha256_file

PLANIR_ORACLE_VERSION = "barun-mobile-grounded-planir-oracle-v1"
CONSTRUCTION_USE = "training-and-feasibility-only"
CONSTRUCTION_MANIFEST_SHA256 = "800a3ba0a7f0215e5cf95c77c32e47abed48b2f07f375954b0e0b08296a33d10"
CONSTRUCTION_ROWS = 5_745
ACCEPTED_ROWS = 5_744
ACCEPTED_MEMBERSHIP_SHA256 = "27f9acb8905db132420ddb5d11a20bed138f2e2927aec41672f6852b3895452d"
EXPECTED_DATE_OPERATOR_COUNTS = {
    "ABSOLUTE_DATE": 1_333,
    "MONTH_DAY_NEXT": 357,
    "RELATIVE_DAY": 138,
    "WEEKDAY": 241,
}
EXPECTED_CLOCK_REFERENCES = 2_070
EXPECTED_REJECTION_ID = "mobile-actions-04894-6bd7643b1bc95697"
EXPECTED_REJECTION_CODE = "temporal_reference_mismatch"
PINNED_TOKENIZER_SHA256 = "70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6"

DateOperator = Literal["ABSOLUTE_DATE", "MONTH_DAY_NEXT", "RELATIVE_DAY", "WEEKDAY"]

_ENGLISH_MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_ENGLISH_WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_MONTH_NUMBERS = {
    name.casefold(): index for index, name in enumerate(_ENGLISH_MONTH_NAMES, start=1)
}
_MONTH_PATTERN = "|".join(map(re.escape, _ENGLISH_MONTH_NAMES))
_WEEKDAY_NUMBERS = {name.casefold(): index for index, name in enumerate(_ENGLISH_WEEKDAY_NAMES)}
_WEEKDAY_PATTERN = "|".join(map(re.escape, _WEEKDAY_NUMBERS))
_ORDINAL_DAY = r"(?P<day>\d{1,2})(?P<suffix>st|nd|rd|th)?"

_ABSOLUTE_DATE_RES = (
    re.compile(r"(?<!\d)(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})(?!\d)"),
    re.compile(
        rf"\b(?P<month>{_MONTH_PATTERN})\s+{_ORDINAL_DAY},\s*(?P<year>\d{{4}})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_ORDINAL_DAY}(?:\s+of)?\s+(?P<month>{_MONTH_PATTERN}),\s*"
        rf"(?P<year>\d{{4}})\b",
        re.IGNORECASE,
    ),
)
_MONTH_DAY_RES = (
    re.compile(
        rf"\b(?P<month>{_MONTH_PATTERN})\s+{_ORDINAL_DAY}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_ORDINAL_DAY}(?:\s+of)?\s+(?P<month>{_MONTH_PATTERN})\b",
        re.IGNORECASE,
    ),
)
_RELATIVE_DATE_RE = re.compile(
    r"\b(?:day\s+after\s+tomorrow|tomorrow|tonight|today|"
    r"this\s+(?:morning|afternoon|evening)|in\s+\d+\s+days?|"
    r"\d+\s+days?\s+from\s+now)\b",
    re.IGNORECASE,
)
_RELATIVE_OFFSET_RE = re.compile(
    r"\A(?:in\s+(?P<in_days>\d+)\s+days?|(?P<from_days>\d+)\s+days?\s+from\s+now)\Z",
    re.IGNORECASE,
)
_WEEKDAY_RE = re.compile(rf"\b(?:{_WEEKDAY_PATTERN})\b", re.IGNORECASE)

_MERIDIEM_TIME_RE = re.compile(
    r"(?<![\d:])(?P<hour>1[0-2]|0?[1-9])"
    r"(?::(?P<minute>[0-5]\d))?(?::(?P<second>[0-5]\d))?"
    # The terminal period is deliberately outside the selected quote.  This is
    # the immutable quote-v1 choice; the compiler accepts the resulting ``a.m``
    # or ``p.m`` substring exactly, and changing it alters frozen token evidence.
    r"\s*(?P<meridiem>a\.?m\.?|p\.?m\.?)\b",
    re.IGNORECASE,
)
_TWENTY_FOUR_HOUR_TIME_RE = re.compile(
    r"(?<![\d:])(?P<hour>[01]\d|2[0-3]):(?P<minute>[0-5]\d)"
    r"(?::(?P<second>[0-5]\d))?(?![\d:])"
)
_DAYPART_TIME_RE = re.compile(
    r"(?<!\w)(?P<hour>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve)(?:\s+(?P<minute>thirty))?\s+in\s+the\s+"
    r"(?P<meridiem>morning|afternoon|evening|night)(?!\w)",
    re.IGNORECASE,
)
_NOON_MIDNIGHT_RE = re.compile(r"(?<!\w)(?P<clock>noon|midnight)(?!\w)", re.IGNORECASE)
_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


class MobilePlanIRError(ValueError):
    """The construction-only PlanIR oracle contract was violated."""


@dataclass(frozen=True, slots=True)
class SpanRef:
    quote: str
    occurrence: int

    def as_dict(self) -> dict[str, str | int]:
        return {"quote": self.quote, "occurrence": self.occurrence}


@dataclass(frozen=True, slots=True)
class DateCandidate:
    operator: DateOperator
    ref: SpanRef
    start: int
    resolved: date
    ordinal: int | None = None
    compatibility_tier: int = 0

    def as_plan_value(self) -> dict[str, object]:
        value: dict[str, object] = {"op": self.operator, "ref": self.ref.as_dict()}
        if self.operator == "WEEKDAY":
            assert self.ordinal in (1, 2)
            value["ordinal"] = self.ordinal
        return value


@dataclass(frozen=True, slots=True)
class ClockCandidate:
    ref: SpanRef
    start: int
    resolved: time
    compatibility_tier: int = 0

    def as_plan_value(self) -> dict[str, object]:
        return {"ref": self.ref.as_dict()}


@dataclass(frozen=True, slots=True)
class GroundingCandidates:
    dates: tuple[DateCandidate, ...]
    clocks: tuple[ClockCandidate, ...]


@dataclass(frozen=True, slots=True)
class OracleDerivation:
    example_id: str
    accepted: bool
    code: str
    plan: Mapping[str, Any] | None
    plan_json: str | None
    selected_date_operator: DateOperator | None
    selected_date_ref: SpanRef | None
    selected_clock_ref: SpanRef | None
    date_candidates: int
    clock_candidates: int
    compiled_action_ir: str | None


@dataclass(frozen=True, slots=True)
class ConstructionOracleAudit:
    use: str
    manifest_sha256: str
    source_rows: int
    accepted_rows: int
    rejected_rows: int
    accepted_membership_sha256: str
    date_operator_counts: Mapping[str, int]
    clock_references: int
    rejection_codes: Mapping[str, int]
    rejections: tuple[OracleDerivation, ...]


@dataclass(frozen=True, slots=True)
class TokenLengthDistribution:
    total: int
    mean: float
    median: float
    p90: int
    maximum: int


@dataclass(frozen=True, slots=True)
class ConstructionTokenLengthAudit:
    use: str
    manifest_sha256: str
    tokenizer_sha256: str
    source_rows: int
    compared_rows: int
    rejected_rows: int
    accepted_membership_sha256: str
    rejection_ids: tuple[str, ...]
    max_sequence_length: int
    direct_targets: TokenLengthDistribution
    planir_targets: TokenLengthDistribution
    rows_planir_exceeds_direct: int
    maximum_direct_sequence: int
    maximum_planir_sequence: int
    direct_sequences_over_limit: int
    planir_sequences_over_limit: int


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_now(raw: object) -> datetime:
    if not isinstance(raw, str):
        raise MobilePlanIRError("NOW must be canonical YYYY-MM-DDTHH:MM:SS")
    try:
        parsed = datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S")  # noqa: DTZ007
    except ValueError as error:
        raise MobilePlanIRError("NOW must be canonical YYYY-MM-DDTHH:MM:SS") from error
    if parsed.isoformat(timespec="seconds") != raw:
        raise MobilePlanIRError("NOW must be canonical YYYY-MM-DDTHH:MM:SS")
    return parsed


def _validate_request(request: object) -> str:
    if not isinstance(request, str):
        raise MobilePlanIRError("request must be a string")
    try:
        request.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise MobilePlanIRError("request must be valid UTF-8") from error
    if unicodedata.normalize("NFC", request) != request:
        raise MobilePlanIRError("request must be NFC before candidates are derived")
    return request


def _span_ref(request: str, start: int, end: int) -> SpanRef:
    normalized = unicodedata.normalize("NFC", request)
    if normalized != request:
        raise MobilePlanIRError("request must be NFC before span references are derived")
    quote = request[start:end]
    occurrence = 0
    cursor = 0
    while True:
        found = request.find(quote, cursor)
        if found < 0 or found == start:
            break
        occurrence += 1
        cursor = found + len(quote)
    if found != start:
        raise MobilePlanIRError("span is not an exact request substring")
    return SpanRef(quote=quote, occurrence=occurrence)


def _month_number(raw: str) -> int:
    if raw.isdigit():
        return int(raw)
    return _MONTH_NUMBERS[raw.casefold()]


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _ordinal_is_valid(day: int, suffix: str | None) -> bool:
    if suffix is None:
        return True
    if 10 <= day % 100 <= 20:
        expected = "th"
    else:
        expected = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return suffix.casefold() == expected


def _next_month_day(now: datetime, month: int, day: int) -> date | None:
    for year in range(now.year, min(now.year + 8, 9999) + 1):
        candidate = _safe_date(year, month, day)
        if candidate is not None and candidate >= now.date():
            return candidate
    return None


def _candidate_key(candidate: DateCandidate | ClockCandidate) -> tuple[object, ...]:
    operator_priority = {
        "ABSOLUTE_DATE": 0,
        "MONTH_DAY_NEXT": 1,
        "RELATIVE_DAY": 2,
        "WEEKDAY": 3,
    }
    operator = getattr(candidate, "operator", None)
    return (
        operator_priority.get(operator, 0),
        len(candidate.ref.quote),
        candidate.start,
        candidate.ref.quote,
        candidate.ref.occurrence,
        getattr(candidate, "ordinal", 0) or 0,
    )


def _oracle_candidate_key(candidate: DateCandidate | ClockCandidate) -> tuple[object, ...]:
    """Preserve quote-v1 winners while making compiler-aligned extensions visible.

    Candidate enumeration remains complete and target-free.  Forms added while
    hardening alignment with the already-frozen compiler receive tier 1, so they
    are selected only when no historical tier-0 candidate resolves to the gold
    value.  This prevents silently rederiving quote-v1 targets and token evidence.
    """

    return (candidate.compatibility_tier, *_candidate_key(candidate))


def _absolute_date_candidates(request: str) -> list[DateCandidate]:
    candidates: list[DateCandidate] = []
    for pattern in _ABSOLUTE_DATE_RES:
        for match in pattern.finditer(request):
            day = int(match.group("day"))
            if not _ordinal_is_valid(day, match.groupdict().get("suffix")):
                continue
            resolved = _safe_date(
                int(match.group("year")),
                _month_number(match.group("month")),
                day,
            )
            if resolved is not None:
                compatibility_tier = int(not 2000 <= resolved.year <= 2099)
                candidates.append(
                    DateCandidate(
                        operator="ABSOLUTE_DATE",
                        ref=_span_ref(request, match.start(), match.end()),
                        start=match.start(),
                        resolved=resolved,
                        compatibility_tier=compatibility_tier,
                    )
                )
    return candidates


def _month_day_candidates(request: str, now: datetime) -> list[DateCandidate]:
    candidates: list[DateCandidate] = []
    for pattern in _MONTH_DAY_RES:
        for match in pattern.finditer(request):
            day = int(match.group("day"))
            if not _ordinal_is_valid(day, match.groupdict().get("suffix")):
                continue
            month = _month_number(match.group("month"))
            resolved = _next_month_day(now, month, day)
            if resolved is not None:
                legacy_resolved = _safe_date(now.year, month, day)
                if legacy_resolved is not None and legacy_resolved < now.date():
                    legacy_resolved = _safe_date(
                        legacy_resolved.year + 1,
                        legacy_resolved.month,
                        legacy_resolved.day,
                    )
                candidates.append(
                    DateCandidate(
                        operator="MONTH_DAY_NEXT",
                        ref=_span_ref(request, match.start(), match.end()),
                        start=match.start(),
                        resolved=resolved,
                        compatibility_tier=int(legacy_resolved != resolved),
                    )
                )
    return candidates


def _relative_date_candidates(request: str, now: datetime) -> list[DateCandidate]:
    candidates: list[DateCandidate] = []
    for match in _RELATIVE_DATE_RE.finditer(request):
        phrase = match.group().casefold()
        if phrase == "tomorrow":
            delta = 1
        elif phrase == "day after tomorrow":
            delta = 2
        elif phrase in {"today", "tonight", "this morning", "this afternoon", "this evening"}:
            delta = 0
        else:
            offset_match = _RELATIVE_OFFSET_RE.fullmatch(phrase)
            if offset_match is None:
                continue
            raw_days = offset_match.group("in_days") or offset_match.group("from_days")
            assert raw_days is not None
            if len(raw_days) > 3 or int(raw_days) > 366:
                continue
            delta = int(raw_days)
        compatibility_tier = int(_RELATIVE_OFFSET_RE.fullmatch(phrase) is not None)
        try:
            resolved = now.date() + timedelta(days=delta)
        except OverflowError:
            continue
        candidates.append(
            DateCandidate(
                operator="RELATIVE_DAY",
                ref=_span_ref(request, match.start(), match.end()),
                start=match.start(),
                resolved=resolved,
                compatibility_tier=compatibility_tier,
            )
        )
    return candidates


def _weekday_candidates(request: str, now: datetime) -> list[DateCandidate]:
    candidates: list[DateCandidate] = []
    for match in _WEEKDAY_RE.finditer(request):
        if re.search(
            r"\b(?:last|previous)\s*$", request[: match.start()], re.IGNORECASE
        ) or re.match(r"\s+ago\b", request[match.end() :], re.IGNORECASE):
            continue
        weekday = _WEEKDAY_NUMBERS[match.group().casefold()]
        first_delta = (weekday - now.weekday()) % 7 or 7
        for ordinal in (1, 2):
            try:
                resolved = now.date() + timedelta(days=first_delta + 7 * (ordinal - 1))
            except OverflowError:
                continue
            candidates.append(
                DateCandidate(
                    operator="WEEKDAY",
                    ref=_span_ref(request, match.start(), match.end()),
                    start=match.start(),
                    resolved=resolved,
                    ordinal=ordinal,
                )
            )
    return candidates


def _clock_candidates(request: str) -> list[ClockCandidate]:
    candidates: list[ClockCandidate] = []
    for match in _MERIDIEM_TIME_RE.finditer(request):
        hour = int(match.group("hour")) % 12
        if match.group("meridiem").casefold().startswith("p"):
            hour += 12
        resolved = time(
            hour,
            int(match.group("minute") or 0),
            int(match.group("second") or 0),
        )
        candidates.append(
            ClockCandidate(
                ref=_span_ref(request, match.start(), match.end()),
                start=match.start(),
                resolved=resolved,
            )
        )
    for match in _TWENTY_FOUR_HOUR_TIME_RE.finditer(request):
        resolved = time(
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second") or 0),
        )
        candidates.append(
            ClockCandidate(
                ref=_span_ref(request, match.start(), match.end()),
                start=match.start(),
                resolved=resolved,
            )
        )
    for match in _DAYPART_TIME_RE.finditer(request):
        raw_hour = match.group("hour").casefold()
        hour = int(raw_hour) if raw_hour.isdigit() else _WORD_NUMBERS[raw_hour]
        if not 1 <= hour <= 12:
            continue
        if match.group("meridiem").casefold() == "night" and (hour == 12 or 1 <= hour <= 6):
            continue
        hour %= 12
        if match.group("meridiem").casefold() in {"afternoon", "evening", "night"}:
            hour += 12
        resolved = time(hour, 30 if match.group("minute") is not None else 0)
        compatibility_tier = int(
            match.group("meridiem").casefold() == "night"
            or (raw_hour.isdigit() and match.group("minute") is not None)
        )
        candidates.append(
            ClockCandidate(
                ref=_span_ref(request, match.start(), match.end()),
                start=match.start(),
                resolved=resolved,
                compatibility_tier=compatibility_tier,
            )
        )
    for match in _NOON_MIDNIGHT_RE.finditer(request):
        resolved = time(12, 0) if match.group("clock").casefold() == "noon" else time(0, 0)
        candidates.append(
            ClockCandidate(
                ref=_span_ref(request, match.start(), match.end()),
                start=match.start(),
                resolved=resolved,
                compatibility_tier=1,
            )
        )
    unique = {
        (candidate.ref.quote, candidate.ref.occurrence, candidate.resolved): candidate
        for candidate in candidates
    }
    return sorted(unique.values(), key=_candidate_key)


def enumerate_grounding_candidates(request: str, now: str) -> GroundingCandidates:
    """Derive all grounding candidates without consulting a target or model output."""

    request = _validate_request(request)
    parsed_now = _parse_now(now)
    dates = [
        *_absolute_date_candidates(request),
        *_month_day_candidates(request, parsed_now),
        *_relative_date_candidates(request, parsed_now),
        *_weekday_candidates(request, parsed_now),
    ]
    unique_dates = {
        (
            candidate.operator,
            candidate.ref.quote,
            candidate.ref.occurrence,
            candidate.resolved,
            candidate.ordinal,
        ): candidate
        for candidate in dates
    }
    return GroundingCandidates(
        dates=tuple(sorted(unique_dates.values(), key=_candidate_key)),
        clocks=tuple(_clock_candidates(request)),
    )


def _parse_prompt(example: SFTExample) -> tuple[str, str, datetime]:
    if not isinstance(example.prompt, str):
        raise MobilePlanIRError(f"example {example.example_id!r} prompt must be a string")
    try:
        prompt = parse_construction_prompt(example.prompt)
    except Exception as error:
        raise MobilePlanIRError(
            f"example {example.example_id!r} violates the canonical prompt contract"
        ) from error
    if prompt.prompt_contract != ACTION_PROMPT_CONTRACT or prompt.rendered_table is not None:
        raise MobilePlanIRError(
            f"example {example.example_id!r} must be an unaugmented ACTION_IR_V1 prompt"
        )
    request = _validate_request(prompt.request)
    return request, prompt.now, _parse_now(prompt.now)


def _parse_target(example: SFTExample) -> dict[str, Any]:
    try:
        target = json.loads(example.target)
    except (json.JSONDecodeError, TypeError) as error:
        raise MobilePlanIRError(f"example {example.example_id!r} target is not JSON") from error
    if not isinstance(target, dict):
        raise MobilePlanIRError(f"example {example.example_id!r} target is not an object")
    try:
        canonical_target = _canonical_json(target)
    except (TypeError, ValueError) as error:
        raise MobilePlanIRError(
            f"example {example.example_id!r} target contains an invalid JSON value"
        ) from error
    if canonical_target != example.target:
        raise MobilePlanIRError(f"example {example.example_id!r} target is not canonical JSON")
    try:
        parse_action_ir(example.target, MOBILE_TOOL_SCHEMAS)
    except ActionIRError as error:
        raise MobilePlanIRError(
            f"example {example.example_id!r} target is not valid Action IR: {error.code}"
        ) from error
    return target


def _plan_with_references(
    target: Mapping[str, Any],
    *,
    date_candidate: DateCandidate | None,
    clock_candidate: ClockCandidate | None,
) -> dict[str, Any]:
    plan = json.loads(_canonical_json(target))
    plan["schema_version"] = "grounded-plan-ir-v1"
    for call in plan.get("calls", []):
        if call["tool"] == "create_calendar_event":
            if date_candidate is None or clock_candidate is None:
                raise MobilePlanIRError("calendar plans require both date and clock references")
            call["args"]["datetime"] = {
                "date": date_candidate.as_plan_value(),
                "time": clock_candidate.as_plan_value(),
            }
    return plan


def _validate_construction_metadata(example: SFTExample) -> None:
    """Defense in depth; only the pinned bulk-manifest hash is authoritative."""

    required = {
        "adapter_schema_version": "barun-mobile-actions-adapter-v2",
        "dataset": "google/mobile-actions",
        "derived_split": "train",
        "prompt_contract_version": "barun-action-prompt-v1",
        "source_split": "train",
    }
    for key, expected in required.items():
        if example.metadata.get(key) != expected:
            raise MobilePlanIRError(
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
            raise MobilePlanIRError(
                f"example {example.example_id!r} metadata.{key} is forbidden construction input"
            )


def _derive_oracle(example: SFTExample) -> OracleDerivation:
    """Derive and compile one gold-supervised oracle after target-free enumeration."""

    _validate_construction_metadata(example)
    request, now_string, _now = _parse_prompt(example)
    candidates = enumerate_grounding_candidates(request, now_string)
    target = _parse_target(example)
    calendar_calls = [
        call for call in target.get("calls", []) if call["tool"] == "create_calendar_event"
    ]
    if len(calendar_calls) > 1:
        raise MobilePlanIRError("the construction contract permits at most one calendar call")

    selected_date: DateCandidate | None = None
    selected_clock: ClockCandidate | None = None
    if calendar_calls:
        raw_datetime = calendar_calls[0]["args"].get("datetime")
        try:
            gold_datetime = datetime.fromisoformat(raw_datetime)
        except (TypeError, ValueError) as error:
            raise MobilePlanIRError("calendar gold datetime must be ISO local datetime") from error
        selected_date = min(
            (
                candidate
                for candidate in candidates.dates
                if candidate.resolved == gold_datetime.date()
            ),
            key=_oracle_candidate_key,
            default=None,
        )
        selected_clock = min(
            (
                candidate
                for candidate in candidates.clocks
                if candidate.resolved == gold_datetime.time()
            ),
            key=_oracle_candidate_key,
            default=None,
        )
        if selected_date is None or selected_clock is None:
            return OracleDerivation(
                example_id=example.example_id,
                accepted=False,
                code="temporal_reference_mismatch",
                plan=None,
                plan_json=None,
                selected_date_operator=selected_date.operator if selected_date else None,
                selected_date_ref=selected_date.ref if selected_date else None,
                selected_clock_ref=selected_clock.ref if selected_clock else None,
                date_candidates=len(candidates.dates),
                clock_candidates=len(candidates.clocks),
                compiled_action_ir=None,
            )

    plan = _plan_with_references(
        target,
        date_candidate=selected_date,
        clock_candidate=selected_clock,
    )
    plan_json = _canonical_json(plan)
    compiled = compile_mobile_plan(
        plan_json,
        request=request,
        now=now_string,
        schemas=MOBILE_TOOL_SCHEMAS,
    )
    if not compiled.ok or compiled.action_ir != example.target:
        code = "compiler_rejected_oracle" if not compiled.ok else "compiled_target_mismatch"
        return OracleDerivation(
            example_id=example.example_id,
            accepted=False,
            code=code,
            plan=plan,
            plan_json=plan_json,
            selected_date_operator=selected_date.operator if selected_date else None,
            selected_date_ref=selected_date.ref if selected_date else None,
            selected_clock_ref=selected_clock.ref if selected_clock else None,
            date_candidates=len(candidates.dates),
            clock_candidates=len(candidates.clocks),
            compiled_action_ir=compiled.action_ir,
        )
    return OracleDerivation(
        example_id=example.example_id,
        accepted=True,
        code="accepted",
        plan=plan,
        plan_json=plan_json,
        selected_date_operator=selected_date.operator if selected_date else None,
        selected_date_ref=selected_date.ref if selected_date else None,
        selected_clock_ref=selected_clock.ref if selected_clock else None,
        date_candidates=len(candidates.dates),
        clock_candidates=len(candidates.clocks),
        compiled_action_ir=compiled.action_ir,
    )


def _membership_sha256(example_ids: Sequence[str]) -> str:
    if len(example_ids) != len(set(example_ids)):
        raise MobilePlanIRError("accepted oracle membership contains duplicate IDs")
    payload = "\n".join(sorted(example_ids)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_pinned_construction_manifest(path: str | Path) -> tuple[SFTExample, ...]:
    examples = load_manifest(
        path,
        expected_sha256=CONSTRUCTION_MANIFEST_SHA256,
        expected_derived_split="train",
    )
    if len(examples) != CONSTRUCTION_ROWS:
        raise MobilePlanIRError(
            f"construction row count changed: {len(examples)} != {CONSTRUCTION_ROWS}"
        )
    for example in examples:
        _validate_construction_metadata(example)
    return tuple(examples)


def _token_length_distribution(values: Sequence[int]) -> TokenLengthDistribution:
    if not values:
        raise MobilePlanIRError("cannot summarize an empty token-length population")
    ordered = sorted(values)
    p90_index = max(0, (9 * len(ordered) + 9) // 10 - 1)
    return TokenLengthDistribution(
        total=sum(values),
        mean=statistics.fmean(values),
        median=float(statistics.median(values)),
        p90=ordered[p90_index],
        maximum=ordered[-1],
    )


def audit_construction_target_lengths(
    manifest_path: str | Path,
    tokenizer_path: str | Path,
    *,
    max_sequence_length: int = 2_048,
) -> ConstructionTokenLengthAudit:
    """Compare direct and oracle target lengths without reading a held-out population."""

    if (
        not isinstance(max_sequence_length, int)
        or isinstance(max_sequence_length, bool)
        or max_sequence_length <= 0
    ):
        raise MobilePlanIRError("max_sequence_length must be a positive integer")
    actual_tokenizer_sha256 = sha256_file(tokenizer_path)
    if actual_tokenizer_sha256 != PINNED_TOKENIZER_SHA256:
        raise MobilePlanIRError(
            f"tokenizer SHA-256 changed: {actual_tokenizer_sha256} != {PINNED_TOKENIZER_SHA256}"
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    examples = _load_pinned_construction_manifest(manifest_path)

    direct_lengths: list[int] = []
    plan_lengths: list[int] = []
    direct_sequences: list[int] = []
    plan_sequences: list[int] = []
    rejected_rows = 0
    accepted_ids: list[str] = []
    rejection_ids: list[str] = []
    for example in examples:
        derivation = _derive_oracle(example)
        if not derivation.accepted:
            rejected_rows += 1
            rejection_ids.append(derivation.example_id)
            continue
        accepted_ids.append(derivation.example_id)
        assert derivation.plan_json is not None
        prompt_length = len(tokenizer.encode(example.prompt, add_special_tokens=False).ids)
        direct_length = len(tokenizer.encode(example.target, add_special_tokens=False).ids)
        plan_length = len(tokenizer.encode(derivation.plan_json, add_special_tokens=False).ids)
        direct_lengths.append(direct_length)
        plan_lengths.append(plan_length)
        direct_sequences.append(prompt_length + direct_length + 1)
        plan_sequences.append(prompt_length + plan_length + 1)

    membership = _membership_sha256(accepted_ids)
    if (
        len(direct_lengths) != ACCEPTED_ROWS
        or rejected_rows != 1
        or membership != ACCEPTED_MEMBERSHIP_SHA256
        or rejection_ids != [EXPECTED_REJECTION_ID]
    ):
        raise MobilePlanIRError("token audit oracle membership changed")
    return ConstructionTokenLengthAudit(
        use=CONSTRUCTION_USE,
        manifest_sha256=CONSTRUCTION_MANIFEST_SHA256,
        tokenizer_sha256=actual_tokenizer_sha256,
        source_rows=len(examples),
        compared_rows=len(direct_lengths),
        rejected_rows=rejected_rows,
        accepted_membership_sha256=membership,
        rejection_ids=tuple(rejection_ids),
        max_sequence_length=max_sequence_length,
        direct_targets=_token_length_distribution(direct_lengths),
        planir_targets=_token_length_distribution(plan_lengths),
        rows_planir_exceeds_direct=sum(
            plan > direct for direct, plan in zip(direct_lengths, plan_lengths, strict=True)
        ),
        maximum_direct_sequence=max(direct_sequences),
        maximum_planir_sequence=max(plan_sequences),
        direct_sequences_over_limit=sum(
            length > max_sequence_length for length in direct_sequences
        ),
        planir_sequences_over_limit=sum(length > max_sequence_length for length in plan_sequences),
    )


def audit_construction_oracles(path: str | Path) -> ConstructionOracleAudit:
    """Audit the pinned former-v3 construction file as training/feasibility data only."""

    examples = _load_pinned_construction_manifest(path)
    derivations = tuple(_derive_oracle(example) for example in examples)
    accepted = tuple(item for item in derivations if item.accepted)
    rejected = tuple(item for item in derivations if not item.accepted)
    membership = _membership_sha256([item.example_id for item in accepted])
    operator_counts = Counter(
        item.selected_date_operator for item in accepted if item.selected_date_operator is not None
    )
    clock_references = sum(item.selected_clock_ref is not None for item in derivations)
    rejection_codes = Counter(item.code for item in rejected)

    expected_rejections = ((EXPECTED_REJECTION_ID, EXPECTED_REJECTION_CODE),)
    observed_rejections = tuple((item.example_id, item.code) for item in rejected)
    if len(accepted) != ACCEPTED_ROWS:
        raise MobilePlanIRError(f"accepted row count changed: {len(accepted)} != {ACCEPTED_ROWS}")
    if membership != ACCEPTED_MEMBERSHIP_SHA256:
        raise MobilePlanIRError("accepted oracle membership changed")
    if dict(operator_counts) != EXPECTED_DATE_OPERATOR_COUNTS:
        raise MobilePlanIRError(
            f"date operator counts changed: {dict(operator_counts)} != "
            f"{EXPECTED_DATE_OPERATOR_COUNTS}"
        )
    if clock_references != EXPECTED_CLOCK_REFERENCES:
        raise MobilePlanIRError(
            f"clock reference count changed: {clock_references} != {EXPECTED_CLOCK_REFERENCES}"
        )
    if observed_rejections != expected_rejections:
        raise MobilePlanIRError(f"construction rejection evidence changed: {observed_rejections!r}")
    return ConstructionOracleAudit(
        use=CONSTRUCTION_USE,
        manifest_sha256=CONSTRUCTION_MANIFEST_SHA256,
        source_rows=len(examples),
        accepted_rows=len(accepted),
        rejected_rows=len(rejected),
        accepted_membership_sha256=membership,
        date_operator_counts=dict(operator_counts),
        clock_references=clock_references,
        rejection_codes=dict(rejection_codes),
        rejections=rejected,
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
    "PLANIR_ORACLE_VERSION",
    "ConstructionOracleAudit",
    "ConstructionTokenLengthAudit",
    "GroundingCandidates",
    "MobilePlanIRError",
    "OracleDerivation",
    "TokenLengthDistribution",
    "audit_construction_oracles",
    "audit_construction_target_lengths",
    "enumerate_grounding_candidates",
]
