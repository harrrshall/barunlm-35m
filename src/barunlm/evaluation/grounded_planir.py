"""Strict Grounded PlanIR v1 compiler for the Mobile Actions research lane.

Grounded PlanIR is an experimental, model-facing representation.  It does not
change the production Action IR contract.  The compiler is deliberately total,
pure, and failure-closed: it performs no repair, retries, tool execution, clock
reads, context lookup, or network access.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from types import MappingProxyType
from typing import Any

from .action_ir import (
    ActionIRError,
    ActionIRParseError,
    Decision,
    ToolSchema,
    decode_json_object,
    validate_action_ir,
)
from .mobile_action_schemas import MOBILE_TOOL_SCHEMAS

SCHEMA_VERSION = "grounded-plan-ir-v1"


class GroundedPlanIRError(ValueError):
    """Internal compiler failure with a stable public code and JSON path."""

    def __init__(self, code: str, message: str, path: str = "$") -> None:
        super().__init__(f"{path}: {message}")
        self.code = code
        self.message = message
        self.path = path


@dataclass(frozen=True, slots=True)
class CompileFailure:
    """One serializable, structured compilation failure."""

    code: str
    message: str
    path: str


@dataclass(frozen=True, slots=True)
class CompileResult:
    """Exactly one of a canonical Action IR string or a structured failure."""

    ok: bool
    action_ir: str | None = None
    error: CompileFailure | None = None

    def __post_init__(self) -> None:
        if self.ok:
            if self.action_ir is None or self.error is not None:
                raise ValueError("successful CompileResult requires only action_ir")
        elif self.action_ir is not None or self.error is None:
            raise ValueError("failed CompileResult requires only error")


_MOBILE_TOOL_SCHEMA_BY_NAME: Mapping[str, ToolSchema] = MappingProxyType(
    {schema.name: schema for schema in MOBILE_TOOL_SCHEMAS}
)


@dataclass(frozen=True, slots=True)
class _ArgumentContract:
    required: frozenset[str]
    optional: frozenset[str] = frozenset()
    datetime_argument: str | None = None

    @property
    def names(self) -> frozenset[str]:
        return self.required | self.optional


_DATETIME_ARGUMENT_BY_TOOL: Mapping[str, str] = MappingProxyType(
    {"create_calendar_event": "datetime"}
)
_TOOL_CONTRACTS: Mapping[str, _ArgumentContract] = MappingProxyType(
    {
        schema.name: _ArgumentContract(
            required=schema.required,
            optional=frozenset(schema.arguments).difference(schema.required),
            datetime_argument=_DATETIME_ARGUMENT_BY_TOOL.get(schema.name),
        )
        for schema in MOBILE_TOOL_SCHEMAS
    }
)

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
_MONTHS = {name.casefold(): number for number, name in enumerate(_ENGLISH_MONTH_NAMES, start=1)}
_WEEKDAYS = {name.casefold(): number for number, name in enumerate(_ENGLISH_WEEKDAY_NAMES)}
_ORDINAL_RE = r"(?P<day>\d{1,2})(?P<suffix>st|nd|rd|th)?"
_MONTH_NAME_RE = "(?:" + "|".join(_ENGLISH_MONTH_NAMES) + ")"
_ABSOLUTE_MONTH_FIRST_RE = re.compile(
    rf"\A(?P<month>{_MONTH_NAME_RE})\s+{_ORDINAL_RE},\s*(?P<year>\d{{4}})\Z",
    re.IGNORECASE,
)
_ABSOLUTE_DAY_FIRST_RE = re.compile(
    rf"\A{_ORDINAL_RE}(?:\s+of)?\s+(?P<month>{_MONTH_NAME_RE}),\s*"
    rf"(?P<year>\d{{4}})\Z",
    re.IGNORECASE,
)
_MONTH_DAY_MONTH_FIRST_RE = re.compile(
    rf"\A(?P<month>{_MONTH_NAME_RE})\s+{_ORDINAL_RE}\Z", re.IGNORECASE
)
_MONTH_DAY_DAY_FIRST_RE = re.compile(
    rf"\A{_ORDINAL_RE}(?:\s+of)?\s+(?P<month>{_MONTH_NAME_RE})\Z",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\A(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\Z")
_RELATIVE_N_RE = re.compile(
    r"\A(?:in\s+(?P<in_days>\d+)\s+days?|(?P<from_days>\d+)\s+days?\s+from\s+now)\Z",
    re.IGNORECASE,
)
_WEEKDAY_RE = re.compile(r"\A(?:" + "|".join(_ENGLISH_WEEKDAY_NAMES) + r")\Z", re.IGNORECASE)

_MERIDIEM_RE = re.compile(
    r"\A(?P<hour>\d{1,2})(?::(?P<minute>\d{2})(?::(?P<second>\d{2}))?)?\s*"
    r"(?P<meridiem>a\.?m\.?|p\.?m\.?)\Z",
    re.IGNORECASE,
)
_TWENTY_FOUR_HOUR_RE = re.compile(r"\A(?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?\Z")
_PERIOD_RE = re.compile(
    r"\A(?P<hour>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve)(?:\s+(?P<word_minute>thirty))?\s+in\s+the\s+"
    r"(?P<period>morning|afternoon|evening|night)\Z",
    re.IGNORECASE,
)
_CLOCK_SEARCH_RE = re.compile(
    r"(?<!\w)(?:"
    r"\d{1,2}(?::\d{2}(?::\d{2})?)?\s*(?:a\.?m\.?|p\.?m\.?)"
    r"|\d{2}:\d{2}(?::\d{2})?"
    r"|(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"(?:\s+thirty)?\s+in\s+the\s+(?:morning|afternoon|evening|night)"
    r"|noon|midnight)(?!\w)",
    re.IGNORECASE,
)
_WORD_HOURS = {
    word: value
    for value, word in enumerate(
        (
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
        )
    )
}


def _exact_fields(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = expected.difference(value)
    if missing:
        raise GroundedPlanIRError("missing_field", f"missing fields: {sorted(missing)!r}", path)
    extra = set(value).difference(expected)
    if extra:
        raise GroundedPlanIRError("unknown_field", f"unknown fields: {sorted(extra)!r}", path)


def _validate_mobile_schemas(
    schemas: Mapping[str, ToolSchema] | Iterable[ToolSchema],
) -> tuple[ToolSchema, ...]:
    """Require semantic identity with the single frozen seven-tool registry."""

    mapping_items: tuple[tuple[object, object], ...] | None = None
    try:
        if isinstance(schemas, Mapping):
            mapping_items = tuple(schemas.items())
            supplied: tuple[object, ...] = tuple(schema for _, schema in mapping_items)
        else:
            supplied = tuple(schemas)
    except Exception as exc:
        raise GroundedPlanIRError(
            "final_action_ir_validation_failure",
            "schemas must be a mapping or iterable of ToolSchema objects",
            "$.schemas",
        ) from exc

    actual: dict[str, ToolSchema] = {}
    for index, raw_schema in enumerate(supplied):
        path = f"$.schemas[{index}]"
        if not isinstance(raw_schema, ToolSchema):
            raise GroundedPlanIRError(
                "final_action_ir_validation_failure",
                "every supplied schema must be a ToolSchema",
                path,
            )
        if mapping_items is not None:
            raw_key = mapping_items[index][0]
            if raw_key != raw_schema.name:
                raise GroundedPlanIRError(
                    "final_action_ir_validation_failure",
                    f"schema mapping key {raw_key!r} does not match {raw_schema.name!r}",
                    path,
                )
        if raw_schema.name in actual:
            raise GroundedPlanIRError(
                "final_action_ir_validation_failure",
                f"duplicate tool schema {raw_schema.name!r}",
                path,
            )
        actual[raw_schema.name] = raw_schema

    expected_names = set(_MOBILE_TOOL_SCHEMA_BY_NAME)
    actual_names = set(actual)
    missing = expected_names.difference(actual_names)
    extra = actual_names.difference(expected_names)
    if missing or extra:
        raise GroundedPlanIRError(
            "final_action_ir_validation_failure",
            f"schemas differ from frozen Mobile tools; missing={sorted(missing)!r}, "
            f"extra={sorted(extra)!r}",
            "$.schemas",
        )
    for name in sorted(expected_names):
        if actual[name] != _MOBILE_TOOL_SCHEMA_BY_NAME[name]:
            raise GroundedPlanIRError(
                "final_action_ir_validation_failure",
                f"schema for {name!r} differs from the frozen Mobile contract",
                f"$.schemas.{name}",
            )
    return MOBILE_TOOL_SCHEMAS


def _mapping(value: object, *, path: str, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GroundedPlanIRError(
            "final_action_ir_validation_failure", f"{label} must be an object", path
        )
    return value


def _text(value: object, *, path: str, label: str, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise GroundedPlanIRError(
            "final_action_ir_validation_failure", f"{label} must be a string", path
        )
    normalized = unicodedata.normalize("NFC", value)
    if nonempty and not normalized:
        raise GroundedPlanIRError(
            "final_action_ir_validation_failure", f"{label} must be non-empty", path
        )
    return normalized


def resolve_span_ref(request: str, ref: object, *, path: str = "$") -> str:
    """Resolve one exact, case-sensitive, zero-based, non-overlapping span reference."""

    if not isinstance(request, str):
        raise GroundedPlanIRError("reference_not_found", "request must be a string", "$.request")
    request = unicodedata.normalize("NFC", request)
    ref_obj = _mapping(ref, path=path, label="SpanRef")
    _exact_fields(ref_obj, {"occurrence", "quote"}, path)
    quote = ref_obj["quote"]
    occurrence = ref_obj["occurrence"]
    if not isinstance(quote, str) or not unicodedata.normalize("NFC", quote):
        raise GroundedPlanIRError(
            "reference_not_found", "quote must be a non-empty string", f"{path}.quote"
        )
    quote = unicodedata.normalize("NFC", quote)
    if not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 0:
        raise GroundedPlanIRError(
            "reference_occurrence_out_of_range",
            "occurrence must be a non-negative integer",
            f"{path}.occurrence",
        )

    positions: list[int] = []
    cursor = 0
    while True:
        position = request.find(quote, cursor)
        if position < 0:
            break
        positions.append(position)
        cursor = position + len(quote)
    if not positions:
        raise GroundedPlanIRError(
            "reference_not_found", "quote does not occur exactly in request", path
        )
    if occurrence >= len(positions):
        raise GroundedPlanIRError(
            "reference_occurrence_out_of_range",
            f"occurrence {occurrence} is outside {len(positions)} exact matches",
            f"{path}.occurrence",
        )
    start = positions[occurrence]
    return request[start : start + len(quote)]


def _parse_now(raw: object) -> datetime:
    if not isinstance(raw, str):
        raise GroundedPlanIRError(
            "invalid_calendar_date", "now must be a naive ISO datetime with seconds", "$.now"
        )
    try:
        parsed = datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S")  # noqa: DTZ007
    except ValueError as exc:
        raise GroundedPlanIRError(
            "invalid_calendar_date", "now must be a naive ISO datetime with seconds", "$.now"
        ) from exc
    if parsed.isoformat(timespec="seconds") != raw:
        raise GroundedPlanIRError(
            "invalid_calendar_date", "now must use canonical YYYY-MM-DDTHH:MM:SS", "$.now"
        )
    return parsed


def _ordinal_is_valid(day: int, suffix: str | None) -> bool:
    if suffix is None:
        return True
    if 10 <= day % 100 <= 20:
        expected = "th"
    else:
        expected = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return suffix.casefold() == expected


def _date_from_parts(match: re.Match[str], *, path: str) -> date:
    day = int(match.group("day"))
    suffix = match.groupdict().get("suffix")
    if not _ordinal_is_valid(day, suffix):
        raise GroundedPlanIRError(
            "invalid_calendar_date", "day has an invalid ordinal suffix", path
        )
    month = _MONTHS[match.group("month").casefold()]
    year = int(match.group("year"))
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise GroundedPlanIRError(
            "invalid_calendar_date", "date does not exist in the Gregorian calendar", path
        ) from exc


def _parse_absolute_date(quote: str, *, path: str) -> date:
    iso = _ISO_DATE_RE.fullmatch(quote)
    if iso is not None:
        try:
            return date(int(iso.group("year")), int(iso.group("month")), int(iso.group("day")))
        except ValueError as exc:
            raise GroundedPlanIRError(
                "invalid_calendar_date", "date does not exist in the Gregorian calendar", path
            ) from exc
    for pattern in (_ABSOLUTE_MONTH_FIRST_RE, _ABSOLUTE_DAY_FIRST_RE):
        match = pattern.fullmatch(quote)
        if match is not None:
            return _date_from_parts(match, path=path)
    raise GroundedPlanIRError(
        "temporal_reference_mismatch", "reference is not an ABSOLUTE_DATE expression", path
    )


def _parse_month_day_next(quote: str, now: datetime, *, path: str) -> date:
    match = None
    for pattern in (_MONTH_DAY_MONTH_FIRST_RE, _MONTH_DAY_DAY_FIRST_RE):
        match = pattern.fullmatch(quote)
        if match is not None:
            break
    if match is None:
        raise GroundedPlanIRError(
            "temporal_reference_mismatch", "reference is not a MONTH_DAY_NEXT expression", path
        )
    day = int(match.group("day"))
    suffix = match.groupdict().get("suffix")
    if not _ordinal_is_valid(day, suffix):
        raise GroundedPlanIRError(
            "invalid_calendar_date", "day has an invalid ordinal suffix", path
        )
    month = _MONTHS[match.group("month").casefold()]
    valid_day_in_any_year = False
    for year in range(now.year, min(now.year + 8, 9999) + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        valid_day_in_any_year = True
        if candidate >= now.date():
            return candidate
    message = (
        "date cannot be resolved within the supported Gregorian range"
        if valid_day_in_any_year
        else "date does not exist in the Gregorian calendar"
    )
    raise GroundedPlanIRError("invalid_calendar_date", message, path)


def _parse_relative_day(quote: str, now: datetime, *, path: str) -> date:
    simple = {
        "today": 0,
        "tonight": 0,
        "this morning": 0,
        "this afternoon": 0,
        "this evening": 0,
        "tomorrow": 1,
        "day after tomorrow": 2,
    }
    folded = quote.casefold()
    if folded in simple:
        days = simple[folded]
    else:
        match = _RELATIVE_N_RE.fullmatch(quote)
        if match is None:
            raise GroundedPlanIRError(
                "temporal_reference_mismatch", "reference is not a RELATIVE_DAY expression", path
            )
        raw_days = match.group("in_days") or match.group("from_days")
        assert raw_days is not None
        if len(raw_days) > 3 or int(raw_days) > 366:
            raise GroundedPlanIRError(
                "invalid_calendar_date", "relative-day offset must be in [0, 366]", path
            )
        days = int(raw_days)
    try:
        return now.date() + timedelta(days=days)
    except OverflowError as exc:
        raise GroundedPlanIRError(
            "invalid_calendar_date", "relative date exceeds the Gregorian range", path
        ) from exc


def _parse_weekday(quote: str, ordinal: object, now: datetime, *, path: str) -> date:
    if _WEEKDAY_RE.fullmatch(quote) is None:
        raise GroundedPlanIRError(
            "temporal_reference_mismatch", "reference is not a weekday name", path
        )
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal not in (1, 2):
        raise GroundedPlanIRError(
            "temporal_reference_mismatch", "WEEKDAY ordinal must be 1 or 2", f"{path}.ordinal"
        )
    weekday = _WEEKDAYS[quote.casefold()]
    delta = (weekday - now.weekday()) % 7
    if delta == 0:
        delta = 7
    delta += 7 * (ordinal - 1)
    try:
        return now.date() + timedelta(days=delta)
    except OverflowError as exc:
        raise GroundedPlanIRError(
            "invalid_calendar_date", "weekday date exceeds the Gregorian range", path
        ) from exc


def _weekday_ref_is_explicitly_past(request: str, ref: object, quote: str) -> bool:
    ref_obj = _mapping(ref, path="$.weekday_ref", label="SpanRef")
    occurrence = ref_obj["occurrence"]
    assert isinstance(occurrence, int) and not isinstance(occurrence, bool)
    positions: list[int] = []
    cursor = 0
    while True:
        position = request.find(quote, cursor)
        if position < 0:
            break
        positions.append(position)
        cursor = position + len(quote)
    start = positions[occurrence]
    end = start + len(quote)
    return bool(
        re.search(r"\b(?:last|previous)\s*$", request[:start], re.IGNORECASE)
        or re.match(r"\s+ago\b", request[end:], re.IGNORECASE)
    )


def _parse_plan_date(value: object, request: str, now: datetime, *, path: str) -> date:
    expr = _mapping(value, path=path, label="date expression")
    op = expr.get("op")
    if op == "WEEKDAY":
        _exact_fields(expr, {"op", "ordinal", "ref"}, path)
    else:
        _exact_fields(expr, {"op", "ref"}, path)
    quote = resolve_span_ref(request, expr["ref"], path=f"{path}.ref")
    if op == "ABSOLUTE_DATE":
        return _parse_absolute_date(quote, path=f"{path}.ref")
    if op == "MONTH_DAY_NEXT":
        return _parse_month_day_next(quote, now, path=f"{path}.ref")
    if op == "RELATIVE_DAY":
        return _parse_relative_day(quote, now, path=f"{path}.ref")
    if op == "WEEKDAY":
        if _weekday_ref_is_explicitly_past(request, expr["ref"], quote):
            raise GroundedPlanIRError(
                "past_datetime",
                "explicitly past weekday expressions cannot become future references",
                f"{path}.ref",
            )
        return _parse_weekday(quote, expr["ordinal"], now, path=path)
    raise GroundedPlanIRError(
        "temporal_reference_mismatch", f"unknown date operation {op!r}", f"{path}.op"
    )


def _clock_from_match(match: re.Match[str], *, twelve_hour: bool, path: str) -> time:
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    second = int(match.group("second") or 0)
    if twelve_hour:
        if not 1 <= hour <= 12:
            raise GroundedPlanIRError(
                "invalid_clock_time", "12-hour clock hour must be in [1, 12]", path
            )
        meridiem = match.group("meridiem").replace(".", "").casefold()
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    if hour > 23 or minute > 59 or second > 59:
        raise GroundedPlanIRError(
            "invalid_clock_time", "clock time is outside its valid range", path
        )
    return time(hour, minute, second)


def _parse_clock_exact(quote: str, *, path: str) -> time:
    folded = quote.casefold()
    if folded == "noon":
        return time(12, 0, 0)
    if folded == "midnight":
        return time(0, 0, 0)
    match = _MERIDIEM_RE.fullmatch(quote)
    if match is not None:
        return _clock_from_match(match, twelve_hour=True, path=path)
    match = _TWENTY_FOUR_HOUR_RE.fullmatch(quote)
    if match is not None:
        return _clock_from_match(match, twelve_hour=False, path=path)
    match = _PERIOD_RE.fullmatch(quote)
    if match is not None:
        raw_hour = match.group("hour").casefold()
        hour = int(raw_hour) if raw_hour.isdigit() else _WORD_HOURS[raw_hour]
        if not 1 <= hour <= 12:
            raise GroundedPlanIRError(
                "invalid_clock_time", "period clock hour must be in [1, 12]", path
            )
        period = match.group("period").casefold()
        if period == "night" and (hour == 12 or 1 <= hour <= 6):
            raise GroundedPlanIRError(
                "ambiguous_temporal_reference",
                "night hours 12 and 1 through 6 require explicit AM/PM, noon, or midnight",
                path,
            )
        if period in {"afternoon", "evening", "night"}:
            hour = hour % 12 + 12
        else:
            hour %= 12
        minute = 30 if match.group("word_minute") is not None else 0
        return time(hour, minute, 0)
    raise GroundedPlanIRError(
        "temporal_reference_mismatch", "reference is not a supported clock expression", path
    )


def _parse_clock(quote: str, *, path: str) -> time:
    found: set[time] = set()
    for match in _CLOCK_SEARCH_RE.finditer(quote):
        try:
            found.add(_parse_clock_exact(match.group(0), path=path))
        except GroundedPlanIRError:
            # A syntactically clock-like but invalid token is handled by the exact parser.
            pass
    if len(found) > 1:
        raise GroundedPlanIRError(
            "ambiguous_temporal_reference", "reference contains multiple distinct clocks", path
        )
    return _parse_clock_exact(quote, path=path)


def _compile_datetime(value: object, request: str, now: datetime, *, path: str) -> str:
    plan_datetime = _mapping(value, path=path, label="PlanDateTime")
    _exact_fields(plan_datetime, {"date", "time"}, path)
    time_expr = _mapping(plan_datetime["time"], path=f"{path}.time", label="time expression")
    _exact_fields(time_expr, {"ref"}, f"{path}.time")
    resolved_date = _parse_plan_date(plan_datetime["date"], request, now, path=f"{path}.date")
    clock_quote = resolve_span_ref(request, time_expr["ref"], path=f"{path}.time.ref")
    resolved_time = _parse_clock(clock_quote, path=f"{path}.time.ref")
    compiled = datetime.combine(resolved_date, resolved_time)
    if compiled < now:
        raise GroundedPlanIRError("past_datetime", "compiled datetime is earlier than NOW", path)
    return compiled.isoformat(timespec="seconds")


def _compile_plan(
    raw_plan: str,
    request: str,
    now: str,
    schemas: Mapping[str, ToolSchema] | Iterable[ToolSchema],
    context: Mapping[str, Any],
) -> str:
    try:
        invalid_context = not isinstance(context, Mapping) or bool(context)
    except Exception as exc:
        raise GroundedPlanIRError(
            "unsupported_context", "context could not be inspected safely", "$.context"
        ) from exc
    if invalid_context:
        raise GroundedPlanIRError(
            "unsupported_context",
            "Grounded PlanIR v1 requires context to be exactly {}",
            "$.context",
        )
    if not isinstance(request, str):
        raise GroundedPlanIRError("reference_not_found", "request must be a string", "$.request")
    validated_schemas = _validate_mobile_schemas(schemas)
    parsed_now = _parse_now(now)
    try:
        plan = decode_json_object(raw_plan)
    except ActionIRParseError as exc:
        raise GroundedPlanIRError("invalid_plan_json", str(exc), exc.path) from exc

    if "schema_version" not in plan:
        raise GroundedPlanIRError("missing_field", "missing fields: ['schema_version']", "$")
    version = plan["schema_version"]
    if version != SCHEMA_VERSION:
        raise GroundedPlanIRError(
            "schema_version_mismatch",
            f"schema_version must equal {SCHEMA_VERSION!r}",
            "$.schema_version",
        )
    if "decision" not in plan:
        raise GroundedPlanIRError("missing_field", "missing fields: ['decision']", "$")
    decision = plan["decision"]
    if decision == Decision.ABSTAIN.value:
        _exact_fields(plan, {"decision", "schema_version"}, "$")
        action_obj: Mapping[str, Any] = {"decision": decision}
    elif decision == Decision.CLARIFY.value:
        _exact_fields(plan, {"decision", "missing", "schema_version"}, "$")
        missing = plan["missing"]
        if not isinstance(missing, tuple):
            raise GroundedPlanIRError(
                "final_action_ir_validation_failure", "missing must be an array", "$.missing"
            )
        action_obj = {"decision": decision, "missing": missing}
    elif decision in (Decision.CALL.value, Decision.CONFIRM.value):
        _exact_fields(plan, {"calls", "decision", "mode", "schema_version"}, "$")
        mode = plan["mode"]
        if not isinstance(mode, str) or mode not in {"SINGLE", "SERIAL", "PARALLEL"}:
            raise GroundedPlanIRError("invalid_mode", f"unknown mode {mode!r}", "$.mode")
        calls = plan["calls"]
        if not isinstance(calls, tuple) or not calls:
            raise GroundedPlanIRError(
                "invalid_call_count", "calls must be a non-empty array", "$.calls"
            )
        if mode == "SINGLE" and len(calls) != 1:
            raise GroundedPlanIRError(
                "invalid_call_count", "SINGLE mode requires exactly one call", "$.calls"
            )
        if mode in {"SERIAL", "PARALLEL"} and len(calls) < 2:
            raise GroundedPlanIRError(
                "invalid_call_count", f"{mode} mode requires at least two calls", "$.calls"
            )
        compiled_calls: list[dict[str, Any]] = []
        for index, raw_call in enumerate(calls):
            call_path = f"$.calls[{index}]"
            call = _mapping(raw_call, path=call_path, label="call")
            _exact_fields(call, {"args", "tool"}, call_path)
            tool = call["tool"]
            if not isinstance(tool, str) or tool not in _TOOL_CONTRACTS:
                raise GroundedPlanIRError(
                    "unknown_tool", f"unknown Mobile tool {tool!r}", f"{call_path}.tool"
                )
            args = _mapping(call["args"], path=f"{call_path}.args", label="args")
            contract = _TOOL_CONTRACTS[tool]
            missing_args = contract.required.difference(args)
            if missing_args:
                raise GroundedPlanIRError(
                    "missing_argument",
                    f"missing required arguments: {sorted(missing_args)!r}",
                    f"{call_path}.args",
                )
            extra_args = set(args).difference(contract.names)
            if extra_args:
                raise GroundedPlanIRError(
                    "extra_argument",
                    f"unexpected arguments: {sorted(extra_args)!r}",
                    f"{call_path}.args",
                )
            compiled_args: dict[str, str] = {}
            for name in sorted(args):
                arg_path = f"{call_path}.args.{name}"
                if name == contract.datetime_argument:
                    compiled_args[name] = _compile_datetime(
                        args[name], request, parsed_now, path=arg_path
                    )
                else:
                    compiled_args[name] = _text(
                        args[name], path=arg_path, label="noncalendar argument"
                    )
            compiled_calls.append({"args": compiled_args, "tool": tool})
        action_obj = {"calls": tuple(compiled_calls), "decision": decision, "mode": mode}
    else:
        raise GroundedPlanIRError(
            "final_action_ir_validation_failure", f"unknown decision {decision!r}", "$.decision"
        )

    try:
        return validate_action_ir(action_obj, validated_schemas).canonical_json()
    except (ActionIRError, TypeError, ValueError) as exc:
        path = exc.path if isinstance(exc, ActionIRError) else "$"
        raise GroundedPlanIRError("final_action_ir_validation_failure", str(exc), path) from exc


def compile_mobile_plan_or_raise(
    raw_plan: str,
    request: str,
    now: str,
    schemas: Mapping[str, ToolSchema] | Iterable[ToolSchema] = MOBILE_TOOL_SCHEMAS,
    context: Mapping[str, Any] = MappingProxyType({}),
) -> str:
    """Compile one plan to canonical Action IR, raising only structured plan errors."""

    return _compile_plan(raw_plan, request, now, schemas, context)


def compile_mobile_plan(
    raw_plan: str,
    request: str,
    now: str,
    schemas: Mapping[str, ToolSchema] | Iterable[ToolSchema] = MOBILE_TOOL_SCHEMAS,
    context: Mapping[str, Any] = MappingProxyType({}),
) -> CompileResult:
    """Total public compiler returning one success or one structured failure."""

    try:
        action_ir = compile_mobile_plan_or_raise(raw_plan, request, now, schemas, context)
    except GroundedPlanIRError as exc:
        return CompileResult(
            ok=False,
            error=CompileFailure(code=exc.code, message=exc.message, path=exc.path),
        )
    return CompileResult(ok=True, action_ir=action_ir)


__all__ = [
    "MOBILE_TOOL_SCHEMAS",
    "SCHEMA_VERSION",
    "CompileFailure",
    "CompileResult",
    "GroundedPlanIRError",
    "compile_mobile_plan",
    "compile_mobile_plan_or_raise",
    "resolve_span_ref",
]
