"""Compact, table-grounded PlanIR v2 for the Mobile Actions research lane.

The model emits ordinary Action IR-shaped JSON except that a calendar datetime is
one canonical placeholder such as ``@R:D00:T00``.  References are derived only
from the NFC request and a canonical, timezone-naive ``NOW``.  The compiler is
pure and failure-closed: it does not repair output, read a clock, use a target,
contact a tool, or access a network.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from types import MappingProxyType
from typing import Any, Literal

from .action_ir import (
    ActionIRError,
    ActionIRParseError,
    ToolSchema,
    canonical_json_value,
    decode_json_object,
    validate_action_ir,
)
from .grounded_planir import (
    MOBILE_TOOL_SCHEMAS,
    CompileFailure,
    CompileResult,
    GroundedPlanIRError,
    _validate_mobile_schemas,
)

PROMPT_CONTRACT = "PLAN_IR_V2"
ACTION_PROMPT_CONTRACT = "ACTION_IR_V1"
TABLE_MARKER = "REFS_V2"
TABLE_VERSION = "barun-mobile-reference-table-v2"
PROMPT_RENDERER_VERSION = "barun-mobile-planir-v2-prompt-renderer-v1"
MAX_REFERENCES_PER_KIND = 100

DateOpcode = Literal["A", "M", "R", "W1", "W2"]


class GroundedPlanIRV2Error(ValueError):
    """One stable, path-addressed PlanIR v2 failure."""

    def __init__(self, code: str, message: str, path: str = "$") -> None:
        super().__init__(f"{path}: {message}")
        self.code = code
        self.message = message
        self.path = path


@dataclass(frozen=True, slots=True)
class SpanRef:
    quote: str
    occurrence: int
    start: int


@dataclass(frozen=True, slots=True)
class DateCandidate:
    opcode: DateOpcode
    ref: SpanRef
    resolved: date


@dataclass(frozen=True, slots=True)
class ClockCandidate:
    ref: SpanRef
    resolved: time


@dataclass(frozen=True, slots=True)
class GroundingCandidates:
    dates: tuple[DateCandidate, ...]
    clocks: tuple[ClockCandidate, ...]
    date_distractors: tuple[SpanRef, ...] = ()


@dataclass(frozen=True, slots=True)
class PromptEvidence:
    """Hashes the supplied prompt bytes and the independently rebuilt reference table.

    This record lets the compiler detect mismatched supplied evidence.  It does not
    prove that inference actually showed those bytes to a model; only the bound
    generation/evaluator receipt can establish that provenance claim.
    """

    renderer_version: str
    prompt_contract: str
    prompt_sha256: str
    request_sha256: str
    now: str
    table_sha256: str


@dataclass(frozen=True, slots=True)
class ParsedConstructionPrompt:
    """Canonical components of one model-visible construction prompt."""

    prompt_contract: str
    system_body: str
    request: str
    now: str
    rendered_table: str | None


@dataclass(frozen=True, slots=True)
class ReferenceTable:
    """Input-derived references plus the semantic candidates behind them."""

    date_refs: tuple[SpanRef, ...]
    time_refs: tuple[SpanRef, ...]
    candidates: GroundingCandidates

    def payload(self) -> dict[str, list[list[str | int]]]:
        return {
            "D": [[item.quote, item.occurrence] for item in self.date_refs],
            "T": [[item.quote, item.occurrence] for item in self.time_refs],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.payload(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def canonical_line(self) -> str:
        return f"{TABLE_MARKER} {self.canonical_json()}"

    def render(self) -> str | None:
        """Omit the line if and only if both target-independent tables are empty."""

        return self.canonical_line() if self.date_refs or self.time_refs else None

    def sha256(self) -> str:
        payload = f"{TABLE_VERSION}\0{self.canonical_line()}".encode()
        return hashlib.sha256(payload).hexdigest()


_PROMPT_PREFIX = "<bos><system>\n"
_USER_DELIMITER = "\n<user>\n"
_ASSISTANT_SUFFIX = "\n<assistant>\n"
_PROMPT_NOW_RE = re.compile(r"(?m)^NOW (?P<now>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})$")
_RESERVED_REQUEST_MARKERS = (
    "<bos>",
    "<eos>",
    "<system>",
    "<user>",
    "<assistant>",
    "<pad>",
    "<unk>",
    "<mask>",
    ACTION_PROMPT_CONTRACT,
    PROMPT_CONTRACT,
    TABLE_MARKER,
)


def parse_construction_prompt(rendered_prompt: str) -> ParsedConstructionPrompt:
    """Parse one exact prompt without searching/replacing untrusted request text."""

    if not isinstance(rendered_prompt, str):
        raise GroundedPlanIRV2Error(
            "invalid_prompt", "rendered prompt must be a string", "$.rendered_prompt"
        )
    if not rendered_prompt.startswith(_PROMPT_PREFIX) or not rendered_prompt.endswith(
        _ASSISTANT_SUFFIX
    ):
        raise GroundedPlanIRV2Error(
            "invalid_prompt_boundary",
            "prompt must have the exact system prefix and final assistant boundary",
            "$.rendered_prompt",
        )
    for marker, expected in (
        ("<bos>", 1),
        ("<system>", 1),
        ("<user>", 1),
        ("<assistant>", 1),
    ):
        if rendered_prompt.count(marker) != expected:
            raise GroundedPlanIRV2Error(
                "invalid_prompt_boundary",
                f"prompt must contain exactly {expected} {marker!r} marker",
                "$.rendered_prompt",
            )
    if any(marker in rendered_prompt for marker in ("<eos>", "<pad>", "<unk>", "<mask>")):
        raise GroundedPlanIRV2Error(
            "reserved_prompt_marker",
            "prompt contains a forbidden reserved marker",
            "$.rendered_prompt",
        )
    if rendered_prompt.count(_USER_DELIMITER) != 1 or rendered_prompt.count(_ASSISTANT_SUFFIX) != 1:
        raise GroundedPlanIRV2Error(
            "invalid_prompt_boundary",
            "prompt must contain exactly one user and final assistant delimiter",
            "$.rendered_prompt",
        )

    before_user, request_and_suffix = rendered_prompt.split(_USER_DELIMITER, 1)
    request = request_and_suffix[: -len(_ASSISTANT_SUFFIX)]
    for marker in _RESERVED_REQUEST_MARKERS:
        if marker in request:
            raise GroundedPlanIRV2Error(
                "reserved_request_marker",
                f"request contains reserved marker {marker!r}",
                "$.request",
            )

    system = before_user[len(_PROMPT_PREFIX) :]
    if "\n" not in system:
        raise GroundedPlanIRV2Error(
            "invalid_prompt_contract", "prompt contract header is missing", "$.rendered_prompt"
        )
    contract, system_section = system.split("\n", 1)
    if contract not in {ACTION_PROMPT_CONTRACT, PROMPT_CONTRACT}:
        raise GroundedPlanIRV2Error(
            "invalid_prompt_contract", f"unknown prompt contract {contract!r}", "$.rendered_prompt"
        )
    if rendered_prompt.count(ACTION_PROMPT_CONTRACT) != (
        1 if contract == ACTION_PROMPT_CONTRACT else 0
    ) or rendered_prompt.count(PROMPT_CONTRACT) != (1 if contract == PROMPT_CONTRACT else 0):
        raise GroundedPlanIRV2Error(
            "invalid_prompt_contract",
            "prompt contract marker must occur exactly once in the fixed header",
            "$.rendered_prompt",
        )

    lines = system_section.split("\n")
    table_lines = [line for line in lines if line.startswith(f"{TABLE_MARKER} ")]
    if len(table_lines) > 1 or (table_lines and lines[-1] != table_lines[0]):
        raise GroundedPlanIRV2Error(
            "invalid_reference_table_placement",
            "reference table may occur once only as the final system line",
            "$.rendered_prompt",
        )
    rendered_table = table_lines[0] if table_lines else None
    body_lines = lines[:-1] if rendered_table is not None else lines
    system_body = "\n".join(body_lines)
    if not system_body:
        raise GroundedPlanIRV2Error(
            "invalid_prompt_boundary", "system body must be non-empty", "$.rendered_prompt"
        )
    now_matches = list(_PROMPT_NOW_RE.finditer(system_body))
    if len(now_matches) != 1:
        raise GroundedPlanIRV2Error(
            "invalid_prompt_now", "system body must contain exactly one canonical NOW line", "$.now"
        )
    now = now_matches[0].group("now")
    _parse_now(now)

    rebuilt = render_construction_prompt(
        prompt_contract=contract,
        system_body=system_body,
        request=request,
        rendered_table=rendered_table,
    )
    if rebuilt != rendered_prompt:
        raise GroundedPlanIRV2Error(
            "noncanonical_prompt", "prompt bytes are not canonical", "$.rendered_prompt"
        )
    return ParsedConstructionPrompt(
        prompt_contract=contract,
        system_body=system_body,
        request=request,
        now=now,
        rendered_table=rendered_table,
    )


def render_construction_prompt(
    *,
    prompt_contract: str,
    system_body: str,
    request: str,
    rendered_table: str | None,
) -> str:
    """Render canonical components; callers must parse the result before use."""

    system_section = system_body
    if rendered_table is not None:
        system_section = f"{system_section}\n{rendered_table}"
    return (
        f"{_PROMPT_PREFIX}{prompt_contract}\n{system_section}"
        f"{_USER_DELIMITER}{request}{_ASSISTANT_SUFFIX}"
    )


def make_prompt_evidence(
    rendered_prompt: str,
    prompt_contract: str,
    table: ReferenceTable,
    request: str,
    now: str,
) -> PromptEvidence:
    """Create canonical evidence for exact prompt bytes and one rebuilt table."""

    if not isinstance(rendered_prompt, str):
        raise GroundedPlanIRV2Error(
            "invalid_prompt_evidence", "rendered prompt must be a string", "$.prompt_evidence"
        )
    parsed = parse_construction_prompt(rendered_prompt)
    table_render_is_valid = parsed.rendered_table == table.render() or (
        prompt_contract == ACTION_PROMPT_CONTRACT and parsed.rendered_table is None
    )
    if (
        parsed.prompt_contract != prompt_contract
        or parsed.request != request
        or parsed.now != now
        or not table_render_is_valid
    ):
        raise GroundedPlanIRV2Error(
            "prompt_input_mismatch",
            "prompt contract, request, NOW, or rendered table differs from compiler inputs",
            "$.prompt_evidence",
        )
    try:
        encoded = rendered_prompt.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise GroundedPlanIRV2Error(
            "invalid_prompt_evidence", "rendered prompt must be valid UTF-8", "$.prompt_evidence"
        ) from exc
    return PromptEvidence(
        renderer_version=PROMPT_RENDERER_VERSION,
        prompt_contract=prompt_contract,
        prompt_sha256=hashlib.sha256(encoded).hexdigest(),
        request_sha256=hashlib.sha256(request.encode()).hexdigest(),
        now=now,
        table_sha256=table.sha256(),
    )


def verify_prompt_evidence(
    rendered_prompt: str,
    evidence: PromptEvidence,
    table: ReferenceTable,
    request: str,
    now: str,
) -> None:
    """Verify supplied bytes/metadata; this is not a model-observation attestation."""

    if not isinstance(evidence, PromptEvidence):
        raise GroundedPlanIRV2Error(
            "invalid_prompt_evidence",
            "prompt evidence must be a PromptEvidence record",
            "$.prompt_evidence",
        )
    if evidence.renderer_version != PROMPT_RENDERER_VERSION:
        raise GroundedPlanIRV2Error(
            "prompt_renderer_mismatch",
            f"renderer version must equal {PROMPT_RENDERER_VERSION!r}",
            "$.prompt_evidence.renderer_version",
        )
    expected = make_prompt_evidence(rendered_prompt, evidence.prompt_contract, table, request, now)
    if evidence.prompt_sha256 != expected.prompt_sha256:
        raise GroundedPlanIRV2Error(
            "prompt_hash_mismatch",
            "supplied prompt bytes differ from the prompt evidence digest",
            "$.prompt_evidence.prompt_sha256",
        )
    if evidence.table_sha256 != expected.table_sha256:
        raise GroundedPlanIRV2Error(
            "prompt_table_hash_mismatch",
            "prompt evidence table digest differs from the input-derived table",
            "$.prompt_evidence.table_sha256",
        )
    if evidence.request_sha256 != expected.request_sha256 or evidence.now != expected.now:
        raise GroundedPlanIRV2Error(
            "prompt_input_mismatch",
            "prompt evidence request or NOW differs from compiler inputs",
            "$.prompt_evidence",
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
_MONTHS = {name.casefold(): number for number, name in enumerate(_ENGLISH_MONTH_NAMES, 1)}
_WEEKDAYS = {name.casefold(): number for number, name in enumerate(_ENGLISH_WEEKDAY_NAMES)}
_MONTH = "(?:" + "|".join(_ENGLISH_MONTH_NAMES) + ")"
_WEEKDAY = "(?:" + "|".join(_ENGLISH_WEEKDAY_NAMES) + ")"
_DAY = r"(?P<day>\d{1,2})(?P<suffix>st|nd|rd|th)?"

_ABSOLUTE_PATTERNS = (
    re.compile(
        r"(?<!\d)(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})(?!\d)",
    ),
    re.compile(
        rf"\b(?P<month>{_MONTH})\s+{_DAY}(?:\s*,)?\s+(?P<year>\d{{4}})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_DAY}(?:\s+of)?\s+(?P<month>{_MONTH})(?:\s*,)?\s+"
        rf"(?P<year>\d{{4}})\b",
        re.IGNORECASE,
    ),
)
_MONTH_DAY_PATTERNS = (
    re.compile(rf"\b(?P<month>{_MONTH})\s+{_DAY}\b", re.IGNORECASE),
    re.compile(rf"\b{_DAY}(?:\s+of)?\s+(?P<month>{_MONTH})\b", re.IGNORECASE),
)
_RELATIVE_PATTERN = re.compile(
    r"\b(?:day\s+after\s+tomorrow|tomorrow|tonight|today|"
    r"this\s+(?:morning|afternoon|evening)|in\s+\d+\s+days?|"
    r"\d+\s+days?\s+from\s+now)\b",
    re.IGNORECASE,
)
_WEEKDAY_PATTERN = re.compile(rf"\b(?P<weekday>{_WEEKDAY})\b", re.IGNORECASE)

_MERIDIEM_PATTERN = re.compile(
    r"(?<![\d:])(?P<hour>1[0-2]|0?[1-9])"
    r"(?::(?P<minute>[0-5]\d))?(?::(?P<second>[0-5]\d))?\s*"
    r"(?P<meridiem>a\.?m\.?|p\.?m\.?)\b",
    re.IGNORECASE,
)
_TWENTY_FOUR_HOUR_PATTERN = re.compile(
    r"(?<![\d:])(?P<hour>[01]\d|2[0-3]):(?P<minute>[0-5]\d)"
    r"(?::(?P<second>[0-5]\d))?(?![\d:])"
)
_PERIOD_PATTERN = re.compile(
    r"\b(?P<hour>1[0-2]|0?[1-9]|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve)(?:\s+(?P<minute>fifteen|thirty|forty-five))?"
    r"\s+(?:in\s+the\s+)?(?P<period>morning|afternoon|evening|night)\b",
    re.IGNORECASE,
)
_NOON_MIDNIGHT_PATTERN = re.compile(r"\b(?P<special>noon|midnight)\b", re.IGNORECASE)
_WORD_HOURS = {
    word: number
    for number, word in enumerate(
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
_WORD_MINUTES = {None: 0, "fifteen": 15, "thirty": 30, "forty-five": 45}
_PLACEHOLDER_RE = re.compile(r"\A@(?P<opcode>A|M|R|W1|W2):D(?P<date_id>\d+):T(?P<time_id>\d+)\Z")


def _parse_now(raw: object) -> datetime:
    if not isinstance(raw, str):
        raise GroundedPlanIRV2Error(
            "invalid_now", "NOW must be canonical YYYY-MM-DDTHH:MM:SS", "$.now"
        )
    try:
        parsed = datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S")  # noqa: DTZ007
    except ValueError as exc:
        raise GroundedPlanIRV2Error(
            "invalid_now", "NOW must be canonical YYYY-MM-DDTHH:MM:SS", "$.now"
        ) from exc
    if parsed.isoformat(timespec="seconds") != raw:
        raise GroundedPlanIRV2Error(
            "invalid_now", "NOW must be canonical YYYY-MM-DDTHH:MM:SS", "$.now"
        )
    return parsed


def _canonical_request(raw: object) -> str:
    if not isinstance(raw, str):
        raise GroundedPlanIRV2Error("invalid_request", "request must be a string", "$.request")
    try:
        raw.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise GroundedPlanIRV2Error(
            "invalid_request", "request must be valid UTF-8", "$.request"
        ) from exc
    if unicodedata.normalize("NFC", raw) != raw:
        raise GroundedPlanIRV2Error(
            "noncanonical_request", "request must be NFC-normalized", "$.request"
        )
    return raw


def _ordinal_is_valid(day: int, suffix: str | None) -> bool:
    if suffix is None:
        return True
    expected = "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return suffix.casefold() == expected


def _exact_nonoverlapping_positions(request: str, quote: str) -> tuple[int, ...]:
    positions: list[int] = []
    cursor = 0
    while True:
        found = request.find(quote, cursor)
        if found < 0:
            break
        positions.append(found)
        cursor = found + len(quote)
    return tuple(positions)


class _SpanOccurrenceIndex:
    """Cache one exact non-overlapping scan per distinct candidate quote."""

    def __init__(self, request: str) -> None:
        self.request = request
        self._occurrences: dict[str, dict[int, int]] = {}

    def ref(self, match: re.Match[str]) -> SpanRef:
        quote = match.group(0)
        positions = self._occurrences.get(quote)
        if positions is None:
            positions = {
                start: occurrence
                for occurrence, start in enumerate(
                    _exact_nonoverlapping_positions(self.request, quote)
                )
            }
            self._occurrences[quote] = positions
        occurrence = positions.get(match.start())
        if occurrence is None:
            raise GroundedPlanIRV2Error(
                "reference_not_found", "candidate is not an exact non-overlapping span"
            )
        return SpanRef(quote=quote, occurrence=occurrence, start=match.start())


def _record_reference(
    seen: set[tuple[int, str, int]],
    ref: SpanRef,
    *,
    kind: str,
) -> None:
    seen.add((ref.start, ref.quote, ref.occurrence))
    if len(seen) > MAX_REFERENCES_PER_KIND:
        raise GroundedPlanIRV2Error(
            "too_many_references",
            f"{kind} table must contain at most {MAX_REFERENCES_PER_KIND} entries",
            "$.reference_table",
        )


def _date_from_match(match: re.Match[str]) -> date | None:
    day = int(match.group("day"))
    if not _ordinal_is_valid(day, match.groupdict().get("suffix")):
        return None
    raw_month = match.group("month")
    month = int(raw_month) if raw_month.isdigit() else _MONTHS[raw_month.casefold()]
    try:
        return date(int(match.group("year")), month, day)
    except ValueError:
        return None


def _forward_month_day(now: datetime, month: int, day: int) -> date | None:
    # Eight years covers the Gregorian leap-year gap while remaining bounded at 9999.
    for year in range(now.year, min(now.year + 8, 9999) + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate >= now.date():
            return candidate
    return None


def _relative_date(quote: str, now: datetime) -> date | None:
    folded = quote.casefold()
    fixed = {
        "today": 0,
        "tonight": 0,
        "this morning": 0,
        "this afternoon": 0,
        "this evening": 0,
        "tomorrow": 1,
        "day after tomorrow": 2,
    }
    if folded in fixed:
        offset = fixed[folded]
    else:
        match = re.fullmatch(
            r"(?:in\s+(?P<in_days>\d+)\s+days?|"
            r"(?P<from_days>\d+)\s+days?\s+from\s+now)",
            quote,
            re.IGNORECASE,
        )
        if match is None:
            return None
        raw_offset = match.group("in_days") or match.group("from_days")
        assert raw_offset is not None
        if len(raw_offset) > 3 or int(raw_offset) > 366:
            return None
        offset = int(raw_offset)
    try:
        return now.date() + timedelta(days=offset)
    except OverflowError:
        return None


def _has_explicit_past_weekday_modifier(request: str, start: int, end: int) -> bool:
    return bool(
        re.search(r"\b(?:last|previous)\s*$", request[:start], re.IGNORECASE)
        or re.match(r"\s+ago\b", request[end:], re.IGNORECASE)
    )


def _enumerate_dates(
    request: str, now: datetime, spans: _SpanOccurrenceIndex
) -> tuple[tuple[DateCandidate, ...], tuple[SpanRef, ...]]:
    candidates: list[DateCandidate] = []
    distractors: list[SpanRef] = []
    seen_refs: set[tuple[int, str, int]] = set()
    for pattern in _ABSOLUTE_PATTERNS:
        for match in pattern.finditer(request):
            resolved = _date_from_match(match)
            if resolved is not None:
                ref = spans.ref(match)
                _record_reference(seen_refs, ref, kind="D")
                candidates.append(DateCandidate("A", ref, resolved))
    for pattern in _MONTH_DAY_PATTERNS:
        for match in pattern.finditer(request):
            day = int(match.group("day"))
            if not _ordinal_is_valid(day, match.groupdict().get("suffix")):
                continue
            resolved = _forward_month_day(now, _MONTHS[match.group("month").casefold()], day)
            if resolved is not None:
                ref = spans.ref(match)
                _record_reference(seen_refs, ref, kind="D")
                candidates.append(DateCandidate("M", ref, resolved))
    for match in _RELATIVE_PATTERN.finditer(request):
        resolved = _relative_date(match.group(0), now)
        if resolved is not None:
            ref = spans.ref(match)
            _record_reference(seen_refs, ref, kind="D")
            candidates.append(DateCandidate("R", ref, resolved))
    for match in _WEEKDAY_PATTERN.finditer(request):
        if _has_explicit_past_weekday_modifier(request, match.start(), match.end()):
            ref = spans.ref(match)
            _record_reference(seen_refs, ref, kind="D")
            distractors.append(ref)
            continue
        weekday = _WEEKDAYS[match.group("weekday").casefold()]
        first_delta = (weekday - now.weekday()) % 7 or 7
        ref = spans.ref(match)
        _record_reference(seen_refs, ref, kind="D")
        for ordinal in (1, 2):
            try:
                resolved = now.date() + timedelta(days=first_delta + 7 * (ordinal - 1))
            except OverflowError:
                continue
            candidates.append(DateCandidate(f"W{ordinal}", ref, resolved))  # type: ignore[arg-type]
    unique = {
        (item.opcode, item.ref.start, item.ref.quote, item.ref.occurrence, item.resolved): item
        for item in candidates
    }
    priority = {"A": 0, "M": 1, "R": 2, "W1": 3, "W2": 4}
    ordered_candidates = tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                priority[item.opcode],
                len(item.ref.quote),
                item.ref.start,
                item.ref.quote,
                item.ref.occurrence,
            ),
        )
    )
    unique_distractors = {(item.start, item.quote, item.occurrence): item for item in distractors}
    ordered_distractors = tuple(value for _, value in sorted(unique_distractors.items()))
    return ordered_candidates, ordered_distractors


def _clock_from_meridiem(match: re.Match[str]) -> time:
    hour = int(match.group("hour")) % 12
    if match.group("meridiem").replace(".", "").casefold() == "pm":
        hour += 12
    return time(hour, int(match.group("minute") or 0), int(match.group("second") or 0))


def _clock_from_period(match: re.Match[str]) -> time:
    raw_hour = match.group("hour").casefold()
    hour = int(raw_hour) if raw_hour.isdigit() else _WORD_HOURS[raw_hour]
    if match.group("period").casefold() == "night" and (hour == 12 or 1 <= hour <= 6):
        raise GroundedPlanIRV2Error(
            "ambiguous_clock_time",
            "night hours 12 and 1 through 6 require explicit AM/PM, noon, or midnight",
            "$.reference_table",
        )
    hour %= 12
    if match.group("period").casefold() in {"afternoon", "evening", "night"}:
        hour += 12
    return time(hour, _WORD_MINUTES[(match.group("minute") or "").casefold() or None])


def _enumerate_clocks(request: str, spans: _SpanOccurrenceIndex) -> tuple[ClockCandidate, ...]:
    candidates: list[ClockCandidate] = []
    seen_refs: set[tuple[int, str, int]] = set()
    for pattern, resolver in (
        (_MERIDIEM_PATTERN, _clock_from_meridiem),
        (
            _TWENTY_FOUR_HOUR_PATTERN,
            lambda match: time(
                int(match.group("hour")),
                int(match.group("minute")),
                int(match.group("second") or 0),
            ),
        ),
        (_PERIOD_PATTERN, _clock_from_period),
        (
            _NOON_MIDNIGHT_PATTERN,
            lambda match: time(12 if match.group("special").casefold() == "noon" else 0),
        ),
    ):
        for match in pattern.finditer(request):
            ref = spans.ref(match)
            _record_reference(seen_refs, ref, kind="T")
            candidates.append(ClockCandidate(ref, resolver(match)))
    unique = {
        (item.ref.start, item.ref.quote, item.ref.occurrence, item.resolved): item
        for item in candidates
    }
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                len(item.ref.quote),
                item.ref.start,
                item.ref.quote,
                item.ref.occurrence,
            ),
        )
    )


def enumerate_grounding_candidates(request: str, now: str) -> GroundingCandidates:
    """Enumerate executable candidates and lexical date distractors from input only."""

    canonical_request = _canonical_request(request)
    parsed_now = _parse_now(now)
    spans = _SpanOccurrenceIndex(canonical_request)
    dates, date_distractors = _enumerate_dates(canonical_request, parsed_now, spans)
    return GroundingCandidates(
        dates=dates,
        clocks=_enumerate_clocks(canonical_request, spans),
        date_distractors=date_distractors,
    )


def build_reference_table(request: str, now: str) -> ReferenceTable:
    """Build the sole canonical D/T table from input-only state."""

    candidates = enumerate_grounding_candidates(request, now)
    date_by_span = {
        (item.ref.start, item.ref.quote, item.ref.occurrence): item.ref for item in candidates.dates
    }
    date_by_span.update(
        {(item.start, item.quote, item.occurrence): item for item in candidates.date_distractors}
    )
    time_by_span = {
        (item.ref.start, item.ref.quote, item.ref.occurrence): item.ref
        for item in candidates.clocks
    }
    date_refs = tuple(value for _, value in sorted(date_by_span.items()))
    time_refs = tuple(value for _, value in sorted(time_by_span.items()))
    if len(date_refs) > MAX_REFERENCES_PER_KIND or len(time_refs) > MAX_REFERENCES_PER_KIND:
        raise GroundedPlanIRV2Error(
            "too_many_references",
            "D and T tables must each contain at most 100 entries",
            "$.reference_table",
        )
    return ReferenceTable(date_refs=date_refs, time_refs=time_refs, candidates=candidates)


def verify_reference_table(
    request: str,
    now: str,
    rendered_table: str | None,
    table_sha256: str,
) -> ReferenceTable:
    """Regenerate and byte-verify both the rendered prompt table and its digest."""

    table = build_reference_table(request, now)
    if rendered_table != table.render():
        raise GroundedPlanIRV2Error(
            "reference_table_mismatch",
            "supplied rendered table differs from the input-derived table",
            "$.reference_table.render",
        )
    if table_sha256 != table.sha256():
        raise GroundedPlanIRV2Error(
            "reference_table_hash_mismatch",
            "supplied table digest differs from the input-derived digest",
            "$.reference_table.sha256",
        )
    return table


def _reference_at(items: tuple[SpanRef, ...], raw_index: str, prefix: str, path: str) -> SpanRef:
    if len(raw_index) != 2 or not raw_index.isascii():
        raise GroundedPlanIRV2Error(
            "noncanonical_reference_id", f"{prefix} ID must use two ASCII decimal digits", path
        )
    index = int(raw_index)
    canonical = f"{index:02d}"
    if raw_index != canonical:
        raise GroundedPlanIRV2Error(
            "noncanonical_reference_id", f"{prefix} ID must use two decimal digits", path
        )
    if index >= len(items):
        raise GroundedPlanIRV2Error(
            "reference_id_out_of_range", f"{prefix}{raw_index} is outside the table", path
        )
    return items[index]


def _compile_placeholder(
    placeholder: object,
    table: ReferenceTable,
    now: datetime,
    *,
    path: str,
) -> str:
    if not isinstance(placeholder, str):
        raise GroundedPlanIRV2Error(
            "invalid_placeholder", "calendar datetime must be a placeholder string", path
        )
    match = _PLACEHOLDER_RE.fullmatch(placeholder)
    if match is None:
        raise GroundedPlanIRV2Error(
            "invalid_placeholder",
            "calendar datetime must match @A|M|R|W1|W2:D<digits>:T<digits> exactly",
            path,
        )
    opcode = match.group("opcode")
    date_ref = _reference_at(table.date_refs, match.group("date_id"), "D", path)
    time_ref = _reference_at(table.time_refs, match.group("time_id"), "T", path)
    dates = {
        item.resolved
        for item in table.candidates.dates
        if item.opcode == opcode and item.ref == date_ref
    }
    clocks = {item.resolved for item in table.candidates.clocks if item.ref == time_ref}
    if len(dates) != 1:
        code = "incompatible_date_reference" if not dates else "ambiguous_date_reference"
        raise GroundedPlanIRV2Error(code, "opcode and D reference must resolve exactly once", path)
    if len(clocks) != 1:
        code = "incompatible_time_reference" if not clocks else "ambiguous_time_reference"
        raise GroundedPlanIRV2Error(code, "T reference must resolve exactly once", path)
    compiled = datetime.combine(next(iter(dates)), next(iter(clocks)))
    if compiled < now:
        raise GroundedPlanIRV2Error("past_datetime", "compiled datetime is earlier than NOW", path)
    return compiled.isoformat(timespec="seconds")


def _mutable(value: object) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_mutable(child) for child in value]
    return value


def _compile_plan(
    raw_plan: str,
    request: str,
    now: str,
    rendered_table: str | None,
    table_sha256: str,
    prompt_contract: str,
    rendered_prompt: str,
    prompt_evidence: PromptEvidence,
    schemas: Mapping[str, ToolSchema] | Iterable[ToolSchema],
    context: Mapping[str, Any],
) -> str:
    if prompt_contract != PROMPT_CONTRACT:
        raise GroundedPlanIRV2Error(
            "prompt_contract_mismatch",
            f"prompt contract must equal {PROMPT_CONTRACT!r}",
            "$.prompt_contract",
        )
    try:
        invalid_context = not isinstance(context, Mapping) or bool(context)
    except Exception as exc:
        raise GroundedPlanIRV2Error(
            "unsupported_context", "context could not be inspected safely", "$.context"
        ) from exc
    if invalid_context:
        raise GroundedPlanIRV2Error(
            "unsupported_context", "PlanIR v2 requires context to be exactly {}", "$.context"
        )
    table = verify_reference_table(request, now, rendered_table, table_sha256)
    verify_prompt_evidence(rendered_prompt, prompt_evidence, table, request, now)
    if prompt_evidence.prompt_contract != prompt_contract:
        raise GroundedPlanIRV2Error(
            "prompt_contract_mismatch",
            "prompt evidence contract differs from the compiler contract",
            "$.prompt_evidence.prompt_contract",
        )
    parsed_now = _parse_now(now)
    try:
        decoded = decode_json_object(raw_plan)
    except ActionIRParseError as exc:
        raise GroundedPlanIRV2Error("invalid_plan_json", str(exc), exc.path) from exc
    mutable = _mutable(decoded)
    calls = mutable.get("calls") if isinstance(mutable, dict) else None
    if isinstance(calls, list):
        for index, call in enumerate(calls):
            if not isinstance(call, dict) or call.get("tool") != "create_calendar_event":
                continue
            args = call.get("args")
            if not isinstance(args, dict) or "datetime" not in args:
                continue
            args["datetime"] = _compile_placeholder(
                args["datetime"], table, parsed_now, path=f"$.calls[{index}].args.datetime"
            )
    frozen_schemas = _validate_mobile_schemas(schemas)
    try:
        canonical = decode_json_object(canonical_json_value(mutable))
        return validate_action_ir(canonical, frozen_schemas).canonical_json()
    except (ActionIRError, TypeError, ValueError) as exc:
        path = exc.path if isinstance(exc, ActionIRError) else "$"
        raise GroundedPlanIRV2Error("final_action_ir_validation_failure", str(exc), path) from exc


def compile_mobile_plan_or_raise(
    raw_plan: str,
    request: str,
    now: str,
    *,
    prompt_contract: str,
    rendered_prompt: str,
    prompt_evidence: PromptEvidence,
    rendered_table: str | None,
    table_sha256: str,
    schemas: Mapping[str, ToolSchema] | Iterable[ToolSchema] = MOBILE_TOOL_SCHEMAS,
    context: Mapping[str, Any] = MappingProxyType({}),
) -> str:
    """Compile one v2 plan, requiring exact evidence for the shown table."""

    return _compile_plan(
        raw_plan,
        request,
        now,
        rendered_table,
        table_sha256,
        prompt_contract,
        rendered_prompt,
        prompt_evidence,
        schemas,
        context,
    )


def compile_mobile_plan(
    raw_plan: str,
    request: str,
    now: str,
    *,
    prompt_contract: str,
    rendered_prompt: str,
    prompt_evidence: PromptEvidence,
    rendered_table: str | None,
    table_sha256: str,
    schemas: Mapping[str, ToolSchema] | Iterable[ToolSchema] = MOBILE_TOOL_SCHEMAS,
    context: Mapping[str, Any] = MappingProxyType({}),
) -> CompileResult:
    """Return one success or failure after verifying supplied prompt/table evidence.

    Verification detects internally inconsistent bytes and hashes.  It is not proof
    that a model observed the supplied prompt; bind that claim in the immutable
    generation/evaluator receipt.
    """

    try:
        action_ir = compile_mobile_plan_or_raise(
            raw_plan,
            request,
            now,
            prompt_contract=prompt_contract,
            rendered_prompt=rendered_prompt,
            prompt_evidence=prompt_evidence,
            rendered_table=rendered_table,
            table_sha256=table_sha256,
            schemas=schemas,
            context=context,
        )
    except (GroundedPlanIRV2Error, GroundedPlanIRError) as exc:
        return CompileResult(
            ok=False,
            error=CompileFailure(code=exc.code, message=exc.message, path=exc.path),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        return CompileResult(
            ok=False,
            error=CompileFailure(code="internal_compiler_failure", message=str(exc), path="$"),
        )
    return CompileResult(ok=True, action_ir=action_ir)


__all__ = [
    "ACTION_PROMPT_CONTRACT",
    "MAX_REFERENCES_PER_KIND",
    "PROMPT_CONTRACT",
    "PROMPT_RENDERER_VERSION",
    "TABLE_MARKER",
    "TABLE_VERSION",
    "ClockCandidate",
    "DateCandidate",
    "GroundedPlanIRV2Error",
    "GroundingCandidates",
    "ParsedConstructionPrompt",
    "PromptEvidence",
    "ReferenceTable",
    "SpanRef",
    "build_reference_table",
    "compile_mobile_plan",
    "compile_mobile_plan_or_raise",
    "enumerate_grounding_candidates",
    "make_prompt_evidence",
    "parse_construction_prompt",
    "render_construction_prompt",
    "verify_prompt_evidence",
    "verify_reference_table",
]
