"""Outcome-blind statistical planning for proposed BarunAction GVS-v1.

The functions in this module turn externally authored assumptions into reproducible
one-sided binomial and paired McNemar power calculations.  They never accept scored
rows, labels, predictions, candidate traces, or aggregate outcomes.  A report is
structural planning evidence only: it cannot authorize model, label, CUDA, or Jarvis
access and it is not a substitute for the GVS one-shot custody receipt.

Independent connected components from the population firewall are the only supported
analysis unit.  All probabilities and alpha allocations enter as exact rational JSON.
One-sample cutoffs and tails are computed with integer arithmetic.  Paired power uses
the exact conditional McNemar rejection boundary and an exact common-denominator
multinomial calculation for the preregistered discordance/gain assumptions.  Bounded
component and hypothesis rosters are part of the hashed input contract.
"""

from __future__ import annotations

import hashlib
import json
import marshal
import math
import os
import platform
import re
import sys
import types
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

# Capture every external callable that can change validation, exact arithmetic,
# serialization, hashing, source attestation, or runtime attestation.  Calculations
# below use these import-time references instead of looking functions up again on a
# mutable module object.  Their identities and exact behavior probes are included in
# every runtime receipt.
_DECIMAL_TYPE = Decimal
_FRACTION_TYPE = Fraction
_HASH_SHA256 = hashlib.sha256
_JSON_DECODE_ERROR = json.JSONDecodeError
_JSON_DUMPS = json.dumps
_JSON_LOADS = json.loads
_LOCAL_DECIMAL_CONTEXT = localcontext
_MATH_COMB = math.comb
_MATH_GCD = math.gcd
_MATH_ISFINITE = math.isfinite
_MATH_LCM = math.lcm
_OS_FSTAT = os.fstat
_PATH_TYPE = Path
_PLATFORM_IMPLEMENTATION = platform.python_implementation
_PLATFORM_MACHINE = platform.machine
_PLATFORM_RELEASE = platform.release
_PLATFORM_SYSTEM = platform.system
_PLATFORM_VERSION = platform.python_version
_UNICODE_CATEGORY = unicodedata.category
_UNICODE_NORMALIZE = unicodedata.normalize
_CODE_TYPE = types.CodeType
_MARSHAL_VERSION = marshal.version

GVS_POWER_ASSUMPTIONS_SCHEMA_VERSION = "barun-gvs-power-assumptions-v1"
GVS_POWER_REPORT_SCHEMA_VERSION = "barun-gvs-power-report-v1"
GVS_POWER_RUNTIME_SELF_TEST_SCHEMA_VERSION = "barun-gvs-power-runtime-self-test-v1"

ONE_SAMPLE_KIND = "one_sample_binomial_superiority"
PAIRED_KIND = "paired_exact_mcnemar_superiority"
SUPPORTED_ENDPOINT_KINDS = frozenset({ONE_SAMPLE_KIND, PAIRED_KIND})

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_IDENTIFIER_LENGTH = 256
_MAX_ENDPOINTS = 64
_MAX_COMPONENT_ROSTERS = 64
_MAX_COMPONENTS = 10_000
_MAX_PAIRED_COMPONENTS = 4_000
_MAX_EXTERNAL_RATIO_INTEGER_BITS = 64
_MAX_INTERNAL_RATIO_INTEGER_BITS = 128
_MAX_TOTAL_EXACT_WORK_UNITS = 20_000_000
_MAX_TOTAL_SERIALIZED_INTEGER_BITS = 500_000
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_SOURCE_BYTES = 2 * 1024 * 1024

_COMPONENT_UNIT = "gvs_firewall_connected_component"
_ALTERNATIVE_DIRECTION = "candidate_greater"
_SUPPORTED_POPULATION_ROLES = frozenset({"D-support", "S-new", "C-new"})
_MAX_FAMILYWISE_ALPHA = _FRACTION_TYPE(1, 20)
_MIN_TARGET_POWER = _FRACTION_TYPE(4, 5)


class GVSPowerError(ValueError):
    """A GVS power assumption or calculation is unsafe or malformed."""


def _exact_fraction(numerator: int, denominator: int) -> Fraction:
    """Construct a reduced Fraction without a mutable transitive ``math.gcd`` lookup."""

    if type(numerator) is not int or type(denominator) is not int or denominator == 0:
        raise GVSPowerError("internal exact ratio requires integer terms and nonzero denominator")
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    common = _MATH_GCD(abs(numerator), denominator)
    return _FRACTION_TYPE(
        numerator // common,
        denominator // common,
        _normalize=False,
    )


def _fraction_add(left: Fraction, right: Fraction) -> Fraction:
    return _exact_fraction(
        left.numerator * right.denominator + right.numerator * left.denominator,
        left.denominator * right.denominator,
    )


def _fraction_subtract(left: Fraction, right: Fraction) -> Fraction:
    return _exact_fraction(
        left.numerator * right.denominator - right.numerator * left.denominator,
        left.denominator * right.denominator,
    )


def _fraction_divide(left: Fraction, right: Fraction | int) -> Fraction:
    if type(right) is int:
        return _exact_fraction(left.numerator, left.denominator * right)
    if type(right) is not _FRACTION_TYPE:
        raise GVSPowerError("internal exact division requires a Fraction or integer divisor")
    return _exact_fraction(
        left.numerator * right.denominator,
        left.denominator * right.numerator,
    )


def _strict_identifier(value: object, *, label: str) -> str:
    if type(value) is not str or not value or len(value) > _MAX_IDENTIFIER_LENGTH:
        raise GVSPowerError(
            f"{label} must be a nonempty string of at most {_MAX_IDENTIFIER_LENGTH} characters"
        )
    if value != _UNICODE_NORMALIZE("NFC", value):
        raise GVSPowerError(f"{label} must be NFC-normalized")
    if value != value.strip() or any(
        character.isspace() or _UNICODE_CATEGORY(character).startswith("C") for character in value
    ):
        raise GVSPowerError(f"{label} contains forbidden whitespace or control characters")
    return value


def _strict_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise GVSPowerError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_positive_integer(value: object, *, label: str, maximum: int) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise GVSPowerError(f"{label} must be an integer in [1, {maximum}]")
    return value


def _strict_fraction(
    value: object,
    *,
    label: str,
    maximum_integer_bits: int,
) -> Fraction:
    if type(value) is not _FRACTION_TYPE:
        raise GVSPowerError(f"{label} must be an exact fractions.Fraction")
    if (
        value.numerator.bit_length() > maximum_integer_bits
        or value.denominator.bit_length() > maximum_integer_bits
    ):
        raise GVSPowerError(
            f"{label} numerator and denominator must fit in {maximum_integer_bits} bits"
        )
    return value


def _strict_true(value: object, *, label: str) -> None:
    if value is not True:
        raise GVSPowerError(f"{label} must be true")


def _strict_false(value: object, *, label: str) -> None:
    if value is not False:
        raise GVSPowerError(f"{label} must be false")


def _exact_keys(value: object, expected: frozenset[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise GVSPowerError(f"{label} must be an exact JSON object")
    result = value
    actual = set(result)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise GVSPowerError(f"{label} fields changed; missing={missing!r}, extra={extra!r}")
    return result


def _snapshot_json(
    value: object,
    *,
    label: str,
    active_ids: set[int] | None = None,
    depth: int = 0,
) -> object:
    """Take one detached snapshot of exact built-in JSON without normalization."""

    if depth > _MAX_JSON_DEPTH:
        raise GVSPowerError(f"{label} exceeds maximum JSON depth {_MAX_JSON_DEPTH}")
    active = set() if active_ids is None else active_ids
    if value is None or type(value) in {bool, int, float, str}:
        if type(value) is float and not _MATH_ISFINITE(value):
            raise GVSPowerError(f"{label} contains a non-finite number")
        return value
    if type(value) is list:
        identity = id(value)
        if identity in active:
            raise GVSPowerError(f"{label} contains a JSON container cycle")
        active.add(identity)
        try:
            return [
                _snapshot_json(
                    child,
                    label=f"{label}[]",
                    active_ids=active,
                    depth=depth + 1,
                )
                for child in value
            ]
        finally:
            active.remove(identity)
    if type(value) is dict:
        identity = id(value)
        if identity in active:
            raise GVSPowerError(f"{label} contains a JSON container cycle")
        active.add(identity)
        try:
            detached: dict[str, object] = {}
            for key, child in value.items():
                if type(key) is not str:
                    raise GVSPowerError(f"{label} contains a non-string JSON key")
                detached[key] = _snapshot_json(
                    child,
                    label=f"{label}.{key}",
                    active_ids=active,
                    depth=depth + 1,
                )
            return detached
        finally:
            active.remove(identity)
    raise GVSPowerError(f"{label} must contain only exact built-in JSON values")


def _canonical_json(value: object) -> bytes:
    try:
        encoded = _JSON_DUMPS(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise GVSPowerError("power artifact is not strict JSON") from error
    if len(encoded) > _MAX_JSON_BYTES:
        raise GVSPowerError(f"power artifact exceeds {_MAX_JSON_BYTES} bytes")
    return encoded


def _sha256_json(value: object) -> str:
    return _HASH_SHA256(_canonical_json(value)).hexdigest()


def _ratio(value: object, *, label: str, allow_zero: bool = False) -> Fraction:
    fields = _exact_keys(value, frozenset({"denominator", "numerator"}), label=label)
    numerator = fields["numerator"]
    denominator = fields["denominator"]
    if type(numerator) is not int or type(denominator) is not int:
        raise GVSPowerError(f"{label} numerator and denominator must be exact integers")
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise GVSPowerError(f"{label} must lie in [0, 1]")
    if not allow_zero and numerator == 0:
        raise GVSPowerError(f"{label} must be strictly positive")
    if _MATH_GCD(numerator, denominator) != 1:
        raise GVSPowerError(f"{label} must be in lowest terms")
    if (
        numerator.bit_length() > _MAX_EXTERNAL_RATIO_INTEGER_BITS
        or denominator.bit_length() > _MAX_EXTERNAL_RATIO_INTEGER_BITS
    ):
        raise GVSPowerError(
            f"{label} numerator and denominator must fit in {_MAX_EXTERNAL_RATIO_INTEGER_BITS} bits"
        )
    return _exact_fraction(numerator, denominator)


def _ratio_record(value: Fraction) -> dict[str, int]:
    return {"denominator": value.denominator, "numerator": value.numerator}


def _bounded_nonnegative_integer_text(value: int) -> str:
    if type(value) is not int or value < 0:
        raise GVSPowerError("exact probability integer must be non-negative")
    if value == 0:
        return "0"
    chunks: list[int] = []
    while value:
        value, remainder = divmod(value, 1_000_000_000)
        chunks.append(remainder)
    return str(chunks[-1]) + "".join(f"{chunk:09d}" for chunk in reversed(chunks[:-1]))


def _probability_record(numerator: int, denominator: int) -> dict[str, object]:
    if denominator <= 0 or not 0 <= numerator <= denominator:
        raise GVSPowerError("internal probability is outside [0, 1]")
    common = _MATH_GCD(numerator, denominator)
    numerator //= common
    denominator //= common
    numerator_text = _bounded_nonnegative_integer_text(numerator)
    denominator_text = _bounded_nonnegative_integer_text(denominator)
    with _LOCAL_DECIMAL_CONTEXT() as context:
        context.prec = 50
        decimal = _DECIMAL_TYPE(numerator) / _DECIMAL_TYPE(denominator)
    return {
        "decimal_40": format(decimal, ".40f"),
        "denominator": denominator_text,
        "denominator_sha256": _HASH_SHA256(denominator_text.encode("ascii")).hexdigest(),
        "numerator": numerator_text,
        "numerator_sha256": _HASH_SHA256(numerator_text.encode("ascii")).hexdigest(),
    }


def _binomial_upper_tail_integer(
    trials: int,
    probability: Fraction,
    minimum_successes: int,
) -> tuple[int, int]:
    """Return an exact integer numerator/denominator for P[X >= minimum]."""

    if not 0 < probability < 1:
        raise GVSPowerError("binomial probability must lie strictly between zero and one")
    if not 0 <= minimum_successes <= trials + 1:
        raise GVSPowerError("binomial minimum successes is outside [0, n + 1]")
    denominator = probability.denominator**trials
    numerator = _weighted_binomial_upper_tail_integer(
        trials,
        probability.numerator,
        probability.denominator - probability.numerator,
        minimum_successes,
    )
    return numerator, denominator


def _weighted_binomial_upper_tail_integer(
    trials: int,
    success_weight: int,
    failure_weight: int,
    minimum_successes: int,
) -> int:
    """Return ``sum(comb(n,k)*s**k*f**(n-k), k >= minimum)`` exactly."""

    if type(trials) is not int or trials < 0:
        raise GVSPowerError("weighted binomial trials must be a non-negative integer")
    if type(success_weight) is not int or type(failure_weight) is not int:
        raise GVSPowerError("weighted binomial weights must be exact integers")
    if success_weight <= 0 or failure_weight <= 0:
        raise GVSPowerError("weighted binomial weights must be strictly positive")
    if not 0 <= minimum_successes <= trials + 1:
        raise GVSPowerError("weighted binomial minimum successes is outside [0, n + 1]")
    if minimum_successes == 0:
        return (success_weight + failure_weight) ** trials
    if minimum_successes == trials + 1:
        return 0

    index = minimum_successes
    term = _MATH_COMB(trials, index) * success_weight**index * failure_weight ** (trials - index)
    total = term
    while index < trials:
        dividend = term * (trials - index) * success_weight
        divisor = (index + 1) * failure_weight
        term, remainder = divmod(dividend, divisor)
        if remainder:
            raise AssertionError("binomial integer recurrence lost exact divisibility")
        total += term
        index += 1
    return total


def _tail_at_most(
    trials: int,
    probability: Fraction,
    minimum_successes: int,
    threshold: Fraction,
) -> bool:
    numerator, denominator = _binomial_upper_tail_integer(trials, probability, minimum_successes)
    return numerator * threshold.denominator <= threshold.numerator * denominator


def exact_critical_successes(
    *,
    trials: int,
    null_probability: Fraction,
    one_sided_alpha: Fraction,
) -> int:
    """Smallest success count whose exact null upper tail is at most alpha.

    ``trials + 1`` is returned when no observable success count can attain alpha.
    """

    _strict_positive_integer(trials, label="trials", maximum=_MAX_COMPONENTS)
    null_probability = _strict_fraction(
        null_probability,
        label="null_probability",
        maximum_integer_bits=_MAX_INTERNAL_RATIO_INTEGER_BITS,
    )
    one_sided_alpha = _strict_fraction(
        one_sided_alpha,
        label="one_sided_alpha",
        maximum_integer_bits=_MAX_INTERNAL_RATIO_INTEGER_BITS,
    )
    if not 0 < null_probability < 1:
        raise GVSPowerError("null_probability must lie strictly between zero and one")
    if not 0 < one_sided_alpha < 1:
        raise GVSPowerError("one_sided_alpha must lie strictly between zero and one")
    lower = 0
    upper = trials + 1
    while lower < upper:
        midpoint = (lower + upper) // 2
        if _tail_at_most(trials, null_probability, midpoint, one_sided_alpha):
            upper = midpoint
        else:
            lower = midpoint + 1
    return lower


def _one_sample_result(
    *,
    component_count: int,
    null_rate: Fraction,
    alternative_rate: Fraction,
    endpoint_alpha: Fraction,
    target_power: Fraction,
) -> dict[str, object]:
    if not 0 < null_rate < alternative_rate < 1:
        raise GVSPowerError("one-sample rates must satisfy 0 < null < alternative < 1")
    cutoff = exact_critical_successes(
        trials=component_count,
        null_probability=null_rate,
        one_sided_alpha=endpoint_alpha,
    )
    null_numerator, null_denominator = _binomial_upper_tail_integer(
        component_count, null_rate, cutoff
    )
    power_numerator, power_denominator = _binomial_upper_tail_integer(
        component_count, alternative_rate, cutoff
    )
    attainable = cutoff <= component_count
    meets_target = (
        power_numerator * target_power.denominator >= target_power.numerator * power_denominator
    )
    return {
        "alternative_power": _probability_record(power_numerator, power_denominator),
        "attainable_rejection_boundary": attainable,
        "critical_rate": _ratio_record(_exact_fraction(cutoff, component_count)),
        "critical_successes": cutoff,
        "meets_planning_power_target": meets_target,
        "null_tail_at_boundary": _probability_record(null_numerator, null_denominator),
        "test": "exact one-sided binomial upper tail",
    }


def _paired_category_weights(
    assumed_discordance: Fraction,
    assumed_gain: Fraction,
) -> tuple[Fraction, Fraction, Fraction, int, int, int, int]:
    candidate_win = _fraction_divide(_fraction_add(assumed_discordance, assumed_gain), 2)
    reference_win = _fraction_divide(
        _fraction_subtract(assumed_discordance, assumed_gain),
        2,
    )
    tie = _fraction_subtract(_exact_fraction(1, 1), assumed_discordance)
    common_denominator = _MATH_LCM(
        candidate_win.denominator,
        reference_win.denominator,
        tie.denominator,
    )
    candidate_weight = candidate_win.numerator * (common_denominator // candidate_win.denominator)
    reference_weight = reference_win.numerator * (common_denominator // reference_win.denominator)
    tie_weight = tie.numerator * (common_denominator // tie.denominator)
    if (
        candidate_weight <= 0
        or reference_weight <= 0
        or tie_weight <= 0
        or candidate_weight + reference_weight + tie_weight != common_denominator
    ):
        raise AssertionError("paired category weights are internally inconsistent")
    return (
        candidate_win,
        reference_win,
        tie,
        candidate_weight,
        reference_weight,
        tie_weight,
        common_denominator,
    )


def _paired_result(
    *,
    component_count: int,
    assumed_discordance: Fraction,
    assumed_gain: Fraction,
    endpoint_alpha: Fraction,
    target_power: Fraction,
) -> dict[str, object]:
    if not 0 < assumed_discordance < 1:
        raise GVSPowerError("assumed_discordance must lie strictly between zero and one")
    if not 0 < assumed_gain < assumed_discordance:
        raise GVSPowerError("assumed_gain must lie strictly between zero and discordance")

    (
        candidate_win,
        reference_win,
        tie,
        candidate_weight,
        reference_weight,
        tie_weight,
        common_denominator,
    ) = _paired_category_weights(assumed_discordance, assumed_gain)
    conditional_candidate_win = _fraction_divide(candidate_win, assumed_discordance)
    critical_boundaries_sha = _HASH_SHA256()
    power_numerator = 0

    for discordant_count in range(component_count + 1):
        cutoff = exact_critical_successes(
            trials=max(1, discordant_count),
            null_probability=_exact_fraction(1, 2),
            one_sided_alpha=endpoint_alpha,
        )
        if discordant_count == 0:
            cutoff = 1
        critical_boundaries_sha.update(f"{discordant_count}:{cutoff}\n".encode("ascii"))
        discordant_rejection_weight = _weighted_binomial_upper_tail_integer(
            discordant_count,
            candidate_weight,
            reference_weight,
            cutoff,
        )
        power_numerator += (
            _MATH_COMB(component_count, discordant_count)
            * tie_weight ** (component_count - discordant_count)
            * discordant_rejection_weight
        )
    power_denominator = common_denominator**component_count
    if not 0 <= power_numerator <= power_denominator:
        raise AssertionError("exact paired power is outside [0, 1]")
    meets_target = (
        power_numerator * target_power.denominator >= target_power.numerator * power_denominator
    )

    return {
        "alternative_power": _probability_record(power_numerator, power_denominator),
        "arithmetic": "exact integer multinomial common denominator",
        "assumed_candidate_only_rate": _ratio_record(candidate_win),
        "assumed_conditional_candidate_win_rate": _ratio_record(conditional_candidate_win),
        "assumed_reference_only_rate": _ratio_record(reference_win),
        "assumed_tie_rate": _ratio_record(tie),
        "critical_boundaries_sha256": critical_boundaries_sha.hexdigest(),
        "meets_planning_power_target": meets_target,
        "represented_probability_mass": _probability_record(
            power_denominator,
            power_denominator,
        ),
        "test": "exact conditional one-sided McNemar boundaries",
        "unconditional_power_method": "exact multinomial rejection-region sum",
    }


@dataclass(frozen=True, slots=True)
class _ComponentRoster:
    roster_id: str
    population_role: str
    stratum_id: str
    scope: str
    component_count: int
    eligible_component_ids_sha256: str
    population_firewall_report_sha256: str


@dataclass(frozen=True, slots=True)
class _Endpoint:
    endpoint_id: str
    kind: str
    alternative_direction: str
    component_roster_id: str
    metric_id: str
    reference_id: str
    source_id: str
    null_rate: Fraction | None = None
    alternative_rate: Fraction | None = None
    assumed_discordance: Fraction | None = None
    assumed_gain: Fraction | None = None


def _parse_component_roster(value: object, *, index: int) -> _ComponentRoster:
    fields = _exact_keys(
        value,
        frozenset(
            {
                "component_count",
                "eligible_component_ids_sha256",
                "eligible_only",
                "outcomes_absent",
                "population_firewall_report_sha256",
                "population_role",
                "preoutcome_frozen",
                "roster_id",
                "scope",
                "stratum_id",
                "unit",
            }
        ),
        label=f"component_rosters[{index}]",
    )
    roster_id = _strict_identifier(fields["roster_id"], label="roster_id")
    population_role = _strict_identifier(fields["population_role"], label="population_role")
    if population_role not in _SUPPORTED_POPULATION_ROLES:
        raise GVSPowerError(
            f"population_role must be one of {sorted(_SUPPORTED_POPULATION_ROLES)!r}"
        )
    if fields["unit"] != _COMPONENT_UNIT:
        raise GVSPowerError("component roster unit must be the GVS firewall connected component")
    _strict_true(fields["eligible_only"], label="component roster eligible_only")
    _strict_true(fields["outcomes_absent"], label="component roster outcomes_absent")
    _strict_true(fields["preoutcome_frozen"], label="component roster preoutcome_frozen")
    return _ComponentRoster(
        roster_id=roster_id,
        population_role=population_role,
        stratum_id=_strict_identifier(fields["stratum_id"], label="stratum_id"),
        scope=_strict_identifier(fields["scope"], label="scope"),
        component_count=_strict_positive_integer(
            fields["component_count"],
            label="component roster component_count",
            maximum=_MAX_COMPONENTS,
        ),
        eligible_component_ids_sha256=_strict_sha256(
            fields["eligible_component_ids_sha256"],
            label="eligible_component_ids_sha256",
        ),
        population_firewall_report_sha256=_strict_sha256(
            fields["population_firewall_report_sha256"],
            label="population_firewall_report_sha256",
        ),
    )


def _parse_endpoint(value: object, *, index: int) -> _Endpoint:
    if type(value) is not dict:
        raise GVSPowerError(f"endpoints[{index}] must be an exact JSON object")
    kind = value.get("kind")
    if kind == ONE_SAMPLE_KIND:
        fields = _exact_keys(
            value,
            frozenset(
                {
                    "alternative_direction",
                    "alternative_rate",
                    "component_roster_id",
                    "endpoint_id",
                    "kind",
                    "metric_id",
                    "null_rate",
                    "reference_id",
                    "source_id",
                }
            ),
            label=f"endpoints[{index}]",
        )
        if fields["alternative_direction"] != _ALTERNATIVE_DIRECTION:
            raise GVSPowerError("endpoint alternative_direction must be candidate_greater")
        return _Endpoint(
            endpoint_id=_strict_identifier(fields["endpoint_id"], label="endpoint_id"),
            kind=ONE_SAMPLE_KIND,
            alternative_direction=_strict_identifier(
                fields["alternative_direction"], label="alternative_direction"
            ),
            component_roster_id=_strict_identifier(
                fields["component_roster_id"], label="component_roster_id"
            ),
            metric_id=_strict_identifier(fields["metric_id"], label="metric_id"),
            reference_id=_strict_identifier(fields["reference_id"], label="reference_id"),
            source_id=_strict_identifier(fields["source_id"], label="source_id"),
            null_rate=_ratio(fields["null_rate"], label="null_rate"),
            alternative_rate=_ratio(fields["alternative_rate"], label="alternative_rate"),
        )
    if kind == PAIRED_KIND:
        fields = _exact_keys(
            value,
            frozenset(
                {
                    "alternative_direction",
                    "assumed_discordance",
                    "assumed_gain",
                    "component_roster_id",
                    "endpoint_id",
                    "kind",
                    "metric_id",
                    "reference_id",
                    "source_id",
                }
            ),
            label=f"endpoints[{index}]",
        )
        if fields["alternative_direction"] != _ALTERNATIVE_DIRECTION:
            raise GVSPowerError("endpoint alternative_direction must be candidate_greater")
        return _Endpoint(
            endpoint_id=_strict_identifier(fields["endpoint_id"], label="endpoint_id"),
            kind=PAIRED_KIND,
            alternative_direction=_strict_identifier(
                fields["alternative_direction"], label="alternative_direction"
            ),
            component_roster_id=_strict_identifier(
                fields["component_roster_id"], label="component_roster_id"
            ),
            metric_id=_strict_identifier(fields["metric_id"], label="metric_id"),
            reference_id=_strict_identifier(fields["reference_id"], label="reference_id"),
            source_id=_strict_identifier(fields["source_id"], label="source_id"),
            assumed_discordance=_ratio(fields["assumed_discordance"], label="assumed_discordance"),
            assumed_gain=_ratio(fields["assumed_gain"], label="assumed_gain"),
        )
    raise GVSPowerError(
        f"endpoints[{index}].kind must be one of {sorted(SUPPORTED_ENDPOINT_KINDS)!r}"
    )


def _stable_source_sha256() -> str:
    try:
        source_path = _PATH_TYPE(__file__).resolve(strict=True)
        with source_path.open("rb") as handle:
            before = _OS_FSTAT(handle.fileno())
            source_bytes = handle.read(_MAX_SOURCE_BYTES + 1)
            after = _OS_FSTAT(handle.fileno())
    except OSError as error:
        raise GVSPowerError("could not snapshot the power-planner source") from error
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or len(source_bytes) != before.st_size:
        raise GVSPowerError("power-planner source changed while it was being snapshotted")
    if len(source_bytes) > _MAX_SOURCE_BYTES:
        raise GVSPowerError("power-planner source exceeds the bounded source size")
    return _HASH_SHA256(source_bytes).hexdigest()


def _update_constant_digest(digest: Any, constant: object) -> None:
    if isinstance(constant, _CODE_TYPE):
        digest.update(b"code\x00")
        _update_code_digest(digest, constant)
        return
    if constant is None:
        digest.update(b"none\x00")
        return
    if constant is Ellipsis:
        digest.update(b"ellipsis\x00")
        return
    if type(constant) is bool:
        digest.update(b"bool\x00true\x00" if constant else b"bool\x00false\x00")
        return
    if type(constant) is int:
        digest.update(b"int\x00" + str(constant).encode("ascii") + b"\x00")
        return
    if type(constant) is float:
        digest.update(b"float\x00" + constant.hex().encode("ascii") + b"\x00")
        return
    if type(constant) is complex:
        digest.update(
            b"complex\x00"
            + constant.real.hex().encode("ascii")
            + b":"
            + constant.imag.hex().encode("ascii")
            + b"\x00"
        )
        return
    if type(constant) is str:
        encoded = constant.encode("utf-8")
        digest.update(b"str\x00" + len(encoded).to_bytes(8, "big") + encoded)
        return
    if type(constant) is bytes:
        digest.update(b"bytes\x00" + len(constant).to_bytes(8, "big") + constant)
        return
    if type(constant) is tuple:
        digest.update(b"tuple\x00" + len(constant).to_bytes(8, "big"))
        for child in constant:
            _update_constant_digest(digest, child)
        return
    if type(constant) is list:
        digest.update(b"list\x00" + len(constant).to_bytes(8, "big"))
        for child in constant:
            _update_constant_digest(digest, child)
        return
    if type(constant) is dict:
        digest.update(b"dict\x00" + len(constant).to_bytes(8, "big"))
        for key in sorted(constant):
            if type(key) is not str:
                raise GVSPowerError("runtime callable metadata contains a non-string mapping key")
            _update_constant_digest(digest, key)
            _update_constant_digest(digest, constant[key])
        return
    if type(constant) is frozenset:
        child_digests: list[bytes] = []
        for child in constant:
            child_digest = _HASH_SHA256()
            _update_constant_digest(child_digest, child)
            child_digests.append(child_digest.digest())
        digest.update(b"frozenset\x00" + len(child_digests).to_bytes(8, "big"))
        for child_digest in sorted(child_digests):
            digest.update(child_digest)
        return
    raise GVSPowerError(
        f"loaded power-planner bytecode contains unsupported constant {type(constant).__name__}"
    )


def _update_code_digest(digest: Any, code: types.CodeType) -> None:
    for value in (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        code.co_firstlineno,
    ):
        digest.update(f"{value}:".encode("ascii"))
    for value in (
        code.co_name,
        getattr(code, "co_qualname", code.co_name),
        *code.co_names,
        *code.co_varnames,
        *code.co_freevars,
        *code.co_cellvars,
    ):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    for value in (
        code.co_code,
        getattr(code, "co_linetable", code.co_lnotab),
        getattr(code, "co_exceptiontable", b""),
    ):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    for constant in code.co_consts:
        _update_constant_digest(digest, constant)


def _loaded_code_sha256() -> str:
    digest = _HASH_SHA256()
    functions = [
        (name, value)
        for name, value in globals().items()
        if callable(value)
        and getattr(value, "__module__", None) == __name__
        and hasattr(value, "__code__")
    ]
    for name, function in sorted(functions):
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        _update_code_digest(digest, function.__code__)
        _update_constant_digest(digest, getattr(function, "__defaults__", None))
        _update_constant_digest(digest, getattr(function, "__kwdefaults__", None))
        closure = getattr(function, "__closure__", None)
        if closure is None:
            _update_constant_digest(digest, None)
        else:
            _update_constant_digest(
                digest,
                tuple(cell.cell_contents for cell in closure),
            )
        digest.update(b"\x00")
    return digest.hexdigest()


def _callable_dependency_record(dependency_id: str, value: object) -> dict[str, object]:
    """Return stable identity evidence for one captured external callable."""

    if not callable(value):
        raise GVSPowerError(f"captured dependency {dependency_id!r} is no longer callable")
    value_type = type(value)
    record: dict[str, object] = {
        "dependency_id": dependency_id,
        "module": getattr(value, "__module__", None),
        "qualname": getattr(value, "__qualname__", getattr(value, "__name__", None)),
        "type_module": value_type.__module__,
        "type_qualname": value_type.__qualname__,
    }
    code = getattr(value, "__code__", None)
    if code is None:
        record["python_callable_code_sha256"] = None
    elif type(code) is _CODE_TYPE:
        digest = _HASH_SHA256()
        _update_code_digest(digest, code)
        _update_constant_digest(digest, getattr(value, "__defaults__", None))
        _update_constant_digest(digest, getattr(value, "__kwdefaults__", None))
        closure = getattr(value, "__closure__", None)
        _update_constant_digest(
            digest,
            None if closure is None else tuple(cell.cell_contents for cell in closure),
        )
        record["python_callable_code_sha256"] = digest.hexdigest()
    else:
        raise GVSPowerError(
            f"captured dependency {dependency_id!r} exposes a malformed code object"
        )
    record["dependency_identity_sha256"] = _sha256_json(record)
    return record


def _behavior_dependency_snapshot() -> dict[str, object]:
    """Bind the complete import-time dependency roster used by this module."""

    dependencies = (
        ("decimal.Decimal", _DECIMAL_TYPE),
        ("decimal.localcontext", _LOCAL_DECIMAL_CONTEXT),
        ("fractions.Fraction", _FRACTION_TYPE),
        ("hashlib.sha256", _HASH_SHA256),
        ("json.JSONDecodeError", _JSON_DECODE_ERROR),
        ("json.dumps", _JSON_DUMPS),
        ("json.loads", _JSON_LOADS),
        ("math.comb", _MATH_COMB),
        ("math.gcd", _MATH_GCD),
        ("math.isfinite", _MATH_ISFINITE),
        ("math.lcm", _MATH_LCM),
        ("os.fstat", _OS_FSTAT),
        ("pathlib.Path", _PATH_TYPE),
        ("platform.machine", _PLATFORM_MACHINE),
        ("platform.python_implementation", _PLATFORM_IMPLEMENTATION),
        ("platform.python_version", _PLATFORM_VERSION),
        ("platform.release", _PLATFORM_RELEASE),
        ("platform.system", _PLATFORM_SYSTEM),
        ("types.CodeType", _CODE_TYPE),
        ("unicodedata.category", _UNICODE_CATEGORY),
        ("unicodedata.normalize", _UNICODE_NORMALIZE),
    )
    records = [
        _callable_dependency_record(dependency_id, value) for dependency_id, value in dependencies
    ]
    snapshot: dict[str, object] = {
        "dependencies": records,
        "dependency_count": len(records),
    }
    snapshot["dependency_roster_sha256"] = _sha256_json(snapshot)
    return snapshot


def _behavior_constant_snapshot() -> dict[str, object]:
    """Bind every mutable module-level constant that affects accepted inputs or work."""

    contract: dict[str, object] = {
        "alternative_direction": _ALTERNATIVE_DIRECTION,
        "component_unit": _COMPONENT_UNIT,
        "limits": {
            "external_ratio_integer_bits": _MAX_EXTERNAL_RATIO_INTEGER_BITS,
            "identifier_length": _MAX_IDENTIFIER_LENGTH,
            "internal_ratio_integer_bits": _MAX_INTERNAL_RATIO_INTEGER_BITS,
            "json_bytes": _MAX_JSON_BYTES,
            "json_depth": _MAX_JSON_DEPTH,
            "paired_components": _MAX_PAIRED_COMPONENTS,
            "source_bytes": _MAX_SOURCE_BYTES,
            "total_exact_work_units": _MAX_TOTAL_EXACT_WORK_UNITS,
            "total_serialized_integer_bits": _MAX_TOTAL_SERIALIZED_INTEGER_BITS,
            "component_rosters": _MAX_COMPONENT_ROSTERS,
            "components": _MAX_COMPONENTS,
            "endpoints": _MAX_ENDPOINTS,
        },
        "maximum_familywise_alpha": _ratio_record(_MAX_FAMILYWISE_ALPHA),
        "minimum_target_power": _ratio_record(_MIN_TARGET_POWER),
        "schema_versions": {
            "assumptions": GVS_POWER_ASSUMPTIONS_SCHEMA_VERSION,
            "report": GVS_POWER_REPORT_SCHEMA_VERSION,
            "runtime_self_test": GVS_POWER_RUNTIME_SELF_TEST_SCHEMA_VERSION,
        },
        "sha256_pattern": {
            "flags": _SHA256_RE.flags,
            "pattern": _SHA256_RE.pattern,
        },
        "supported_endpoint_kinds": sorted(SUPPORTED_ENDPOINT_KINDS),
        "supported_population_roles": sorted(_SUPPORTED_POPULATION_ROLES),
    }
    contract["behavior_constant_sha256"] = _sha256_json(contract)
    return contract


def _assert_import_time_dependency_bindings() -> None:
    """Reject replacement of a captured callable or its originating module slot."""

    identity_checks = (
        ("decimal.Decimal", Decimal, _DECIMAL_TYPE),
        ("decimal.localcontext", localcontext, _LOCAL_DECIMAL_CONTEXT),
        ("fractions.Fraction", Fraction, _FRACTION_TYPE),
        ("hashlib.sha256", hashlib.sha256, _HASH_SHA256),
        ("json.JSONDecodeError", json.JSONDecodeError, _JSON_DECODE_ERROR),
        ("json.dumps", json.dumps, _JSON_DUMPS),
        ("json.loads", json.loads, _JSON_LOADS),
        ("math.comb", math.comb, _MATH_COMB),
        ("math.gcd", math.gcd, _MATH_GCD),
        ("math.isfinite", math.isfinite, _MATH_ISFINITE),
        ("math.lcm", math.lcm, _MATH_LCM),
        ("os.fstat", os.fstat, _OS_FSTAT),
        ("pathlib.Path", Path, _PATH_TYPE),
        ("platform.machine", platform.machine, _PLATFORM_MACHINE),
        (
            "platform.python_implementation",
            platform.python_implementation,
            _PLATFORM_IMPLEMENTATION,
        ),
        ("platform.python_version", platform.python_version, _PLATFORM_VERSION),
        ("platform.release", platform.release, _PLATFORM_RELEASE),
        ("platform.system", platform.system, _PLATFORM_SYSTEM),
        ("types.CodeType", types.CodeType, _CODE_TYPE),
        ("unicodedata.category", unicodedata.category, _UNICODE_CATEGORY),
        ("unicodedata.normalize", unicodedata.normalize, _UNICODE_NORMALIZE),
    )
    changed = [name for name, current, captured in identity_checks if current is not captured]
    if marshal.version != _MARSHAL_VERSION:
        changed.append("marshal.version")
    if changed:
        raise GVSPowerError(f"import-time dependency binding changed: {', '.join(sorted(changed))}")


def _exact_runtime_self_test() -> dict[str, object]:
    """Run immutable exact vectors spanning every arithmetic/receipt primitive."""

    expected: dict[str, object] = {
        "binomial_10_half_ge_9": {"denominator": 1024, "numerator": 11},
        "canonical_json_sha256": (
            "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
        ),
        "critical_10_half_alpha_1_20": 9,
        "decimal_one_eighth_40": "0.1250000000000000000000000000000000000000",
        "fraction_one_third_plus_one_sixth": {"denominator": 2, "numerator": 1},
        "json_loads_canonical_probe": {"a": 1, "b": 2},
        "math_comb_10_3": 120,
        "math_gcd_84_30": 6,
        "math_isfinite_probe": [True, False],
        "math_lcm_21_6": 42,
        "paired_n10_d1_2_g1_4_alpha1_40": {
            "denominator": 1_073_741_824,
            "numerator": 64_116_279,
        },
        "sha256_probe": "b5126d901d7c2dfe96b34e5e95178cad703c501202b9ec047972e4eafb8642f9",
        "unicode_probe": {"category": "Cf", "normalized": "é"},
        "weighted_binomial_n4_s3_f1_ge3": 189,
    }
    try:
        canonical_probe = _canonical_json({"b": 2, "a": 1})
        fraction_probe = _fraction_add(_exact_fraction(1, 3), _exact_fraction(1, 6))
        binomial_numerator, binomial_denominator = _binomial_upper_tail_integer(
            10,
            _exact_fraction(1, 2),
            9,
        )
        paired_probe = _paired_result(
            component_count=10,
            assumed_discordance=_exact_fraction(1, 2),
            assumed_gain=_exact_fraction(1, 4),
            endpoint_alpha=_exact_fraction(1, 40),
            target_power=_exact_fraction(4, 5),
        )["alternative_power"]
        with _LOCAL_DECIMAL_CONTEXT() as context:
            context.prec = 50
            decimal_probe = _DECIMAL_TYPE(1) / _DECIMAL_TYPE(8)
        actual: dict[str, object] = {
            "binomial_10_half_ge_9": {
                "denominator": binomial_denominator,
                "numerator": binomial_numerator,
            },
            "canonical_json_sha256": _HASH_SHA256(canonical_probe).hexdigest(),
            "critical_10_half_alpha_1_20": exact_critical_successes(
                trials=10,
                null_probability=_exact_fraction(1, 2),
                one_sided_alpha=_exact_fraction(1, 20),
            ),
            "decimal_one_eighth_40": format(decimal_probe, ".40f"),
            "fraction_one_third_plus_one_sixth": {
                "denominator": fraction_probe.denominator,
                "numerator": fraction_probe.numerator,
            },
            "json_loads_canonical_probe": _JSON_LOADS(canonical_probe.decode("utf-8")),
            "math_comb_10_3": _MATH_COMB(10, 3),
            "math_gcd_84_30": _MATH_GCD(84, 30),
            "math_isfinite_probe": [_MATH_ISFINITE(1.0), _MATH_ISFINITE(float("inf"))],
            "math_lcm_21_6": _MATH_LCM(21, 6),
            "paired_n10_d1_2_g1_4_alpha1_40": {
                "denominator": int(paired_probe["denominator"]),
                "numerator": int(paired_probe["numerator"]),
            },
            "sha256_probe": _HASH_SHA256(b"barun-gvs-power-self-test-v1").hexdigest(),
            "unicode_probe": {
                "category": _UNICODE_CATEGORY("\u202e"),
                "normalized": _UNICODE_NORMALIZE("NFC", "e\u0301"),
            },
            "weighted_binomial_n4_s3_f1_ge3": _weighted_binomial_upper_tail_integer(
                4,
                3,
                1,
                3,
            ),
        }
    except Exception as error:
        raise GVSPowerError("exact runtime dependency self-test raised an exception") from error
    if actual != expected:
        raise GVSPowerError("exact runtime dependency self-test failed")
    evidence: dict[str, object] = {
        "schema_version": GVS_POWER_RUNTIME_SELF_TEST_SCHEMA_VERSION,
        "vectors": actual,
    }
    evidence["vectors_sha256"] = _sha256_json(actual)
    return evidence


def _runtime_identity_snapshot() -> dict[str, object]:
    try:
        _assert_import_time_dependency_bindings()
        behavior_dependencies = _behavior_dependency_snapshot()
        behavior_constants = _behavior_constant_snapshot()
        exact_self_test = _exact_runtime_self_test()
        loaded_code_sha256 = _loaded_code_sha256()
        module_source_sha256 = _stable_source_sha256()
        if loaded_code_sha256 != _INITIAL_LOADED_CODE_SHA256:
            raise GVSPowerError("power-planner loaded code changed after import")
        if module_source_sha256 != _INITIAL_SOURCE_SHA256:
            raise GVSPowerError("power-planner source changed after import")
        identity: dict[str, object] = {
            "arithmetic_contract": (
                "exact Python integer/Fraction/Decimal; no binary floating point"
            ),
            "behavior_constants": behavior_constants,
            "behavior_dependencies": behavior_dependencies,
            "exact_runtime_self_test": exact_self_test,
            "loaded_module_code_sha256": loaded_code_sha256,
            "marshal_version": _MARSHAL_VERSION,
            "module": __name__,
            "module_source_sha256": module_source_sha256,
            "platform_machine": _PLATFORM_MACHINE(),
            "platform_release": _PLATFORM_RELEASE(),
            "platform_system": _PLATFORM_SYSTEM(),
            "python_byteorder": sys.byteorder,
            "python_cache_tag": sys.implementation.cache_tag,
            "python_hexversion": sys.hexversion,
            "python_implementation": _PLATFORM_IMPLEMENTATION(),
            "python_version": _PLATFORM_VERSION(),
        }
        identity["runtime_identity_sha256"] = _sha256_json(identity)
        _assert_import_time_dependency_bindings()
    except GVSPowerError:
        raise
    except Exception as error:
        raise GVSPowerError("could not attest the exact power runtime") from error
    return identity


def _endpoint_work_preflight(
    endpoints: Sequence[_Endpoint],
    rosters_by_id: Mapping[str, _ComponentRoster],
) -> dict[str, int]:
    work_units = 0
    serialized_integer_bits = 0
    for endpoint in endpoints:
        roster = rosters_by_id[endpoint.component_roster_id]
        component_count = roster.component_count
        if endpoint.kind == ONE_SAMPLE_KIND:
            assert endpoint.null_rate is not None and endpoint.alternative_rate is not None
            if not 0 < endpoint.null_rate < endpoint.alternative_rate < 1:
                raise GVSPowerError("one-sample rates must satisfy 0 < null < alternative < 1")
            work_units += component_count * (component_count + 1).bit_length()
            maximum_denominator_bits = max(
                endpoint.null_rate.denominator.bit_length(),
                endpoint.alternative_rate.denominator.bit_length(),
            )
            work_units += (
                component_count
                * (component_count + 1).bit_length()
                * max(0, (maximum_denominator_bits - 1) // 16)
            )
            serialized_integer_bits += 4 * component_count * maximum_denominator_bits
        else:
            assert endpoint.assumed_discordance is not None and endpoint.assumed_gain is not None
            if not 0 < endpoint.assumed_discordance < 1:
                raise GVSPowerError("assumed_discordance must lie strictly between zero and one")
            if not 0 < endpoint.assumed_gain < endpoint.assumed_discordance:
                raise GVSPowerError("assumed_gain must lie strictly between zero and discordance")
            if component_count > _MAX_PAIRED_COMPONENTS:
                raise GVSPowerError(
                    f"paired endpoint component_count exceeds {_MAX_PAIRED_COMPONENTS}"
                )
            *_, common_denominator = _paired_category_weights(
                endpoint.assumed_discordance,
                endpoint.assumed_gain,
            )
            denominator_work_factor = max(1, (common_denominator.bit_length() + 15) // 16)
            work_units += component_count * component_count * denominator_work_factor
            serialized_integer_bits += 2 * component_count * common_denominator.bit_length()
    if work_units > _MAX_TOTAL_EXACT_WORK_UNITS:
        raise GVSPowerError(
            "power design exceeds the bounded aggregate exact-calculation work budget"
        )
    if serialized_integer_bits > _MAX_TOTAL_SERIALIZED_INTEGER_BITS:
        raise GVSPowerError(
            "power design exceeds the bounded aggregate exact-integer report budget"
        )
    return {
        "estimated_exact_work_units": work_units,
        "estimated_serialized_integer_bits": serialized_integer_bits,
        "maximum_exact_work_units": _MAX_TOTAL_EXACT_WORK_UNITS,
        "maximum_serialized_integer_bits": _MAX_TOTAL_SERIALIZED_INTEGER_BITS,
    }


def build_power_report(assumptions: Mapping[str, Any]) -> dict[str, object]:
    """Validate an outcome-free assumption manifest and compute a bound report."""

    runtime_identity = _runtime_identity_snapshot()
    snapshot = _snapshot_json(assumptions, label="assumptions")
    assumptions_bytes = _canonical_json(snapshot)
    assumptions_sha256 = _HASH_SHA256(assumptions_bytes).hexdigest()
    fields = _exact_keys(
        snapshot,
        frozenset(
            {
                "assumption_sources",
                "authorizes_model_or_label_access",
                "component_rosters",
                "design_id",
                "endpoints",
                "hypothesis_roster",
                "independence_contract",
                "multiplicity",
                "outcome_access_prohibited",
                "population_protocol_sha256",
                "protocol_sha256",
                "schema_version",
                "target_power",
                "zero_post_freeze_attrition",
            }
        ),
        label="assumptions",
    )
    if fields["schema_version"] != GVS_POWER_ASSUMPTIONS_SCHEMA_VERSION:
        raise GVSPowerError("assumptions schema_version changed")
    design_id = _strict_identifier(fields["design_id"], label="design_id")
    protocol_sha256 = _strict_sha256(fields["protocol_sha256"], label="protocol_sha256")
    population_protocol_sha256 = _strict_sha256(
        fields["population_protocol_sha256"], label="population_protocol_sha256"
    )
    _strict_true(fields["outcome_access_prohibited"], label="outcome_access_prohibited")
    _strict_false(
        fields["authorizes_model_or_label_access"], label="authorizes_model_or_label_access"
    )
    _strict_true(fields["zero_post_freeze_attrition"], label="zero_post_freeze_attrition")

    independence = _exact_keys(
        fields["independence_contract"],
        frozenset(
            {
                "cross_role_component_closure_required",
                "effective_n_equals_eligible_components",
                "unit",
            }
        ),
        label="independence_contract",
    )
    if independence["unit"] != _COMPONENT_UNIT:
        raise GVSPowerError("independence unit must be the GVS firewall connected component")
    _strict_true(
        independence["cross_role_component_closure_required"],
        label="cross_role_component_closure_required",
    )
    _strict_true(
        independence["effective_n_equals_eligible_components"],
        label="effective_n_equals_eligible_components",
    )

    component_roster_values = fields["component_rosters"]
    if (
        type(component_roster_values) is not list
        or not 0 < len(component_roster_values) <= _MAX_COMPONENT_ROSTERS
    ):
        raise GVSPowerError("component_rosters must be a nonempty bounded JSON array")
    component_rosters = [
        _parse_component_roster(value, index=index)
        for index, value in enumerate(component_roster_values)
    ]
    component_roster_ids = [roster.roster_id for roster in component_rosters]
    if component_roster_ids != sorted(set(component_roster_ids)):
        raise GVSPowerError("component_rosters must have unique canonical roster_id order")
    component_membership_hashes = [
        roster.eligible_component_ids_sha256 for roster in component_rosters
    ]
    if len(component_membership_hashes) != len(set(component_membership_hashes)):
        raise GVSPowerError("component rosters may not duplicate an eligible component roster")
    population_strata = [
        (roster.population_role, roster.stratum_id) for roster in component_rosters
    ]
    if len(population_strata) != len(set(population_strata)):
        raise GVSPowerError("component rosters must have unique population-role strata")
    rosters_by_id = {roster.roster_id: roster for roster in component_rosters}

    multiplicity = _exact_keys(
        fields["multiplicity"],
        frozenset({"endpoint_count", "family_id", "familywise_alpha", "method"}),
        label="multiplicity",
    )
    if multiplicity["method"] != "bonferroni":
        raise GVSPowerError("only the preregistrable Bonferroni multiplicity method is supported")
    endpoint_count = _strict_positive_integer(
        multiplicity["endpoint_count"], label="endpoint_count", maximum=_MAX_ENDPOINTS
    )
    multiplicity_family_id = _strict_identifier(
        multiplicity["family_id"], label="multiplicity family_id"
    )
    familywise_alpha = _ratio(multiplicity["familywise_alpha"], label="familywise_alpha")
    if familywise_alpha > _MAX_FAMILYWISE_ALPHA:
        raise GVSPowerError("familywise_alpha must be at most one twentieth")
    target_power = _ratio(fields["target_power"], label="target_power")
    if target_power < _MIN_TARGET_POWER or target_power >= 1:
        raise GVSPowerError("target_power must lie in [four fifths, one)")
    endpoint_alpha = _fraction_divide(familywise_alpha, endpoint_count)

    hypothesis_roster = _exact_keys(
        fields["hypothesis_roster"],
        frozenset(
            {
                "complete_for_protocol",
                "endpoint_ids",
                "family_id",
                "protocol_sha256",
            }
        ),
        label="hypothesis_roster",
    )
    _strict_true(
        hypothesis_roster["complete_for_protocol"],
        label="hypothesis_roster complete_for_protocol",
    )
    hypothesis_family_id = _strict_identifier(
        hypothesis_roster["family_id"], label="hypothesis roster family_id"
    )
    hypothesis_protocol_sha256 = _strict_sha256(
        hypothesis_roster["protocol_sha256"], label="hypothesis roster protocol_sha256"
    )
    if hypothesis_family_id != multiplicity_family_id:
        raise GVSPowerError("hypothesis and multiplicity family_id values differ")
    if hypothesis_protocol_sha256 != protocol_sha256:
        raise GVSPowerError("hypothesis roster protocol_sha256 differs from the protocol")
    roster_endpoint_ids_value = hypothesis_roster["endpoint_ids"]
    if type(roster_endpoint_ids_value) is not list:
        raise GVSPowerError("hypothesis_roster.endpoint_ids must be an exact JSON array")
    roster_endpoint_ids = [
        _strict_identifier(value, label="hypothesis roster endpoint_id")
        for value in roster_endpoint_ids_value
    ]
    if roster_endpoint_ids != sorted(set(roster_endpoint_ids)):
        raise GVSPowerError("hypothesis_roster.endpoint_ids must have unique canonical order")
    if len(roster_endpoint_ids) != endpoint_count:
        raise GVSPowerError(
            "hypothesis roster must contain exactly multiplicity.endpoint_count endpoint IDs"
        )

    source_values = fields["assumption_sources"]
    if type(source_values) is not list or not 0 < len(source_values) <= _MAX_ENDPOINTS:
        raise GVSPowerError("assumption_sources must be a nonempty bounded JSON array")
    source_ids: list[str] = []
    for index, source_value in enumerate(source_values):
        source = _exact_keys(
            source_value,
            frozenset({"precollection", "role", "sha256", "source_id"}),
            label=f"assumption_sources[{index}]",
        )
        source_id = _strict_identifier(source["source_id"], label="source_id")
        if source["role"] != "external_or_historical_precollection_assumption":
            raise GVSPowerError("assumption source role changed")
        _strict_true(source["precollection"], label="assumption source precollection")
        _strict_sha256(source["sha256"], label="assumption source sha256")
        source_ids.append(source_id)
    if source_ids != sorted(set(source_ids)):
        raise GVSPowerError("assumption_sources must have unique canonical source_id order")

    endpoint_values = fields["endpoints"]
    if type(endpoint_values) is not list or len(endpoint_values) != endpoint_count:
        raise GVSPowerError("endpoints must be an array matching multiplicity.endpoint_count")
    endpoints = [_parse_endpoint(value, index=index) for index, value in enumerate(endpoint_values)]
    endpoint_ids = [endpoint.endpoint_id for endpoint in endpoints]
    if endpoint_ids != sorted(set(endpoint_ids)):
        raise GVSPowerError("endpoints must have unique canonical endpoint_id order")
    if endpoint_ids != roster_endpoint_ids:
        raise GVSPowerError("endpoints differ from the complete hypothesis roster")
    used_sources = {endpoint.source_id for endpoint in endpoints}
    if not used_sources.issubset(source_ids):
        raise GVSPowerError("an endpoint references an unknown assumption source")
    if used_sources != set(source_ids):
        raise GVSPowerError("every assumption source must support at least one endpoint")
    used_component_rosters = {endpoint.component_roster_id for endpoint in endpoints}
    if not used_component_rosters.issubset(rosters_by_id):
        raise GVSPowerError("an endpoint references an unknown component roster")
    if used_component_rosters != set(rosters_by_id):
        raise GVSPowerError("every component roster must support at least one endpoint")
    structured_hypotheses = [
        (
            endpoint.component_roster_id,
            endpoint.metric_id,
            endpoint.reference_id,
            endpoint.alternative_direction,
        )
        for endpoint in endpoints
    ]
    if len(set(structured_hypotheses)) != len(structured_hypotheses):
        raise GVSPowerError("structured hypotheses must be unique within the multiplicity family")
    work_preflight = _endpoint_work_preflight(endpoints, rosters_by_id)

    endpoint_reports: list[dict[str, object]] = []
    for endpoint in endpoints:
        component_roster = rosters_by_id[endpoint.component_roster_id]
        if endpoint.kind == ONE_SAMPLE_KIND:
            assert endpoint.null_rate is not None and endpoint.alternative_rate is not None
            calculation = _one_sample_result(
                component_count=component_roster.component_count,
                null_rate=endpoint.null_rate,
                alternative_rate=endpoint.alternative_rate,
                endpoint_alpha=endpoint_alpha,
                target_power=target_power,
            )
            assumptions_record = {
                "alternative_rate": _ratio_record(endpoint.alternative_rate),
                "null_rate": _ratio_record(endpoint.null_rate),
            }
        else:
            assert endpoint.assumed_discordance is not None and endpoint.assumed_gain is not None
            calculation = _paired_result(
                component_count=component_roster.component_count,
                assumed_discordance=endpoint.assumed_discordance,
                assumed_gain=endpoint.assumed_gain,
                endpoint_alpha=endpoint_alpha,
                target_power=target_power,
            )
            assumptions_record = {
                "assumed_discordance": _ratio_record(endpoint.assumed_discordance),
                "assumed_gain": _ratio_record(endpoint.assumed_gain),
            }
        endpoint_reports.append(
            {
                "assumptions": assumptions_record,
                "alternative_direction": endpoint.alternative_direction,
                "calculation": calculation,
                "component_count": component_roster.component_count,
                "component_roster_id": component_roster.roster_id,
                "eligible_component_ids_sha256": (component_roster.eligible_component_ids_sha256),
                "endpoint_id": endpoint.endpoint_id,
                "kind": endpoint.kind,
                "metric_id": endpoint.metric_id,
                "population_firewall_report_sha256": (
                    component_roster.population_firewall_report_sha256
                ),
                "population_role": component_roster.population_role,
                "reference_id": endpoint.reference_id,
                "scope": component_roster.scope,
                "source_id": endpoint.source_id,
                "stratum_id": component_roster.stratum_id,
            }
        )

    component_roster_records = [
        {
            "component_count": roster.component_count,
            "eligible_component_ids_sha256": roster.eligible_component_ids_sha256,
            "eligible_only": True,
            "outcomes_absent": True,
            "population_firewall_report_sha256": roster.population_firewall_report_sha256,
            "population_role": roster.population_role,
            "preoutcome_frozen": True,
            "roster_id": roster.roster_id,
            "scope": roster.scope,
            "stratum_id": roster.stratum_id,
            "unit": _COMPONENT_UNIT,
        }
        for roster in component_rosters
    ]
    structured_hypothesis_records = [
        {
            "alternative_direction": endpoint_report["alternative_direction"],
            "component_roster_id": endpoint_report["component_roster_id"],
            "endpoint_id": endpoint_report["endpoint_id"],
            "kind": endpoint_report["kind"],
            "metric_id": endpoint_report["metric_id"],
            "planning_assumptions": endpoint_report["assumptions"],
            "reference_id": endpoint_report["reference_id"],
            "source_id": endpoint_report["source_id"],
        }
        for endpoint_report in endpoint_reports
    ]
    hypothesis_roster_record = {
        "complete_for_protocol": True,
        "endpoint_ids": endpoint_ids,
        "family_id": multiplicity_family_id,
        "hypotheses": structured_hypothesis_records,
        "protocol_sha256": protocol_sha256,
    }
    runtime_identity_after = _runtime_identity_snapshot()
    if runtime_identity_after != runtime_identity:
        raise GVSPowerError("power-planner source or loaded runtime changed during calculation")
    report: dict[str, object] = {
        "assumption_sources_are_caller_attested_only": True,
        "assumption_sources_sha256": _sha256_json(source_values),
        "assumptions_sha256": assumptions_sha256,
        "authorizes_cuda_or_jarvis_access": False,
        "authorizes_model_or_label_access": False,
        "component_roster_authentication_verified": False,
        "component_roster_sha256": _sha256_json(component_roster_records),
        "component_rosters": component_roster_records,
        "design_id": design_id,
        "endpoint_alpha": _ratio_record(endpoint_alpha),
        "endpoints": endpoint_reports,
        "familywise_alpha": _ratio_record(familywise_alpha),
        "hypothesis_roster": hypothesis_roster_record,
        "hypothesis_roster_completeness_authenticated": False,
        "hypothesis_roster_sha256": _sha256_json(hypothesis_roster_record),
        "independence_contract": independence,
        "launch_authorized": False,
        "multiplicity": {
            "endpoint_count": endpoint_count,
            "family_id": multiplicity_family_id,
            "method": "bonferroni",
        },
        "outcome_blindness_authenticated": False,
        "population_protocol_sha256": population_protocol_sha256,
        "protocol_sha256": protocol_sha256,
        "requires_authenticated_one_shot_custody_receipt": True,
        "requires_independent_power_review": True,
        "runtime_identity": runtime_identity,
        "schema_version": GVS_POWER_REPORT_SCHEMA_VERSION,
        "source_sha256": runtime_identity["module_source_sha256"],
        "structural_planning_only": True,
        "target_power": _ratio_record(target_power),
        "work_preflight": work_preflight,
        "zero_post_freeze_attrition": True,
    }
    report["report_sha256"] = _sha256_json(report)
    return report


def _load_strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_JSON_BYTES:
        raise GVSPowerError(f"{label} bytes are empty, non-bytes, or oversized")

    def pairs_hook(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise GVSPowerError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = _JSON_LOADS(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                GVSPowerError(f"non-finite JSON constant {token!r}")
            ),
        )
    except GVSPowerError:
        raise
    except (UnicodeDecodeError, _JSON_DECODE_ERROR, ValueError, RecursionError) as error:
        raise GVSPowerError(f"{label} is not strict UTF-8 JSON") from error
    snapshot = _snapshot_json(value, label=f"{label} JSON")
    if type(snapshot) is not dict:
        raise GVSPowerError(f"{label} root must be an exact JSON object")
    return snapshot


def load_power_assumptions_json(raw: bytes) -> dict[str, Any]:
    """Strict bounded assumptions loader that rejects duplicates and non-finite constants."""

    return _load_strict_json_object(raw, label="power assumptions")


def load_power_report_json(raw: bytes) -> dict[str, Any]:
    """Load a bounded report; call :func:`verify_power_report` before trusting it."""

    return _load_strict_json_object(raw, label="power report")


def verify_power_report(
    report: Mapping[str, Any],
    assumptions: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute and byte-compare a report from one detached assumptions snapshot."""

    report_snapshot = _snapshot_json(report, label="power report")
    assumptions_snapshot = _snapshot_json(assumptions, label="power assumptions")
    if type(report_snapshot) is not dict:
        raise GVSPowerError("power report must be an exact JSON object")
    if type(assumptions_snapshot) is not dict:
        raise GVSPowerError("power assumptions must be an exact JSON object")
    _canonical_json(report_snapshot)
    stored_report_sha256 = _strict_sha256(
        report_snapshot.get("report_sha256"),
        label="report_sha256",
    )
    unhashed_report = dict(report_snapshot)
    del unhashed_report["report_sha256"]
    if _sha256_json(unhashed_report) != stored_report_sha256:
        raise GVSPowerError("power report SHA-256 does not match its exact body")
    expected = build_power_report(assumptions_snapshot)
    if _canonical_json(report_snapshot) != _canonical_json(expected):
        raise GVSPowerError("power report differs from an exact recomputation")
    return report_snapshot


__all__ = [
    "GVS_POWER_ASSUMPTIONS_SCHEMA_VERSION",
    "GVS_POWER_REPORT_SCHEMA_VERSION",
    "GVS_POWER_RUNTIME_SELF_TEST_SCHEMA_VERSION",
    "ONE_SAMPLE_KIND",
    "PAIRED_KIND",
    "GVSPowerError",
    "build_power_report",
    "exact_critical_successes",
    "load_power_assumptions_json",
    "load_power_report_json",
    "verify_power_report",
]


# Reports must bind the code and source that were actually imported, not merely
# whatever stable replacement happens to be present when a report is built.
_INITIAL_SOURCE_SHA256 = _stable_source_sha256()
_INITIAL_LOADED_CODE_SHA256 = _loaded_code_sha256()
