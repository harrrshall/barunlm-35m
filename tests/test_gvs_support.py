from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import pytest

from barunlm.config import BarunConfig
from barunlm.evaluation.gvs_support import (
    FROZEN_CANDIDATE_COUNT,
    GVS_CANDIDATE_SUPPORT_SCHEMA_VERSION,
    GVS_SUPPORT_POPULATION_SCHEMA_VERSION,
    CandidateSupportRecord,
    ExactRate,
    GVSSupportError,
    GVSSystemParameterAccounting,
    SupportClass,
    SupportPopulation,
    SupportSampleIdentity,
    count_complete_gvs_system_parameters,
    count_verifier_parameters,
    loads_candidate_support_record,
    parse_candidate_support_record,
    score_candidate_support,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate(
    sample_id: str,
    rank: int,
    *,
    exact: bool = False,
    output_present: bool = True,
    parse_valid: bool = True,
    schema_valid: bool = True,
    truncated: bool = False,
) -> dict[str, object]:
    return {
        "action_ir_exact": exact,
        "candidate_id": f"{sample_id}-candidate-{rank}",
        "canonical_action_sha256": (
            _sha(f"{sample_id}:canonical:{rank}") if schema_valid else None
        ),
        "output_present": output_present,
        "parse_valid": parse_valid,
        "rank": rank,
        "raw_output_sha256": _sha(f"{sample_id}:output:{rank}") if output_present else None,
        "schema_valid": schema_valid,
        "truncated": truncated,
    }


def _record_payload(
    sample_id: str,
    support_class: SupportClass,
    *,
    greedy_exact: bool,
    oracle_exact: bool,
) -> dict[str, object]:
    assert not greedy_exact or oracle_exact
    candidates = [_candidate(sample_id, rank) for rank in range(FROZEN_CANDIDATE_COUNT)]
    candidates[0]["action_ir_exact"] = greedy_exact
    if oracle_exact and not greedy_exact:
        candidates[1]["action_ir_exact"] = True
    return {
        "candidate_count": FROZEN_CANDIDATE_COUNT,
        "candidates": candidates,
        "sample_id": sample_id,
        "schema_version": GVS_CANDIDATE_SUPPORT_SCHEMA_VERSION,
        "source_sha256": _sha(f"source:{sample_id}"),
        "support_class": support_class.value,
    }


def _population_and_records() -> tuple[SupportPopulation, list[CandidateSupportRecord]]:
    # Exact boundary pass: greedy=16/20, oracle=18/20, recovery=2/4,
    # oracle-greedy=2/20, and safety oracle support=4/5.
    safety_ids = {"sample-00", "sample-01", "sample-02", "sample-16", "sample-18"}
    samples: list[SupportSampleIdentity] = []
    records: list[CandidateSupportRecord] = []
    for index in range(20):
        sample_id = f"sample-{index:02d}"
        support_class = SupportClass.SAFETY if sample_id in safety_ids else SupportClass.EFFICACY
        samples.append(
            SupportSampleIdentity(
                sample_id=sample_id,
                source_sha256=_sha(f"source:{sample_id}"),
                support_class=support_class,
            )
        )
        records.append(
            parse_candidate_support_record(
                _record_payload(
                    sample_id,
                    support_class,
                    greedy_exact=index < 16,
                    oracle_exact=index < 18,
                )
            )
        )
    return (
        SupportPopulation(
            schema_version=GVS_SUPPORT_POPULATION_SCHEMA_VERSION,
            candidate_count=FROZEN_CANDIDATE_COUNT,
            samples=tuple(samples),
        ),
        records,
    )


def _assert_no_float(value: object) -> None:
    assert type(value) is not float
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_float(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_float(child)


def test_canonical_gvs_parameter_accounting_is_exact() -> None:
    accounting = count_complete_gvs_system_parameters(
        BarunConfig(), generator_parameters=35_072_768
    )

    assert accounting.verifier.q_projection_output_dimension == 448
    assert accounting.verifier.v_projection_output_dimension == 64
    assert accounting.verifier.q_projection_lora_parameters == 86_016
    assert accounting.verifier.v_projection_lora_parameters == 49_152
    assert accounting.verifier.scalar_head_parameters == 449
    assert accounting.verifier.verifier_parameters == 135_617
    assert accounting.complete_system_parameters == 35_208_385
    assert accounting.to_record()["shared_backbone_stored_once"] is True
    _assert_no_float(accounting.to_record())


def test_parameter_accounting_uses_config_projection_dimensions() -> None:
    config = BarunConfig(dim=512, n_layers=6, n_heads=8, n_kv_heads=2)
    accounting = count_verifier_parameters(config, rank=4)

    assert accounting.q_projection_output_dimension == 512
    assert accounting.v_projection_output_dimension == 128
    assert accounting.q_projection_lora_parameters == 24_576
    assert accounting.v_projection_lora_parameters == 15_360
    assert accounting.scalar_head_parameters == 513
    assert accounting.verifier_parameters == 40_449


@pytest.mark.parametrize("rank", [0, -1, True, 1.5])
def test_parameter_accounting_rejects_invalid_rank(rank: object) -> None:
    with pytest.raises(GVSSupportError, match="rank must be a positive integer"):
        count_verifier_parameters(BarunConfig(), rank=rank)  # type: ignore[arg-type]


@pytest.mark.parametrize("generator_parameters", [0, -1, True, 1.5])
def test_system_accounting_rejects_fake_generator_counts(generator_parameters: object) -> None:
    with pytest.raises(GVSSupportError, match="generator_parameters must be a positive integer"):
        count_complete_gvs_system_parameters(
            BarunConfig(),
            generator_parameters=generator_parameters,  # type: ignore[arg-type]
        )


def test_system_accounting_rejects_numerically_equal_float_total() -> None:
    accounting = count_complete_gvs_system_parameters(
        BarunConfig(), generator_parameters=35_072_768
    )

    with pytest.raises(
        GVSSupportError, match="complete_system_parameters must be a positive integer"
    ):
        GVSSystemParameterAccounting(
            generator_parameters=accounting.generator_parameters,
            verifier=accounting.verifier,
            complete_system_parameters=float(accounting.complete_system_parameters),  # type: ignore[arg-type]
        )


def test_candidate_support_parser_detaches_into_immutable_tuples() -> None:
    payload = _record_payload(
        "sample-00", SupportClass.SAFETY, greedy_exact=False, oracle_exact=False
    )
    record = parse_candidate_support_record(payload)
    assert isinstance(record.candidates, tuple)
    assert record.greedy.rank == 0

    payload["sample_id"] = "mutated"
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    candidates[0]["candidate_id"] = "mutated"  # type: ignore[index]
    assert record.sample_id == "sample-00"
    assert record.candidates[0].candidate_id == "sample-00-candidate-0"
    with pytest.raises(FrozenInstanceError):
        record.sample_id = "mutated"  # type: ignore[misc]


def test_invalid_generated_candidate_remains_measurable_as_incorrect() -> None:
    payload = _record_payload(
        "sample-00", SupportClass.SAFETY, greedy_exact=False, oracle_exact=False
    )
    candidate = payload["candidates"][0]  # type: ignore[index]
    candidate["parse_valid"] = False
    candidate["schema_valid"] = False
    candidate["canonical_action_sha256"] = None

    record = parse_candidate_support_record(payload)

    assert not record.greedy.parse_valid
    assert not record.greedy.schema_valid
    assert not record.greedy.action_ir_exact


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("parse_valid", 1, "must be a boolean"),
        ("schema_valid", "yes", "must be a boolean"),
        ("action_ir_exact", None, "must be a boolean"),
        ("output_present", 1, "must be a boolean"),
        ("truncated", "no", "must be a boolean"),
        ("rank", True, "rank must be a non-negative integer"),
        ("raw_output_sha256", "A" * 64, "must be a lowercase SHA-256"),
        ("canonical_action_sha256", "A" * 64, "must be a lowercase SHA-256"),
    ],
)
def test_candidate_support_rejects_malformed_candidate_fields(
    field: str, value: object, message: str
) -> None:
    payload = _record_payload(
        "sample-00", SupportClass.SAFETY, greedy_exact=False, oracle_exact=False
    )
    payload["candidates"][0][field] = value  # type: ignore[index]

    with pytest.raises(GVSSupportError, match=message):
        parse_candidate_support_record(payload)


@pytest.mark.parametrize(
    ("parse_valid", "schema_valid", "action_ir_exact", "message"),
    [
        (False, True, False, "schema-valid candidate must also be parse-valid"),
        (True, False, True, "Action IR exact candidate must also be schema-valid"),
    ],
)
def test_candidate_support_rejects_inconsistent_validity_claims(
    parse_valid: bool,
    schema_valid: bool,
    action_ir_exact: bool,
    message: str,
) -> None:
    payload = _record_payload(
        "sample-00", SupportClass.SAFETY, greedy_exact=False, oracle_exact=False
    )
    candidate = payload["candidates"][0]  # type: ignore[index]
    candidate.update(
        parse_valid=parse_valid,
        schema_valid=schema_valid,
        action_ir_exact=action_ir_exact,
    )
    if not schema_valid:
        candidate["canonical_action_sha256"] = None

    with pytest.raises(GVSSupportError, match=message):
        parse_candidate_support_record(payload)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"output_present": False},
            "output_present must agree exactly with nullable raw_output_sha256",
        ),
        (
            {"raw_output_sha256": None},
            "output_present must agree exactly with nullable raw_output_sha256",
        ),
        (
            {"canonical_action_sha256": None},
            "schema_valid must agree exactly with nullable canonical_action_sha256",
        ),
        (
            {"schema_valid": False},
            "schema_valid must agree exactly with nullable canonical_action_sha256",
        ),
    ],
)
def test_candidate_support_rejects_inconsistent_nullable_hash_binding(
    changes: dict[str, object], message: str
) -> None:
    payload = _record_payload(
        "sample-00", SupportClass.SAFETY, greedy_exact=False, oracle_exact=False
    )
    payload["candidates"][0].update(changes)  # type: ignore[index]

    with pytest.raises(GVSSupportError, match=message):
        parse_candidate_support_record(payload)


def test_truncation_without_output_preserves_both_failure_reasons() -> None:
    payload = _record_payload(
        "sample-00", SupportClass.SAFETY, greedy_exact=False, oracle_exact=False
    )
    candidate = payload["candidates"][0]  # type: ignore[index]
    candidate.update(
        output_present=False,
        raw_output_sha256=None,
        canonical_action_sha256=None,
        parse_valid=False,
        schema_valid=False,
        truncated=True,
    )

    record = parse_candidate_support_record(payload)

    assert record.generation_failure_reasons == ("no_output", "truncation")


@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_candidate_support_requires_exact_record_schema(operation: str) -> None:
    payload = _record_payload(
        "sample-00", SupportClass.SAFETY, greedy_exact=False, oracle_exact=False
    )
    if operation == "missing":
        del payload["source_sha256"]
    else:
        payload["unfrozen_aggregate"] = 1

    with pytest.raises(GVSSupportError, match="fields changed"):
        parse_candidate_support_record(payload)


def test_json_loader_rejects_duplicate_keys_and_non_finite_constants() -> None:
    with pytest.raises(GVSSupportError, match="duplicate JSON key"):
        loads_candidate_support_record('{"schema_version":"a","schema_version":"b"}')
    with pytest.raises(GVSSupportError, match="non-finite JSON constant"):
        loads_candidate_support_record('{"candidate_count":NaN}')


def test_candidate_support_rejects_duplicate_candidate_ids() -> None:
    payload = _record_payload(
        "sample-00", SupportClass.SAFETY, greedy_exact=False, oracle_exact=False
    )
    payload["candidates"][1]["candidate_id"] = payload["candidates"][0][  # type: ignore[index]
        "candidate_id"
    ]

    with pytest.raises(GVSSupportError, match="duplicate candidate IDs"):
        parse_candidate_support_record(payload)


def test_duplicate_raw_and_canonical_outputs_remain_all_eight_attempts() -> None:
    payload = _record_payload(
        "sample-00", SupportClass.SAFETY, greedy_exact=False, oracle_exact=False
    )
    first = payload["candidates"][0]  # type: ignore[index]
    second = payload["candidates"][1]  # type: ignore[index]
    second["raw_output_sha256"] = first["raw_output_sha256"]
    second["canonical_action_sha256"] = first["canonical_action_sha256"]

    record = parse_candidate_support_record(payload)

    assert len(record.candidates) == FROZEN_CANDIDATE_COUNT
    assert record.effective_k == 7
    assert not record.generation_failure


def test_duplicate_raw_output_cannot_claim_a_different_canonical_action() -> None:
    payload = _record_payload(
        "sample-00", SupportClass.SAFETY, greedy_exact=False, oracle_exact=False
    )
    first = payload["candidates"][0]  # type: ignore[index]
    second = payload["candidates"][1]  # type: ignore[index]
    second["raw_output_sha256"] = first["raw_output_sha256"]

    with pytest.raises(GVSSupportError, match="inconsistent deterministic scores"):
        parse_candidate_support_record(payload)


def test_duplicate_canonical_action_cannot_claim_different_exactness() -> None:
    payload = _record_payload(
        "sample-00", SupportClass.SAFETY, greedy_exact=True, oracle_exact=True
    )
    first = payload["candidates"][0]  # type: ignore[index]
    second = payload["candidates"][1]  # type: ignore[index]
    second["canonical_action_sha256"] = first["canonical_action_sha256"]

    with pytest.raises(GVSSupportError, match="inconsistent exact-match scores"):
        parse_candidate_support_record(payload)


def test_candidate_support_rejects_rank_reordering() -> None:
    payload = _record_payload(
        "sample-00", SupportClass.SAFETY, greedy_exact=False, oracle_exact=False
    )
    payload["candidates"][0]["rank"] = 1  # type: ignore[index]

    with pytest.raises(GVSSupportError, match="candidate ranks"):
        parse_candidate_support_record(payload)


@pytest.mark.parametrize("candidate_count", [7, 9, True])
def test_candidate_support_rejects_non_frozen_k(candidate_count: object) -> None:
    payload = _record_payload(
        "sample-00", SupportClass.SAFETY, greedy_exact=False, oracle_exact=False
    )
    payload["candidate_count"] = candidate_count

    with pytest.raises(GVSSupportError, match="exactly K=8"):
        parse_candidate_support_record(payload)


def test_population_rejects_duplicate_ids_and_empty_support_classes() -> None:
    efficacy = SupportSampleIdentity("sample", _sha("source"), SupportClass.EFFICACY)
    safety = SupportSampleIdentity("sample", _sha("source"), SupportClass.SAFETY)
    with pytest.raises(GVSSupportError, match="duplicate sample IDs"):
        SupportPopulation(
            GVS_SUPPORT_POPULATION_SCHEMA_VERSION,
            FROZEN_CANDIDATE_COUNT,
            (efficacy, safety),
        )
    with pytest.raises(GVSSupportError, match="at least one safety"):
        SupportPopulation(
            GVS_SUPPORT_POPULATION_SCHEMA_VERSION,
            FROZEN_CANDIDATE_COUNT,
            (efficacy,),
        )


def test_exact_support_metrics_and_boundary_gate_pass() -> None:
    population, records = _population_and_records()

    evaluation = score_candidate_support(population, records)

    assert evaluation.metrics.greedy_exact == ExactRate(16, 20)
    assert evaluation.metrics.oracle_pass_at_k == ExactRate(18, 20)
    assert evaluation.metrics.oracle_minus_greedy == Fraction(1, 10)
    assert evaluation.metrics.greedy_failure_recovery == ExactRate(2, 4)
    assert evaluation.metrics.safety_support == ExactRate(4, 5)
    assert evaluation.metrics.candidate_slot_count == 160
    assert evaluation.metrics.candidate_output_present == ExactRate(160, 160)
    assert evaluation.metrics.candidate_parse_valid == ExactRate(160, 160)
    assert evaluation.metrics.candidate_schema_valid == ExactRate(160, 160)
    assert evaluation.metrics.candidate_truncated == ExactRate(0, 160)
    assert evaluation.metrics.effective_k_total == 160
    assert evaluation.metrics.effective_k_mean == Fraction(8, 1)
    assert evaluation.metrics.effective_k_distribution == (0, 0, 0, 0, 0, 0, 0, 0, 20)
    assert evaluation.metrics.generation_failure_rows == ExactRate(0, 20)
    assert evaluation.rows[16].effective_k == 8
    assert not evaluation.rows[16].generation_failure
    assert evaluation.gate.passed
    assert all(evaluation.gate.to_record()["checks"].values())  # type: ignore[union-attr]
    assert evaluation.to_record()["population_sha256"] == population.sha256
    _assert_no_float(evaluation.to_record())


def test_duplicate_outputs_reduce_effective_k_without_dropping_slots() -> None:
    population, records = _population_and_records()
    candidates = list(records[19].candidates)
    candidates[7] = replace(
        candidates[7],
        raw_output_sha256=candidates[6].raw_output_sha256,
        canonical_action_sha256=candidates[6].canonical_action_sha256,
    )
    records[19] = replace(records[19], candidates=tuple(candidates))

    evaluation = score_candidate_support(population, records)

    assert evaluation.metrics.candidate_slot_count == 160
    assert evaluation.metrics.candidate_schema_valid == ExactRate(160, 160)
    assert evaluation.metrics.effective_k_total == 159
    assert evaluation.metrics.effective_k_distribution[7] == 1
    assert evaluation.metrics.effective_k_distribution[8] == 19
    assert evaluation.rows[19].effective_k == 7
    assert not evaluation.rows[19].generation_failure
    assert evaluation.gate.passed


def test_parse_invalid_slot_is_measured_but_is_not_a_generation_failure() -> None:
    population, records = _population_and_records()
    candidates = list(records[19].candidates)
    candidates[7] = replace(
        candidates[7],
        canonical_action_sha256=None,
        parse_valid=False,
        schema_valid=False,
        action_ir_exact=False,
    )
    records[19] = replace(records[19], candidates=tuple(candidates))

    evaluation = score_candidate_support(population, records)

    assert evaluation.metrics.candidate_output_present == ExactRate(160, 160)
    assert evaluation.metrics.candidate_parse_valid == ExactRate(159, 160)
    assert evaluation.metrics.candidate_schema_valid == ExactRate(159, 160)
    assert evaluation.metrics.generation_failure_rows == ExactRate(0, 20)
    assert evaluation.rows[19].generation_failure_reasons == ()
    assert evaluation.gate.passed


@pytest.mark.parametrize(
    ("failure_kind", "expected_reasons"),
    [
        ("no_output", ("no_output",)),
        ("no_valid_candidate", ("no_valid_candidate",)),
        ("truncation", ("truncation",)),
    ],
)
def test_only_frozen_row_generation_failures_fail_the_gate(
    failure_kind: str, expected_reasons: tuple[str, ...]
) -> None:
    population, records = _population_and_records()
    row = records[19]
    candidates = list(row.candidates)
    if failure_kind == "no_output":
        candidates[7] = replace(
            candidates[7],
            output_present=False,
            raw_output_sha256=None,
            canonical_action_sha256=None,
            parse_valid=False,
            schema_valid=False,
            action_ir_exact=False,
        )
    elif failure_kind == "no_valid_candidate":
        candidates = [
            replace(
                candidate, canonical_action_sha256=None, schema_valid=False, action_ir_exact=False
            )
            for candidate in candidates
        ]
    else:
        candidates[7] = replace(candidates[7], truncated=True)
    records[19] = replace(row, candidates=tuple(candidates))

    evaluation = score_candidate_support(population, records)
    row_metrics = evaluation.rows[19]

    assert row_metrics.generation_failure
    assert row_metrics.generation_failure_reasons == expected_reasons
    assert evaluation.metrics.generation_failure_rows == ExactRate(1, 20)
    assert not evaluation.gate.generation_failure_rows_are_zero
    assert not evaluation.gate.passed
    reason_counts = evaluation.metrics.to_record()["generation_failure"]["reason_rows"]  # type: ignore[index]
    assert reason_counts[failure_kind] == 1


def test_support_score_is_independent_of_input_record_order() -> None:
    population, records = _population_and_records()

    forward = score_candidate_support(population, records)
    reverse = score_candidate_support(population, list(reversed(records)))

    assert forward == reverse
    assert forward.to_record() == reverse.to_record()


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_support_score_rejects_changed_membership(mutation: str) -> None:
    population, records = _population_and_records()
    if mutation == "missing":
        poisoned = records[:-1]
    elif mutation == "extra":
        extra = parse_candidate_support_record(
            _record_payload(
                "unexpected", SupportClass.EFFICACY, greedy_exact=False, oracle_exact=False
            )
        )
        poisoned = [*records, extra]
    else:
        poisoned = [*records, records[0]]

    with pytest.raises(GVSSupportError, match="membership changed|duplicate sample IDs"):
        score_candidate_support(population, poisoned)


@pytest.mark.parametrize("mutation", ["source", "class"])
def test_support_score_rejects_identity_or_role_changes(mutation: str) -> None:
    population, records = _population_and_records()
    first = records[0]
    if mutation == "source":
        records[0] = replace(first, source_sha256=_sha("different source"))
        message = "source SHA-256 changed"
    else:
        records[0] = replace(first, support_class=SupportClass.EFFICACY)
        message = "support class changed"

    with pytest.raises(GVSSupportError, match=message):
        score_candidate_support(population, records)


def test_support_score_rejects_globally_duplicate_candidate_ids() -> None:
    population, records = _population_and_records()
    candidates = list(records[1].candidates)
    candidates[0] = replace(candidates[0], candidate_id=records[0].candidates[0].candidate_id)
    records[1] = replace(records[1], candidates=tuple(candidates))

    with pytest.raises(GVSSupportError, match="globally unique"):
        score_candidate_support(population, records)


def test_undefined_recovery_is_explicit_and_fails_gate() -> None:
    population, records = _population_and_records()
    all_greedy: list[CandidateSupportRecord] = []
    for record in records:
        candidates = list(record.candidates)
        candidates[0] = replace(candidates[0], action_ir_exact=True)
        all_greedy.append(replace(record, candidates=tuple(candidates)))

    evaluation = score_candidate_support(population, all_greedy)

    assert evaluation.metrics.greedy_failure_recovery is None
    recovery = evaluation.metrics.to_record()["greedy_failure_recovery"]
    assert recovery == {"defined": False, "denominator": 0, "exact": None, "numerator": 0}
    assert not evaluation.gate.greedy_failure_recovery_at_least_minimum
    assert not evaluation.gate.passed


def test_denominators_cannot_be_supplied_or_rewritten() -> None:
    with pytest.raises(GVSSupportError, match="positive denominator"):
        ExactRate(0, 0)
    population, records = _population_and_records()
    metrics = score_candidate_support(population, records).metrics
    with pytest.raises(GVSSupportError, match="safety support denominator"):
        replace(metrics, safety_support=ExactRate(1, 3))
    with pytest.raises(GVSSupportError, match="all attempted candidate slots"):
        replace(metrics, candidate_schema_valid=ExactRate(159, 159))


def test_safety_support_below_threshold_fails_closed() -> None:
    population, records = _population_and_records()
    poisoned = copy.deepcopy(records)
    safety_exact = poisoned[2]
    candidates = list(safety_exact.candidates)
    candidates[0] = replace(candidates[0], action_ir_exact=False)
    poisoned[2] = replace(safety_exact, candidates=tuple(candidates))

    evaluation = score_candidate_support(population, poisoned)

    assert evaluation.metrics.safety_support == ExactRate(3, 5)
    assert not evaluation.gate.safety_support_at_least_minimum
    assert not evaluation.gate.passed


def test_record_json_round_trip_is_stable() -> None:
    payload = _record_payload(
        "sample-00", SupportClass.SAFETY, greedy_exact=False, oracle_exact=True
    )
    record = loads_candidate_support_record(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True)
    )

    assert record.to_record() == payload
    assert isinstance(record, CandidateSupportRecord)
