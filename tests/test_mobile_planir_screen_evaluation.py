from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, replace

import pytest

import barunlm.evaluation.mobile_planir_screen as screen_module
from barunlm.datasets.mobile_planir_screen import (
    SCREEN_MANIFEST_ROW_VERSION as PRODUCER_ROW_VERSION,
)
from barunlm.datasets.mobile_planir_screen import (
    SPLIT_VERSION as PRODUCER_SPLIT_VERSION,
)
from barunlm.datasets.mobile_planir_screen import _arm_record as producer_arm_record
from barunlm.datasets.mobile_planir_v2 import MaterializedArmExample
from barunlm.evaluation.grounded_planir_v2 import (
    ACTION_PROMPT_CONTRACT,
    PROMPT_CONTRACT,
    build_reference_table,
    make_prompt_evidence,
    render_construction_prompt,
)
from barunlm.evaluation.mobile_planir_screen import (
    ARMS,
    PINNED_MANIFEST_CONTENT_SHA256,
    PINNED_POPULATION_SHA256,
    PINNED_PROMPT_POPULATION_SHA256,
    PINNED_TABLE_POPULATION_SHA256,
    PREDEFINED_SUBSETS,
    SCREEN_MANIFEST_ROW_VERSION,
    SCREEN_METADATA_VERSION,
    SCREEN_PREDICTION_ROW_VERSION,
    SCREEN_SPLIT_VERSION,
    SEEDS,
    SFT_EXAMPLE_VERSION,
    ExactRate,
    MobilePlanIRScreenError,
    RunMetrics,
    ScreenEvaluation,
    SubsetMetrics,
    build_evidence_receipt,
    evaluate_screen_gate,
    score_screen,
    validate_evidence_receipt,
)
from barunlm.training.data import SFTExample

NOW = "2026-08-17T09:00:00"
SYSTEM_BODY = f"NOW {NOW}\nTOOLS\nUse the frozen seven Mobile tools."


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _assert_no_float(value: object) -> None:
    assert not isinstance(value, float)
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_float(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_float(child)


def _call(tool: str, args: dict[str, object]) -> dict[str, object]:
    return {"args": args, "tool": tool}


SOURCE_CASES: tuple[tuple[str, str, list[dict[str, object]], str], ...] = (
    (
        "calendar",
        "Schedule Team Sync tomorrow at noon",
        [_call("create_calendar_event", {"datetime": "2026-08-18T12:00:00", "title": "Team Sync"})],
        "SINGLE",
    ),
    (
        "contact",
        "Create contact Ada Lovelace",
        [_call("create_contact", {"first_name": "Ada", "last_name": "Lovelace"})],
        "SINGLE",
    ),
    ("wifi", "Open Wi-Fi settings", [_call("open_wifi_settings", {})], "SINGLE"),
    (
        "email",
        "Email the status to ada@example.com",
        [_call("send_email", {"subject": "Status", "to": "ada@example.com"})],
        "SINGLE",
    ),
    ("map", "Show the map for Pune", [_call("show_map", {"query": "Pune"})], "SINGLE"),
    ("flashoff", "Turn off the flashlight", [_call("turn_off_flashlight", {})], "SINGLE"),
    ("flashon", "Turn on the flashlight", [_call("turn_on_flashlight", {})], "SINGLE"),
    (
        "multi",
        "Show Pune and email the route to ada@example.com",
        [
            _call("show_map", {"query": "Pune"}),
            _call("send_email", {"subject": "Route", "to": "ada@example.com"}),
        ],
        "SERIAL",
    ),
)


def _gold(calls: list[dict[str, object]], mode: str) -> str:
    return _canonical({"calls": calls, "decision": "CALL", "mode": mode})


def _plan(gold: str, source_id: str) -> str:
    if source_id != "calendar":
        return gold
    decoded = json.loads(gold)
    decoded["calls"][0]["args"]["datetime"] = "@R:D00:T00"
    return _canonical(decoded)


def _manifest_and_plans() -> tuple[list[dict[str, object]], dict[str, str]]:
    rows: list[dict[str, object]] = []
    plans: dict[str, str] = {}
    for source_id, request, calls, mode in SOURCE_CASES:
        gold = _gold(calls, mode)
        plans[source_id] = _plan(gold, source_id)
        table = build_reference_table(request, NOW)
        source_prompt = render_construction_prompt(
            prompt_contract=ACTION_PROMPT_CONTRACT,
            system_body=SYSTEM_BODY,
            request=request,
            rendered_table=None,
        )
        source_content_sha = _sha(f"source-content:{source_id}")
        for arm in ARMS:
            contract = PROMPT_CONTRACT if arm == "C" else ACTION_PROMPT_CONTRACT
            rendered_table = table.render() if arm in {"B", "C"} else None
            prompt = render_construction_prompt(
                prompt_contract=contract,
                system_body=SYSTEM_BODY,
                request=request,
                rendered_table=rendered_table,
            )
            evidence = make_prompt_evidence(prompt, contract, table, request, NOW)
            rows.append(
                {
                    "schema_version": SFT_EXAMPLE_VERSION,
                    "id": source_id,
                    "prompt": prompt,
                    "target": gold,
                    "metadata": {
                        "adapter_schema_version": "barun-mobile-actions-adapter-v2",
                        "dataset": "google/mobile-actions",
                        "derived_split": "dev",
                        "prompt_contract_version": "barun-action-prompt-v1",
                        "prompt_sha256": _sha(prompt),
                        "source_split": "train",
                        "target_sha256": _sha(gold),
                        "mobile_planir_screen": {
                            "schema_version": SCREEN_METADATA_VERSION,
                            "split_version": SCREEN_SPLIT_VERSION,
                            "arm": arm,
                            "population": "screen",
                            "source_id": source_id,
                            "source_content_sha256": source_content_sha,
                            "source_metadata_sha256": _sha(f"source-metadata:{source_id}"),
                            "source_prompt_sha256": _sha(source_prompt),
                            "source_target_sha256": _sha(gold),
                            "output_prompt_sha256": _sha(prompt),
                            "output_target_sha256": _sha(gold),
                            "prompt_evidence": asdict(evidence),
                        },
                    },
                }
            )
    return rows, plans


def _prediction_rows(
    manifest: list[dict[str, object]], plans: dict[str, str]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in manifest:
        metadata = item["metadata"]["mobile_planir_screen"]  # type: ignore[index]
        arm = metadata["arm"]
        source_id = metadata["source_id"]
        raw = item["target"] if arm in {"A", "B"} else plans[source_id]
        assert isinstance(raw, str)
        for seed in SEEDS:
            rows.append(
                {
                    "schema_version": SCREEN_PREDICTION_ROW_VERSION,
                    "id": item["id"],
                    "source_id": source_id,
                    "arm": arm,
                    "seed": seed,
                    "prompt_sha256": metadata["output_prompt_sha256"],
                    "table_sha256": metadata["prompt_evidence"]["table_sha256"],
                    "checkpoint_sha256": _sha(f"checkpoint:{arm}:{seed}"),
                    "decoding_sha256": _sha("unconstrained-greedy-v1"),
                    "prediction_raw": raw,
                    "prediction_raw_sha256": _sha(raw),
                    "missing": False,
                    "truncated": False,
                    "generation_failure": None,
                }
            )
    return rows


def _fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    manifest, plans = _manifest_and_plans()
    return manifest, _prediction_rows(manifest, plans)


def _replace_raw(row: dict[str, object], raw: str | None) -> None:
    row["prediction_raw"] = raw
    row["prediction_raw_sha256"] = None if raw is None else _sha(raw)
    row["missing"] = raw is None


def test_scores_direct_arms_and_only_compiled_c_action_ir() -> None:
    manifest, predictions = _fixture()
    result = score_screen(manifest, predictions, enforce_pinned=False)

    assert len(result.rows) == len(SOURCE_CASES) * len(ARMS) * len(SEEDS)
    assert len(result.runs) == len(ARMS) * len(SEEDS)
    assert all(run.ast_exact.numerator == run.sample_count for run in result.runs)
    assert all(run.argument_value_exact.numerator == run.sample_count for run in result.runs)
    direct = next(row for row in result.rows if row.arm == "A" and row.source_id == "calendar")
    compiled = next(row for row in result.rows if row.arm == "C" and row.source_id == "calendar")
    assert direct.compiler_success is None
    assert compiled.prediction_raw is not None and "@R:D00:T00" in compiled.prediction_raw
    assert compiled.prediction_action_ir is not None
    assert "2026-08-18T12:00:00" in compiled.prediction_action_ir
    assert compiled.compiler_attempted and compiled.compiler_success
    assert compiled.assessed_call and compiled.simulator_success is True
    assert compiled.catastrophic_unauthorized_action is False
    assert compiled.compiler_error_code is None and compiled.compiler_error_path is None
    assert compiled.prompt_sha256 != compiled.table_sha256
    assert set(compiled.subsets) == {
        "calendar",
        "single_call",
        "contains_tool:create_calendar_event",
    }
    assert "not_proof_of_model_presentation" in result.to_record()["prompt_evidence_scope"]


def test_public_scoring_and_receipt_apis_reject_alternate_populations_by_default() -> None:
    manifest, predictions = _fixture()
    with pytest.raises(MobilePlanIRScreenError, match="frozen 1,148-source"):
        score_screen(manifest, predictions)
    with pytest.raises(MobilePlanIRScreenError, match="frozen 1,148-source"):
        build_evidence_receipt(manifest, predictions, **SOURCE_HASHES)

    unpinned = build_evidence_receipt(manifest, predictions, enforce_pinned=False, **SOURCE_HASHES)
    assert unpinned["pinned_manifest_enforced"] is False
    with pytest.raises(MobilePlanIRScreenError, match="frozen 1,148-source"):
        validate_evidence_receipt(unpinned, manifest, predictions, **SOURCE_HASHES)
    with pytest.raises(MobilePlanIRScreenError, match="enforce_pinned must be boolean"):
        score_screen(manifest, predictions, enforce_pinned=1)  # type: ignore[arg-type]


def test_pinned_hashes_reject_same_size_matched_label_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, predictions = _fixture()
    baseline = score_screen(manifest, predictions, enforce_pinned=False)
    monkeypatch.setattr(screen_module, "PINNED_MANIFEST_ROWS", len(manifest))
    monkeypatch.setattr(screen_module, "PINNED_SOURCE_ROWS", len(SOURCE_CASES))
    monkeypatch.setattr(
        screen_module, "PINNED_MANIFEST_CONTENT_SHA256", baseline.manifest_content_sha256
    )
    monkeypatch.setattr(
        screen_module, "PINNED_PROMPT_POPULATION_SHA256", baseline.prompt_population_sha256
    )
    monkeypatch.setattr(
        screen_module, "PINNED_TABLE_POPULATION_SHA256", baseline.table_population_sha256
    )
    monkeypatch.setattr(screen_module, "PINNED_POPULATION_SHA256", baseline.population_sha256)
    assert score_screen(manifest, predictions).pinned_manifest_enforced

    substituted = copy.deepcopy(manifest)
    replacement = _canonical(
        {
            "calls": [_call("show_map", {"query": "Mumbai"})],
            "decision": "CALL",
            "mode": "SINGLE",
        }
    )
    replacement_sha = _sha(replacement)
    for row in substituted:
        if row["id"] != "map":
            continue
        row["target"] = replacement
        metadata = row["metadata"]
        screen = metadata["mobile_planir_screen"]  # type: ignore[index]
        metadata["target_sha256"] = replacement_sha  # type: ignore[index]
        screen["source_target_sha256"] = replacement_sha
        screen["output_target_sha256"] = replacement_sha
        screen["source_content_sha256"] = _sha("substituted-map-content")
        screen["source_metadata_sha256"] = _sha("substituted-map-metadata")
    with pytest.raises(MobilePlanIRScreenError, match="frozen 1,148-source"):
        score_screen(substituted, predictions)


def test_consumes_exact_materializer_screen_records_without_transform() -> None:
    assert PRODUCER_ROW_VERSION == SCREEN_MANIFEST_ROW_VERSION
    assert PRODUCER_SPLIT_VERSION == SCREEN_SPLIT_VERSION
    source_id, request, calls, mode = SOURCE_CASES[0]
    gold = _gold(calls, mode)
    source_prompt = render_construction_prompt(
        prompt_contract=ACTION_PROMPT_CONTRACT,
        system_body=SYSTEM_BODY,
        request=request,
        rendered_table=None,
    )
    source = SFTExample(
        example_id=source_id,
        prompt=source_prompt,
        target=gold,
        metadata={
            "adapter_schema_version": "barun-mobile-actions-adapter-v2",
            "dataset": "google/mobile-actions",
            "derived_split": "train",
            "prompt_contract_version": "barun-action-prompt-v1",
            "prompt_sha256": _sha(source_prompt),
            "source_split": "train",
            "target_sha256": _sha(gold),
        },
        content_sha256=_sha(f"source-content:{source_id}"),
    )
    table = build_reference_table(request, NOW)
    materialized: list[MaterializedArmExample] = []
    for arm in ARMS:
        contract = PROMPT_CONTRACT if arm == "C" else ACTION_PROMPT_CONTRACT
        rendered_table = table.render() if arm in {"B", "C"} else None
        prompt = render_construction_prompt(
            prompt_contract=contract,
            system_body=SYSTEM_BODY,
            request=request,
            rendered_table=rendered_table,
        )
        materialized.append(
            MaterializedArmExample(
                example_id=source_id,
                arm=arm,
                prompt=prompt,
                target=_plan(gold, source_id) if arm == "C" else gold,
                prompt_evidence=make_prompt_evidence(prompt, contract, table, request, NOW),
            )
        )
    manifest = [producer_arm_record(source, item, population="screen") for item in materialized]
    predictions = _prediction_rows(manifest, {source_id: _plan(gold, source_id)})
    result = score_screen(manifest, predictions, enforce_pinned=False)
    assert len(result.rows) == 9
    assert all(row.ast_exact for row in result.rows)


def test_compile_failure_and_markdown_fence_are_not_repaired() -> None:
    manifest, predictions = _fixture()
    literal = next(
        row
        for row in predictions
        if row["arm"] == "C" and row["seed"] == 17 and row["source_id"] == "calendar"
    )
    raw_literal = next(
        row["target"]
        for row in manifest
        if row["id"] == "calendar" and row["metadata"]["mobile_planir_screen"]["arm"] == "C"  # type: ignore[index]
    )
    assert isinstance(raw_literal, str)
    _replace_raw(literal, raw_literal)
    fenced = next(
        row
        for row in predictions
        if row["arm"] == "C" and row["seed"] == 29 and row["source_id"] == "map"
    )
    assert isinstance(fenced["prediction_raw"], str)
    _replace_raw(fenced, f"```json\n{fenced['prediction_raw']}\n```")

    result = score_screen(manifest, predictions, enforce_pinned=False)
    literal_evidence = next(
        row
        for row in result.rows
        if row.arm == "C" and row.seed == 17 and row.source_id == "calendar"
    )
    fenced_evidence = next(
        row for row in result.rows if row.arm == "C" and row.seed == 29 and row.source_id == "map"
    )
    assert literal_evidence.raw_json_parse
    assert literal_evidence.compiler_success is False
    assert literal_evidence.compiler_error_code == "invalid_placeholder"
    assert literal_evidence.compiler_error_path == "$.calls[0].args.datetime"
    assert not literal_evidence.ast_exact
    assert not literal_evidence.assessed_call
    assert literal_evidence.simulator_success is None
    assert not fenced_evidence.raw_json_parse
    assert fenced_evidence.compiler_success is False
    assert fenced_evidence.compiler_error_code == "invalid_plan_json"
    assert fenced_evidence.prediction_action_ir is None


def test_row_complete_argument_values_keep_call_identity_but_not_outer_mode() -> None:
    manifest, predictions = _fixture()
    wrong_noarg_tool = next(
        row
        for row in predictions
        if row["arm"] == "A" and row["seed"] == 17 and row["source_id"] == "wifi"
    )
    _replace_raw(
        wrong_noarg_tool,
        _canonical(
            {
                "calls": [_call("turn_on_flashlight", {})],
                "decision": "CALL",
                "mode": "SINGLE",
            }
        ),
    )
    wrong_mode = next(
        row
        for row in predictions
        if row["arm"] == "A" and row["seed"] == 17 and row["source_id"] == "map"
    )
    assert isinstance(wrong_mode["prediction_raw"], str)
    decoded = json.loads(wrong_mode["prediction_raw"])
    decoded["mode"] = "SERIAL"
    _replace_raw(wrong_mode, _canonical(decoded))

    result = score_screen(manifest, predictions, enforce_pinned=False)
    noarg_evidence = next(
        row for row in result.rows if row.arm == "A" and row.seed == 17 and row.source_id == "wifi"
    )
    mode_evidence = next(
        row for row in result.rows if row.arm == "A" and row.seed == 17 and row.source_id == "map"
    )
    assert not noarg_evidence.argument_value_exact
    assert not noarg_evidence.ast_exact
    assert mode_evidence.argument_value_exact
    assert not mode_evidence.ast_exact
    assert mode_evidence.assessed_call
    assert mode_evidence.simulator_success is False


def test_strict_manifest_and_prediction_bindings_reject_swaps() -> None:
    manifest, predictions = _fixture()
    bad_prompt = copy.deepcopy(predictions)
    first = bad_prompt[0]
    first["prompt_sha256"] = _sha("a different prompt")
    with pytest.raises(MobilePlanIRScreenError, match="prompt/table binding"):
        score_screen(manifest, bad_prompt, enforce_pinned=False)

    bad_manifest = copy.deepcopy(manifest)
    bad_manifest[0]["metadata"]["mobile_planir_screen"]["population"] = "selection"  # type: ignore[index]
    with pytest.raises(MobilePlanIRScreenError, match="not a screen population"):
        score_screen(bad_manifest, predictions, enforce_pinned=False)

    extra_prediction_field = copy.deepcopy(predictions)
    extra_prediction_field[0]["repaired_output"] = extra_prediction_field[0]["prediction_raw"]
    with pytest.raises(MobilePlanIRScreenError, match="fields differ"):
        score_screen(manifest, extra_prediction_field, enforce_pinned=False)

    caller_catastrophic = copy.deepcopy(predictions)
    caller_catastrophic[0]["catastrophic_unauthorized_action"] = True
    with pytest.raises(MobilePlanIRScreenError, match="fields differ"):
        score_screen(manifest, caller_catastrophic, enforce_pinned=False)

    mismatched_decoding = copy.deepcopy(predictions)
    for row in mismatched_decoding:
        if row["arm"] == "C" and row["seed"] == 43:
            row["decoding_sha256"] = _sha("different decoding")
    with pytest.raises(MobilePlanIRScreenError, match="identical decoding"):
        score_screen(manifest, mismatched_decoding, enforce_pinned=False)


def test_caller_populations_are_exact_containers_and_receipts_are_detached() -> None:
    manifest, predictions = _fixture()

    class HostileSequence(Sequence[dict[str, object]]):
        def __init__(self, values: list[dict[str, object]]) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int) -> dict[str, object]:
            return self.values[index]

        def __iter__(self) -> Iterator[dict[str, object]]:
            return iter(self.values)

    with pytest.raises(MobilePlanIRScreenError, match="exact list or tuple"):
        score_screen(HostileSequence(manifest), predictions, enforce_pinned=False)
    with pytest.raises(MobilePlanIRScreenError, match="exact list or tuple"):
        score_screen(manifest, HostileSequence(predictions), enforce_pinned=False)

    non_json_container = copy.deepcopy(manifest)
    non_json_container[0]["metadata"]["call_names"] = ("create_calendar_event",)  # type: ignore[index]
    with pytest.raises(MobilePlanIRScreenError, match="exact built-in JSON"):
        score_screen(non_json_container, predictions, enforce_pinned=False)

    receipt = build_evidence_receipt(
        tuple(manifest), tuple(predictions), enforce_pinned=False, **SOURCE_HASHES
    )
    detached_hash = receipt["manifest_content_sha256"]
    manifest[0]["prompt"] = "caller mutation after receipt"
    assert receipt["manifest_content_sha256"] == detached_hash


SOURCE_HASHES = {
    "config_sha256": _sha("config"),
    "compiler_source_sha256": _sha("compiler"),
    "evaluator_source_sha256": _sha("evaluator"),
    "renderer_source_sha256": _sha("renderer"),
}


def test_receipt_rejects_swapped_prompt_prediction_and_checkpoint_evidence() -> None:
    manifest, predictions = _fixture()
    receipt = build_evidence_receipt(manifest, predictions, enforce_pinned=False, **SOURCE_HASHES)
    assert (
        validate_evidence_receipt(
            receipt, manifest, predictions, enforce_pinned=False, **SOURCE_HASHES
        )
        == receipt
    )
    assert "cannot by itself observe" in receipt["prompt_evidence_scope"]

    swapped_outputs = copy.deepcopy(predictions)
    first = next(
        row
        for row in swapped_outputs
        if row["arm"] == "A" and row["seed"] == 17 and row["source_id"] == "map"
    )
    second = next(
        row
        for row in swapped_outputs
        if row["arm"] == "A" and row["seed"] == 17 and row["source_id"] == "email"
    )
    first_raw, second_raw = first["prediction_raw"], second["prediction_raw"]
    assert isinstance(first_raw, str) and isinstance(second_raw, str)
    _replace_raw(first, second_raw)
    _replace_raw(second, first_raw)
    with pytest.raises(MobilePlanIRScreenError, match="does not bind"):
        validate_evidence_receipt(
            receipt, manifest, swapped_outputs, enforce_pinned=False, **SOURCE_HASHES
        )

    swapped_checkpoints = copy.deepcopy(predictions)
    for row in swapped_checkpoints:
        if row["arm"] == "A" and row["seed"] == 17:
            row["checkpoint_sha256"] = _sha("substituted checkpoint")
    with pytest.raises(MobilePlanIRScreenError, match="does not bind"):
        validate_evidence_receipt(
            receipt, manifest, swapped_checkpoints, enforce_pinned=False, **SOURCE_HASHES
        )

    swapped_prompts = copy.deepcopy(manifest)
    prompt_bound_predictions = copy.deepcopy(predictions)
    source_id = "map"
    request = next(item[1] for item in SOURCE_CASES if item[0] == source_id)
    table = build_reference_table(request, NOW)
    replacement_body = f"{SYSTEM_BODY}\nReceipt substitution probe."
    source_prompt = render_construction_prompt(
        prompt_contract=ACTION_PROMPT_CONTRACT,
        system_body=replacement_body,
        request=request,
        rendered_table=None,
    )
    for row in swapped_prompts:
        screen = row["metadata"]["mobile_planir_screen"]  # type: ignore[index]
        if screen["source_id"] != source_id:
            continue
        arm = screen["arm"]
        contract = PROMPT_CONTRACT if arm == "C" else ACTION_PROMPT_CONTRACT
        shown_table = table.render() if arm in {"B", "C"} else None
        prompt = render_construction_prompt(
            prompt_contract=contract,
            system_body=replacement_body,
            request=request,
            rendered_table=shown_table,
        )
        prompt_sha = _sha(prompt)
        row["prompt"] = prompt
        row["metadata"]["prompt_sha256"] = prompt_sha  # type: ignore[index]
        screen["output_prompt_sha256"] = prompt_sha
        screen["source_prompt_sha256"] = _sha(source_prompt)
        screen["source_content_sha256"] = _sha("substituted source content")
        screen["source_metadata_sha256"] = _sha("substituted source metadata")
        screen["prompt_evidence"] = asdict(
            make_prompt_evidence(prompt, contract, table, request, NOW)
        )
        for prediction in prompt_bound_predictions:
            if prediction["source_id"] == source_id and prediction["arm"] == arm:
                prediction["prompt_sha256"] = prompt_sha
    with pytest.raises(MobilePlanIRScreenError, match="does not bind"):
        validate_evidence_receipt(
            receipt,
            swapped_prompts,
            prompt_bound_predictions,
            enforce_pinned=False,
            **SOURCE_HASHES,
        )

    for path in (
        ("prompt_population_sha256",),
        ("prediction_content_sha256",),
        ("runs", "A:s17", "checkpoint_sha256"),
    ):
        swapped = copy.deepcopy(receipt)
        cursor = swapped
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = _sha(f"swapped:{':'.join(path)}")
        with pytest.raises(MobilePlanIRScreenError, match="does not bind"):
            validate_evidence_receipt(
                swapped, manifest, predictions, enforce_pinned=False, **SOURCE_HASHES
            )

    with pytest.raises(MobilePlanIRScreenError, match="does not bind"):
        validate_evidence_receipt(
            receipt,
            manifest,
            predictions,
            enforce_pinned=False,
            **{**SOURCE_HASHES, "config_sha256": _sha("swapped-config")},
        )


def _gate_run(
    arm: str,
    seed: int,
    *,
    ast: int,
    arguments: int,
    denominator: int = 1_000,
    parse: int = 1_000,
    compiler: int = 1_000,
    conditional_schema: int | None = None,
    subset_override: tuple[str, int] | None = None,
    failure: str | None = None,
) -> RunMetrics:
    subset_scores = {
        name: SubsetMetrics(
            sample_count=denominator,
            ast_exact=ExactRate(ast, denominator),
            argument_value_exact=ExactRate(arguments, denominator),
        )
        for name in PREDEFINED_SUBSETS
    }
    if subset_override is not None:
        name, numerator = subset_override
        subset_scores[name] = SubsetMetrics(
            sample_count=denominator,
            ast_exact=ExactRate(numerator, denominator),
            argument_value_exact=ExactRate(arguments, denominator),
        )
    if conditional_schema is None:
        conditional_schema = compiler
    failures = {
        "raw_parse_failure": denominator - parse,
        "action_ir_schema_failure": 0 if arm != "C" else denominator - conditional_schema,
        "compiler_failure": 0 if arm != "C" else denominator - compiler,
        "missing": 0,
        "truncation": 0,
        "generation_failure": 0,
        "catastrophic_unauthorized_action": 0,
    }
    if failure is not None:
        failures[failure] = 1
    return RunMetrics(
        arm=arm,  # type: ignore[arg-type]
        seed=seed,
        population_sha256=PINNED_POPULATION_SHA256,
        checkpoint_sha256=_sha(f"checkpoint:{arm}:{seed}"),
        decoding_sha256=_sha("decoding"),
        sample_count=denominator,
        ast_exact=ExactRate(ast, denominator),
        argument_value_exact=ExactRate(arguments, denominator),
        raw_json_parse=ExactRate(parse, denominator),
        action_ir_schema_valid=ExactRate(
            denominator if arm != "C" else conditional_schema, denominator
        ),
        compiler_success=ExactRate(compiler, denominator) if arm == "C" else None,
        conditional_compiled_schema_valid=(
            ExactRate(conditional_schema, compiler) if arm == "C" else None
        ),
        failure_counts=failures,
        subsets=subset_scores,
    )


def _gate_evaluation(
    *,
    c_ast: tuple[int, int, int] = (930, 930, 930),
    c_arguments: tuple[int, int, int] = (900, 900, 900),
    c_parse: tuple[int, int, int] = (995, 995, 995),
    c_compiler: tuple[int, int, int] = (995, 995, 995),
    c_schema: tuple[int, int, int] = (995, 995, 995),
    subset_override: tuple[str, int] | None = None,
    failure: str | None = None,
) -> ScreenEvaluation:
    runs: list[RunMetrics] = []
    for arm in ARMS:
        for index, seed in enumerate(SEEDS):
            if arm == "C":
                runs.append(
                    _gate_run(
                        arm,
                        seed,
                        ast=c_ast[index],
                        arguments=c_arguments[index],
                        parse=c_parse[index],
                        compiler=c_compiler[index],
                        conditional_schema=c_schema[index],
                        subset_override=subset_override,
                        failure=failure,
                    )
                )
            else:
                runs.append(_gate_run(arm, seed, ast=900, arguments=850))
    return ScreenEvaluation(
        schema_version="barun-mobile-planir-screen-score-v1",
        manifest_content_sha256=PINNED_MANIFEST_CONTENT_SHA256,
        prompt_population_sha256=PINNED_PROMPT_POPULATION_SHA256,
        table_population_sha256=PINNED_TABLE_POPULATION_SHA256,
        population_sha256=PINNED_POPULATION_SHA256,
        rows=(),
        runs=tuple(runs),
        pinned_manifest_enforced=True,
    )


def test_gate_accepts_exact_thresholds_and_two_of_three_seed_wins() -> None:
    result = evaluate_screen_gate(
        _gate_evaluation(c_ast=(960, 900, 930), c_arguments=(900, 900, 900))
    )
    assert result.passed
    assert result.observed["positive_ast_seeds"] == [17, 43]
    assert result.observed["comparisons"]["A"]["ast_exact_delta"] == {  # type: ignore[index]
        "numerator": 3,
        "denominator": 100,
    }
    assert result.observed["comparisons"]["B"]["argument_value_exact_delta"] == {  # type: ignore[index]
        "numerator": 1,
        "denominator": 20,
    }
    _assert_no_float(result.to_record())


def test_gate_rejects_an_unpinned_evaluation_even_when_metrics_pass() -> None:
    evaluation = replace(_gate_evaluation(), pinned_manifest_enforced=False)
    with pytest.raises(MobilePlanIRScreenError, match="requires a score with pinned"):
        evaluate_screen_gate(evaluation)


@pytest.mark.parametrize(
    "field",
    (
        "manifest_content_sha256",
        "population_sha256",
        "prompt_population_sha256",
        "table_population_sha256",
    ),
)
def test_gate_rejects_a_forged_pinned_boolean_with_wrong_identity(field: str) -> None:
    evaluation = replace(_gate_evaluation(), **{field: _sha(f"wrong:{field}")})
    with pytest.raises(MobilePlanIRScreenError, match="exact frozen manifest"):
        evaluate_screen_gate(evaluation)


@pytest.mark.parametrize(
    ("kwargs", "failed_check"),
    [
        ({"c_ast": (929, 929, 929)}, "ast_margin_vs_A_at_least_3_points"),
        (
            {"c_arguments": (899, 899, 899)},
            "argument_margin_vs_A_at_least_5_points",
        ),
        (
            {"c_ast": (960, 900, 900)},
            "c_beats_both_ast_in_at_least_2_seeds",
        ),
        ({"c_parse": (994, 995, 995)}, "c_raw_json_parse_at_least_99_5_percent"),
        (
            {"c_compiler": (994, 995, 995), "c_schema": (994, 995, 995)},
            "c_compiler_success_at_least_99_5_percent",
        ),
        (
            {"c_schema": (994, 995, 995)},
            "c_conditional_compiled_schema_valid_100_percent",
        ),
        ({"failure": "truncation"}, "every_arm_zero_truncation"),
    ],
)
def test_gate_fails_each_exact_boundary(kwargs: dict[str, object], failed_check: str) -> None:
    result = evaluate_screen_gate(_gate_evaluation(**kwargs))  # type: ignore[arg-type]
    assert not result.passed
    assert result.checks[failed_check] is False


def test_gate_rejects_three_point_predefined_subset_loss() -> None:
    result = evaluate_screen_gate(_gate_evaluation(subset_override=("contains_tool:show_map", 870)))
    assert not result.passed
    assert result.checks["no_predefined_subset_ast_loss_over_2_points"] is False
    comparison = result.observed["subset_comparisons"]["contains_tool:show_map"]["A"]  # type: ignore[index]
    assert comparison["delta"] == {"numerator": -3, "denominator": 100}


def test_run_metric_contract_rejects_conditional_schema_denominator_mismatch() -> None:
    run = _gate_run("C", 17, ast=930, arguments=900, compiler=995)
    with pytest.raises(ValueError, match="conditional schema denominator"):
        replace(run, conditional_compiled_schema_valid=ExactRate(994, 1_000))
