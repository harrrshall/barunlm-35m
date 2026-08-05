"""CPU-hermetic tests for the matched-adaptation scale-sweep rules (attempt 5).

Covers the fresh grouped split derivation, the gold token-length audit and
generation-budget rule, the raw prompt transport and termination contract, the
preregistered learning-rate screen, the adoption decision rule, the immutable
v5 configuration binding, the in-run candidate-v2 reference contract, the
protected-machine-ID enforcement, the axolotl/CPython 3.11.10 runtime
attestation gate, the isolated-venv / flash_attn fail-closed preflight, the
real-scorer outcome join, the complete challenger snapshot hash binding, the
explicit frozen decoding overrides, the per-fit measured-failure semantics, and
the environment version binding.  No network, GPU, or workstation-specific paths
are used; committed repository files are the only fixtures.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from barunaction.candidate import CANDIDATE_CHECKPOINT_SHA256
from barunlm.baselines.mobile_scale_sweep import (
    ATTEMPT_1_CONFIG_SHA256,
    ATTEMPT_1_INFRASTRUCTURE_FAILURE_SHA256,
    ATTEMPT_1_NO_GO_SHA256,
    ATTEMPT_2_CONFIG_SHA256,
    ATTEMPT_2_NO_GO_SHA256,
    ATTEMPT_3_CONFIG_SHA256,
    ATTEMPT_3_GO_SHA256,
    ATTEMPT_4_CONFIG_SHA256,
    ATTEMPT_4_GO_SHA256,
    ATTEMPT_4_INFRASTRUCTURE_FAILURE_SHA256,
    CONFIG_PATH,
    CONFIG_SHA256,
    ENVIRONMENT_PACKAGES,
    GOLD_AUDIT_FROZEN_FIELDS,
    REQUIRED_INCLUDE_SYSTEM_SITE_PACKAGES,
    REQUIRED_PROTECTED_EVIDENCE_IDS,
    REQUIRED_PROVIDER_TEMPLATE,
    REQUIRED_PYTHON_IMPLEMENTATION,
    REQUIRED_PYTHON_VERSION,
    RUN_ID,
    SCALE_SWEEP_CONFIG_SCHEMA_VERSION,
    SELECTION_POLICY_VERSION,
    SNAPSHOT_ALLOW_PATTERNS,
    SNAPSHOT_REQUIRED_FILES,
    TRAIN_ROWS,
    ArmOutcome,
    LearningRateFit,
    MeasuredFitFailure,
    ScaleSweepError,
    build_decision,
    build_parser,
    decide_adoption,
    decoding_kwargs,
    enforce_machine_id,
    enforce_runtime_attestation,
    enforce_venv_isolation,
    environment_versions,
    fit_outcome_counts,
    gold_token_length_audit,
    load_frozen_config,
    max_new_tokens_from_audit,
    parse_pyvenv_cfg,
    partition_frozen_train_manifest,
    read_include_system_site_packages,
    select_learning_rate,
    selection_fold_for_cluster,
    tokenize_raw_rows,
    validate_snapshot_pins,
    verify_challenger_snapshot,
    verify_context_fit,
    verify_gold_audit_frozen,
    verify_partition_against_config,
    verify_reference_checkpoint,
    verify_termination_contract,
    write_partition_artifacts,
)
from barunlm.evaluation.mobile_actions import write_scores
from barunlm.training.data import SFTExample

REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "experiments" / "runs" / RUN_ID


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _row(example_id: str, cluster_id: str, *, split: str = "train") -> str:
    record = {
        "schema_version": "barun-sft-example-v1",
        "id": example_id,
        "prompt": f"<bos><system>\nprompt {example_id}",
        "target": f"target {example_id}",
        "metadata": {"cluster_id": cluster_id, "derived_split": split},
    }
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _write_manifest(path: Path, lines: list[str]) -> str:
    payload = "".join(lines).encode("utf-8")
    path.write_bytes(payload)
    return _sha256_bytes(payload)


class FakeTokenizer:
    """Whitespace tokenizer with a tokenizers-like encode surface."""

    def __init__(self, vocab_size: int = 1000) -> None:
        self._vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> object:
        assert add_special_tokens is False
        ids = [1 + (hash(word) % (self._vocab_size - 1)) for word in text.split()]

        class Encoding:
            pass

        encoding = Encoding()
        encoding.ids = ids
        return encoding

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        return self._vocab_size


def _sft(example_id: str, prompt: str, target: str) -> SFTExample:
    return SFTExample(
        example_id=example_id,
        prompt=prompt,
        target=target,
        metadata={},
        content_sha256="0" * 64,
    )


# ---------------------------------------------------------------------------
# Fold assignment
# ---------------------------------------------------------------------------


def test_selection_fold_is_deterministic_and_policy_namespaced() -> None:
    first = selection_fold_for_cluster("cluster-a")
    assert first == selection_fold_for_cluster("cluster-a")
    assert 0 <= first < 10
    other_policy = selection_fold_for_cluster("cluster-a", policy_version="different-policy-v9")
    digest = hashlib.sha256(f"{SELECTION_POLICY_VERSION}:cluster-a".encode()).digest()
    assert first == int.from_bytes(digest[:8], "big") % 10
    assert isinstance(other_policy, int)


def test_selection_fold_rejects_bad_inputs() -> None:
    with pytest.raises(ScaleSweepError):
        selection_fold_for_cluster("")
    with pytest.raises(ScaleSweepError):
        selection_fold_for_cluster("cluster-a", folds=1)


# ---------------------------------------------------------------------------
# Partition derivation
# ---------------------------------------------------------------------------


def _find_cluster_ids(count: int, *, fold: int) -> list[str]:
    found: list[str] = []
    index = 0
    while len(found) < count:
        candidate = f"synthetic-{fold}-{index}"
        if selection_fold_for_cluster(candidate) == fold:
            found.append(candidate)
        index += 1
    return found


def test_partition_groups_whole_clusters_and_preserves_bytes(tmp_path: Path) -> None:
    selection_clusters = _find_cluster_ids(2, fold=0)
    train_clusters = [candidate for candidate in _find_cluster_ids(30, fold=3)][:3]
    lines = [
        _row("row-1", train_clusters[0]),
        _row("row-2", selection_clusters[0]),
        _row("row-3", train_clusters[0]),
        _row("row-4", selection_clusters[1]),
        _row("row-5", train_clusters[1]),
        _row("row-6", train_clusters[2]),
        _row("row-7", selection_clusters[0]),
    ]
    manifest = tmp_path / "train.jsonl"
    digest = _write_manifest(manifest, lines)
    partition = partition_frozen_train_manifest(
        manifest, expected_sha256=digest, expected_rows=len(lines)
    )
    assert partition.selection_ids == ("row-2", "row-4", "row-7")
    assert partition.sweep_train_ids == ("row-1", "row-3", "row-5", "row-6")
    assert partition.selection_clusters == 2
    assert partition.sweep_train_clusters == 3
    assert partition.sweep_train_bytes + b"" == "".join(
        lines[index] for index in (0, 2, 4, 5)
    ).encode("utf-8")
    assert partition.selection_bytes == "".join(lines[index] for index in (1, 3, 6)).encode("utf-8")
    assert not set(partition.sweep_train_ids) & set(partition.selection_ids)


def test_partition_rejects_hash_mismatch_and_bad_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    digest = _write_manifest(manifest, [_row("row-1", _find_cluster_ids(1, fold=1)[0])])
    with pytest.raises(ScaleSweepError, match="SHA-256"):
        partition_frozen_train_manifest(manifest, expected_sha256="0" * 64, expected_rows=1)
    with pytest.raises(ScaleSweepError, match="rows"):
        partition_frozen_train_manifest(manifest, expected_sha256=digest, expected_rows=2)

    non_train = tmp_path / "bad-split.jsonl"
    digest = _write_manifest(
        non_train, [_row("row-1", _find_cluster_ids(1, fold=1)[0], split="dev")]
    )
    with pytest.raises(ScaleSweepError, match="non-train"):
        partition_frozen_train_manifest(non_train, expected_sha256=digest, expected_rows=1)

    duplicate = tmp_path / "duplicate.jsonl"
    row = _row("row-1", _find_cluster_ids(1, fold=1)[0])
    digest = _write_manifest(duplicate, [row, row])
    with pytest.raises(ScaleSweepError, match="duplicate"):
        partition_frozen_train_manifest(duplicate, expected_sha256=digest, expected_rows=2)

    missing_cluster = tmp_path / "missing-cluster.jsonl"
    record = {
        "schema_version": "barun-sft-example-v1",
        "id": "row-1",
        "prompt": "p",
        "target": "t",
        "metadata": {"derived_split": "train"},
    }
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    digest = _write_manifest(missing_cluster, [line])
    with pytest.raises(ScaleSweepError, match="cluster_id"):
        partition_frozen_train_manifest(missing_cluster, expected_sha256=digest, expected_rows=1)


def test_partition_requires_both_sides(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    lines = [_row(f"row-{index}", _find_cluster_ids(4, fold=2)[index]) for index in range(4)]
    digest = _write_manifest(manifest, lines)
    with pytest.raises(ScaleSweepError, match="non-empty"):
        partition_frozen_train_manifest(manifest, expected_sha256=digest, expected_rows=4)


def test_write_partition_artifacts_refuses_overwrite(tmp_path: Path) -> None:
    selection = _find_cluster_ids(1, fold=0)[0]
    train = _find_cluster_ids(1, fold=5)[0]
    manifest = tmp_path / "train.jsonl"
    lines = [_row("row-1", train), _row("row-2", selection)]
    digest = _write_manifest(manifest, lines)
    partition = partition_frozen_train_manifest(manifest, expected_sha256=digest, expected_rows=2)
    output = tmp_path / "fresh-split"
    artifacts = write_partition_artifacts(partition, output)
    receipt = artifacts["receipt"]
    assert receipt["sweep_train_rows"] == 1
    assert receipt["selection_rows"] == 1
    written = (output / "selection.jsonl").read_bytes()
    assert written == lines[1].encode("utf-8")
    with pytest.raises(ScaleSweepError, match="refusing to overwrite"):
        write_partition_artifacts(partition, output)


# ---------------------------------------------------------------------------
# Gold token-length audit and generation budget
# ---------------------------------------------------------------------------


def test_gold_token_length_audit_measures_maxima() -> None:
    rows = [
        _sft("a", "one two three", "x y"),
        _sft("b", "one two", "x y z w"),
    ]
    audit = gold_token_length_audit(rows, FakeTokenizer())
    assert audit["rows"] == 2
    assert audit["prompt_tokens"]["max"] == 3
    assert audit["target_tokens"]["max"] == 4
    assert audit["max_target_tokens_with_eos"] == 5
    assert audit["total_with_eos"]["max"] == max(3 + 2 + 1, 2 + 4 + 1)


def test_gold_token_length_audit_rejects_zero_token_rows() -> None:
    with pytest.raises(ScaleSweepError, match="zero tokens"):
        gold_token_length_audit([_sft("a", "   ", "x")], FakeTokenizer())


def test_max_new_tokens_rule() -> None:
    assert max_new_tokens_from_audit(197) == 256
    assert max_new_tokens_from_audit(100) == 192
    assert max_new_tokens_from_audit(1) == 64
    assert max_new_tokens_from_audit(224) == 256
    assert max_new_tokens_from_audit(225) == 320
    with pytest.raises(ScaleSweepError):
        max_new_tokens_from_audit(0)


def test_verify_context_fit_gates_both_windows() -> None:
    verify_context_fit(
        prompt_tokens_max=487, total_with_eos_max=660, max_new_tokens=256, context_limit=2048
    )
    with pytest.raises(ScaleSweepError, match="training row"):
        verify_context_fit(
            prompt_tokens_max=100, total_with_eos_max=2049, max_new_tokens=64, context_limit=2048
        )
    with pytest.raises(ScaleSweepError, match="max_new_tokens"):
        verify_context_fit(
            prompt_tokens_max=1900, total_with_eos_max=2000, max_new_tokens=256, context_limit=2048
        )


# ---------------------------------------------------------------------------
# Raw transport and termination contract
# ---------------------------------------------------------------------------


def test_termination_contract_requires_matching_eos() -> None:
    verify_termination_contract(
        appended_eos_id=0, generation_eos_id=0, pad_token_id=0, vocab_size=50304
    )
    with pytest.raises(ScaleSweepError, match="SmolLM2 0/756"):
        verify_termination_contract(
            appended_eos_id=0, generation_eos_id=2, pad_token_id=0, vocab_size=50304
        )
    with pytest.raises(ScaleSweepError, match="vocabulary"):
        verify_termination_contract(
            appended_eos_id=50304, generation_eos_id=50304, pad_token_id=0, vocab_size=50304
        )
    with pytest.raises(ScaleSweepError, match="non-negative"):
        verify_termination_contract(
            appended_eos_id=-1, generation_eos_id=-1, pad_token_id=0, vocab_size=8
        )


def test_tokenize_raw_rows_appends_exactly_one_eos() -> None:
    class SequentialTokenizer(FakeTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> object:
            assert add_special_tokens is False
            ids = [10 + index for index, _word in enumerate(text.split())]

            class Encoding:
                pass

            encoding = Encoding()
            encoding.ids = ids
            return encoding

    rows = [_sft("a", "one two three", "x y")]
    tokenized = tokenize_raw_rows(rows, SequentialTokenizer(), eos_token_id=0, max_seq_len=64)
    assert len(tokenized) == 1
    example = tokenized[0]
    assert example.prompt_tokens == 3
    assert example.input_ids[-1] == 0
    assert example.input_ids[:3] == (10, 11, 12)
    assert example.labels[:3] == (-100, -100, -100)
    assert example.labels[3:] == (10, 11, 0)
    assert example.target_tokens == 3


def test_tokenize_raw_rows_aborts_on_overlength() -> None:
    rows = [_sft("a", "one two three four five six", "x y")]
    with pytest.raises(ScaleSweepError, match="overlength"):
        tokenize_raw_rows(rows, FakeTokenizer(), eos_token_id=0, max_seq_len=4)


# ---------------------------------------------------------------------------
# Learning-rate screen
# ---------------------------------------------------------------------------


def test_select_learning_rate_prefers_higher_exact_then_lower_rate() -> None:
    rates = [3e-5, 1e-4]
    winner = select_learning_rate(
        [
            LearningRateFit(learning_rate=3e-5, exact_match_count=400, completed=True),
            LearningRateFit(learning_rate=1e-4, exact_match_count=410, completed=True),
        ],
        allowed_rates=rates,
    )
    assert winner == 1e-4
    tie = select_learning_rate(
        [
            LearningRateFit(learning_rate=3e-5, exact_match_count=400, completed=True),
            LearningRateFit(learning_rate=1e-4, exact_match_count=400, completed=True),
        ],
        allowed_rates=rates,
    )
    assert tie == 3e-5


def test_select_learning_rate_failed_fit_cannot_win() -> None:
    rates = [3e-5, 1e-4]
    winner = select_learning_rate(
        [
            LearningRateFit(learning_rate=1e-4, exact_match_count=999, completed=False),
            LearningRateFit(learning_rate=3e-5, exact_match_count=1, completed=True),
        ],
        allowed_rates=rates,
    )
    assert winner == 3e-5
    with pytest.raises(ScaleSweepError, match="measured failure"):
        select_learning_rate(
            [
                LearningRateFit(learning_rate=1e-4, exact_match_count=0, completed=False),
                LearningRateFit(learning_rate=3e-5, exact_match_count=0, completed=False),
            ],
            allowed_rates=rates,
        )


def test_select_learning_rate_requires_exact_frozen_set() -> None:
    with pytest.raises(ScaleSweepError, match="frozen set"):
        select_learning_rate(
            [LearningRateFit(learning_rate=2e-5, exact_match_count=1, completed=True)],
            allowed_rates=[3e-5, 1e-4],
        )


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------


def _outcome(
    arm_id: str,
    parameters: int,
    exact: int,
    *,
    schema_valid: int = 725,
    truncated: int = 0,
    missing: int = 0,
    failures: int = 0,
) -> ArmOutcome:
    return ArmOutcome(
        arm_id=arm_id,
        unique_parameters=parameters,
        rows=725,
        exact_match_count=exact,
        schema_valid_count=schema_valid,
        truncated_count=truncated,
        missing_prediction_count=missing,
        generation_failure_count=failures,
    )


def test_decision_adopts_smallest_passing_arm() -> None:
    reference = 80.0  # needs >= 83.0% => >= 602 exact of 725
    decision = decide_adoption(
        [
            _outcome("smollm2-360m", 361_821_120, 700),
            _outcome("pythia-70m-deduped", 70_426_624, 500),
            _outcome("smollm2-135m", 134_515_008, 640),
        ],
        reference_exact_percent=reference,
    )
    assert decision["adopted_arm_id"] == "smollm2-135m"
    assert decision["all_arms_falsified"] is False
    ordered = [entry["arm_id"] for entry in decision["arms"]]
    assert ordered == ["pythia-70m-deduped", "smollm2-135m", "smollm2-360m"]


def test_decision_gates_validity_truncation_and_missing() -> None:
    reference = 50.0
    decision = decide_adoption(
        [
            _outcome("a", 1, 700, schema_valid=600),
            _outcome("b", 2, 700, truncated=1),
            _outcome("c", 3, 700, missing=1),
            _outcome("d", 4, 700, failures=1),
        ],
        reference_exact_percent=reference,
    )
    assert decision["adopted_arm_id"] is None
    assert decision["all_arms_falsified"] is True
    checks = {entry["arm_id"]: entry["checks"] for entry in decision["arms"]}
    assert checks["a"]["schema_validity"] is False
    assert checks["b"]["zero_truncations"] is False
    assert checks["c"]["zero_missing_predictions"] is False
    assert checks["d"]["zero_generation_failures"] is False


def test_decision_rejects_bad_reference_and_duplicate_arms() -> None:
    with pytest.raises(ScaleSweepError, match="reference"):
        decide_adoption([_outcome("a", 1, 1)], reference_exact_percent=101.0)
    with pytest.raises(ScaleSweepError, match="unique"):
        decide_adoption(
            [_outcome("a", 1, 1), _outcome("a", 2, 2)],
            reference_exact_percent=50.0,
        )


# ---------------------------------------------------------------------------
# Immutable configuration binding
# ---------------------------------------------------------------------------


def test_frozen_config_loads_and_matches_hash() -> None:
    config = load_frozen_config()
    assert config["run_id"] == RUN_ID
    assert all(value is False for value in config["authorization"].values())
    payload = CONFIG_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == CONFIG_SHA256


def test_frozen_config_rejects_tampering(tmp_path: Path) -> None:
    tampered = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    tampered["authorization"]["launch_authorized"] = True
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ScaleSweepError, match="SHA-256"):
        load_frozen_config(tampered_path)


def test_config_fresh_split_matches_committed_receipt() -> None:
    config = load_frozen_config()
    receipt = json.loads((RUN_DIR / "split-receipt.json").read_text(encoding="utf-8"))
    frozen = config["data"]["fresh_split"]
    for key in (
        "sweep_train_rows",
        "selection_rows",
        "sweep_train_clusters",
        "selection_clusters",
        "sweep_train_manifest_sha256",
        "selection_manifest_sha256",
        "sweep_train_membership_sha256",
        "selection_membership_sha256",
    ):
        assert frozen[key] == receipt[key], key
    assert frozen["sweep_train_rows"] + frozen["selection_rows"] == TRAIN_ROWS
    assert receipt["policy_version"] == SELECTION_POLICY_VERSION


def test_config_membership_files_match_frozen_hashes() -> None:
    config = load_frozen_config()
    frozen = config["data"]["fresh_split"]
    for filename, key, rows_key in (
        ("sweep-train-membership.txt", "sweep_train_membership_sha256", "sweep_train_rows"),
        ("selection-membership.txt", "selection_membership_sha256", "selection_rows"),
    ):
        payload = (RUN_DIR / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == frozen[key], filename
        ids = payload.decode("utf-8").splitlines()
        assert len(ids) == frozen[rows_key]
        assert ids == sorted(ids)
    sweep_ids = set(
        (RUN_DIR / "sweep-train-membership.txt").read_text(encoding="utf-8").splitlines()
    )
    selection_ids = set(
        (RUN_DIR / "selection-membership.txt").read_text(encoding="utf-8").splitlines()
    )
    assert not sweep_ids & selection_ids


def test_config_generation_budget_matches_audit_rule() -> None:
    config = load_frozen_config()
    audit = config["gold_token_audit"]
    observed_max = max(
        entry[split]["max_target_tokens_with_eos"]
        for entry in audit["per_tokenizer"].values()
        for split in ("sweep_train", "selection")
    )
    assert observed_max == audit["global_max_target_tokens_with_eos"]
    assert max_new_tokens_from_audit(observed_max) == audit["max_new_tokens"]
    assert config["evaluation"]["max_new_tokens"] == audit["max_new_tokens"]
    committed = json.loads((RUN_DIR / "gold-token-audit.json").read_text(encoding="utf-8"))
    assert committed["max_new_tokens"] == audit["max_new_tokens"]
    for key, entry in audit["per_tokenizer"].items():
        for split in ("sweep_train", "selection"):
            committed_split = committed["per_tokenizer"][key][split]
            assert (
                committed_split["max_target_tokens_with_eos"]
                == entry[split]["max_target_tokens_with_eos"]
            )
            assert committed_split["prompt_tokens"]["max"] == entry[split]["prompt_tokens_max"]
            assert committed_split["total_with_eos"]["max"] == entry[split]["total_with_eos_max"]
    for entry in audit["per_tokenizer"].values():
        for split in ("sweep_train", "selection"):
            verify_context_fit(
                prompt_tokens_max=entry[split]["prompt_tokens_max"],
                total_with_eos_max=entry[split]["total_with_eos_max"],
                max_new_tokens=audit["max_new_tokens"],
                context_limit=2048,
            )


def test_config_optimizer_step_budget_matches_sweep_rows() -> None:
    config = load_frozen_config()
    rows = config["data"]["fresh_split"]["sweep_train_rows"]
    optimization = config["optimization"]
    microbatches = math.ceil(rows / optimization["per_device_batch_size"])
    assert (
        math.ceil(microbatches / optimization["gradient_accumulation_steps"])
        == optimization["expected_optimizer_steps"]
    )


def test_config_protected_ids_extend_prior_denylist() -> None:
    config = load_frozen_config()
    source = json.loads(
        (REPO / "configs" / "mobile_sub100m_off_the_shelf_v1.json").read_text(encoding="utf-8")
    )
    prior = source["compute"]["protected_machine_ids"]
    protected = config["compute"]["protected_machine_ids"]
    assert protected == sorted(set(prior) | set(REQUIRED_PROTECTED_EVIDENCE_IDS))
    assert 465072 in protected
    assert 465155 in protected
    assert 465183 in protected
    assert 465186 in protected


def test_config_roster_is_pinned_and_matches_metadata_receipt() -> None:
    config = load_frozen_config()
    receipt = json.loads((RUN_DIR / "roster-metadata.json").read_text(encoding="utf-8"))
    receipt_by_repo = {entry["repo_id"]: entry for entry in receipt["models"]}
    assert len(config["roster"]) == 3
    for arm in config["roster"]:
        pinned = receipt_by_repo[arm["repo_id"]]
        assert arm["revision"] == pinned["pinned_revision"]
        assert arm["safetensors_total_parameters"] == pinned["safetensors_total_parameters"]
        assert 1 <= arm["unique_trainable_parameters"] <= arm["safetensors_total_parameters"]
        assert (
            arm["tokenizer"]["tokenizer_json_sha256"]
            == pinned["downloaded_metadata_files"]["tokenizer.json"]["sha256"]
        )
        assert arm["tokenizer"]["eos_token_id"] == pinned["eos_token_id_config"]
        assert arm["tokenizer"]["vocab_size"] == pinned["vocab_size"]
        assert pinned["chat_template_present"] is False
        assert arm["transport"] == "raw_prompt_no_chat_template"
        verify_termination_contract(
            appended_eos_id=arm["tokenizer"]["eos_token_id"],
            generation_eos_id=arm["tokenizer"]["eos_token_id"],
            pad_token_id=arm["tokenizer"]["pad_token_id"],
            vocab_size=arm["tokenizer"]["vocab_size"],
        )


def test_verify_partition_against_config_detects_drift(tmp_path: Path) -> None:
    selection = _find_cluster_ids(1, fold=0)[0]
    train = _find_cluster_ids(1, fold=5)[0]
    manifest = tmp_path / "train.jsonl"
    lines = [_row("row-1", train), _row("row-2", selection)]
    digest = _write_manifest(manifest, lines)
    partition = partition_frozen_train_manifest(manifest, expected_sha256=digest, expected_rows=2)
    config = load_frozen_config()
    with pytest.raises(ScaleSweepError, match="fresh split"):
        verify_partition_against_config(partition, config)


# ---------------------------------------------------------------------------
# Attempt-5 lineage: v1–v4 artifacts immutable; v5 cites attempt-4 infra failure + spent go
# ---------------------------------------------------------------------------


def test_v1_through_v4_configs_untouched_and_v5_cites_lineage() -> None:
    repo_v1 = REPO / "configs" / "mobile_scale_sweep_v1.json"
    assert hashlib.sha256(repo_v1.read_bytes()).hexdigest() == ATTEMPT_1_CONFIG_SHA256
    no_go_1 = RUN_DIR / "prelaunch-audit-attempt-1-no-go.json"
    assert hashlib.sha256(no_go_1.read_bytes()).hexdigest() == ATTEMPT_1_NO_GO_SHA256
    repo_v2 = REPO / "configs" / "mobile_scale_sweep_v2.json"
    assert hashlib.sha256(repo_v2.read_bytes()).hexdigest() == ATTEMPT_2_CONFIG_SHA256
    no_go_2 = RUN_DIR / "prelaunch-audit-attempt-2-no-go.json"
    assert hashlib.sha256(no_go_2.read_bytes()).hexdigest() == ATTEMPT_2_NO_GO_SHA256
    repo_v3 = REPO / "configs" / "mobile_scale_sweep_v3.json"
    assert hashlib.sha256(repo_v3.read_bytes()).hexdigest() == ATTEMPT_3_CONFIG_SHA256
    go_3 = RUN_DIR / "prelaunch-audit-attempt-3-go.json"
    assert hashlib.sha256(go_3.read_bytes()).hexdigest() == ATTEMPT_3_GO_SHA256
    infra_1 = RUN_DIR / "attempt-1-infrastructure-failure.json"
    assert (
        hashlib.sha256(infra_1.read_bytes()).hexdigest() == ATTEMPT_1_INFRASTRUCTURE_FAILURE_SHA256
    )
    repo_v4 = REPO / "configs" / "mobile_scale_sweep_v4.json"
    assert hashlib.sha256(repo_v4.read_bytes()).hexdigest() == ATTEMPT_4_CONFIG_SHA256
    go_4 = RUN_DIR / "prelaunch-audit-attempt-4-go.json"
    assert hashlib.sha256(go_4.read_bytes()).hexdigest() == ATTEMPT_4_GO_SHA256
    infra_4 = RUN_DIR / "attempt-4-infrastructure-failure.json"
    assert (
        hashlib.sha256(infra_4.read_bytes()).hexdigest() == ATTEMPT_4_INFRASTRUCTURE_FAILURE_SHA256
    )

    config = load_frozen_config()
    assert CONFIG_PATH.name == "mobile_scale_sweep_v5.json"
    assert (
        config["schema_version"]
        == SCALE_SWEEP_CONFIG_SCHEMA_VERSION
        == ("barun-mobile-scale-sweep-config-v5")
    )
    supersedes = config["supersedes"]
    assert supersedes["config_sha256"] == ATTEMPT_4_CONFIG_SHA256
    assert supersedes["attempt"] == 5
    assert supersedes["spent_attempt_4_go"]["sha256"] == ATTEMPT_4_GO_SHA256
    assert supersedes["spent_attempt_4_go"]["reuse_authorized"] is False
    assert (
        supersedes["attempt_4_infrastructure_failure"]["sha256"]
        == ATTEMPT_4_INFRASTRUCTURE_FAILURE_SHA256
    )
    assert supersedes["attempt_4_infrastructure_failure"]["create_machine_id"] == 465183
    assert supersedes["attempt_4_infrastructure_failure"]["resume_migrant_machine_id"] == 465186
    assert supersedes["attempt_4_lineage"]["config_sha256"] == ATTEMPT_4_CONFIG_SHA256
    assert supersedes["attempt_3_lineage"]["config_sha256"] == ATTEMPT_3_CONFIG_SHA256
    assert supersedes["attempt_2_lineage"]["config_sha256"] == ATTEMPT_2_CONFIG_SHA256
    assert supersedes["attempt_1_lineage"]["config_sha256"] == ATTEMPT_1_CONFIG_SHA256
    assert (
        len(
            {
                CONFIG_SHA256,
                ATTEMPT_1_CONFIG_SHA256,
                ATTEMPT_2_CONFIG_SHA256,
                ATTEMPT_3_CONFIG_SHA256,
                ATTEMPT_4_CONFIG_SHA256,
            }
        )
        == 5
    )


def test_v5_scientific_bindings_match_attempt_4() -> None:
    v4 = json.loads((REPO / "configs" / "mobile_scale_sweep_v4.json").read_text(encoding="utf-8"))
    v5 = load_frozen_config()
    for key in (
        "hypothesis",
        "decision_rule",
        "reference_evaluation",
        "anchors",
        "data",
        "roster",
        "transport",
        "gold_token_audit",
        "optimization",
        "evaluation",
    ):
        assert v5[key] == v4[key], key
    assert all(value is False for value in v5["authorization"].values())
    # Reference score from attempt-4 evidence must not become a CLI float.
    assert v5["reference_evaluation"]["cli_override_forbidden"] is True
    assert v5["reference_evaluation"]["scored_before_challenger_arms"] is True


def test_v4_config_remains_immutable_rejected_active_path() -> None:
    # v4 may still exist on disk but must never be the active CONFIG_PATH.
    assert CONFIG_PATH.name != "mobile_scale_sweep_v4.json"
    assert (
        hashlib.sha256((REPO / "configs" / "mobile_scale_sweep_v4.json").read_bytes()).hexdigest()
        == ATTEMPT_4_CONFIG_SHA256
    )


# ---------------------------------------------------------------------------
# P0-2: protected machine ID enforcement
# ---------------------------------------------------------------------------


def test_enforce_machine_id_rejects_every_protected_id() -> None:
    config = load_frozen_config()
    protected = config["compute"]["protected_machine_ids"]
    assert 463058 in protected
    for machine_id in protected:
        with pytest.raises(ScaleSweepError, match="protected"):
            enforce_machine_id(machine_id, protected)
    assert enforce_machine_id(999_999, protected) == 999_999


def test_enforce_machine_id_rejects_malformed_inputs() -> None:
    config = load_frozen_config()
    protected = config["compute"]["protected_machine_ids"]
    for bad_id in (0, -5, "463058", 1.5, None, True):
        with pytest.raises(ScaleSweepError):
            enforce_machine_id(bad_id, protected)
    with pytest.raises(ScaleSweepError, match="denylist"):
        enforce_machine_id(999_999, [])
    with pytest.raises(ScaleSweepError, match="denylist"):
        enforce_machine_id(999_999, [463058, 463058])
    with pytest.raises(ScaleSweepError, match="denylist"):
        enforce_machine_id(999_999, [1, 2, 3])
    with pytest.raises(ScaleSweepError, match="denylist"):
        enforce_machine_id(999_999, [463058, 400000])
    # Prior denylist without the new evidence IDs is rejected at validation.
    prior_only = [mid for mid in protected if mid not in REQUIRED_PROTECTED_EVIDENCE_IDS]
    with pytest.raises(ScaleSweepError, match="denylist"):
        enforce_machine_id(999_999, prior_only)


# ---------------------------------------------------------------------------
# Attempt-4 INFRA retained: axolotl / CPython 3.11.10 runtime attestation
# ---------------------------------------------------------------------------


def test_config_freezes_axolotl_cpython_31110_runtime() -> None:
    config = load_frozen_config()
    compute = config["compute"]
    assert compute["template"] == REQUIRED_PROVIDER_TEMPLATE == "axolotl"
    assert compute["python_implementation"] == REQUIRED_PYTHON_IMPLEMENTATION == "CPython"
    assert compute["python_version"] == REQUIRED_PYTHON_VERSION == "3.11.10"
    assert compute["provider"] == "JarvisLabs"
    assert compute["gpu"] == "H200"
    assert compute["num_gpus"] == 1
    assert compute["region"] == "IN2"
    assert compute["is_spot"] is False
    assert compute["storage_gb"] == 100
    assert compute["max_gpu_job_minutes"] == 360


def test_enforce_runtime_attestation_accepts_frozen_axolotl_identity() -> None:
    config = load_frozen_config()
    receipt = enforce_runtime_attestation(
        compute=config["compute"],
        observed_template="axolotl",
        observed_python_implementation="CPython",
        observed_python_version="3.11.10",
    )
    assert receipt == {
        "template": "axolotl",
        "python_implementation": "CPython",
        "python_version": "3.11.10",
    }


@pytest.mark.parametrize(
    ("field", "bad_value", "match"),
    [
        ("observed_template", "pytorch", "template"),
        ("observed_python_implementation", "PyPy", "implementation"),
        ("observed_python_version", "3.10.20", "version"),
        ("observed_python_version", "3.11.9", "version"),
    ],
)
def test_enforce_runtime_attestation_rejects_mismatched_live_identity(
    field: str, bad_value: str, match: str
) -> None:
    config = load_frozen_config()
    kwargs = {
        "observed_template": "axolotl",
        "observed_python_implementation": "CPython",
        "observed_python_version": "3.11.10",
    }
    kwargs[field] = bad_value
    with pytest.raises(ScaleSweepError, match=match):
        enforce_runtime_attestation(compute=config["compute"], **kwargs)


def test_enforce_runtime_attestation_rejects_tampered_compute_contract() -> None:
    config = load_frozen_config()
    tampered = dict(config["compute"])
    tampered["template"] = "pytorch"
    with pytest.raises(ScaleSweepError, match="compute.template"):
        enforce_runtime_attestation(
            compute=tampered,
            observed_template="axolotl",
            observed_python_implementation="CPython",
            observed_python_version="3.11.10",
        )


# ---------------------------------------------------------------------------
# Attempt-5 INFRA: isolated venv / flash_attn fail-closed preflight
# ---------------------------------------------------------------------------


def _write_pyvenv_cfg(directory: Path, *, include_system_site_packages: bool) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    cfg = directory / "pyvenv.cfg"
    flag = "true" if include_system_site_packages else "false"
    cfg.write_text(
        f"home = /usr\ninclude-system-site-packages = {flag}\nversion = 3.11.10\n",
        encoding="utf-8",
    )
    return cfg


def test_config_freezes_isolated_venv_without_system_site_packages() -> None:
    config = load_frozen_config()
    isolation = config["compute"]["venv_isolation"]
    assert isolation["approach"] == "isolated_project_venv_without_system_site_packages"
    assert isolation["include_system_site_packages"] is REQUIRED_INCLUDE_SYSTEM_SITE_PACKAGES
    assert isolation["include_system_site_packages"] is False
    assert isolation["flash_attn_must_be_unimportable"] is True
    assert isolation["fail_closed_before"] == [
        "torch_import",
        "challenger_AutoModelForCausalLM_from_pretrained",
    ]
    assert "flash_attn" in isolation["provider_hazard"]
    assert "system-site-packages" in isolation["provider_hazard"]


def test_parse_pyvenv_cfg_and_system_site_flag(tmp_path: Path) -> None:
    cfg = _write_pyvenv_cfg(tmp_path / "isolated", include_system_site_packages=False)
    values = parse_pyvenv_cfg(cfg)
    assert values["include-system-site-packages"] == "false"
    assert read_include_system_site_packages(cfg) is False
    contaminated = _write_pyvenv_cfg(tmp_path / "contaminated", include_system_site_packages=True)
    assert read_include_system_site_packages(contaminated) is True


def test_enforce_venv_isolation_accepts_isolated_venv_without_flash_attn(
    tmp_path: Path,
) -> None:
    config = load_frozen_config()
    venv = tmp_path / "project-venv"
    _write_pyvenv_cfg(venv, include_system_site_packages=False)
    receipt = enforce_venv_isolation(
        compute=config["compute"],
        virtual_env=str(venv),
        flash_attn_importable=False,
    )
    assert receipt["include_system_site_packages"] is False
    assert receipt["flash_attn_importable"] is False
    assert receipt["virtual_env"] == str(venv.resolve())


def test_enforce_venv_isolation_rejects_system_site_packages_like_attempt_4(
    tmp_path: Path,
) -> None:
    """This is the exact attempt-4 failure mode: jl uv venv --system-site-packages."""

    config = load_frozen_config()
    venv = tmp_path / "contaminated-venv"
    _write_pyvenv_cfg(venv, include_system_site_packages=True)
    with pytest.raises(ScaleSweepError, match="include-system-site-packages"):
        enforce_venv_isolation(
            compute=config["compute"],
            virtual_env=str(venv),
            flash_attn_importable=False,
        )


def test_enforce_venv_isolation_rejects_importable_flash_attn(tmp_path: Path) -> None:
    config = load_frozen_config()
    venv = tmp_path / "isolated-but-flash"
    _write_pyvenv_cfg(venv, include_system_site_packages=False)
    with pytest.raises(ScaleSweepError, match="flash_attn"):
        enforce_venv_isolation(
            compute=config["compute"],
            virtual_env=str(venv),
            flash_attn_importable=True,
        )


def test_enforce_venv_isolation_rejects_missing_virtual_env(tmp_path: Path) -> None:
    config = load_frozen_config()
    with pytest.raises(ScaleSweepError, match="VIRTUAL_ENV"):
        enforce_venv_isolation(
            compute=config["compute"],
            virtual_env="",
            flash_attn_importable=False,
        )
    with pytest.raises(ScaleSweepError, match="pyvenv.cfg"):
        enforce_venv_isolation(
            compute=config["compute"],
            virtual_env=str(tmp_path / "no-cfg"),
            flash_attn_importable=False,
        )


def test_enforce_venv_isolation_rejects_tampered_isolation_contract(tmp_path: Path) -> None:
    config = load_frozen_config()
    tampered = dict(config["compute"])
    isolation = dict(tampered["venv_isolation"])
    isolation["include_system_site_packages"] = True
    tampered["venv_isolation"] = isolation
    venv = tmp_path / "venv"
    _write_pyvenv_cfg(venv, include_system_site_packages=False)
    with pytest.raises(ScaleSweepError, match="include_system_site_packages"):
        enforce_venv_isolation(
            compute=tampered,
            virtual_env=str(venv),
            flash_attn_importable=False,
        )


# ---------------------------------------------------------------------------
# P0-1: in-run reference evaluation; no unauthenticated score path
# ---------------------------------------------------------------------------


def _base_cli(tmp_path: Path) -> list[str]:
    return [
        "--run-id",
        RUN_ID,
        "--jarvis-machine-id",
        "999999",
        "--jarvis-template",
        "axolotl",
        "--train-manifest",
        str(tmp_path / "train.jsonl"),
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--reference-checkpoint",
        str(tmp_path / "checkpoint"),
    ]


def test_cli_rejects_reference_float_and_batch_size_overrides(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(_base_cli(tmp_path))
    assert args.jarvis_machine_id == 999_999
    assert args.jarvis_template == "axolotl"
    assert args.reference_checkpoint == tmp_path / "checkpoint"
    assert not hasattr(args, "reference_exact_percent")
    assert not hasattr(args, "generation_batch_size")


def test_cli_requires_jarvis_template(tmp_path: Path) -> None:
    parser = build_parser()
    base = [
        "--run-id",
        RUN_ID,
        "--jarvis-machine-id",
        "999999",
        "--train-manifest",
        str(tmp_path / "train.jsonl"),
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--reference-checkpoint",
        str(tmp_path / "checkpoint"),
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(base)
    with pytest.raises(SystemExit):
        parser.parse_args([*_base_cli(tmp_path), "--reference-exact-percent", "83.0"])
    with pytest.raises(SystemExit):
        parser.parse_args([*_base_cli(tmp_path), "--generation-batch-size", "32"])
    without_checkpoint = _base_cli(tmp_path)[:-2]
    with pytest.raises(SystemExit):
        parser.parse_args(without_checkpoint)
    non_positive = _base_cli(tmp_path)
    non_positive[3] = "0"
    with pytest.raises(SystemExit):
        parser.parse_args(non_positive)


def _write_checkpoint(directory: Path, contents: dict[str, bytes]) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, payload in contents.items():
        (directory / name).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema_version": "barun-release-checkpoint-v1",
        "file_sha256": hashes,
    }
    (directory / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return hashes


def test_reference_checkpoint_verification_binds_hashes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    hashes = _write_checkpoint(
        checkpoint,
        {
            "model.safetensors": b"stand-in weights",
            "barun_config.json": b"{}",
            "tokenizer.json": b'{"model": {}}',
        },
    )
    verified = verify_reference_checkpoint(checkpoint, expected_sha256=hashes)
    assert verified == hashes

    (checkpoint / "model.safetensors").write_bytes(b"tampered weights")
    with pytest.raises(ScaleSweepError, match="reference checkpoint verification failed"):
        verify_reference_checkpoint(checkpoint, expected_sha256=hashes)


def test_reference_checkpoint_verification_rejects_wrong_pins(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    hashes = _write_checkpoint(
        checkpoint,
        {
            "model.safetensors": b"stand-in weights",
            "barun_config.json": b"{}",
            "tokenizer.json": b'{"model": {}}',
        },
    )
    forged = dict(hashes)
    forged["model.safetensors"] = "0" * 64
    with pytest.raises(ScaleSweepError, match="reference checkpoint verification failed"):
        verify_reference_checkpoint(checkpoint, expected_sha256=forged)
    with pytest.raises(ScaleSweepError, match="reference checkpoint verification failed"):
        verify_reference_checkpoint(tmp_path / "missing", expected_sha256=hashes)


def test_config_reference_contract_binds_committed_candidate_pin() -> None:
    config = load_frozen_config()
    reference = config["reference_evaluation"]
    assert reference["candidate_id"] == "candidate-v2"
    assert reference["checkpoint_sha256"] == dict(CANDIDATE_CHECKPOINT_SHA256)
    assert reference["checkpoint_sha256"] == config["anchors"]["candidate_v2"]["checkpoint_sha256"]
    assert reference["scored_before_challenger_arms"] is True
    assert reference["cli_override_forbidden"] is True
    assert reference["gold_token_audit_key"] in config["gold_token_audit"]["per_tokenizer"]
    assert set(reference["checkpoint_sha256"]) == {
        "barun_config.json",
        "model.safetensors",
        "tokenizer.json",
    }


# ---------------------------------------------------------------------------
# P1-1: pythia unique trainable parameter binding
# ---------------------------------------------------------------------------


def test_pythia_unique_trainable_count_is_exact() -> None:
    config = load_frozen_config()
    arms = {arm["arm_id"]: arm for arm in config["roster"]}
    hidden, layers, intermediate, vocab = 512, 6, 2048, 50304
    embeddings = 2 * vocab * hidden  # untied embed_in + embed_out
    final_norm = 2 * hidden
    per_layer = (
        2 * (2 * hidden)  # input and post-attention LayerNorms
        + (hidden * 3 * hidden + 3 * hidden)  # fused qkv projection
        + (hidden * hidden + hidden)  # attention dense
        + (hidden * intermediate + intermediate)  # mlp h_to_4h
        + (intermediate * hidden + hidden)  # mlp 4h_to_h
    )
    analytic = embeddings + final_norm + layers * per_layer
    assert analytic == 70_426_624
    pythia = arms["pythia-70m-deduped"]
    assert pythia["unique_trainable_parameters"] == analytic
    # The safetensors total additionally counts persisted non-trainable buffers:
    # six 2048x2048 causal-mask attention.bias tensors plus 48 rotary inv_freq entries.
    assert pythia["safetensors_total_parameters"] - analytic == 6 * 2048 * 2048 + 48
    for arm_id in ("smollm2-135m", "smollm2-360m"):
        arm = arms[arm_id]
        assert arm["unique_trainable_parameters"] == arm["safetensors_total_parameters"]


# ---------------------------------------------------------------------------
# P1-2 / P3-2: outcome join against a real scorer aggregate
# ---------------------------------------------------------------------------

SCORER_PROMPT = (
    "<bos><system>\nACTION_IR_V1\nNOW 2026-08-03T16:30:00\nTOOLS\n"
    "set_timer(minutes:integer!): Start a timer.\n"
    "<user>\nSet a five minute timer.\n<assistant>\n"
)
SCORER_GOLD = (
    '{"calls":[{"args":{"minutes":5},"tool":"set_timer"}],"decision":"CALL","mode":"SINGLE"}'
)
SCORER_WRONG = (
    '{"calls":[{"args":{"minutes":10},"tool":"set_timer"}],"decision":"CALL","mode":"SINGLE"}'
)


def _write_scored_fixture(tmp_path: Path) -> dict[str, object]:
    manifest_rows = [
        {
            "id": sample_id,
            "prompt": SCORER_PROMPT,
            "target": SCORER_GOLD,
            "metadata": {"call_names": ["set_timer"]},
        }
        for sample_id in ("correct", "wrong", "cut")
    ]
    prediction_rows = [
        {"id": "correct", "prediction_raw": SCORER_GOLD, "truncated": False},
        {"id": "wrong", "prediction_raw": SCORER_WRONG, "truncated": False},
        {"id": "cut", "prediction_raw": SCORER_GOLD, "truncated": True},
    ]
    manifest_path = tmp_path / "selection.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    for path, rows in ((manifest_path, manifest_rows), (predictions_path, prediction_rows)):
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    write_scores(manifest_path, predictions_path, tmp_path / "scores")
    return json.loads((tmp_path / "scores" / "aggregate.json").read_text(encoding="utf-8"))


def test_fit_outcome_counts_joins_real_scorer_aggregate(tmp_path: Path) -> None:
    aggregate = _write_scored_fixture(tmp_path)
    # Regression guard for the audited defect: the real scorer emits schema_valid,
    # never schema_validity, and every rate carries exact integer numerators.
    assert "schema_valid" in aggregate
    assert "schema_validity" not in aggregate
    generation = {"examples": 3, "generated": 3, "failed": 0, "truncated": 1}
    counts = fit_outcome_counts(generation, aggregate)
    assert counts == {
        "rows": 3,
        "exact_match_count": 2,
        "schema_valid_count": 3,
        "truncated_count": 1,
        "generation_failure_count": 0,
        "missing_prediction_count": 0,
    }


def test_fit_outcome_counts_rejects_inconsistent_join(tmp_path: Path) -> None:
    aggregate = _write_scored_fixture(tmp_path)
    with pytest.raises(ScaleSweepError, match="truncation"):
        fit_outcome_counts({"examples": 3, "generated": 3, "failed": 0, "truncated": 0}, aggregate)
    with pytest.raises(ScaleSweepError, match="denominator"):
        fit_outcome_counts({"examples": 4, "generated": 4, "failed": 0, "truncated": 1}, aggregate)
    renamed = {
        "schema_validity" if key == "schema_valid" else key: value
        for key, value in aggregate.items()
    }
    with pytest.raises(ScaleSweepError, match="schema_valid"):
        fit_outcome_counts({"examples": 3, "generated": 3, "failed": 0, "truncated": 1}, renamed)
    with pytest.raises(ScaleSweepError, match="zero examples"):
        fit_outcome_counts({"examples": 0, "generated": 0, "failed": 0, "truncated": 0}, aggregate)


# ---------------------------------------------------------------------------
# P2-1: gold-audit drift check covers every frozen field
# ---------------------------------------------------------------------------


def _audit_pair() -> tuple[dict[str, object], dict[str, dict[str, int]]]:
    rows = [_sft("a", "one two three", "x y"), _sft("b", "one two", "x y z w")]
    audit = {
        "sweep_train": gold_token_length_audit(rows, FakeTokenizer()),
        "selection": gold_token_length_audit(rows, FakeTokenizer()),
    }
    frozen = {
        split: {
            "max_target_tokens_with_eos": audit[split]["max_target_tokens_with_eos"],
            "prompt_tokens_max": audit[split]["prompt_tokens"]["max"],
            "total_with_eos_max": audit[split]["total_with_eos"]["max"],
        }
        for split in ("sweep_train", "selection")
    }
    return audit, frozen


def test_gold_audit_drift_check_covers_every_field() -> None:
    audit, frozen = _audit_pair()
    verify_gold_audit_frozen(audit, frozen, label="test")
    assert set(GOLD_AUDIT_FROZEN_FIELDS) == {
        "max_target_tokens_with_eos",
        "prompt_tokens_max",
        "total_with_eos_max",
    }
    for field in GOLD_AUDIT_FROZEN_FIELDS:
        for split in ("sweep_train", "selection"):
            _, tampered = _audit_pair()
            tampered[split][field] += 1
            with pytest.raises(ScaleSweepError, match=field):
                verify_gold_audit_frozen(audit, tampered, label="test")


# ---------------------------------------------------------------------------
# P2-2: generation batch size is config-bound
# ---------------------------------------------------------------------------


def test_generation_batch_size_is_frozen_in_config() -> None:
    config = load_frozen_config()
    assert config["evaluation"]["generation_batch_size"] == 64
    assert "no CLI" in config["evaluation"]["generation_batch_size_binding"]


# ---------------------------------------------------------------------------
# P0-3: complete challenger snapshot hash binding
# ---------------------------------------------------------------------------

_SYNTHETIC_SNAPSHOT_CONTENTS = {
    "config.json": b'{"architectures": ["SyntheticForCausalLM"]}',
    "generation_config.json": b'{"eos_token_id": 0}',
    "model.safetensors": b"synthetic weight bytes",
    "special_tokens_map.json": b"{}",
    "tokenizer.json": b'{"model": {}}',
    "tokenizer_config.json": b'{"eos_token": "<x>"}',
}


def _synthetic_snapshot(tmp_path: Path) -> tuple[dict[str, object], Path]:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    pins: dict[str, dict[str, object]] = {}
    for name, payload in _SYNTHETIC_SNAPSHOT_CONTENTS.items():
        (snapshot / name).write_bytes(payload)
        pins[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    arm = {
        "arm_id": "synthetic",
        "snapshot_allow_patterns": list(SNAPSHOT_ALLOW_PATTERNS),
        "snapshot_files": pins,
        "tokenizer": {
            "tokenizer_json_sha256": pins["tokenizer.json"]["sha256"],
            "tokenizer_config_sha256": pins["tokenizer_config.json"]["sha256"],
            "config_json_sha256": pins["config.json"]["sha256"],
        },
    }
    return arm, snapshot


def test_verify_challenger_snapshot_accepts_exact_pins(tmp_path: Path) -> None:
    arm, snapshot = _synthetic_snapshot(tmp_path)
    verified = verify_challenger_snapshot(snapshot, arm)
    assert verified == {name: entry["sha256"] for name, entry in arm["snapshot_files"].items()}


def test_every_pinned_snapshot_file_is_enforced(tmp_path: Path) -> None:
    """Enforcement iterates the pin table itself: corrupting any single pinned
    file (same byte length, different bytes) is rejected by name, so no
    advertised pin can be left unenforced."""

    arm, _snapshot = _synthetic_snapshot(tmp_path / "reference")
    assert set(arm["snapshot_files"]) >= set(SNAPSHOT_REQUIRED_FILES)
    for index, name in enumerate(sorted(arm["snapshot_files"])):
        _, snapshot = _synthetic_snapshot(tmp_path / f"case-{index}")
        original = _SYNTHETIC_SNAPSHOT_CONTENTS[name]
        corrupted = bytes(byte ^ 0xFF for byte in original)
        assert len(corrupted) == len(original)
        (snapshot / name).write_bytes(corrupted)
        with pytest.raises(ScaleSweepError, match=name.replace(".", r"\.")):
            verify_challenger_snapshot(snapshot, arm)


def test_verify_challenger_snapshot_rejects_set_and_size_drift(tmp_path: Path) -> None:
    arm, snapshot = _synthetic_snapshot(tmp_path / "extra")
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"smuggled shard")
    with pytest.raises(ScaleSweepError, match="file set"):
        verify_challenger_snapshot(snapshot, arm)

    arm, snapshot = _synthetic_snapshot(tmp_path / "missing")
    (snapshot / "model.safetensors").unlink()
    with pytest.raises(ScaleSweepError, match="file set"):
        verify_challenger_snapshot(snapshot, arm)

    arm, snapshot = _synthetic_snapshot(tmp_path / "size")
    payload = _SYNTHETIC_SNAPSHOT_CONTENTS["model.safetensors"] + b"!"
    (snapshot / "model.safetensors").write_bytes(payload)
    with pytest.raises(ScaleSweepError, match="bytes"):
        verify_challenger_snapshot(snapshot, arm)


def test_validate_snapshot_pins_rejects_malformed_tables(tmp_path: Path) -> None:
    def broken(mutate) -> dict[str, object]:
        arm, _snapshot = _synthetic_snapshot(tmp_path / "template")
        mutate(arm)
        return arm

    with pytest.raises(ScaleSweepError, match="snapshot_files"):
        validate_snapshot_pins(broken(lambda arm: arm.pop("snapshot_files")))
    with pytest.raises(ScaleSweepError, match="model.safetensors"):
        validate_snapshot_pins(broken(lambda arm: arm["snapshot_files"].pop("model.safetensors")))
    with pytest.raises(ScaleSweepError, match="64-hex"):
        validate_snapshot_pins(
            broken(lambda arm: arm["snapshot_files"]["config.json"].update(sha256="ABC123"))
        )
    with pytest.raises(ScaleSweepError, match="byte size"):
        validate_snapshot_pins(
            broken(lambda arm: arm["snapshot_files"]["config.json"].update(bytes=0))
        )
    with pytest.raises(ScaleSweepError, match="allow pattern"):
        validate_snapshot_pins(
            broken(
                lambda arm: arm["snapshot_files"].update(
                    {"vocab.json": {"sha256": "0" * 64, "bytes": 1}}
                )
            )
        )
    with pytest.raises(ScaleSweepError, match="snapshot_allow_patterns"):
        validate_snapshot_pins(broken(lambda arm: arm.update(snapshot_allow_patterns=[])))
    # An advertised tokenizer/config hash that disagrees with the snapshot pin
    # is exactly the advertised-but-unenforced defect class; it must be rejected.
    with pytest.raises(ScaleSweepError, match="config_json_sha256"):
        validate_snapshot_pins(
            broken(lambda arm: arm["tokenizer"].update(config_json_sha256="0" * 64))
        )
    with pytest.raises(ScaleSweepError, match="tokenizer_json_sha256"):
        validate_snapshot_pins(
            broken(lambda arm: arm["tokenizer"].update(tokenizer_json_sha256="0" * 64))
        )


def test_config_pins_full_challenger_snapshots_and_matches_receipts() -> None:
    config = load_frozen_config()
    receipt = json.loads((RUN_DIR / "snapshot-pins.json").read_text(encoding="utf-8"))
    receipt_by_arm = {entry["arm_id"]: entry for entry in receipt["arms"]}
    roster_receipt = json.loads((RUN_DIR / "roster-metadata.json").read_text(encoding="utf-8"))
    roster_by_repo = {entry["repo_id"]: entry for entry in roster_receipt["models"]}
    weight_hashes = set()
    for arm in config["roster"]:
        pins = arm["snapshot_files"]
        pinned_receipt = receipt_by_arm[arm["arm_id"]]
        assert pinned_receipt["revision"] == arm["revision"]
        assert arm["snapshot_allow_patterns"] == list(SNAPSHOT_ALLOW_PATTERNS)
        assert pinned_receipt["allow_patterns"] == list(SNAPSHOT_ALLOW_PATTERNS)
        # The frozen pin table equals the read-only HF metadata receipt exactly.
        assert set(pins) == set(pinned_receipt["files"])
        for name, entry in pins.items():
            assert entry["sha256"] == pinned_receipt["files"][name]["sha256"], name
            assert entry["bytes"] == pinned_receipt["files"][name]["bytes"], name
        for required in SNAPSHOT_REQUIRED_FILES:
            assert required in pins
        weight_hashes.add(pins["model.safetensors"]["sha256"])
        # Attempt-1 roster metadata hashes stay consistent with the pin table.
        downloaded = roster_by_repo[arm["repo_id"]]["downloaded_metadata_files"]
        for name, meta in downloaded.items():
            assert pins[name]["sha256"] == meta["sha256"], name
            assert pins[name]["bytes"] == meta["bytes"], name
    assert len(weight_hashes) == 3, "each arm pins a distinct model.safetensors"
    arms = {arm["arm_id"]: arm for arm in config["roster"]}
    # pythia has no generation_config.json at its pinned revision: its absence is
    # pinned through exact file-set equality, never silently tolerated.
    assert "generation_config.json" not in arms["pythia-70m-deduped"]["snapshot_files"]
    for arm_id in ("smollm2-135m", "smollm2-360m"):
        assert "generation_config.json" in arms[arm_id]["snapshot_files"]


# ---------------------------------------------------------------------------
# P2-A: frozen decoding overrides are validated and passed explicitly
# ---------------------------------------------------------------------------


def test_decoding_kwargs_passes_frozen_greedy_contract() -> None:
    config = load_frozen_config()
    kwargs = decoding_kwargs(config["evaluation"])
    assert kwargs == {
        "do_sample": False,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    assert None not in kwargs.values()


def test_decoding_kwargs_rejects_any_contract_drift() -> None:
    config = load_frozen_config()
    frozen = config["evaluation"]["generation_config_overrides"]

    def evaluation_with(overrides: dict[str, object]) -> dict[str, object]:
        return {"generation_config_overrides": overrides}

    for mutate in (
        lambda o: o.update(do_sample=True),
        lambda o: o.update(temperature=0.7),
        lambda o: o.update(num_beams=4),
        lambda o: o.update(repetition_penalty=1.2),
        lambda o: o.pop("no_repeat_ngram_size"),
        lambda o: o.update(extra_field=1),
    ):
        overrides = dict(frozen)
        mutate(overrides)
        with pytest.raises(ScaleSweepError, match="frozen greedy contract"):
            decoding_kwargs(evaluation_with(overrides))
    with pytest.raises(ScaleSweepError, match="frozen greedy contract"):
        decoding_kwargs({})


# ---------------------------------------------------------------------------
# P2-B: per-fit measured-failure semantics
# ---------------------------------------------------------------------------


def test_measured_fit_failure_is_a_scale_sweep_error_subclass() -> None:
    assert issubclass(MeasuredFitFailure, ScaleSweepError)
    with pytest.raises(ScaleSweepError):
        raise MeasuredFitFailure("non-finite training loss at optimizer step 1")


def test_config_freezes_per_fit_measured_failure_semantics() -> None:
    config = load_frozen_config()
    semantics = config["optimization"]["fit_failure_semantics"]
    assert semantics["per_fit_measured_failure"] is True
    assert semantics["measured_failure_classes"] == [
        "cuda_out_of_memory",
        "non_finite_training_loss",
    ]
    assert "abort" in semantics["rule"]
    assert "manual_seed" in config["optimization"]["seed_scope"]


def test_build_decision_records_measured_failed_arms() -> None:
    decision = build_decision(
        [
            _outcome("smollm2-135m", 134_515_008, 640),
            _outcome("smollm2-360m", 361_821_120, 700),
        ],
        measured_failed_arm_ids=["pythia-70m-deduped"],
        reference_exact_percent=80.0,
        margin_points=3.0,
        schema_validity_floor=0.95,
    )
    assert decision["adopted_arm_id"] == "smollm2-135m"
    assert decision["measured_failed_arm_ids"] == ["pythia-70m-deduped"]
    assert "never be adopted" in decision["measured_failure_rule"]


def test_build_decision_handles_all_arms_measured_failure() -> None:
    decision = build_decision(
        [],
        measured_failed_arm_ids=["a", "b", "c"],
        reference_exact_percent=80.0,
        margin_points=3.0,
        schema_validity_floor=0.95,
    )
    assert decision["adopted_arm_id"] is None
    assert decision["all_arms_falsified"] is True
    assert decision["arms"] == []
    assert decision["measured_failed_arm_ids"] == ["a", "b", "c"]


def test_build_decision_rejects_overlap_duplicates_and_empty() -> None:
    with pytest.raises(ScaleSweepError, match="unique"):
        build_decision(
            [],
            measured_failed_arm_ids=["a", "a"],
            reference_exact_percent=80.0,
            margin_points=3.0,
            schema_validity_floor=0.95,
        )
    with pytest.raises(ScaleSweepError, match="at least one"):
        build_decision(
            [],
            measured_failed_arm_ids=[],
            reference_exact_percent=80.0,
            margin_points=3.0,
            schema_validity_floor=0.95,
        )
    with pytest.raises(ScaleSweepError, match="both scored and measured"):
        build_decision(
            [_outcome("a", 1, 700)],
            measured_failed_arm_ids=["a"],
            reference_exact_percent=80.0,
            margin_points=3.0,
            schema_validity_floor=0.95,
        )
    with pytest.raises(ScaleSweepError, match="\\[0, 100\\]"):
        build_decision(
            [],
            measured_failed_arm_ids=["a"],
            reference_exact_percent=101.0,
            margin_points=3.0,
            schema_validity_floor=0.95,
        )


# ---------------------------------------------------------------------------
# P3-B: environment version binding
# ---------------------------------------------------------------------------


def test_environment_versions_bind_required_packages() -> None:
    assert ENVIRONMENT_PACKAGES == (
        "huggingface_hub",
        "safetensors",
        "tokenizers",
        "torch",
        "transformers",
    )
    versions = environment_versions(packages=("tokenizers",))
    assert set(versions) == {"python", "tokenizers"}
    assert all(isinstance(value, str) and value for value in versions.values())
    with pytest.raises(ScaleSweepError, match="not installed"):
        environment_versions(packages=("barun-nonexistent-package",))
