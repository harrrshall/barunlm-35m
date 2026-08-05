"""CPU-hermetic tests for the matched-adaptation scale-sweep rules (attempt 2).

Covers the fresh grouped split derivation, the gold token-length audit and
generation-budget rule, the raw prompt transport and termination contract, the
preregistered learning-rate screen, the adoption decision rule, the immutable
v2 configuration binding, the in-run candidate-v2 reference contract, the
protected-machine-ID enforcement, and the real-scorer outcome join.  No
network, GPU, or workstation-specific paths are used; committed repository
files are the only fixtures.
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
    ATTEMPT_1_NO_GO_SHA256,
    CONFIG_PATH,
    CONFIG_SHA256,
    GOLD_AUDIT_FROZEN_FIELDS,
    RUN_ID,
    SCALE_SWEEP_CONFIG_SCHEMA_VERSION,
    SELECTION_POLICY_VERSION,
    TRAIN_ROWS,
    ArmOutcome,
    LearningRateFit,
    ScaleSweepError,
    build_parser,
    decide_adoption,
    enforce_machine_id,
    fit_outcome_counts,
    gold_token_length_audit,
    load_frozen_config,
    max_new_tokens_from_audit,
    partition_frozen_train_manifest,
    select_learning_rate,
    selection_fold_for_cluster,
    tokenize_raw_rows,
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


def test_config_protected_ids_copied_from_sub100m_config() -> None:
    config = load_frozen_config()
    source = json.loads(
        (REPO / "configs" / "mobile_sub100m_off_the_shelf_v1.json").read_text(encoding="utf-8")
    )
    assert config["compute"]["protected_machine_ids"] == source["compute"]["protected_machine_ids"]


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
# Attempt-2 lineage: v1 artifacts immutable, v2 successor cites the no-go
# ---------------------------------------------------------------------------


def test_v1_artifacts_untouched_and_v2_cites_lineage() -> None:
    repo_v1 = REPO / "configs" / "mobile_scale_sweep_v1.json"
    assert hashlib.sha256(repo_v1.read_bytes()).hexdigest() == ATTEMPT_1_CONFIG_SHA256
    no_go = RUN_DIR / "prelaunch-audit-attempt-1-no-go.json"
    assert hashlib.sha256(no_go.read_bytes()).hexdigest() == ATTEMPT_1_NO_GO_SHA256

    config = load_frozen_config()
    assert CONFIG_PATH.name == "mobile_scale_sweep_v2.json"
    assert (
        config["schema_version"]
        == SCALE_SWEEP_CONFIG_SCHEMA_VERSION
        == ("barun-mobile-scale-sweep-config-v2")
    )
    supersedes = config["supersedes"]
    assert supersedes["config_sha256"] == ATTEMPT_1_CONFIG_SHA256
    assert supersedes["prelaunch_audit_no_go_sha256"] == ATTEMPT_1_NO_GO_SHA256
    assert supersedes["attempt"] == 2
    assert CONFIG_SHA256 != ATTEMPT_1_CONFIG_SHA256


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


# ---------------------------------------------------------------------------
# P0-1: in-run reference evaluation; no unauthenticated score path
# ---------------------------------------------------------------------------


def _base_cli(tmp_path: Path) -> list[str]:
    return [
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


def test_cli_rejects_reference_float_and_batch_size_overrides(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(_base_cli(tmp_path))
    assert args.jarvis_machine_id == 999_999
    assert args.reference_checkpoint == tmp_path / "checkpoint"
    assert not hasattr(args, "reference_exact_percent")
    assert not hasattr(args, "generation_batch_size")
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
