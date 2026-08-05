"""Matched-adaptation base-model size/token sweep for Mobile Actions.

This module is the CPU-buildable core of run ``20260805-1554-mobile-scale-sweep-s17``.
It derives the fresh grouped selection split from the frozen 7,937-row internal
train manifest, audits gold token lengths per roster tokenizer, freezes the raw
prompt transport and termination contract for base (non-chat) checkpoints, and
binds the preregistered learning-rate screen and adoption decision rule.

The sealed 961-row official Mobile Actions evaluation tail is never an input:
this runner accepts only the already-derived, hash-pinned internal-train
manifests plus the adapter audit that proves the tail stayed opaque.  The reused
756-row development probe is likewise not a selection metric here; scoring uses
only the fresh selection partition derived below.

``transformers`` and ``huggingface_hub`` are imported only inside :func:`run`,
keeping every preregistered rule CPU-testable without network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from barunlm.baselines.mobile_matched import count_unique_parameters
from barunlm.evaluation.mobile_actions import MOBILE_ACTIONS_SCORER_VERSION, write_scores
from barunlm.training.data import (
    SFTExample,
    TokenizedExample,
    load_manifest,
    sha256_file,
    tokenize_examples,
)

SCALE_SWEEP_CONFIG_SCHEMA_VERSION = "barun-mobile-scale-sweep-config-v1"
RESULT_SCHEMA_VERSION = "barun-mobile-scale-sweep-result-v1"
RAW_TRANSPORT_VERSION = "barun-raw-prompt-transport-v1"
SELECTION_POLICY_VERSION = "barun-mobile-scale-sweep-selection-v1"

RUN_ID = "20260805-1554-mobile-scale-sweep-s17"
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "mobile_scale_sweep_v1.json"
# Frozen after the CPU build, before any baseline weight download or training.
CONFIG_SHA256 = "d3ee897f9afeefe1e01ec32fe9b2721479b7496785e742d0954ad081758953b8"

# Upstream frozen Mobile Actions inputs (identical to the matched-baseline lane).
MOBILE_DATASET_REVISION = "e920309bc2acbc2e99a5e3201cf37df2b9fd9151"
OFFICIAL_SOURCE_SHA256 = "91d251ee958cfd295af6c4504c236a3a1ad19517de240c3bc680bacfcbf7e7d9"
TRAIN_MANIFEST_SHA256 = "131473ccb5bfb51cac0439b42159e72ec4c598025e50364a52b122b056c2e84e"
TRAIN_MEMBERSHIP_SHA256 = "4cdfc3649c21a9f0d3dc4d8d9cffcae3cb5c6abc4414a9fb09636e9220afe96f"
AUDIT_SHA256 = "dc756f97c0a7ef706ec8ffefe2d57cf7e16d75a6ccd932f906d16b6a6ee2f83c"
TRAIN_ROWS = 7_937
DEV_ROWS = 756
OFFICIAL_EVAL_ROWS = 961

SELECTION_FOLDS = 10
SELECTION_FOLD = 0

_RUN_ID_PATTERN = re.compile(r"\d{8}-\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*-s\d+")


class ScaleSweepError(RuntimeError):
    """A frozen scale-sweep invariant was violated."""


class ExistingScaleSweepRunError(FileExistsError):
    """An immutable run directory already exists."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _membership_sha256(example_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(example_ids)) + "\n"
    return _sha256_bytes(payload.encode("utf-8"))


# ---------------------------------------------------------------------------
# Fresh grouped selection split
# ---------------------------------------------------------------------------


def selection_fold_for_cluster(
    cluster_id: str,
    *,
    policy_version: str = SELECTION_POLICY_VERSION,
    folds: int = SELECTION_FOLDS,
) -> int:
    """Deterministic fold assignment for one connected-component cluster.

    The digest input is namespaced by the policy version, so this partition is
    independent of the original train/dev fold assignment even where cluster IDs
    coincide.  Every row of a cluster lands on one side; grouped leakage across
    the boundary is therefore structurally impossible.
    """

    if not isinstance(cluster_id, str) or not cluster_id:
        raise ScaleSweepError("cluster_id must be a non-empty string")
    if folds < 2:
        raise ScaleSweepError("selection folds must be at least 2")
    digest = hashlib.sha256(f"{policy_version}:{cluster_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


@dataclass(frozen=True, slots=True)
class TrainPartition:
    """Byte-exact partition of the frozen train manifest into sweep-train/selection."""

    sweep_train_lines: tuple[str, ...]
    selection_lines: tuple[str, ...]
    sweep_train_ids: tuple[str, ...]
    selection_ids: tuple[str, ...]
    sweep_train_clusters: int
    selection_clusters: int

    @property
    def sweep_train_bytes(self) -> bytes:
        return "".join(self.sweep_train_lines).encode("utf-8")

    @property
    def selection_bytes(self) -> bytes:
        return "".join(self.selection_lines).encode("utf-8")

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "barun-mobile-scale-sweep-split-v1",
            "policy_version": SELECTION_POLICY_VERSION,
            "folds": SELECTION_FOLDS,
            "selection_fold": SELECTION_FOLD,
            "source_train_manifest_sha256": TRAIN_MANIFEST_SHA256,
            "source_train_rows": TRAIN_ROWS,
            "sweep_train_rows": len(self.sweep_train_ids),
            "selection_rows": len(self.selection_ids),
            "sweep_train_clusters": self.sweep_train_clusters,
            "selection_clusters": self.selection_clusters,
            "sweep_train_manifest_sha256": _sha256_bytes(self.sweep_train_bytes),
            "selection_manifest_sha256": _sha256_bytes(self.selection_bytes),
            "sweep_train_membership_sha256": _membership_sha256(self.sweep_train_ids),
            "selection_membership_sha256": _membership_sha256(self.selection_ids),
        }


def partition_frozen_train_manifest(
    train_manifest: str | Path,
    *,
    expected_sha256: str = TRAIN_MANIFEST_SHA256,
    expected_rows: int = TRAIN_ROWS,
) -> TrainPartition:
    """Split the hash-pinned train manifest by whole clusters, preserving bytes.

    Rows keep their original manifest order and exact serialized bytes, so the
    partition manifests remain loadable by the strict manifest loader and their
    hashes are reproducible from the frozen train manifest alone.  The frozen
    defaults bind the production manifest; tests may pin synthetic fixtures.
    """

    manifest_path = Path(train_manifest)
    actual = sha256_file(manifest_path)
    if actual != expected_sha256:
        raise ScaleSweepError(
            f"train manifest SHA-256 changed: expected {expected_sha256}, got {actual}"
        )
    sweep_lines: list[str] = []
    selection_lines: list[str] = []
    sweep_ids: list[str] = []
    selection_ids: list[str] = []
    cluster_side: dict[str, bool] = {}
    seen_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ScaleSweepError(f"line {line_number}: manifest line lacks a newline")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ScaleSweepError(f"line {line_number}: invalid JSON") from error
            if not isinstance(payload, dict):
                raise ScaleSweepError(f"line {line_number}: row is not an object")
            example_id = payload.get("id")
            metadata = payload.get("metadata")
            if not isinstance(example_id, str) or not example_id:
                raise ScaleSweepError(f"line {line_number}: missing id")
            if example_id in seen_ids:
                raise ScaleSweepError(f"line {line_number}: duplicate id {example_id!r}")
            seen_ids.add(example_id)
            if not isinstance(metadata, dict):
                raise ScaleSweepError(f"line {line_number}: missing metadata")
            if metadata.get("derived_split") != "train":
                raise ScaleSweepError(f"line {line_number}: non-train row in train manifest")
            cluster_id = metadata.get("cluster_id")
            if not isinstance(cluster_id, str) or not cluster_id:
                raise ScaleSweepError(f"line {line_number}: missing cluster_id")
            is_selection = selection_fold_for_cluster(cluster_id) == SELECTION_FOLD
            previous = cluster_side.get(cluster_id)
            if previous is not None and previous != is_selection:
                raise ScaleSweepError(  # pragma: no cover - determinism invariant
                    f"cluster {cluster_id!r} was assigned to both sides"
                )
            cluster_side[cluster_id] = is_selection
            if is_selection:
                selection_lines.append(line)
                selection_ids.append(example_id)
            else:
                sweep_lines.append(line)
                sweep_ids.append(example_id)
    if len(sweep_ids) + len(selection_ids) != expected_rows:
        raise ScaleSweepError(
            f"partition covers {len(sweep_ids) + len(selection_ids)} rows, expected {expected_rows}"
        )
    if not sweep_ids or not selection_ids:
        raise ScaleSweepError("both partition sides must be non-empty")
    if set(sweep_ids) & set(selection_ids):  # pragma: no cover - construction invariant
        raise ScaleSweepError("partition sides overlap")
    return TrainPartition(
        sweep_train_lines=tuple(sweep_lines),
        selection_lines=tuple(selection_lines),
        sweep_train_ids=tuple(sweep_ids),
        selection_ids=tuple(selection_ids),
        sweep_train_clusters=sum(1 for side in cluster_side.values() if not side),
        selection_clusters=sum(1 for side in cluster_side.values() if side),
    )


def write_partition_artifacts(partition: TrainPartition, output_dir: str | Path) -> dict[str, Any]:
    """Materialize the two partition manifests plus the split receipt, refusing overwrite."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "sweep_train_manifest": output / "sweep-train.jsonl",
        "selection_manifest": output / "selection.jsonl",
        "receipt": output / "split-receipt.json",
    }
    for path in paths.values():
        if path.exists():
            raise ScaleSweepError(f"refusing to overwrite {path}")
    paths["sweep_train_manifest"].write_bytes(partition.sweep_train_bytes)
    paths["selection_manifest"].write_bytes(partition.selection_bytes)
    receipt = partition.receipt()
    paths["receipt"].write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"paths": {key: str(path) for key, path in paths.items()}, "receipt": receipt}


def verify_partition_against_config(partition: TrainPartition, config: Mapping[str, Any]) -> None:
    """Require the derived partition to match the immutable frozen split exactly."""

    frozen = config["data"]["fresh_split"]
    receipt = partition.receipt()
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
        if receipt[key] != frozen[key]:
            raise ScaleSweepError(
                f"fresh split {key} changed: expected {frozen[key]!r}, got {receipt[key]!r}"
            )


# ---------------------------------------------------------------------------
# Gold token-length audit and generation-budget rule
# ---------------------------------------------------------------------------


class EncodesText(Protocol):
    """Subset of a ``tokenizers.Tokenizer`` used by the audit and transport."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> Any: ...


def _encode_ids(tokenizer: EncodesText, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = getattr(encoded, "ids", encoded)
    if not isinstance(ids, (list, tuple)) or any(type(item) is not int for item in ids):
        raise ScaleSweepError("tokenizer.encode must return integer ids or an object with .ids")
    return list(ids)


def _length_stats(lengths: Sequence[int]) -> dict[str, int]:
    ordered = sorted(lengths)
    if not ordered:
        raise ScaleSweepError("cannot audit an empty row set")

    def percentile(fraction: float) -> int:
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def gold_token_length_audit(
    rows: Sequence[SFTExample],
    tokenizer: EncodesText,
    *,
    eos_reserved: int = 1,
) -> dict[str, Any]:
    """Measure prompt/target token lengths for one tokenizer over the given rows.

    ``max_target_tokens_with_eos`` is the quantity the generation budget must
    dominate: no gold output may be truncatable at the frozen ``max_new_tokens``.
    """

    if eos_reserved < 1:
        raise ScaleSweepError("at least one EOS token must be reserved")
    prompt_lengths: list[int] = []
    target_lengths: list[int] = []
    for row in rows:
        prompt_ids = _encode_ids(tokenizer, row.prompt)
        target_ids = _encode_ids(tokenizer, row.target)
        if not prompt_ids or not target_ids:
            raise ScaleSweepError(f"example {row.example_id!r} encoded to zero tokens")
        prompt_lengths.append(len(prompt_ids))
        target_lengths.append(len(target_ids))
    totals = [
        prompt + target + eos_reserved
        for prompt, target in zip(prompt_lengths, target_lengths, strict=True)
    ]
    return {
        "rows": len(rows),
        "eos_reserved": eos_reserved,
        "prompt_tokens": _length_stats(prompt_lengths),
        "target_tokens": _length_stats(target_lengths),
        "total_with_eos": _length_stats(totals),
        "max_target_tokens_with_eos": max(target_lengths) + eos_reserved,
    }


def max_new_tokens_from_audit(
    max_target_tokens_with_eos: int,
    *,
    margin: int = 32,
    multiple: int = 64,
) -> int:
    """Frozen budget rule: smallest multiple of 64 at or above the gold max plus 32."""

    if max_target_tokens_with_eos < 1:
        raise ScaleSweepError("gold maximum must be positive")
    if margin < 0 or multiple < 1:
        raise ScaleSweepError("margin must be non-negative and multiple positive")
    return multiple * math.ceil((max_target_tokens_with_eos + margin) / multiple)


def verify_context_fit(
    *,
    prompt_tokens_max: int,
    total_with_eos_max: int,
    max_new_tokens: int,
    context_limit: int,
) -> None:
    """Both the longest training row and generation window must fit the context."""

    if total_with_eos_max > context_limit:
        raise ScaleSweepError(
            f"longest training row ({total_with_eos_max} tokens) exceeds the "
            f"{context_limit}-token context"
        )
    if prompt_tokens_max + max_new_tokens > context_limit:
        raise ScaleSweepError(
            f"longest prompt ({prompt_tokens_max} tokens) plus max_new_tokens "
            f"({max_new_tokens}) exceeds the {context_limit}-token context"
        )


# ---------------------------------------------------------------------------
# Raw prompt transport and termination contract for base checkpoints
# ---------------------------------------------------------------------------


def verify_termination_contract(
    *,
    appended_eos_id: int,
    generation_eos_id: int,
    pad_token_id: int,
    vocab_size: int,
) -> None:
    """The supervised EOS must be the exact generation stop token.

    This is the preregistered guard against the SmolLM2 0/756 recipe artifact,
    where 407/756 generations never emitted the stop token that training
    supervised and were counted as truncations.
    """

    for label, value in (
        ("appended_eos_id", appended_eos_id),
        ("generation_eos_id", generation_eos_id),
        ("pad_token_id", pad_token_id),
    ):
        if type(value) is not int or value < 0:
            raise ScaleSweepError(f"{label} must be a non-negative integer")
        if value >= vocab_size:
            raise ScaleSweepError(f"{label} is outside the tokenizer vocabulary")
    if appended_eos_id != generation_eos_id:
        raise ScaleSweepError(
            "training-appended EOS differs from the generation stop token; "
            "this reproduces the SmolLM2 0/756 truncation artifact"
        )


def tokenize_raw_rows(
    rows: Sequence[SFTExample],
    tokenizer: Any,
    *,
    eos_token_id: int,
    max_seq_len: int,
) -> list[TokenizedExample]:
    """Raw transport: full frozen prompt bytes, no chat template, one appended EOS.

    Base checkpoints receive byte-identical model-visible prompts to BarunAction
    candidate-v2 (including the literal role-marker text), encoded with their own
    tokenizer with ``add_special_tokens=False``.  Prompt and target are encoded
    separately, exactly as at generation time.  Any overlength row aborts the
    run; nothing is dropped or truncated.
    """

    accepted, rejected = tokenize_examples(
        rows, tokenizer, eos_token_id=eos_token_id, max_seq_len=max_seq_len
    )
    if rejected:
        raise ScaleSweepError(
            "raw transport produced overlength rows: "
            + ", ".join(sorted(row.example_id for row in rejected))
        )
    if len(accepted) != len(rows):  # pragma: no cover - tokenize_examples invariant
        raise ScaleSweepError("raw transport lost rows")
    for row, example in zip(accepted, rows, strict=True):
        prompt_length = row.prompt_tokens
        if tuple(row.input_ids[prompt_length:])[-1] != eos_token_id:
            raise ScaleSweepError(f"example {example.example_id!r} lacks the appended EOS")
        if row.input_ids[:prompt_length] != row.input_ids[: row.prompt_tokens]:
            raise ScaleSweepError("prompt prefix invariant failed")  # pragma: no cover
    return accepted


# ---------------------------------------------------------------------------
# Preregistered learning-rate screen
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LearningRateFit:
    """Selection-split outcome of one (arm, learning-rate) fit."""

    learning_rate: float
    exact_match_count: int
    completed: bool


def select_learning_rate(
    fits: Sequence[LearningRateFit], *, allowed_rates: Sequence[float]
) -> float:
    """Frozen screen rule: highest selection exact match; ties go to the lower rate.

    Every allowed rate must be represented exactly once.  A fit that failed to
    complete is a measured failure and can never win; if no fit completed the
    arm itself is a measured failure.
    """

    expected = sorted(allowed_rates)
    if len(set(expected)) != len(expected) or not expected:
        raise ScaleSweepError("allowed_rates must be non-empty and unique")
    observed = sorted(fit.learning_rate for fit in fits)
    if observed != expected:
        raise ScaleSweepError(f"screen rates {observed} do not match the frozen set {expected}")
    completed = [fit for fit in fits if fit.completed]
    if not completed:
        raise ScaleSweepError("no learning-rate fit completed; the arm is a measured failure")
    for fit in completed:
        if fit.exact_match_count < 0:
            raise ScaleSweepError("exact-match counts cannot be negative")
    best = max(completed, key=lambda fit: (fit.exact_match_count, -fit.learning_rate))
    return best.learning_rate


# ---------------------------------------------------------------------------
# Preregistered adoption decision rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    """Selection-split outcome of one arm's screen-selected fit."""

    arm_id: str
    unique_parameters: int
    rows: int
    exact_match_count: int
    schema_valid_count: int
    truncated_count: int
    missing_prediction_count: int
    generation_failure_count: int


def _arm_passes(
    arm: ArmOutcome,
    *,
    reference_exact_percent: float,
    margin_points: float,
    schema_validity_floor: float,
) -> dict[str, Any]:
    if arm.rows < 1:
        raise ScaleSweepError(f"arm {arm.arm_id!r} scored zero rows")
    exact_percent = 100.0 * arm.exact_match_count / arm.rows
    schema_validity = arm.schema_valid_count / arm.rows
    checks = {
        "margin": exact_percent >= reference_exact_percent + margin_points,
        "schema_validity": schema_validity >= schema_validity_floor,
        "zero_truncations": arm.truncated_count == 0,
        "zero_missing_predictions": arm.missing_prediction_count == 0,
        "zero_generation_failures": arm.generation_failure_count == 0,
    }
    return {
        "arm_id": arm.arm_id,
        "unique_parameters": arm.unique_parameters,
        "exact_match_percent": exact_percent,
        "schema_validity": schema_validity,
        "checks": checks,
        "passes": all(checks.values()),
    }


def decide_adoption(
    arms: Sequence[ArmOutcome],
    *,
    reference_exact_percent: float,
    margin_points: float = 3.0,
    schema_validity_floor: float = 0.95,
) -> dict[str, Any]:
    """Adopt the smallest passing configuration; otherwise report falsifications.

    ``reference_exact_percent`` is BarunAction candidate-v2 scored on the same
    fresh selection split.  Candidate-v2 trained on every selection row, so its
    reference score is upward-biased and the +3.0-point margin is conservative
    for the challenger arms, which never see selection rows.
    """

    if not arms:
        raise ScaleSweepError("decision rule requires at least one arm outcome")
    if not math.isfinite(reference_exact_percent) or not 0 <= reference_exact_percent <= 100:
        raise ScaleSweepError("reference exact-match percent must be in [0, 100]")
    evaluations = [
        _arm_passes(
            arm,
            reference_exact_percent=reference_exact_percent,
            margin_points=margin_points,
            schema_validity_floor=schema_validity_floor,
        )
        for arm in sorted(arms, key=lambda arm: (arm.unique_parameters, arm.arm_id))
    ]
    if len({evaluation["arm_id"] for evaluation in evaluations}) != len(evaluations):
        raise ScaleSweepError("arm IDs must be unique")
    adopted = next((evaluation for evaluation in evaluations if evaluation["passes"]), None)
    return {
        "schema_version": "barun-mobile-scale-sweep-decision-v1",
        "reference_exact_percent": reference_exact_percent,
        "margin_points": margin_points,
        "schema_validity_floor": schema_validity_floor,
        "ordering": "ascending unique parameter count; smallest passing arm is adopted",
        "arms": evaluations,
        "adopted_arm_id": None if adopted is None else adopted["arm_id"],
        "all_arms_falsified": adopted is None,
    }


# ---------------------------------------------------------------------------
# Frozen configuration binding
# ---------------------------------------------------------------------------


def load_frozen_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the immutable scientific config for this sweep."""

    config_path = Path(path)
    actual = sha256_file(config_path)
    if actual != CONFIG_SHA256:
        raise ScaleSweepError(
            f"scale-sweep config SHA-256 changed: expected {CONFIG_SHA256}, got {actual}"
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScaleSweepError("config must be a JSON object")
    if payload.get("schema_version") != SCALE_SWEEP_CONFIG_SCHEMA_VERSION:
        raise ScaleSweepError("config schema version changed")
    if payload.get("run_id") != RUN_ID or _RUN_ID_PATTERN.fullmatch(RUN_ID) is None:
        raise ScaleSweepError("config run_id is not the frozen immutable run ID")

    authorization = payload.get("authorization")
    if not isinstance(authorization, dict) or not authorization:
        raise ScaleSweepError("config lacks authorization flags")
    for flag, value in authorization.items():
        if value is not False:
            raise ScaleSweepError(
                f"authorization flag {flag!r} must be false until the independent "
                "prelaunch audit passes"
            )

    data = payload.get("data")
    if not isinstance(data, dict):
        raise ScaleSweepError("config lacks the data section")
    frozen_data = {
        "dataset_revision": (data.get("dataset_revision"), MOBILE_DATASET_REVISION),
        "source_sha256": (data.get("source_sha256"), OFFICIAL_SOURCE_SHA256),
        "train_manifest_sha256": (data.get("train_manifest_sha256"), TRAIN_MANIFEST_SHA256),
        "train_membership_sha256": (
            data.get("train_membership_sha256"),
            TRAIN_MEMBERSHIP_SHA256,
        ),
        "audit_sha256": (data.get("audit_sha256"), AUDIT_SHA256),
        "train_rows": (data.get("train_rows"), TRAIN_ROWS),
        "official_eval_rows": (data.get("official_eval_rows"), OFFICIAL_EVAL_ROWS),
    }
    for label, (observed, expected) in frozen_data.items():
        if observed != expected:
            raise ScaleSweepError(f"data.{label} changed: expected {expected!r}, got {observed!r}")
    fresh_split = data.get("fresh_split")
    if not isinstance(fresh_split, dict):
        raise ScaleSweepError("config lacks data.fresh_split")
    if fresh_split.get("policy_version") != SELECTION_POLICY_VERSION:
        raise ScaleSweepError("fresh split policy version changed")
    if fresh_split.get("folds") != SELECTION_FOLDS or fresh_split.get("fold") != SELECTION_FOLD:
        raise ScaleSweepError("fresh split fold parameters changed")
    if fresh_split.get("sweep_train_rows", 0) + fresh_split.get("selection_rows", 0) != TRAIN_ROWS:
        raise ScaleSweepError("fresh split row counts do not sum to the frozen train rows")

    roster = payload.get("roster")
    if not isinstance(roster, list) or len(roster) != 3:
        raise ScaleSweepError("config roster must list exactly the three sweep arms")
    for arm in roster:
        if not isinstance(arm, dict):
            raise ScaleSweepError("each roster arm must be an object")
        revision = arm.get("revision")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ScaleSweepError(f"arm {arm.get('arm_id')!r} lacks a pinned 40-hex revision")
        if arm.get("license") != "apache-2.0":
            raise ScaleSweepError(f"arm {arm.get('arm_id')!r} license changed")

    optimization = payload.get("optimization")
    if not isinstance(optimization, dict):
        raise ScaleSweepError("config lacks the optimization section")
    screen = optimization.get("learning_rate_screen")
    if not isinstance(screen, list) or sorted(screen) != sorted({3e-5, 1e-4}):
        raise ScaleSweepError("learning-rate screen changed from the frozen {1e-4, 3e-5} set")
    for label, expected in (
        ("seed", 17),
        ("epochs", 1),
        ("per_device_batch_size", 21),
        ("gradient_accumulation_steps", 3),
        ("max_seq_len", 2048),
    ):
        if optimization.get(label) != expected:
            raise ScaleSweepError(f"optimization.{label} changed from the frozen value")

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ScaleSweepError("config lacks the evaluation section")
    if evaluation.get("scorer_version") != MOBILE_ACTIONS_SCORER_VERSION:
        raise ScaleSweepError("scorer version binding changed")
    if evaluation.get("decoding") != "unconstrained_deterministic_greedy":
        raise ScaleSweepError("primary decoding must stay unconstrained and greedy")
    if evaluation.get("population") != "fresh_selection_split_only":
        raise ScaleSweepError("selection metric population changed")

    compute = payload.get("compute")
    if not isinstance(compute, dict):
        raise ScaleSweepError("config lacks the compute section")
    protected = compute.get("protected_machine_ids")
    if not isinstance(protected, list) or not protected:
        raise ScaleSweepError("config lacks the protected machine ID denylist")

    return payload


# ---------------------------------------------------------------------------
# Remote GPU runner (exercised only after an independent prelaunch audit)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _RawPromptRow:
    """Prompt-only view of one selection row for batched greedy generation."""

    example_id: str
    prompt_ids: tuple[int, ...]

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_ids)


class _RawTokenizerAdapter:
    """Adapter giving a ``tokenizers.Tokenizer`` the pad/eos/decode surface."""

    def __init__(self, tokenizer: Any, *, eos_token_id: int) -> None:
        self._tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.pad_token_id = eos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> Any:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=with_added_tokens)

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _train_one_fit(
    *,
    model: Any,
    examples: Sequence[TokenizedExample],
    pad_token_id: int,
    optimization: Mapping[str, Any],
    learning_rate: float,
    expected_rows: int,
    expected_optimizer_steps: int,
    output_dir: Path,
) -> dict[str, Any]:
    """One full-parameter response-only SFT fit (mirrors the matched-lane loop)."""

    import torch

    from barunlm.training.data import collate_sft, deterministic_batches

    device = torch.device("cuda", 0)
    model.train()
    if bool(optimization["gradient_checkpointing"]):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(float(optimization["beta1"]), float(optimization["beta2"])),
        eps=float(optimization["adam_epsilon"]),
        weight_decay=float(optimization["weight_decay"]),
        foreach=False,
        fused=False,
    )
    microbatch_size = int(optimization["per_device_batch_size"])
    accumulation = int(optimization["gradient_accumulation_steps"])
    batches = list(
        deterministic_batches(
            examples,
            batch_size=microbatch_size,
            seed=int(optimization["seed"]),
            epoch=0,
            shuffle=True,
        )
    )
    total_steps = math.ceil(len(batches) / accumulation)
    if total_steps != expected_optimizer_steps:
        raise ScaleSweepError(
            f"optimizer-step budget changed: expected {expected_optimizer_steps}, got {total_steps}"
        )
    warmup_steps = int(optimization["warmup_steps"])
    minimum_ratio = float(optimization["min_learning_rate_ratio"])
    metrics_path = output_dir / "metrics.jsonl"
    started = time.monotonic()
    observed_ids: list[str] = []
    with metrics_path.open("x", encoding="utf-8", newline="\n") as metrics:
        for optimizer_step, group_start in enumerate(range(0, len(batches), accumulation), start=1):
            group = batches[group_start : group_start + accumulation]
            total_target_tokens = sum(
                row.target_tokens for microbatch in group for row in microbatch
            )
            optimizer.zero_grad(set_to_none=True)
            weighted_loss = 0.0
            step_example_ids: list[str] = []
            for microbatch in group:
                batch = collate_sft(microbatch, pad_token_id=pad_token_id).to(device)
                step_example_ids.extend(batch.example_ids)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        labels=batch.labels,
                        use_cache=False,
                    )
                if not bool(torch.isfinite(output.loss).item()):
                    raise ScaleSweepError(
                        f"non-finite training loss at optimizer step {optimizer_step}"
                    )
                scaled_loss = output.loss * (batch.target_tokens / total_target_tokens)
                scaled_loss.backward()
                weighted_loss += float(output.loss.detach().float()) * batch.target_tokens
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(optimization["gradient_clip_norm"]),
                error_if_nonfinite=True,
                foreach=False,
            )
            if optimizer_step <= warmup_steps:
                multiplier = optimizer_step / warmup_steps
            else:
                progress = (optimizer_step - warmup_steps) / (total_steps - warmup_steps)
                multiplier = minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )
            for group_params in optimizer.param_groups:
                group_params["lr"] = learning_rate * multiplier
            optimizer.step()
            observed_ids.extend(step_example_ids)
            record = {
                "optimizer_step": optimizer_step,
                "examples": len(step_example_ids),
                "target_tokens": total_target_tokens,
                "response_nll": weighted_loss / total_target_tokens,
                "gradient_norm": float(gradient_norm.detach().float()),
                "learning_rate": learning_rate * multiplier,
                "elapsed_seconds": time.monotonic() - started,
            }
            metrics.write(_canonical_json(record) + "\n")
            metrics.flush()
    if len(observed_ids) != expected_rows or set(observed_ids) != {
        row.example_id for row in examples
    }:
        raise ScaleSweepError("training did not present every sweep-train ID exactly once")
    return {
        "schema_version": "barun-mobile-scale-sweep-training-v1",
        "method": "full_parameter_response_only_sft",
        "learning_rate": learning_rate,
        "epochs": 1,
        "examples_presented": len(observed_ids),
        "optimizer_steps": total_steps,
        "presentation_order_sha256": _sha256_bytes(("\n".join(observed_ids) + "\n").encode()),
        "elapsed_seconds": time.monotonic() - started,
    }


def _generate_selection(
    *,
    model: Any,
    adapter: _RawTokenizerAdapter,
    rows: Sequence[_RawPromptRow],
    max_seq_len: int,
    max_new_tokens: int,
    batch_size: int,
    output_path: Path,
) -> dict[str, Any]:
    """Greedy deterministic generation over the fresh selection split."""

    import torch

    device = torch.device("cuda", 0)
    model.eval()
    model.config.use_cache = True
    records_by_id: dict[str, dict[str, Any]] = {}
    eligible: list[_RawPromptRow] = []
    for row in rows:
        if row.prompt_tokens + max_new_tokens > max_seq_len:
            records_by_id[row.example_id] = {
                "id": row.example_id,
                "prediction_raw": None,
                "truncated": False,
                "generation_failure": "context_overflow",
                "prompt_tokens": row.prompt_tokens,
                "generated_tokens": 0,
            }
        else:
            eligible.append(row)
    eligible.sort(key=lambda row: (row.prompt_tokens, row.example_id))
    started = time.monotonic()
    for start in range(0, len(eligible), batch_size):
        batch = eligible[start : start + batch_size]
        width = max(row.prompt_tokens for row in batch)
        input_ids = torch.full(
            (len(batch), width), adapter.pad_token_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros((len(batch), width), dtype=torch.bool, device=device)
        for index, row in enumerate(batch):
            input_ids[index, width - row.prompt_tokens :] = torch.tensor(
                row.prompt_ids, dtype=torch.long, device=device
            )
            attention_mask[index, width - row.prompt_tokens :] = True
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=adapter.eos_token_id,
                pad_token_id=adapter.pad_token_id,
                use_cache=True,
            )
        for batch_index, row in enumerate(batch):
            continuation = output_ids[batch_index, width:].tolist()
            eos_position = next(
                (
                    position
                    for position, token_id in enumerate(continuation)
                    if token_id == adapter.eos_token_id
                ),
                None,
            )
            truncated = eos_position is None
            content_ids = continuation if truncated else continuation[:eos_position]
            records_by_id[row.example_id] = {
                "id": row.example_id,
                "prediction_raw": adapter.decode(content_ids, skip_special_tokens=False),
                "truncated": truncated,
                "generation_failure": None,
                "prompt_tokens": row.prompt_tokens,
                "generated_tokens": len(continuation) if truncated else eos_position + 1,
            }
    ordered = [records_by_id[row.example_id] for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in ordered:
            handle.write(_canonical_json(record) + "\n")
    return {
        "schema_version": "barun-mobile-scale-sweep-generation-v1",
        "decoding": "unconstrained_deterministic_greedy",
        "grammar_constrained": False,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "examples": len(rows),
        "generated": sum(record["generation_failure"] is None for record in ordered),
        "failed": sum(record["generation_failure"] is not None for record in ordered),
        "truncated": sum(bool(record["truncated"]) for record in ordered),
        "predictions_sha256": sha256_file(output_path),
        "elapsed_seconds": time.monotonic() - started,
    }


def _score_aggregate(scores_dir: Path) -> dict[str, Any]:
    return json.loads((scores_dir / "aggregate.json").read_text(encoding="utf-8"))


def _fit_outcome_counts(
    generation: Mapping[str, Any], aggregate: Mapping[str, Any]
) -> dict[str, int]:
    exact = aggregate["ast_exact_match"]["value"]
    schema_valid = aggregate["schema_validity"]["value"]
    rows = int(generation["examples"])
    return {
        "rows": rows,
        "exact_match_count": round(float(exact) * rows),
        "schema_valid_count": round(float(schema_valid) * rows),
        "truncated_count": int(generation["truncated"]),
        "generation_failure_count": int(generation["failed"]),
        "missing_prediction_count": rows - int(generation["generated"]) - int(generation["failed"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the frozen sweep: per-arm LR screen on the fresh selection split.

    Requires CUDA and the frozen remote environment; runs only after the
    independent prelaunch audit flips no flag in this module but issues a
    separate signed authorization. Never reads the sealed official rows.
    """

    import torch

    config = load_frozen_config(args.config)
    if args.run_id != config["run_id"]:
        raise ScaleSweepError("CLI run_id differs from the frozen configuration")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ScaleSweepError("the scale sweep requires CUDA with bfloat16 support")

    try:
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer
        from transformers import AutoModelForCausalLM
    except ImportError as error:  # pragma: no cover - exercised on the frozen remote stage
        raise ScaleSweepError("install the frozen remote requirements first") from error

    run_root = Path(args.artifact_root) / args.run_id
    Path(args.artifact_root).mkdir(parents=True, exist_ok=True)
    try:
        run_root.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ExistingScaleSweepRunError(f"refusing to overwrite {run_root}") from error
    export = run_root / "export"
    export.mkdir()
    shutil.copy2(args.config, export / "preregistration.json")

    train_rows = tuple(
        load_manifest(
            args.train_manifest,
            expected_sha256=TRAIN_MANIFEST_SHA256,
            expected_derived_split="train",
        )
    )
    partition = partition_frozen_train_manifest(args.train_manifest)
    verify_partition_against_config(partition, config)
    split_dir = export / "fresh-split"
    write_partition_artifacts(partition, split_dir)
    rows_by_id = {row.example_id: row for row in train_rows}
    sweep_rows = tuple(rows_by_id[example_id] for example_id in partition.sweep_train_ids)
    selection_rows = tuple(rows_by_id[example_id] for example_id in partition.selection_ids)

    optimization = config["optimization"]
    evaluation = config["evaluation"]
    max_seq_len = int(optimization["max_seq_len"])
    max_new_tokens = int(evaluation["max_new_tokens"])
    selection_manifest_path = split_dir / "selection.jsonl"

    arm_results: list[dict[str, Any]] = []
    outcomes: list[ArmOutcome] = []
    for arm in config["roster"]:
        arm_id = arm["arm_id"]
        arm_dir = export / "arms" / arm_id
        arm_dir.mkdir(parents=True)
        snapshot = Path(
            snapshot_download(
                repo_id=arm["repo_id"],
                revision=arm["revision"],
                allow_patterns=[
                    "config.json",
                    "generation_config.json",
                    "model.safetensors",
                    "model-*.safetensors",
                    "model.safetensors.index.json",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ],
                cache_dir=args.cache_dir,
            )
        ).resolve()
        tokenizer_path = snapshot / "tokenizer.json"
        actual_tokenizer_sha = sha256_file(tokenizer_path)
        if actual_tokenizer_sha != arm["tokenizer"]["tokenizer_json_sha256"]:
            raise ScaleSweepError(f"arm {arm_id!r}: pinned tokenizer.json hash changed")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        eos_token_id = int(arm["tokenizer"]["eos_token_id"])
        adapter = _RawTokenizerAdapter(tokenizer, eos_token_id=eos_token_id)
        verify_termination_contract(
            appended_eos_id=eos_token_id,
            generation_eos_id=adapter.eos_token_id,
            pad_token_id=adapter.pad_token_id,
            vocab_size=tokenizer.get_vocab_size(with_added_tokens=True),
        )

        audit = {
            "sweep_train": gold_token_length_audit(sweep_rows, adapter),
            "selection": gold_token_length_audit(selection_rows, adapter),
        }
        frozen_audit = config["gold_token_audit"]["per_tokenizer"][arm["tokenizer"]["audit_key"]]
        for split_name in ("sweep_train", "selection"):
            for field in ("max_target_tokens_with_eos",):
                if audit[split_name][field] != frozen_audit[split_name][field]:
                    raise ScaleSweepError(
                        f"arm {arm_id!r}: {split_name} gold token audit drifted from the "
                        "frozen CPU audit"
                    )
        for split_name in ("sweep_train", "selection"):
            verify_context_fit(
                prompt_tokens_max=audit[split_name]["prompt_tokens"]["max"],
                total_with_eos_max=audit[split_name]["total_with_eos"]["max"],
                max_new_tokens=max_new_tokens,
                context_limit=min(max_seq_len, int(arm["context_length"])),
            )
        _write_json(arm_dir / "gold-token-audit.json", audit)

        tokenized = tokenize_raw_rows(
            sweep_rows, adapter, eos_token_id=eos_token_id, max_seq_len=max_seq_len
        )
        prompt_rows = [
            _RawPromptRow(
                example_id=row.example_id,
                prompt_ids=tuple(_encode_ids(adapter, row.prompt)),
            )
            for row in selection_rows
        ]

        fits: list[LearningRateFit] = []
        fit_records: dict[str, dict[str, Any]] = {}
        for learning_rate in sorted(optimization["learning_rate_screen"]):
            fit_key = f"lr{learning_rate:.0e}"
            fit_dir = arm_dir / fit_key
            fit_dir.mkdir()
            model = AutoModelForCausalLM.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            ).to(torch.device("cuda", 0))
            unique_parameters = count_unique_parameters(model)
            if unique_parameters != int(arm["safetensors_total_parameters"]):
                raise ScaleSweepError(
                    f"arm {arm_id!r}: unique parameter count changed: expected "
                    f"{arm['safetensors_total_parameters']}, got {unique_parameters}"
                )
            training = _train_one_fit(
                model=model,
                examples=tokenized,
                pad_token_id=adapter.pad_token_id,
                optimization=optimization,
                learning_rate=float(learning_rate),
                expected_rows=len(sweep_rows),
                expected_optimizer_steps=int(optimization["expected_optimizer_steps"]),
                output_dir=fit_dir,
            )
            predictions_path = fit_dir / "predictions.jsonl"
            generation = _generate_selection(
                model=model,
                adapter=adapter,
                rows=prompt_rows,
                max_seq_len=max_seq_len,
                max_new_tokens=max_new_tokens,
                batch_size=args.generation_batch_size,
                output_path=predictions_path,
            )
            write_scores(selection_manifest_path, predictions_path, fit_dir / "scores")
            aggregate = _score_aggregate(fit_dir / "scores")
            counts = _fit_outcome_counts(generation, aggregate)
            fit_records[fit_key] = {
                "learning_rate": float(learning_rate),
                "training": training,
                "generation": generation,
                "aggregate": aggregate,
                "counts": counts,
            }
            fits.append(
                LearningRateFit(
                    learning_rate=float(learning_rate),
                    exact_match_count=counts["exact_match_count"],
                    completed=True,
                )
            )
            del model
            torch.cuda.empty_cache()
        selected_rate = select_learning_rate(
            fits, allowed_rates=list(optimization["learning_rate_screen"])
        )
        selected_key = f"lr{selected_rate:.0e}"
        selected_counts = fit_records[selected_key]["counts"]
        outcomes.append(
            ArmOutcome(
                arm_id=arm_id,
                unique_parameters=int(arm["safetensors_total_parameters"]),
                rows=selected_counts["rows"],
                exact_match_count=selected_counts["exact_match_count"],
                schema_valid_count=selected_counts["schema_valid_count"],
                truncated_count=selected_counts["truncated_count"],
                missing_prediction_count=selected_counts["missing_prediction_count"],
                generation_failure_count=selected_counts["generation_failure_count"],
            )
        )
        arm_record = {
            "arm_id": arm_id,
            "repo_id": arm["repo_id"],
            "revision": arm["revision"],
            "selected_learning_rate": selected_rate,
            "fits": fit_records,
        }
        _write_json(arm_dir / "arm-result.json", arm_record)
        arm_results.append(arm_record)

    reference = config["decision_rule"]["reference"]
    reference_exact_percent = float(args.reference_exact_percent)
    decision = decide_adoption(
        outcomes,
        reference_exact_percent=reference_exact_percent,
        margin_points=float(config["decision_rule"]["margin_points"]),
        schema_validity_floor=float(config["decision_rule"]["schema_validity_floor"]),
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "jarvis_machine_id": args.jarvis_machine_id,
        "status": "completed",
        "config_sha256": CONFIG_SHA256,
        "fresh_split": partition.receipt(),
        "arms": arm_results,
        "reference": {
            "description": reference,
            "exact_match_percent": reference_exact_percent,
        },
        "decision": decision,
    }
    _write_json(export / "result.json", result)
    return result


def _run_id_argument(value: str) -> str:
    if _RUN_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must match YYYYMMDD-HHMM-lowercase-name-sN")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=_run_id_argument, required=True)
    parser.add_argument("--jarvis-machine-id", type=int, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument(
        "--reference-exact-percent",
        type=float,
        required=True,
        help="candidate-v2 exact match on the fresh selection split, measured in the "
        "same run before any challenger arm is scored",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, allow_nan=False, sort_keys=True))


__all__ = [
    "AUDIT_SHA256",
    "CONFIG_PATH",
    "CONFIG_SHA256",
    "MOBILE_DATASET_REVISION",
    "OFFICIAL_EVAL_ROWS",
    "OFFICIAL_SOURCE_SHA256",
    "RAW_TRANSPORT_VERSION",
    "RESULT_SCHEMA_VERSION",
    "RUN_ID",
    "SCALE_SWEEP_CONFIG_SCHEMA_VERSION",
    "SELECTION_FOLD",
    "SELECTION_FOLDS",
    "SELECTION_POLICY_VERSION",
    "TRAIN_MANIFEST_SHA256",
    "TRAIN_MEMBERSHIP_SHA256",
    "TRAIN_ROWS",
    "ArmOutcome",
    "ExistingScaleSweepRunError",
    "LearningRateFit",
    "ScaleSweepError",
    "TrainPartition",
    "build_parser",
    "decide_adoption",
    "gold_token_length_audit",
    "load_frozen_config",
    "main",
    "max_new_tokens_from_audit",
    "partition_frozen_train_manifest",
    "select_learning_rate",
    "selection_fold_for_cluster",
    "tokenize_raw_rows",
    "verify_context_fit",
    "verify_partition_against_config",
    "verify_termination_contract",
    "write_partition_artifacts",
]
