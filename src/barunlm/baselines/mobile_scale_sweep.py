"""Matched-adaptation base-model size/token sweep for Mobile Actions.

This module is the CPU-buildable core of run ``20260805-1554-mobile-scale-sweep-s17``
(attempt 6).  It derives the fresh grouped selection split from the frozen
7,937-row internal train manifest, audits gold token lengths per roster
tokenizer, freezes the raw prompt transport and termination contract for base
(non-chat) checkpoints, and binds the preregistered learning-rate screen and
adoption decision rule.  Attempts 1 and 2 were rejected by independent
prelaunch audits; attempt 3 received a go and then failed as an inconclusive
infrastructure mismatch on H200 465155 (``pytorch`` template / CPython 3.10.20
versus required CPython 3.11.10).  Attempt 4 attested ``axolotl`` / CPython
3.11.10 on H200 lineage 465183→465186, completed the in-run candidate-v2
reference evaluation, then aborted on the first challenger because ``jl run``
created ``uv venv --system-site-packages`` and Transformers imported the
axolotl image ``flash_attn_2_cuda`` ABI-mismatched against venv torch 2.13.0.
Attempt 5 / frozen v5 was rejected at independent prelaunch audit
(SHA-256 ``05d40c7d…``) for a forgeable isolation attestation.  This attempt
binds ``mobile_scale_sweep_v6.json``, which keeps every scientific binding
attempt-3/4/5 accepted, retains axolotl / CPython 3.11.10 and
``safe_run --isolated-project-venv``, and closes the three P0 attestation
defects: stdlib-first gate before Torch, interpreter-bound ``pyvenv.cfg``, and
``ModuleNotFoundError``-only / ``find_spec`` flash_attn absence.  Spent
attempt-4 go and rejected v5 are never reused.

The sealed 961-row official Mobile Actions evaluation tail is never an input:
this runner accepts only the already-derived, hash-pinned internal-train
manifests plus the adapter audit that proves the tail stayed opaque.  The reused
756-row development probe is likewise not a selection metric here; scoring uses
only the fresh selection partition derived below.

``transformers``, ``huggingface_hub``, and the Barun generation stack are
imported only inside the functions that need them, keeping every preregistered
rule CPU-testable without network access.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import site
import sys
import sysconfig
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

SCALE_SWEEP_CONFIG_SCHEMA_VERSION = "barun-mobile-scale-sweep-config-v6"
RESULT_SCHEMA_VERSION = "barun-mobile-scale-sweep-result-v6"
RAW_TRANSPORT_VERSION = "barun-raw-prompt-transport-v1"
SELECTION_POLICY_VERSION = "barun-mobile-scale-sweep-selection-v1"

RUN_ID = "20260805-1554-mobile-scale-sweep-s17"
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "mobile_scale_sweep_v6.json"
# Frozen after the attempt-6 CPU build, before any baseline weight download or training.
CONFIG_SHA256 = "0885b32f14751e77539f0bf58cae6f89b7c72e1d80c1b8a6d3fe61cff9c55540"

# Immutable prior evidence; never edited, never loaded by this runner as the active config.
ATTEMPT_1_CONFIG_SHA256 = "d3ee897f9afeefe1e01ec32fe9b2721479b7496785e742d0954ad081758953b8"
ATTEMPT_1_NO_GO_SHA256 = "9bc9d3af9e633b08b6e0d0e1c3bfeedfa660cbe7755443ad58987f47e86e99e0"
ATTEMPT_2_CONFIG_SHA256 = "c8d57f84013198094c27d06d35851e5320f66a5106e6f1cbc6407bfd5e78f593"
ATTEMPT_2_NO_GO_SHA256 = "e693302843678bd6622b149a74320d8ca3bbbba77ea8ab4fb8a065b986ec8ef3"
ATTEMPT_3_CONFIG_SHA256 = "a67959b9b95aa72a6c9153234bb502490c800dba2c539ba9163e37cf4e539451"
ATTEMPT_3_GO_SHA256 = "d74d01e2078322c969db1158c54ebdc7343432635d223d78523226bae452f03b"
ATTEMPT_4_CONFIG_SHA256 = "c89b5c77a5531f617f1acc23e754c27336cf039830e9de7f00d44c8353e8dcb0"
ATTEMPT_4_GO_SHA256 = "9bf60db79cc543e35eb5e664d1a79e761337720c0dbd663da64a8b62c6f4dac2"
ATTEMPT_5_CONFIG_SHA256 = "57dcfe573c17759404545c272f1fc945aabb82b2893f697615eb7fe428d1f2d7"
ATTEMPT_5_NO_GO_SHA256 = "05d40c7d6ab73594fb8e60fb15de94f676e76d62f97ecae43dfcaba7225c0500"
ATTEMPT_1_INFRASTRUCTURE_FAILURE_SHA256 = (
    "429ba84586a9b1ea503138e8defab9e595bdad7a08ec4fd148eee2743eee2855"
)
ATTEMPT_4_INFRASTRUCTURE_FAILURE_SHA256 = (
    "adad51dad1e3216272496dd83072c6d0514d6585da16fe66d56d4944184fa5e0"
)
# Exact attempt-4 remote ImportError shape (flash_attn_2_cuda undefined symbol).
ATTEMPT_4_FLASH_ATTN_IMPORT_ERROR = (
    "undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib"
)

# Frozen provider runtime identity (matches successful axolotl H200 scientific runs).
REQUIRED_PROVIDER_TEMPLATE = "axolotl"
REQUIRED_PYTHON_IMPLEMENTATION = "CPython"
REQUIRED_PYTHON_VERSION = "3.11.10"
REQUIRED_PROTECTED_EVIDENCE_IDS = (465072, 465155, 465183, 465186)
REQUIRED_INCLUDE_SYSTEM_SITE_PACKAGES = False
PYVENV_SYSTEM_SITE_KEY = "include-system-site-packages"

# The exact snapshot_download allow patterns; frozen in the config and validated equal.
SNAPSHOT_ALLOW_PATTERNS = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
# Every challenger snapshot must pin at least these evidence-affecting files.
SNAPSHOT_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)

# Every frozen gold-token-audit field the GPU runner must re-verify per arm.
GOLD_AUDIT_FROZEN_FIELDS = (
    "max_target_tokens_with_eos",
    "prompt_tokens_max",
    "total_with_eos_max",
)
GOLD_AUDIT_SPLITS = ("sweep_train", "selection")

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


class MeasuredFitFailure(ScaleSweepError):
    """One learning-rate fit failed in a preregistered measured-failure class.

    Raised for training divergence (non-finite loss); the run loop also maps
    CUDA out-of-memory errors during a fit into this class.  Integrity
    violations (hash mismatches, budget drift, contract violations) stay plain
    :class:`ScaleSweepError` and abort the whole run.
    """


class ExistingScaleSweepRunError(FileExistsError):
    """An immutable run directory already exists."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash one file without importing the Torch-bearing training package."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    rows: Sequence[Any],
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


def verify_gold_audit_frozen(
    audit: Mapping[str, Any],
    frozen: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Abort on drift in any frozen gold-token-audit field on either split.

    ``audit`` maps split name to a fresh :func:`gold_token_length_audit` result;
    ``frozen`` is the config's per-tokenizer entry, which binds exactly the
    fields in :data:`GOLD_AUDIT_FROZEN_FIELDS` per split.  Every field is
    compared; a single-field check is not sufficient because prompt-side drift
    changes the generation window even when the target maximum is unchanged.
    """

    for split_name in GOLD_AUDIT_SPLITS:
        observed = {
            "max_target_tokens_with_eos": audit[split_name]["max_target_tokens_with_eos"],
            "prompt_tokens_max": audit[split_name]["prompt_tokens"]["max"],
            "total_with_eos_max": audit[split_name]["total_with_eos"]["max"],
        }
        for field in GOLD_AUDIT_FROZEN_FIELDS:
            if observed[field] != frozen[split_name][field]:
                raise ScaleSweepError(
                    f"{label}: {split_name} gold token audit field {field} drifted from "
                    f"the frozen CPU audit: expected {frozen[split_name][field]!r}, "
                    f"got {observed[field]!r}"
                )


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
    rows: Sequence[Any],
    tokenizer: Any,
    *,
    eos_token_id: int,
    max_seq_len: int,
) -> list[Any]:
    """Raw transport: full frozen prompt bytes, no chat template, one appended EOS.

    Base checkpoints receive byte-identical model-visible prompts to BarunAction
    candidate-v2 (including the literal role-marker text), encoded with their own
    tokenizer with ``add_special_tokens=False``.  Prompt and target are encoded
    separately, exactly as at generation time.  Any overlength row aborts the
    run; nothing is dropped or truncated.
    """

    # Import lazily: ``barunlm.training.data`` imports Torch.  Production calls
    # this function only after the stdlib-only isolation boundary has passed.
    from barunlm.training.data import tokenize_examples

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
        target_region = tuple(row.input_ids[row.prompt_tokens :])
        if not target_region:
            raise ScaleSweepError(f"example {example.example_id!r} has an empty supervised region")
        if target_region[-1] != eos_token_id:
            raise ScaleSweepError(f"example {example.example_id!r} lacks the appended EOS")
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

    # These project modules are deliberately imported only after the production
    # entrypoint's stdlib isolation preflight.  Neither imports Torch, but keeping
    # them lazy makes the preflight boundary independently inspectable.
    from barunaction.candidate import CANDIDATE_CHECKPOINT_SHA256, CANDIDATE_ID
    from barunlm.evaluation.mobile_actions import MOBILE_ACTIONS_SCORER_VERSION

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
        unique = arm.get("unique_trainable_parameters")
        total = arm.get("safetensors_total_parameters")
        if type(unique) is not int or unique < 1:
            raise ScaleSweepError(
                f"arm {arm.get('arm_id')!r} lacks a positive unique_trainable_parameters "
                "binding (the value count_unique_parameters is asserted against)"
            )
        if type(total) is not int or total < unique:
            raise ScaleSweepError(
                f"arm {arm.get('arm_id')!r} safetensors_total_parameters must be an integer "
                "at or above the unique trainable count (buffers are non-trainable)"
            )
        validate_snapshot_pins(arm)

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
    fit_semantics = optimization.get("fit_failure_semantics")
    if (
        not isinstance(fit_semantics, dict)
        or fit_semantics.get("per_fit_measured_failure") is not True
    ):
        raise ScaleSweepError(
            "config must freeze per-fit measured-failure semantics "
            "(optimization.fit_failure_semantics.per_fit_measured_failure)"
        )
    if fit_semantics.get("measured_failure_classes") != [
        "cuda_out_of_memory",
        "non_finite_training_loss",
    ]:
        raise ScaleSweepError("frozen measured-failure classes changed")

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ScaleSweepError("config lacks the evaluation section")
    if evaluation.get("scorer_version") != MOBILE_ACTIONS_SCORER_VERSION:
        raise ScaleSweepError("scorer version binding changed")
    if evaluation.get("decoding") != "unconstrained_deterministic_greedy":
        raise ScaleSweepError("primary decoding must stay unconstrained and greedy")
    if evaluation.get("population") != "fresh_selection_split_only":
        raise ScaleSweepError("selection metric population changed")
    for label in ("generation_batch_size", "max_new_tokens"):
        value = evaluation.get(label)
        if type(value) is not int or value < 1:
            raise ScaleSweepError(
                f"evaluation.{label} must be a positive frozen integer; the runner "
                "accepts no CLI override for it"
            )
    decoding_kwargs(evaluation)

    reference = payload.get("reference_evaluation")
    if not isinstance(reference, dict):
        raise ScaleSweepError("config lacks the reference_evaluation contract section")
    if reference.get("candidate_id") != CANDIDATE_ID:
        raise ScaleSweepError("reference evaluation is not bound to candidate-v2")
    if reference.get("scored_before_challenger_arms") is not True:
        raise ScaleSweepError("reference evaluation must be scored before any challenger arm")
    if reference.get("cli_override_forbidden") is not True:
        raise ScaleSweepError(
            "the reference score must never be suppliable through an unauthenticated path"
        )
    reference_hashes = reference.get("checkpoint_sha256")
    if not isinstance(reference_hashes, dict) or dict(reference_hashes) != dict(
        CANDIDATE_CHECKPOINT_SHA256
    ):
        raise ScaleSweepError(
            "reference checkpoint hashes differ from the committed candidate-v2 pin"
        )
    anchors = payload.get("anchors")
    if not isinstance(anchors, dict) or not isinstance(anchors.get("candidate_v2"), dict):
        raise ScaleSweepError("config lacks the candidate_v2 anchor")
    if anchors["candidate_v2"].get("checkpoint_sha256") != dict(CANDIDATE_CHECKPOINT_SHA256):
        raise ScaleSweepError("anchor candidate-v2 checkpoint hashes changed")

    gold_audit = payload.get("gold_token_audit")
    if not isinstance(gold_audit, dict) or not isinstance(gold_audit.get("per_tokenizer"), dict):
        raise ScaleSweepError("config lacks the frozen gold token audit")
    for audit_key, entry in gold_audit["per_tokenizer"].items():
        for split_name in GOLD_AUDIT_SPLITS:
            split_entry = entry.get(split_name) if isinstance(entry, dict) else None
            if not isinstance(split_entry, dict):
                raise ScaleSweepError(f"gold token audit {audit_key!r} lacks {split_name}")
            for field in GOLD_AUDIT_FROZEN_FIELDS:
                if type(split_entry.get(field)) is not int or split_entry[field] < 1:
                    raise ScaleSweepError(
                        f"gold token audit {audit_key!r} {split_name}.{field} must be a "
                        "positive frozen integer"
                    )
    if reference.get("gold_token_audit_key") not in gold_audit["per_tokenizer"]:
        raise ScaleSweepError("reference gold token audit key is not frozen in the config")

    compute = payload.get("compute")
    if not isinstance(compute, dict):
        raise ScaleSweepError("config lacks the compute section")
    _validate_protected_ids(compute.get("protected_machine_ids"))
    _validate_frozen_runtime_attestation(compute)

    return payload


def _validate_protected_ids(protected: Any) -> list[int]:
    """The frozen denylist must be a sorted, duplicate-free, non-empty int list."""

    if (
        not isinstance(protected, list)
        or not protected
        or any(type(value) is not int or value < 1 for value in protected)
        or protected != sorted(set(protected))
        or 463058 not in protected
        or any(machine_id not in protected for machine_id in REQUIRED_PROTECTED_EVIDENCE_IDS)
    ):
        raise ScaleSweepError("config protected machine ID denylist is invalid")
    return protected


def _validate_frozen_runtime_attestation(compute: Mapping[str, Any]) -> None:
    """Config load fails closed unless the provider runtime identity is frozen."""

    expected = {
        "template": REQUIRED_PROVIDER_TEMPLATE,
        "python_implementation": REQUIRED_PYTHON_IMPLEMENTATION,
        "python_version": REQUIRED_PYTHON_VERSION,
        "provider": "JarvisLabs",
        "gpu": "H200",
        "num_gpus": 1,
        "region": "IN2",
        "is_spot": False,
        "storage_gb": 100,
        "max_gpu_job_minutes": 360,
    }
    for field, value in expected.items():
        if compute.get(field) != value:
            raise ScaleSweepError(
                f"compute.{field} must be frozen to {value!r} "
                f"(axolotl / CPython 3.11.10 attestation contract); "
                f"got {compute.get(field)!r}"
            )
    _validate_frozen_venv_isolation(compute.get("venv_isolation"))


def _validate_frozen_venv_isolation(isolation: Any) -> dict[str, Any]:
    """Config load fails closed unless the flash_attn isolation contract is frozen."""

    if not isinstance(isolation, Mapping):
        raise ScaleSweepError("compute.venv_isolation must be a frozen object")
    if isolation.get("approach") != "isolated_project_venv_without_system_site_packages":
        raise ScaleSweepError(
            "compute.venv_isolation.approach must be "
            "'isolated_project_venv_without_system_site_packages'"
        )
    if isolation.get("include_system_site_packages") is not REQUIRED_INCLUDE_SYSTEM_SITE_PACKAGES:
        raise ScaleSweepError("compute.venv_isolation.include_system_site_packages must be false")
    if isolation.get("flash_attn_must_be_unimportable") is not True:
        raise ScaleSweepError("compute.venv_isolation.flash_attn_must_be_unimportable must be true")
    if isolation.get("flash_attn_absence_rule") != "module_not_found_only":
        raise ScaleSweepError(
            "compute.venv_isolation.flash_attn_absence_rule must be 'module_not_found_only'"
        )
    if isolation.get("interpreter_binding") != "sys.prefix_and_sys.executable":
        raise ScaleSweepError(
            "compute.venv_isolation.interpreter_binding must be 'sys.prefix_and_sys.executable'"
        )
    if isolation.get("virtual_env_policy") != "when_set_must_equal_sys.prefix":
        raise ScaleSweepError(
            "compute.venv_isolation.virtual_env_policy must be 'when_set_must_equal_sys.prefix'"
        )
    if isolation.get("discovery") != "importlib.util.find_spec_without_import":
        raise ScaleSweepError(
            "compute.venv_isolation.discovery must be 'importlib.util.find_spec_without_import'"
        )
    fail_closed_before = isolation.get("fail_closed_before")
    if fail_closed_before != [
        "torch_import",
        "challenger_AutoModelForCausalLM_from_pretrained",
    ]:
        raise ScaleSweepError(
            "compute.venv_isolation.fail_closed_before must list torch_import and "
            "challenger_AutoModelForCausalLM_from_pretrained"
        )
    return dict(isolation)


def parse_pyvenv_cfg(path: str | Path) -> dict[str, str]:
    """Parse a ``pyvenv.cfg`` into a flat key/value map (CPU-testable)."""

    cfg_path = Path(path)
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ScaleSweepError(f"unable to read pyvenv.cfg at {cfg_path}: {error}") from error
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def read_include_system_site_packages(pyvenv_cfg: str | Path) -> bool:
    """Return the boolean ``include-system-site-packages`` flag from ``pyvenv.cfg``."""

    values = parse_pyvenv_cfg(pyvenv_cfg)
    raw = values.get(PYVENV_SYSTEM_SITE_KEY)
    if raw is None:
        raise ScaleSweepError(
            f"pyvenv.cfg lacks {PYVENV_SYSTEM_SITE_KEY!r}: isolation contract unenforceable"
        )
    normalized = raw.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ScaleSweepError(
        f"pyvenv.cfg {PYVENV_SYSTEM_SITE_KEY} must be exactly true or false; got {raw!r}"
    )


@dataclass(frozen=True, slots=True)
class FlashAttnSpecObservation:
    """Metadata returned by ``find_spec`` without executing ``flash_attn``."""

    origin: str | None
    search_locations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VenvIsolationObservation:
    """Detached live-process facts consumed by the pure isolation assessor."""

    virtual_env: str | None
    executable: str
    executable_realpath: str
    prefix: str
    base_prefix: str
    base_stdlib_roots: tuple[str, ...]
    cwd: str
    repo_root: str
    source_root: str
    runner_module: str
    runner_module_sha256: str
    pyvenv_cfg: str
    include_system_site_packages: bool
    sys_path: tuple[str, ...]
    python_path: str | None
    user_site_enabled: bool | None
    user_site: str
    flash_attn_spec: FlashAttnSpecObservation | None
    torch_already_imported: bool


def _lexical_absolute_path(value: str, *, label: str) -> str:
    """Return one absolute normalized path without following its final symlink."""

    if not isinstance(value, str) or not value:
        raise ScaleSweepError(f"live {label} must be a nonempty path")
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return os.path.abspath(path)


def _resolved_live_path(value: str, *, label: str) -> str:
    """Resolve one required live path and fail closed when it does not exist."""

    lexical = _lexical_absolute_path(value, label=label)
    try:
        return str(Path(lexical).resolve(strict=True))
    except (OSError, RuntimeError) as error:
        raise ScaleSweepError(f"live {label} cannot be resolved: {lexical}") from error


def _normalized_real_path(value: str, *, label: str) -> str:
    """Resolve existing path components while permitting a missing final path."""

    lexical = _lexical_absolute_path(value, label=label)
    try:
        return str(Path(lexical).resolve(strict=False))
    except (OSError, RuntimeError) as error:
        raise ScaleSweepError(f"live {label} cannot be normalized: {lexical}") from error


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_site_packages_path(path: Path) -> bool:
    return any(part.casefold() in {"site-packages", "dist-packages"} for part in path.parts)


def discover_flash_attn_spec() -> FlashAttnSpecObservation | None:
    """Discover ``flash_attn`` without importing it or executing its native extension."""

    try:
        spec = importlib.util.find_spec("flash_attn")
    except Exception as error:  # fail closed on custom/broken import finders
        raise ScaleSweepError("flash_attn metadata discovery failed") from error
    if spec is None:
        return None
    locations = tuple(
        _normalized_real_path(str(location), label="flash_attn search location")
        for location in (spec.submodule_search_locations or ())
    )
    origin = spec.origin
    if isinstance(origin, str) and origin not in {"built-in", "frozen"}:
        origin = _normalized_real_path(origin, label="flash_attn spec origin")
    return FlashAttnSpecObservation(origin=origin, search_locations=locations)


def classify_flash_attn_import_error(error: BaseException) -> None:
    """Fail closed unless ``error`` is a clean top-level ``ModuleNotFoundError``.

    Attempt-4's ABI mismatch raises ``ImportError`` (undefined symbol on
    ``flash_attn_2_cuda``).  V5 treated every ``ImportError`` as absence; v6
    accepts only ``ModuleNotFoundError`` for top-level ``flash_attn``.
    """

    if isinstance(error, ModuleNotFoundError):
        name = error.name
        if name in {None, "flash_attn"}:
            return
        raise ScaleSweepError(
            "flash_attn ModuleNotFoundError is not a clean top-level absence: "
            f"missing submodule {name!r}"
        ) from error
    if isinstance(error, ImportError):
        detail = str(error)
        raise ScaleSweepError(
            "flash_attn raised ImportError indicating a partial/broken extension "
            "(attempt-4 class: undefined symbol on flash_attn_2_cuda); fail closed: "
            f"{detail}"
        ) from error
    raise ScaleSweepError(
        f"flash_attn probe observed unexpected exception type {type(error).__name__}"
    ) from error


def probe_flash_attn_importable() -> bool:
    """Return whether ``flash_attn`` is discoverable, without importing it.

    Compatibility name retained for tests; discoverable packages are treated as
    importable hazards even when a later import would raise ``ImportError``.
    """

    return discover_flash_attn_spec() is not None


def assert_flash_attn_absent() -> None:
    """Fail closed unless ``find_spec`` proves top-level ``flash_attn`` absent.

    This boundary never imports ``flash_attn``: importing an installed but
    ABI-broken extension would execute the exact attempt-4 failure path before
    the isolation decision exists.
    """

    spec = discover_flash_attn_spec()
    if spec is not None:
        raise ScaleSweepError(
            "flash_attn is discoverable before Torch/CUDA or challenger model load: "
            f"origin={spec.origin!r}, search_locations={list(spec.search_locations)!r}"
        )


def collect_venv_isolation_observation(
    *, require_torch_absent: bool = True
) -> VenvIsolationObservation:
    """Snapshot the actual interpreter/environment for the isolation gate.

    The production entrypoint loads this module via
    ``barunlm.scale_sweep_stdlib_loader`` so ``require_torch_absent=True`` can
    hold before ``baselines`` package init imports Torch.  ``run()`` may re-attest
    the venv/flash_attn contract after that intentional import with
    ``require_torch_absent=False``.
    """

    torch_present = "torch" in sys.modules
    if require_torch_absent and torch_present:
        raise ScaleSweepError(
            "torch is already present in sys.modules at isolation-gate entry; "
            "the stdlib-first boundary was violated"
        )

    # Resolve the venv from the executing interpreter — never trust VIRTUAL_ENV alone.
    prefix = _resolved_live_path(sys.prefix, label="sys.prefix")
    base_prefix = _resolved_live_path(sys.base_prefix, label="sys.base_prefix")
    if prefix == base_prefix:
        raise ScaleSweepError("sys.prefix equals sys.base_prefix; no active isolated venv")
    cfg_path = str(Path(prefix) / "pyvenv.cfg")
    if not Path(cfg_path).is_file():
        raise ScaleSweepError(f"active sys.prefix lacks pyvenv.cfg: {cfg_path}")

    env_root = os.environ.get("VIRTUAL_ENV")
    virtual_env: str | None
    if env_root is None or not str(env_root).strip():
        virtual_env = None
    else:
        if not isinstance(env_root, str):
            raise ScaleSweepError("VIRTUAL_ENV must be a string when set")
        virtual_env = _resolved_live_path(env_root, label="VIRTUAL_ENV")
        if virtual_env != prefix:
            raise ScaleSweepError(
                f"VIRTUAL_ENV {virtual_env} disagrees with executing sys.prefix {prefix}"
            )

    executable = _lexical_absolute_path(sys.executable, label="sys.executable")
    if not Path(executable).is_file():
        raise ScaleSweepError(f"live sys.executable is not a file: {executable}")
    if not _path_is_within(Path(executable), Path(prefix)):
        raise ScaleSweepError(f"sys.executable {executable} is outside active sys.prefix {prefix}")

    normalized_sys_path: list[str] = []
    for index, entry in enumerate(sys.path):
        if not isinstance(entry, str):
            raise ScaleSweepError(f"live sys.path[{index}] is not a string")
        normalized_sys_path.append(
            _normalized_real_path(entry or str(Path.cwd()), label=f"sys.path[{index}]")
        )
    user_site = site.getusersitepackages()
    if not isinstance(user_site, str) or not user_site:
        raise ScaleSweepError("live user-site path is unavailable")
    runner_module = _resolved_live_path(__file__, label="scale-sweep runner module")
    source_root = str(Path(runner_module).parents[2])
    repo_root = str(Path(runner_module).parents[3])
    base_paths = sysconfig.get_paths(vars={"base": base_prefix, "platbase": base_prefix})
    stdlib_roots = {
        _normalized_real_path(value, label=f"sys.base_prefix {name}")
        for name, value in base_paths.items()
        if name in {"stdlib", "platstdlib"}
    }
    for root in tuple(stdlib_roots):
        root_path = Path(root)
        stdlib_roots.add(
            str(root_path.parent / f"python{sys.version_info.major}{sys.version_info.minor}.zip")
        )

    # Confirm absence with ModuleNotFoundError-only classification after find_spec.
    assert_flash_attn_absent()

    return VenvIsolationObservation(
        virtual_env=virtual_env,
        executable=executable,
        executable_realpath=_resolved_live_path(sys.executable, label="sys.executable realpath"),
        prefix=prefix,
        base_prefix=base_prefix,
        base_stdlib_roots=tuple(sorted(stdlib_roots)),
        cwd=_resolved_live_path(str(Path.cwd()), label="current working directory"),
        repo_root=repo_root,
        source_root=source_root,
        runner_module=runner_module,
        runner_module_sha256=sha256_file(runner_module),
        pyvenv_cfg=cfg_path,
        include_system_site_packages=read_include_system_site_packages(cfg_path),
        sys_path=tuple(normalized_sys_path),
        python_path=os.environ.get("PYTHONPATH"),
        user_site_enabled=site.ENABLE_USER_SITE,
        user_site=_normalized_real_path(user_site, label="user-site"),
        flash_attn_spec=None,
        torch_already_imported=torch_present if require_torch_absent else False,
    )


def _observed_absolute_path(value: Any, *, label: str) -> Path:
    """Validate an already-normalized path without touching the filesystem."""

    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ScaleSweepError(f"observed {label} must be a nonempty absolute path")
    normalized = Path(os.path.normpath(value))
    if str(normalized) != value:
        raise ScaleSweepError(f"observed {label} must be normalized")
    return normalized


def assess_venv_isolation(
    *, compute: Mapping[str, Any], observation: VenvIsolationObservation
) -> dict[str, Any]:
    """Purely assess a detached observation against the frozen isolation contract."""

    isolation = _validate_frozen_venv_isolation(compute.get("venv_isolation"))
    if not isinstance(observation, VenvIsolationObservation):
        raise ScaleSweepError("venv isolation observation has the wrong type")
    if observation.torch_already_imported:
        raise ScaleSweepError(
            "torch is already present in sys.modules at isolation-gate entry; "
            "the stdlib-first boundary was violated"
        )

    executable = _observed_absolute_path(observation.executable, label="sys.executable")
    executable_realpath = _observed_absolute_path(
        observation.executable_realpath, label="sys.executable realpath"
    )
    prefix = _observed_absolute_path(observation.prefix, label="sys.prefix")
    base_prefix = _observed_absolute_path(observation.base_prefix, label="sys.base_prefix")
    cwd = _observed_absolute_path(observation.cwd, label="current working directory")
    repo_root = _observed_absolute_path(observation.repo_root, label="staged repository root")
    source_root = _observed_absolute_path(observation.source_root, label="staged source root")
    runner_module = _observed_absolute_path(
        observation.runner_module, label="scale-sweep runner module"
    )
    cfg_path = _observed_absolute_path(observation.pyvenv_cfg, label="pyvenv.cfg")

    if observation.virtual_env is None:
        venv_path = prefix
        virtual_env_receipt: str | None = None
    else:
        venv_path = _observed_absolute_path(observation.virtual_env, label="VIRTUAL_ENV")
        if prefix != venv_path:
            raise ScaleSweepError(
                f"active sys.prefix {prefix} does not equal VIRTUAL_ENV {venv_path}"
            )
        virtual_env_receipt = str(venv_path)

    if prefix == base_prefix:
        raise ScaleSweepError("sys.prefix equals sys.base_prefix; no active isolated venv")
    if cwd != repo_root:
        raise ScaleSweepError(
            f"current working directory {cwd} does not equal staged repository root {repo_root}"
        )
    if source_root != repo_root / "src":
        raise ScaleSweepError("staged source root is not the exact src child of repository root")
    expected_runner = source_root / "barunlm" / "baselines" / "mobile_scale_sweep.py"
    if runner_module != expected_runner:
        raise ScaleSweepError(
            f"runner module {runner_module} is not the expected staged source {expected_runner}"
        )
    if (
        not isinstance(observation.runner_module_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", observation.runner_module_sha256) is None
    ):
        raise ScaleSweepError("runner module observation lacks an exact SHA-256")
    expected_runner_sha256 = isolation.get("runner_module_sha256")
    if expected_runner_sha256 is not None:
        if (
            not isinstance(expected_runner_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_runner_sha256) is None
        ):
            raise ScaleSweepError("frozen runner_module_sha256 is invalid")
        if observation.runner_module_sha256 != expected_runner_sha256:
            raise ScaleSweepError(
                "live runner module SHA-256 differs from the frozen isolation contract"
            )
    if not _path_is_within(executable, prefix):
        raise ScaleSweepError(f"sys.executable {executable} is outside active sys.prefix {prefix}")
    if not _path_is_within(executable_realpath, base_prefix):
        raise ScaleSweepError(
            "sys.executable realpath is outside the attested sys.base_prefix: "
            f"{executable_realpath} not under {base_prefix}"
        )
    if cfg_path != prefix / "pyvenv.cfg":
        raise ScaleSweepError("observed pyvenv.cfg is not the active sys.prefix configuration")
    if observation.include_system_site_packages is not REQUIRED_INCLUDE_SYSTEM_SITE_PACKAGES:
        raise ScaleSweepError(
            "live pyvenv.cfg include-system-site-packages does not match the frozen "
            f"isolation contract: observed {observation.include_system_site_packages!r}, "
            f"expected {REQUIRED_INCLUDE_SYSTEM_SITE_PACKAGES!r} "
            "(attempt-4 failure class: jl uv venv --system-site-packages exposed "
            "flash_attn_2_cuda into Transformers)"
        )
    if observation.python_path not in {None, ""}:
        raise ScaleSweepError(
            "PYTHONPATH must be unset for the frozen venv isolation contract; observed "
            f"{observation.python_path!r}"
        )
    if observation.user_site_enabled is not False:
        raise ScaleSweepError(
            f"Python user-site loading must be disabled; observed {observation.user_site_enabled!r}"
        )

    user_site = _observed_absolute_path(observation.user_site, label="user-site")
    base_stdlib_roots = tuple(
        _observed_absolute_path(value, label=f"base stdlib root[{index}]")
        for index, value in enumerate(observation.base_stdlib_roots)
    )
    if not base_stdlib_roots:
        raise ScaleSweepError("no sys.base_prefix stdlib roots were observed")
    for root in base_stdlib_roots:
        if not _path_is_within(root, base_prefix) or _is_site_packages_path(root):
            raise ScaleSweepError(f"invalid sys.base_prefix stdlib root: {root}")
        if root.suffix == ".zip":
            valid_shape = root.parent == base_prefix / "lib" and re.fullmatch(
                r"python\d+\.zip", root.name
            )
        else:
            valid_shape = root.parent == base_prefix / "lib" and re.fullmatch(
                r"python\d+\.\d+", root.name
            )
        if not valid_shape:
            raise ScaleSweepError(f"unrecognized sys.base_prefix stdlib root shape: {root}")
    observed_sys_path = tuple(
        _observed_absolute_path(value, label=f"sys.path[{index}]")
        for index, value in enumerate(observation.sys_path)
    )
    if not observed_sys_path:
        raise ScaleSweepError("live sys.path must not be empty")
    venv_site_paths = tuple(
        path
        for path in observed_sys_path
        if _path_is_within(path, prefix) and _is_site_packages_path(path)
    )
    if not venv_site_paths:
        raise ScaleSweepError("live sys.path lacks the active venv site-packages directory")
    allowed_project_paths = {repo_root, source_root}
    disallowed_sys_paths = tuple(
        path
        for path in observed_sys_path
        if not _path_is_within(path, prefix)
        and not any(_path_is_within(path, root) for root in base_stdlib_roots)
        and path not in allowed_project_paths
    )
    if disallowed_sys_paths:
        raise ScaleSweepError(
            "sys.path contains entries outside the frozen venv/base-stdlib/project allowlist: "
            + ", ".join(str(path) for path in disallowed_sys_paths)
        )
    if any(path == user_site or _path_is_within(path, user_site) for path in observed_sys_path):
        raise ScaleSweepError(f"user-site path is visible on sys.path: {user_site}")

    if observation.flash_attn_spec is not None:
        spec = observation.flash_attn_spec
        if not isinstance(spec, FlashAttnSpecObservation):
            raise ScaleSweepError("flash_attn spec observation has the wrong type")
        raise ScaleSweepError(
            "flash_attn is discoverable before Torch/CUDA or challenger model load: "
            f"origin={spec.origin!r}, search_locations={list(spec.search_locations)!r}"
        )

    return {
        "approach": isolation["approach"],
        "virtual_env": virtual_env_receipt,
        "pyvenv_cfg": str(cfg_path),
        "sys_executable": str(executable),
        "sys_executable_realpath": str(executable_realpath),
        "sys_prefix": str(prefix),
        "sys_base_prefix": str(base_prefix),
        "base_stdlib_roots": [str(path) for path in base_stdlib_roots],
        "cwd": str(cwd),
        "repo_root": str(repo_root),
        "source_root": str(source_root),
        "runner_module": str(runner_module),
        "runner_module_sha256": observation.runner_module_sha256,
        "include_system_site_packages": False,
        "python_path": None,
        "user_site_enabled": False,
        "user_site": str(user_site),
        "sys_path": [str(path) for path in observed_sys_path],
        "external_site_packages": [],
        "flash_attn_discoverable": False,
        "flash_attn_spec": None,
        "torch_already_imported": False,
    }


def enforce_venv_isolation(
    *, compute: Mapping[str, Any], require_torch_absent: bool = True
) -> dict[str, Any]:
    """Production boundary: assess only actual live-process observations."""

    return assess_venv_isolation(
        compute=compute,
        observation=collect_venv_isolation_observation(require_torch_absent=require_torch_absent),
    )


def _load_preflight_compute(path: str | Path) -> dict[str, Any]:
    """Load only the config fields needed by the stdlib-first isolation boundary."""

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
    if payload.get("run_id") != RUN_ID:
        raise ScaleSweepError("config run_id is not the frozen immutable run ID")
    compute = payload.get("compute")
    if not isinstance(compute, Mapping):
        raise ScaleSweepError("config compute contract is missing")
    _validate_frozen_runtime_attestation(compute)
    return dict(compute)


def stdlib_isolation_preflight(
    path: str | Path = CONFIG_PATH, *, require_torch_absent: bool = True
) -> dict[str, Any]:
    """Run the production isolation boundary without importing any project dependency."""

    return enforce_venv_isolation(
        compute=_load_preflight_compute(path),
        require_torch_absent=require_torch_absent,
    )


def enforce_machine_id(machine_id: Any, protected_machine_ids: Any) -> int:
    """Reject any protected or malformed JarvisLabs machine ID before any work.

    Mirrors the mobile_qwen05b_matched lane: the runner may operate only on a
    fresh, positive, exact project-created machine ID that does not appear in
    the frozen protected denylist.  A protected ID aborts before any download,
    training, or evaluation step.
    """

    protected = _validate_protected_ids(protected_machine_ids)
    if type(machine_id) is not int or machine_id < 1:
        raise ScaleSweepError("jarvis machine ID must be a positive integer")
    if machine_id in protected:
        raise ScaleSweepError(f"jarvis machine ID {machine_id} is protected; refusing to run on it")
    return machine_id


def enforce_runtime_attestation(
    *,
    compute: Mapping[str, Any],
    observed_template: Any,
    observed_python_implementation: Any | None = None,
    observed_python_version: Any | None = None,
) -> dict[str, str]:
    """Fail closed at the earliest gate when live template/Python differ from freeze.

    ``safe_run`` already attests the live JarvisLabs template and remote Python
    identity before upload.  This runner independently re-checks the same
    contract from the frozen config plus the live process identity so a
    mismatched ``pytorch`` / 3.10 host cannot reach CUDA, download, training, or
    scoring even if the controller gate were bypassed.
    """

    _validate_frozen_runtime_attestation(compute)
    if observed_template != REQUIRED_PROVIDER_TEMPLATE:
        raise ScaleSweepError(
            "live provider template attestation does not match the frozen "
            f"axolotl contract: observed {observed_template!r}, "
            f"expected {REQUIRED_PROVIDER_TEMPLATE!r}"
        )
    implementation = (
        REQUIRED_PYTHON_IMPLEMENTATION
        if observed_python_implementation is None
        else observed_python_implementation
    )
    version = (
        platform.python_version() if observed_python_version is None else observed_python_version
    )
    if implementation != REQUIRED_PYTHON_IMPLEMENTATION:
        raise ScaleSweepError(
            "live Python implementation attestation does not match the frozen "
            f"CPython contract: observed {implementation!r}, "
            f"expected {REQUIRED_PYTHON_IMPLEMENTATION!r}"
        )
    if version != REQUIRED_PYTHON_VERSION:
        raise ScaleSweepError(
            "live Python version attestation does not match the frozen "
            f"3.11.10 contract: observed {version!r}, "
            f"expected {REQUIRED_PYTHON_VERSION!r}"
        )
    return {
        "template": REQUIRED_PROVIDER_TEMPLATE,
        "python_implementation": REQUIRED_PYTHON_IMPLEMENTATION,
        "python_version": REQUIRED_PYTHON_VERSION,
    }


# ---------------------------------------------------------------------------
# Challenger snapshot binding (attempt-3 P0-3 correction)
# ---------------------------------------------------------------------------

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def validate_snapshot_pins(arm: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate one roster arm's frozen snapshot pin table.

    Every pinned file must carry an exact lowercase SHA-256 and a positive
    byte size and must be reachable through the frozen download allow
    patterns; the evidence-affecting files (config.json, model.safetensors,
    tokenizer.json, tokenizer_config.json) must all be pinned; and the arm's
    tokenizer/config hash bindings must agree with the pin table, so the
    config cannot advertise a binding that enforcement does not cover.
    """

    arm_id = arm.get("arm_id")
    if arm.get("snapshot_allow_patterns") != list(SNAPSHOT_ALLOW_PATTERNS):
        raise ScaleSweepError(
            f"arm {arm_id!r} snapshot_allow_patterns differ from the frozen download patterns"
        )
    pins = arm.get("snapshot_files")
    if not isinstance(pins, Mapping) or not pins:
        raise ScaleSweepError(f"arm {arm_id!r} lacks the snapshot_files pin table")
    for name, entry in pins.items():
        if not isinstance(name, str) or not any(
            fnmatch.fnmatchcase(name, pattern) for pattern in SNAPSHOT_ALLOW_PATTERNS
        ):
            raise ScaleSweepError(
                f"arm {arm_id!r} pins {name!r}, which no frozen allow pattern covers"
            )
        if not isinstance(entry, Mapping):
            raise ScaleSweepError(f"arm {arm_id!r} snapshot pin {name!r} must be an object")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ScaleSweepError(
                f"arm {arm_id!r} snapshot pin {name!r} lacks a 64-hex lowercase sha256"
            )
        if type(entry.get("bytes")) is not int or entry["bytes"] < 1:
            raise ScaleSweepError(
                f"arm {arm_id!r} snapshot pin {name!r} lacks a positive byte size"
            )
    for required in SNAPSHOT_REQUIRED_FILES:
        if required not in pins:
            raise ScaleSweepError(
                f"arm {arm_id!r} does not pin required snapshot file {required!r}"
            )
    tokenizer_block = arm.get("tokenizer")
    if not isinstance(tokenizer_block, Mapping):
        raise ScaleSweepError(f"arm {arm_id!r} lacks the tokenizer block")
    for key, filename in (
        ("tokenizer_json_sha256", "tokenizer.json"),
        ("tokenizer_config_sha256", "tokenizer_config.json"),
        ("config_json_sha256", "config.json"),
    ):
        if tokenizer_block.get(key) != pins[filename]["sha256"]:
            raise ScaleSweepError(
                f"arm {arm_id!r} tokenizer.{key} disagrees with the snapshot pin for {filename!r}"
            )
    return {str(name): dict(entry) for name, entry in pins.items()}


def verify_challenger_snapshot(snapshot_dir: str | Path, arm: Mapping[str, Any]) -> dict[str, str]:
    """Verify every downloaded challenger snapshot file against the frozen pins.

    Enforcement iterates the config's own ``snapshot_files`` table, so no
    advertised pin can be left unenforced.  Exact file-set equality proves
    both pinned absences (pythia-70m-deduped has no generation_config.json at
    its pinned revision) and the absence of unexpected model-affecting files;
    each present file must match its pinned byte size and SHA-256.  A
    pre-staged cache directory with substituted challenger bytes therefore
    aborts here, before any tokenizer or weight load, and the verified hashes
    are returned for binding into arm-result.json and result.json.
    """

    snapshot = Path(snapshot_dir)
    pins = validate_snapshot_pins(arm)
    arm_id = arm.get("arm_id")
    actual_files = sorted(
        path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*") if path.is_file()
    )
    expected_files = sorted(pins)
    if actual_files != expected_files:
        raise ScaleSweepError(
            f"arm {arm_id!r}: snapshot file set {actual_files} does not equal the "
            f"pinned set {expected_files}"
        )
    verified: dict[str, str] = {}
    for name in expected_files:
        file_path = snapshot / name
        size = file_path.stat().st_size
        if size != pins[name]["bytes"]:
            raise ScaleSweepError(
                f"arm {arm_id!r}: snapshot file {name!r} is {size} bytes; the pin "
                f"froze {pins[name]['bytes']}"
            )
        digest = sha256_file(file_path)
        if digest != pins[name]["sha256"]:
            raise ScaleSweepError(
                f"arm {arm_id!r}: snapshot file {name!r} hash {digest} differs from "
                f"the pinned {pins[name]['sha256']}"
            )
        verified[name] = digest
    return verified


def decoding_kwargs(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Explicit frozen decoding parameters for every challenger ``generate()`` call.

    Every decoding-affecting field is passed explicitly at the call site so a
    snapshot's ``generation_config.json`` (verified or pinned-absent) can never
    influence decoding.  The frozen overrides must stay exactly the greedy
    contract below; the null-valued fields are sampling-only and must remain
    unset under greedy decoding, so they are excluded from the kwargs.
    """

    overrides = evaluation.get("generation_config_overrides")
    frozen = {
        "do_sample": False,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "temperature": None,
        "top_k": None,
        "top_p": None,
    }
    if not isinstance(overrides, Mapping) or dict(overrides) != frozen:
        raise ScaleSweepError(
            "evaluation.generation_config_overrides changed from the frozen greedy contract"
        )
    return {key: value for key, value in overrides.items() if value is not None}


def build_decision(
    outcomes: Sequence[ArmOutcome],
    *,
    measured_failed_arm_ids: Sequence[str],
    reference_exact_percent: float,
    margin_points: float,
    schema_validity_floor: float,
) -> dict[str, Any]:
    """Assemble the final decision with per-fit measured-failure capture.

    Scored arms flow through :func:`decide_adoption` unchanged.  An arm whose
    every learning-rate fit ended in a measured failure can never be adopted
    and is recorded explicitly; if every arm failed, the sweep is falsified by
    measured failure with no adopted arm rather than by silent omission.
    """

    failed = [str(arm_id) for arm_id in measured_failed_arm_ids]
    if len(set(failed)) != len(failed):
        raise ScaleSweepError("measured-failed arm IDs must be unique")
    if not outcomes and not failed:
        raise ScaleSweepError("decision requires at least one scored or measured-failed arm")
    if outcomes:
        decision = decide_adoption(
            outcomes,
            reference_exact_percent=reference_exact_percent,
            margin_points=margin_points,
            schema_validity_floor=schema_validity_floor,
        )
    else:
        if not math.isfinite(reference_exact_percent) or not 0 <= reference_exact_percent <= 100:
            raise ScaleSweepError("reference exact-match percent must be in [0, 100]")
        decision = {
            "schema_version": "barun-mobile-scale-sweep-decision-v1",
            "reference_exact_percent": reference_exact_percent,
            "margin_points": margin_points,
            "schema_validity_floor": schema_validity_floor,
            "ordering": "ascending unique parameter count; smallest passing arm is adopted",
            "arms": [],
            "adopted_arm_id": None,
            "all_arms_falsified": True,
        }
    scored_ids = {entry["arm_id"] for entry in decision["arms"]}
    overlap = scored_ids.intersection(failed)
    if overlap:
        raise ScaleSweepError(f"arms {sorted(overlap)} cannot be both scored and measured failures")
    decision["measured_failed_arm_ids"] = sorted(failed)
    decision["measured_failure_rule"] = (
        "an arm whose every learning-rate fit ended in a measured failure "
        "(non-finite training loss or CUDA out-of-memory) can never be adopted; "
        "the run continues past it and records the failure"
    )
    return decision


# Evidence-affecting packages whose exact versions are bound into result.json.
ENVIRONMENT_PACKAGES = ("huggingface_hub", "safetensors", "tokenizers", "torch", "transformers")


def environment_versions(packages: Sequence[str] = ENVIRONMENT_PACKAGES) -> dict[str, str]:
    """Bind the evidence-affecting dependency versions into the result.

    A missing required package is an abort, never a silent omission; the
    launch tooling's environment.json remains the complete environment record.
    """

    import platform
    from importlib import metadata

    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise ScaleSweepError(f"required package {package!r} is not installed") from error
    return versions


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
    examples: Sequence[Any],
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
                    raise MeasuredFitFailure(
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
    decoding: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Greedy deterministic generation over the fresh selection split.

    ``decoding`` is the exact output of :func:`decoding_kwargs`: every frozen
    decoding-affecting parameter is passed explicitly to ``model.generate`` so
    no snapshot-side generation_config.json default can influence decoding.
    """

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
                max_new_tokens=max_new_tokens,
                eos_token_id=adapter.eos_token_id,
                pad_token_id=adapter.pad_token_id,
                use_cache=True,
                **decoding,
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
        "decoding_overrides_passed_explicitly": dict(decoding),
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


def fit_outcome_counts(
    generation: Mapping[str, Any], aggregate: Mapping[str, Any]
) -> dict[str, int]:
    """Join one generation summary with the frozen scorer's aggregate output.

    Counts are consumed from the scorer's exact ``Rate`` numerators (the
    ``barun-mobile-actions-score-v1`` aggregate serializes ``schema_valid``,
    ``ast_exact_match``, ``truncation``, and ``missing_prediction`` as
    numerator/denominator/value records); nothing is reconstructed from rounded
    float products.  The generation summary and the scorer must agree on the
    row and truncation counts or the join aborts.
    """

    rows = int(generation["examples"])
    if rows < 1:
        raise ScaleSweepError("generation summary reports zero examples")
    rates: dict[str, Mapping[str, Any]] = {}
    for label in ("ast_exact_match", "schema_valid", "truncation", "missing_prediction"):
        rate = aggregate.get(label)
        if not isinstance(rate, Mapping):
            raise ScaleSweepError(f"scorer aggregate lacks the {label} rate record")
        numerator = rate.get("numerator")
        denominator = rate.get("denominator")
        if type(numerator) is not int or type(denominator) is not int:
            raise ScaleSweepError(f"scorer {label} rate lacks exact integer counts")
        if denominator != rows:
            raise ScaleSweepError(
                f"scorer {label} denominator {denominator} disagrees with the {rows} generated rows"
            )
        if not 0 <= numerator <= denominator:
            raise ScaleSweepError(f"scorer {label} numerator is out of range")
        rates[label] = rate
    truncated_count = int(rates["truncation"]["numerator"])
    if truncated_count != int(generation["truncated"]):
        raise ScaleSweepError("scorer truncation count disagrees with the generation summary")
    generation_failure_count = int(generation["failed"])
    missing_count = int(rates["missing_prediction"]["numerator"])
    if missing_count < generation_failure_count:
        raise ScaleSweepError(
            "scorer missing-prediction count is below the recorded generation failures"
        )
    return {
        "rows": rows,
        "exact_match_count": int(rates["ast_exact_match"]["numerator"]),
        "schema_valid_count": int(rates["schema_valid"]["numerator"]),
        "truncated_count": truncated_count,
        "generation_failure_count": generation_failure_count,
        "missing_prediction_count": missing_count - generation_failure_count,
    }


def verify_reference_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_sha256: Mapping[str, str],
) -> dict[str, str]:
    """Verify the candidate-v2 checkpoint against its frozen file hashes.

    Delegates to the release-grade :func:`barunlm.evaluation.generation.verify_checkpoint`,
    which additionally cross-checks the checkpoint's own signed manifest when
    one is present.  Any missing file, hash mismatch, or manifest disagreement
    aborts before the reference score can exist.
    """

    from barunlm.evaluation.generation import GenerationError, verify_checkpoint

    try:
        return verify_checkpoint(checkpoint_dir, expected_sha256=dict(expected_sha256))
    except GenerationError as error:
        raise ScaleSweepError(f"reference checkpoint verification failed: {error}") from error


def _evaluate_reference(
    *,
    config: Mapping[str, Any],
    checkpoint_dir: Path,
    selection_manifest_path: Path,
    sweep_rows: Sequence[Any],
    selection_rows: Sequence[Any],
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    """Score hash-verified candidate-v2 on the fresh selection split, in-run.

    This is the only path that can produce the decision-critical reference
    number: the checkpoint files are verified against the committed pin, the
    frozen barun-tokenizer gold-token audit is recomputed and compared field by
    field, generation uses the same greedy decoding and frozen budget as every
    challenger arm, and the predictions plus scores are bound into the result.
    """

    from tokenizers import Tokenizer

    from barunaction.candidate import CANDIDATE_ID
    from barunlm.evaluation.generation import GenerationError, generate_manifest
    from barunlm.evaluation.mobile_actions import write_scores

    reference_cfg = config["reference_evaluation"]
    evaluation = config["evaluation"]
    expected_hashes = {
        str(name): str(digest) for name, digest in reference_cfg["checkpoint_sha256"].items()
    }
    verified_hashes = verify_reference_checkpoint(checkpoint_dir, expected_sha256=expected_hashes)

    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
    audit = {
        "sweep_train": gold_token_length_audit(sweep_rows, tokenizer),
        "selection": gold_token_length_audit(selection_rows, tokenizer),
    }
    audit_key = reference_cfg["gold_token_audit_key"]
    verify_gold_audit_frozen(
        audit,
        config["gold_token_audit"]["per_tokenizer"][audit_key],
        label=f"reference {CANDIDATE_ID}",
    )

    output_dir.mkdir(parents=True)
    predictions_path = output_dir / "predictions.jsonl"
    try:
        summary = generate_manifest(
            checkpoint_dir=checkpoint_dir,
            manifest_path=selection_manifest_path,
            manifest_sha256=config["evaluation"]["population_manifest_sha256"],
            predictions_path=predictions_path,
            device_name=device_name,
            batch_size=int(evaluation["generation_batch_size"]),
            max_new_tokens=int(evaluation["max_new_tokens"]),
            expected_checkpoint_sha256=expected_hashes,
        )
    except GenerationError as error:
        raise ScaleSweepError(f"reference generation failed: {error}") from error
    generation = summary.to_dict()
    write_scores(selection_manifest_path, predictions_path, output_dir / "scores")
    aggregate = _score_aggregate(output_dir / "scores")
    counts = fit_outcome_counts(generation, aggregate)
    exact_match_percent = 100.0 * counts["exact_match_count"] / counts["rows"]
    record = {
        "schema_version": "barun-mobile-scale-sweep-reference-v1",
        "candidate_id": CANDIDATE_ID,
        "checkpoint_sha256": verified_hashes,
        "gold_token_audit": audit,
        "generation": generation,
        "aggregate": aggregate,
        "counts": counts,
        "exact_match_percent": exact_match_percent,
        "scored_before_challenger_arms": True,
        "bias_note": config["decision_rule"]["reference_bias_note"],
    }
    _write_json(output_dir / "reference-result.json", record)
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the frozen sweep: per-arm LR screen on the fresh selection split.

    Requires CUDA and the frozen remote environment; runs only after the
    independent prelaunch audit flips no flag in this module but issues a
    separate signed authorization. Never reads the sealed official rows.
    """

    # Re-attest venv/flash_attn after the entrypoint's stdlib-first gate. Torch may
    # already be present once this module was imported through ``baselines`` package
    # init; the entrypoint proves the pre-Torch boundary via
    # ``barunlm.scale_sweep_stdlib_loader`` before that import.
    venv_isolation = stdlib_isolation_preflight(args.config, require_torch_absent=False)
    config = load_frozen_config(args.config)
    if args.run_id != config["run_id"]:
        raise ScaleSweepError("CLI run_id differs from the frozen configuration")
    # Earliest attestation gates: protected ID, axolotl/CPython 3.11.10, then
    # isolated venv / flash_attn unimportable. All must pass before Torch/CUDA.
    enforce_machine_id(args.jarvis_machine_id, config["compute"]["protected_machine_ids"])
    runtime_attestation = enforce_runtime_attestation(
        compute=config["compute"],
        observed_template=args.jarvis_template,
        observed_python_implementation=platform.python_implementation(),
        observed_python_version=platform.python_version(),
    )
    # Both modules below import Torch and therefore must remain after the repeated
    # production isolation boundary.
    import torch

    from barunlm.baselines.mobile_matched import count_unique_parameters
    from barunlm.evaluation.mobile_actions import write_scores
    from barunlm.training.data import load_manifest

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ScaleSweepError("the scale sweep requires CUDA with bfloat16 support")
    environment = environment_versions()
    environment["provider_template"] = runtime_attestation["template"]
    environment["python_implementation"] = runtime_attestation["python_implementation"]
    environment["venv_isolation"] = venv_isolation
    # No stochastic module exists on the honest path (no dropout, greedy decoding,
    # no weight init), but the global torch seed is installed anyway so the frozen
    # seed governs every torch RNG, not only the data presentation order.
    torch.manual_seed(int(config["optimization"]["seed"]))
    torch.cuda.manual_seed_all(int(config["optimization"]["seed"]))

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
    # The scientific config keeps its own name; the run preregistration lives in the
    # repository run directory and is a different document.
    shutil.copy2(args.config, export / "config.json")

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
    generation_batch_size = int(evaluation["generation_batch_size"])
    decoding = decoding_kwargs(evaluation)
    selection_manifest_path = split_dir / "selection.jsonl"

    # The decision-critical reference number exists only through this in-run,
    # hash-verified evaluation of candidate-v2, scored before any challenger arm.
    reference_record = _evaluate_reference(
        config=config,
        checkpoint_dir=Path(args.reference_checkpoint).resolve(),
        selection_manifest_path=selection_manifest_path,
        sweep_rows=sweep_rows,
        selection_rows=selection_rows,
        output_dir=export / "reference",
        device_name="cuda",
    )
    reference_exact_percent = float(reference_record["exact_match_percent"])

    arm_results: list[dict[str, Any]] = []
    outcomes: list[ArmOutcome] = []
    measured_failed_arm_ids: list[str] = []
    for arm in config["roster"]:
        arm_id = arm["arm_id"]
        arm_dir = export / "arms" / arm_id
        arm_dir.mkdir(parents=True)
        snapshot = Path(
            snapshot_download(
                repo_id=arm["repo_id"],
                revision=arm["revision"],
                allow_patterns=list(SNAPSHOT_ALLOW_PATTERNS),
                cache_dir=args.cache_dir,
            )
        ).resolve()
        # Every snapshot file the loader can read is verified against the frozen
        # pins before any tokenizer or weight load; the verified hashes are bound
        # into arm-result.json and result.json.
        verified_snapshot = verify_challenger_snapshot(snapshot, arm)
        _write_json(arm_dir / "snapshot-verification.json", verified_snapshot)
        tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
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
        verify_gold_audit_frozen(audit, frozen_audit, label=f"arm {arm_id!r}")
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
            model = None
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    snapshot,
                    local_files_only=True,
                    trust_remote_code=False,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",
                ).to(torch.device("cuda", 0))
                unique_parameters = count_unique_parameters(model)
                if unique_parameters != int(arm["unique_trainable_parameters"]):
                    raise ScaleSweepError(
                        f"arm {arm_id!r}: unique trainable parameter count changed: expected "
                        f"{arm['unique_trainable_parameters']}, got {unique_parameters}"
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
                    batch_size=generation_batch_size,
                    decoding=decoding,
                    output_path=predictions_path,
                )
            except MeasuredFitFailure as error:
                failure = {"class": "non_finite_training_loss", "reason": str(error)}
            except torch.cuda.OutOfMemoryError as error:
                failure = {"class": "cuda_out_of_memory", "reason": str(error)}
            else:
                failure = None
            finally:
                if model is not None:
                    del model
                torch.cuda.empty_cache()
            if failure is not None:
                # Preregistered per-fit measured failure: the fit can never win,
                # the run continues, and the failure is bound into the evidence.
                fit_records[fit_key] = {
                    "learning_rate": float(learning_rate),
                    "completed": False,
                    "measured_failure": failure,
                }
                _write_json(fit_dir / "measured-failure.json", fit_records[fit_key])
                fits.append(
                    LearningRateFit(
                        learning_rate=float(learning_rate),
                        exact_match_count=0,
                        completed=False,
                    )
                )
                continue
            write_scores(selection_manifest_path, predictions_path, fit_dir / "scores")
            aggregate = _score_aggregate(fit_dir / "scores")
            counts = fit_outcome_counts(generation, aggregate)
            fit_records[fit_key] = {
                "learning_rate": float(learning_rate),
                "completed": True,
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
        if any(fit.completed for fit in fits):
            selected_rate = select_learning_rate(
                fits, allowed_rates=list(optimization["learning_rate_screen"])
            )
            selected_key = f"lr{selected_rate:.0e}"
            selected_counts = fit_records[selected_key]["counts"]
            outcomes.append(
                ArmOutcome(
                    arm_id=arm_id,
                    unique_parameters=int(arm["unique_trainable_parameters"]),
                    rows=selected_counts["rows"],
                    exact_match_count=selected_counts["exact_match_count"],
                    schema_valid_count=selected_counts["schema_valid_count"],
                    truncated_count=selected_counts["truncated_count"],
                    missing_prediction_count=selected_counts["missing_prediction_count"],
                    generation_failure_count=selected_counts["generation_failure_count"],
                )
            )
        else:
            selected_rate = None
            measured_failed_arm_ids.append(arm_id)
        arm_record = {
            "arm_id": arm_id,
            "repo_id": arm["repo_id"],
            "revision": arm["revision"],
            "snapshot_sha256": verified_snapshot,
            "measured_failure": selected_rate is None,
            "selected_learning_rate": selected_rate,
            "fits": fit_records,
        }
        _write_json(arm_dir / "arm-result.json", arm_record)
        arm_results.append(arm_record)

    decision = build_decision(
        outcomes,
        measured_failed_arm_ids=measured_failed_arm_ids,
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
        "environment": environment,
        "fresh_split": partition.receipt(),
        "arms": arm_results,
        "reference": reference_record,
        "decision": decision,
    }
    _write_json(export / "result.json", result)
    return result


def _run_id_argument(value: str) -> str:
    if _RUN_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must match YYYYMMDD-HHMM-lowercase-name-sN")
    return value


def _machine_id_argument(value: str) -> int:
    try:
        machine_id = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if machine_id < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return machine_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=_run_id_argument, required=True)
    parser.add_argument("--jarvis-machine-id", type=_machine_id_argument, required=True)
    parser.add_argument(
        "--jarvis-template",
        required=True,
        help="live JarvisLabs provider template attested by safe_run before upload; "
        "must equal the frozen axolotl contract or the runner aborts before CUDA work",
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        required=True,
        help="directory holding the candidate-v2 checkpoint; its files are verified "
        "against the frozen SHA-256 pins before the in-run reference evaluation. "
        "There is deliberately no argument that can supply the reference score "
        "itself, and the generation batch size is read from the frozen config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, allow_nan=False, sort_keys=True))


__all__ = [
    "ATTEMPT_1_CONFIG_SHA256",
    "ATTEMPT_1_INFRASTRUCTURE_FAILURE_SHA256",
    "ATTEMPT_1_NO_GO_SHA256",
    "ATTEMPT_2_CONFIG_SHA256",
    "ATTEMPT_2_NO_GO_SHA256",
    "ATTEMPT_3_CONFIG_SHA256",
    "ATTEMPT_3_GO_SHA256",
    "ATTEMPT_4_CONFIG_SHA256",
    "ATTEMPT_4_GO_SHA256",
    "ATTEMPT_4_INFRASTRUCTURE_FAILURE_SHA256",
    "AUDIT_SHA256",
    "CONFIG_PATH",
    "CONFIG_SHA256",
    "ENVIRONMENT_PACKAGES",
    "GOLD_AUDIT_FROZEN_FIELDS",
    "GOLD_AUDIT_SPLITS",
    "MOBILE_DATASET_REVISION",
    "OFFICIAL_EVAL_ROWS",
    "OFFICIAL_SOURCE_SHA256",
    "PYVENV_SYSTEM_SITE_KEY",
    "RAW_TRANSPORT_VERSION",
    "REQUIRED_INCLUDE_SYSTEM_SITE_PACKAGES",
    "REQUIRED_PROTECTED_EVIDENCE_IDS",
    "REQUIRED_PROVIDER_TEMPLATE",
    "REQUIRED_PYTHON_IMPLEMENTATION",
    "REQUIRED_PYTHON_VERSION",
    "RESULT_SCHEMA_VERSION",
    "RUN_ID",
    "SCALE_SWEEP_CONFIG_SCHEMA_VERSION",
    "SELECTION_FOLD",
    "SELECTION_FOLDS",
    "SELECTION_POLICY_VERSION",
    "SNAPSHOT_ALLOW_PATTERNS",
    "SNAPSHOT_REQUIRED_FILES",
    "TRAIN_MANIFEST_SHA256",
    "TRAIN_MEMBERSHIP_SHA256",
    "TRAIN_ROWS",
    "ArmOutcome",
    "ExistingScaleSweepRunError",
    "LearningRateFit",
    "MeasuredFitFailure",
    "ScaleSweepError",
    "assert_flash_attn_absent",
    "assess_venv_isolation",
    "build_decision",
    "build_parser",
    "classify_flash_attn_import_error",
    "decide_adoption",
    "decoding_kwargs",
    "discover_flash_attn_spec",
    "enforce_machine_id",
    "enforce_runtime_attestation",
    "enforce_venv_isolation",
    "environment_versions",
    "fit_outcome_counts",
    "gold_token_length_audit",
    "load_frozen_config",
    "main",
    "max_new_tokens_from_audit",
    "parse_pyvenv_cfg",
    "partition_frozen_train_manifest",
    "probe_flash_attn_importable",
    "read_include_system_site_packages",
    "run",
    "select_learning_rate",
    "selection_fold_for_cluster",
    "stdlib_isolation_preflight",
    "tokenize_raw_rows",
    "validate_snapshot_pins",
    "verify_challenger_snapshot",
    "verify_context_fit",
    "verify_gold_audit_frozen",
    "verify_partition_against_config",
    "verify_reference_checkpoint",
    "verify_termination_contract",
    "write_partition_artifacts",
]
