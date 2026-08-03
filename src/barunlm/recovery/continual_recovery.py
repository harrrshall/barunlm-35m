"""One-shot Mobile recovery with bounded PRESTO train-only rehearsal.

The implementation deliberately has no development-loss API, checkpoint chooser, retry
branch, or hyperparameter override. Development rows are used only for correctness-blind exact
duplicate exclusion and once for the terminal semantic evaluation performed by the runner.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

import torch
from safetensors.torch import save_model

from barunlm.evaluation.generation import load_verified_model
from barunlm.evaluation.presto import (
    PRESTO_PHENOMENON_TAXONOMY_VERSION,
    PRESTO_SCORER_VERSION,
    PRESTO_USER_REVISION_ALIAS_SET_VERSION,
    PRESTO_USER_REVISION_RAW_LABELS_V2,
    phenomenon_group,
)
from barunlm.model import BarunLM
from barunlm.training.data import (
    SFTExample,
    collate_sft,
    deterministic_batches,
    load_manifest,
    sha256_file,
    tokenize_examples,
)

RECOVERY_CONFIG_VERSION = "barun-continual-recovery-preregistration-v1"
RECOVERY_VIEW_VERSION = "barun-continual-recovery-view-v1"
RECOVERY_TRAINING_VERSION = "barun-continual-recovery-training-v1"
RECOVERY_CONFIG_SHA256 = "b45f6efb28af44e0a2772b9239aff3f321a38765b964f1925e5bfc47c24fbf0d"
RECOVERY_SELECTOR_VERSION = "barun-continual-recovery-replay-v1"
RECOVERY_RECIPE_ID = "barunaction-mobile-presto-recovery-v1"
SHA256_HEX = frozenset("0123456789abcdef")


class RecoveryError(RuntimeError):
    """A frozen recovery input or invariant changed."""


def _fail(message: str) -> NoReturn:
    raise RecoveryError(message)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA256_HEX


def _reject_constant(value: str) -> NoReturn:
    _fail(f"JSON contains non-finite constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except RecoveryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryError(f"cannot read strict JSON from {source}: {error}") from error
    if not isinstance(payload, dict):
        _fail(f"JSON artifact must contain an object: {source}")
    return payload


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be an object")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def _expect(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        _fail(f"{name} changed: expected {expected!r}, got {actual!r}")


def _canonical_json(payload: object) -> str:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise RecoveryError(f"value is not strict JSON: {error}") from error


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        _fail(f"refusing to overwrite recovery artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _membership_sha256(example_ids: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(example_ids)) + "\n").encode("utf-8")).hexdigest()


def _content_identity(example: SFTExample) -> str:
    return hashlib.sha256(
        _canonical_json({"prompt": example.prompt, "target": example.target}).encode("utf-8")
    ).hexdigest()


def load_recovery_config(path: str | Path) -> dict[str, Any]:
    """Load and fail-closed validate the only preregistered recovery recipe."""

    source = Path(path)
    actual_sha256 = sha256_file(source)
    if actual_sha256 != RECOVERY_CONFIG_SHA256:
        _fail(
            "continual-recovery preregistration SHA-256 mismatch: "
            f"expected {RECOVERY_CONFIG_SHA256}, got {actual_sha256}"
        )
    config = _strict_json(source)
    _expect(config.get("schema_version"), RECOVERY_CONFIG_VERSION, "config schema")
    _expect(config.get("recipe_id"), RECOVERY_RECIPE_ID, "recipe ID")
    _expect(config.get("status"), "preregistered_unlaunched", "recipe status")

    lineage = _mapping(config.get("lineage"), "lineage")
    _expect(lineage.get("input_run_id"), "20260803-1920-presto-stage-s17", "input run")
    input_hashes = _mapping(
        lineage.get("input_checkpoint_file_sha256"), "input checkpoint hashes"
    )
    _expect(
        dict(input_hashes),
        {
            "barun_config.json": (
                "9b3a1d71baa95a198744d250f9629231738d942570b8685c44307fd83dd33565"
            ),
            "model.safetensors": (
                "83c78952e719456574d2bb808fb486d0320519dd4f5df3c67736ae17931ee357"
            ),
            "tokenizer.json": (
                "70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6"
            ),
        },
        "input checkpoint lineage",
    )

    data = _mapping(config.get("data"), "data")
    mobile = _mapping(data.get("mobile"), "Mobile data")
    mobile_train = _mapping(mobile.get("train"), "Mobile train")
    mobile_dev = _mapping(mobile.get("development"), "Mobile development")
    _expect(mobile_train.get("rows"), 7_937, "Mobile train rows")
    _expect(
        mobile_train.get("sha256"),
        "131473ccb5bfb51cac0439b42159e72ec4c598025e50364a52b122b056c2e84e",
        "Mobile train hash",
    )
    _expect(mobile_dev.get("rows"), 756, "Mobile development rows")
    _expect(
        mobile_dev.get("sha256"),
        "988bdce5874d1f1a775feeb5ba2b58cd2bdc128f57e73cb9a63d535fae7c1d55",
        "Mobile development hash",
    )
    mobile_firewall = _mapping(
        mobile.get("official_evaluation_firewall"), "Mobile official-evaluation firewall"
    )
    _expect(mobile_firewall.get("accepted_input_paths"), [], "Mobile official paths")
    _expect(mobile_firewall.get("opaque_unparsed"), True, "Mobile official opacity")

    presto = _mapping(data.get("presto"), "PRESTO data")
    _expect(
        presto.get("revision"),
        "fa47167477453afebe698a287409514df5a7dadf",
        "PRESTO revision",
    )
    _expect(_mapping(presto.get("train"), "PRESTO train").get("rows"), 47_806, "PRESTO train rows")
    _expect(
        _mapping(presto.get("development"), "PRESTO development").get("rows"),
        14_288,
        "PRESTO development rows",
    )

    view = _mapping(config.get("recovery_view"), "recovery view")
    _expect(view.get("selector_version"), RECOVERY_SELECTOR_VERSION, "selector version")
    _expect(view.get("mobile_rows"), 7_937, "recovery Mobile rows")
    _expect(view.get("presto_replay_rows"), 2_646, "PRESTO replay rows")
    _expect(view.get("combined_rows"), 10_583, "combined rows")
    _expect(view.get("official_test_rows_read"), 0, "official test access")
    selection = _mapping(view.get("presto_selection"), "PRESTO replay selection")
    _expect(selection.get("development_correctness_used"), False, "development correctness")
    _expect(selection.get("exact_prompt_target_dev_exclusion"), True, "dev exclusion")
    _expect(selection.get("duplicate_source_rows"), False, "duplicate replay source rows")
    _expect(
        set(selection.get("all_four_user_revision_raw_tags_required", ())),
        set(PRESTO_USER_REVISION_RAW_LABELS_V2),
        "full PRESTO user-revision family",
    )

    optimization = _mapping(config.get("optimization"), "optimization")
    frozen_optimization = {
        "seed": 17,
        "epochs": 1,
        "max_seq_len": 2_048,
        "batch_size": 63,
        "gradient_accumulation_steps": 1,
        "expected_optimizer_steps": 168,
        "learning_rate": 5e-5,
        "min_learning_rate_ratio": 0.1,
        "warmup_steps": 16,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "adam_epsilon": 1e-8,
        "gradient_clip_norm": 1.0,
        "checkpoint_policy": "final_only",
        "development_loss_evaluation": False,
        "early_stopping": False,
        "overlength_policy": "error",
    }
    for name, expected in frozen_optimization.items():
        _expect(optimization.get(name), expected, f"optimization.{name}")

    terminal = _mapping(config.get("terminal_evaluation"), "terminal evaluation")
    _expect(terminal.get("checkpoint_evaluations"), 1, "terminal checkpoint evaluations")
    _expect(terminal.get("checkpoint_policy"), "final_only", "terminal checkpoint policy")
    _expect(terminal.get("official_mobile_evaluation_rows_read"), 0, "Mobile official rows")
    _expect(terminal.get("official_presto_test_rows_read"), 0, "PRESTO official rows")
    presto_eval = _mapping(terminal.get("presto"), "PRESTO terminal evaluation")
    _expect(presto_eval.get("scorer_version"), PRESTO_SCORER_VERSION, "PRESTO scorer")
    _expect(
        presto_eval.get("taxonomy_version"),
        PRESTO_PHENOMENON_TAXONOMY_VERSION,
        "PRESTO taxonomy",
    )
    _expect(
        presto_eval.get("revision_alias_set_version"),
        PRESTO_USER_REVISION_ALIAS_SET_VERSION,
        "PRESTO alias set",
    )
    return config


def _contextual(example: SFTExample) -> bool:
    try:
        context_text = example.prompt.split("\nCONTEXT ", 1)[1].split("\nDIALOGUE ", 1)[0]
        dialogue_text = example.prompt.split("\nDIALOGUE ", 1)[1].split("\n<user>\n", 1)[0]
        context = json.loads(context_text)
        dialogue = json.loads(dialogue_text)
    except (IndexError, json.JSONDecodeError) as error:
        raise RecoveryError(
            f"PRESTO replay row {example.example_id!r} has malformed context"
        ) from error
    if not isinstance(context, dict) or not isinstance(dialogue, list):
        _fail(f"PRESTO replay row {example.example_id!r} has malformed context types")
    return bool(dialogue or any(context.get(name) for name in ("contacts", "lists", "notes")))


def _replay_stratum(example: SFTExample) -> str:
    decision = example.metadata.get("policy_decision")
    raw = example.metadata.get("linguistic_phenomenon")
    if decision not in {"ABSTAIN", "CALL", "CONFIRM"} or not isinstance(raw, str):
        _fail(f"PRESTO replay row {example.example_id!r} lacks policy/taxonomy metadata")
    group = phenomenon_group(raw)
    family = f"revision:{raw}" if raw in PRESTO_USER_REVISION_RAW_LABELS_V2 else group
    context = "contextual" if _contextual(example) else "noncontextual"
    return f"{decision}|{family}|{context}"


def _allocate_strata(
    strata: Mapping[str, Sequence[SFTExample]], *, target_rows: int
) -> dict[str, int]:
    nonempty = {name: len(rows) for name, rows in strata.items() if rows}
    if target_rows < len(nonempty):
        _fail("PRESTO replay target cannot cover every nonempty joint stratum")
    available = sum(nonempty.values())
    if target_rows > available:
        _fail(f"PRESTO replay target {target_rows} exceeds {available} eligible train rows")
    allocation = {name: 1 for name in nonempty}
    remaining = target_rows - len(nonempty)
    capacity = {name: count - 1 for name, count in nonempty.items()}
    total_capacity = sum(capacity.values())
    if remaining and total_capacity < 1:
        _fail("PRESTO replay allocation has no remaining capacity")
    fractional: list[tuple[float, str]] = []
    assigned = 0
    for name in sorted(nonempty):
        ideal = remaining * capacity[name] / total_capacity if total_capacity else 0.0
        floor = min(capacity[name], math.floor(ideal))
        allocation[name] += floor
        assigned += floor
        fractional.append((ideal - floor, name))
    leftovers = remaining - assigned
    for _, name in sorted(fractional, key=lambda item: (-item[0], item[1])):
        if leftovers == 0:
            break
        if allocation[name] < nonempty[name]:
            allocation[name] += 1
            leftovers -= 1
    if leftovers:
        _fail("largest-remainder replay allocation did not reach the frozen target")
    if sum(allocation.values()) != target_rows:
        _fail("PRESTO replay allocation changed its total")
    return allocation


def _manifest_record(
    example: SFTExample,
    *,
    example_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "barun-sft-example-v1",
        "id": example.example_id if example_id is None else example_id,
        "prompt": example.prompt,
        "target": example.target,
        "metadata": dict(example.metadata if metadata is None else metadata),
    }


def _write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    if path.exists():
        _fail(f"refusing to overwrite recovery manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(_canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def materialize_recovery_view(
    *,
    mobile_train_path: str | Path,
    mobile_train_sha256: str,
    mobile_train_rows: int,
    mobile_membership_sha256: str,
    presto_train_path: str | Path,
    presto_train_sha256: str,
    presto_train_rows: int,
    presto_dev_path: str | Path,
    presto_dev_sha256: str,
    presto_dev_rows: int,
    replay_rows: int,
    seed: int,
    output_manifest: str | Path,
    output_audit: str | Path,
) -> dict[str, Any]:
    """Create the deterministic one-epoch Mobile plus train-only PRESTO replay view."""

    mobile = load_manifest(
        mobile_train_path,
        expected_sha256=mobile_train_sha256,
        expected_derived_split="train",
    )
    presto_train = load_manifest(
        presto_train_path,
        expected_sha256=presto_train_sha256,
        expected_derived_split="train",
    )
    presto_dev = load_manifest(
        presto_dev_path,
        expected_sha256=presto_dev_sha256,
        expected_derived_split="dev",
    )
    _expect(len(mobile), mobile_train_rows, "Mobile recovery train rows")
    _expect(len(presto_train), presto_train_rows, "PRESTO recovery train rows")
    _expect(len(presto_dev), presto_dev_rows, "PRESTO recovery development rows")
    mobile_ids = [row.example_id for row in mobile]
    _expect(_membership_sha256(mobile_ids), mobile_membership_sha256, "Mobile membership")

    dev_content = {_content_identity(row) for row in presto_dev}
    excluded = [row for row in presto_train if _content_identity(row) in dev_content]
    eligible = [row for row in presto_train if _content_identity(row) not in dev_content]
    if not eligible:
        _fail("PRESTO exact development exclusion removed every train row")
    strata: dict[str, list[SFTExample]] = defaultdict(list)
    for row in eligible:
        strata[_replay_stratum(row)].append(row)
    allocation = _allocate_strata(strata, target_rows=replay_rows)

    selected: list[tuple[str, SFTExample]] = []
    selected_by_stratum: dict[str, list[str]] = {}
    for stratum in sorted(allocation):
        ranked = sorted(
            strata[stratum],
            key=lambda row: (
                hashlib.sha256(
                    f"{seed}:{RECOVERY_SELECTOR_VERSION}:{stratum}:{row.example_id}".encode()
                ).hexdigest(),
                row.example_id,
            ),
        )
        chosen = ranked[: allocation[stratum]]
        selected.extend((stratum, row) for row in chosen)
        selected_by_stratum[stratum] = [row.example_id for row in chosen]
    if len(selected) != replay_rows or len({row.example_id for _, row in selected}) != replay_rows:
        _fail("PRESTO replay must contain the frozen number of unique source rows")

    selected_revision_counts = Counter(
        str(row.metadata.get("linguistic_phenomenon"))
        for _, row in selected
        if row.metadata.get("linguistic_phenomenon") in PRESTO_USER_REVISION_RAW_LABELS_V2
    )
    if set(selected_revision_counts) != set(PRESTO_USER_REVISION_RAW_LABELS_V2):
        _fail("PRESTO replay did not cover the full corrected user-revision family")

    combined: list[Mapping[str, Any]] = [_manifest_record(row) for row in mobile]
    replay_ids: list[str] = []
    for stratum, row in sorted(selected, key=lambda item: (item[0], item[1].example_id)):
        replay_id = f"recovery-presto-{row.example_id}"
        replay_ids.append(replay_id)
        metadata = dict(row.metadata)
        metadata["continual_recovery_replay"] = {
            "selector_version": RECOVERY_SELECTOR_VERSION,
            "source_id": row.example_id,
            "stratum": stratum,
        }
        combined.append(_manifest_record(row, example_id=replay_id, metadata=metadata))
    combined_ids = mobile_ids + replay_ids
    if len(combined_ids) != len(set(combined_ids)):
        _fail("combined recovery view contains duplicate IDs")
    if any(_content_identity(row) in dev_content for _, row in selected):
        _fail("PRESTO development duplicate entered the replay view")

    output_path = Path(output_manifest)
    combined_sha256 = _write_manifest(output_path, combined)
    audit = {
        "schema_version": RECOVERY_VIEW_VERSION,
        "selector_version": RECOVERY_SELECTOR_VERSION,
        "seed": seed,
        "mobile": {
            "source_path": str(Path(mobile_train_path).resolve()),
            "source_sha256": mobile_train_sha256,
            "rows": len(mobile),
            "membership_sha256": _membership_sha256(mobile_ids),
            "presentation_policy": "each source ID exactly once",
        },
        "presto": {
            "train_source_path": str(Path(presto_train_path).resolve()),
            "train_source_sha256": presto_train_sha256,
            "train_rows": len(presto_train),
            "development_source_path": str(Path(presto_dev_path).resolve()),
            "development_source_sha256": presto_dev_sha256,
            "development_rows_read_for_exact_duplicate_exclusion": len(presto_dev),
            "development_correctness_used": False,
            "exact_prompt_target_train_rows_excluded": len(excluded),
            "excluded_source_ids": sorted(row.example_id for row in excluded),
            "eligible_rows": len(eligible),
            "replay_rows": len(selected),
            "replay_source_membership_sha256": _membership_sha256(
                [row.example_id for _, row in selected]
            ),
            "joint_stratum_eligible_counts": {
                name: len(strata[name]) for name in sorted(strata)
            },
            "joint_stratum_replay_counts": dict(sorted(allocation.items())),
            "joint_stratum_selected_membership_sha256": {
                name: _membership_sha256(selected_by_stratum[name])
                for name in sorted(selected_by_stratum)
            },
            "selected_user_revision_raw_tag_counts": dict(
                sorted(selected_revision_counts.items())
            ),
            "corrected_taxonomy_version": PRESTO_PHENOMENON_TAXONOMY_VERSION,
            "revision_alias_set_version": PRESTO_USER_REVISION_ALIAS_SET_VERSION,
            "official_test_rows_read": 0,
        },
        "combined": {
            "manifest": str(output_path.resolve()),
            "sha256": combined_sha256,
            "rows": len(combined),
            "membership_sha256": _membership_sha256(combined_ids),
            "mobile_rows": len(mobile),
            "presto_replay_rows": len(selected),
        },
    }
    _atomic_json(Path(output_audit), audit)
    return audit


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _optimizer(model: BarunLM, optimization: Mapping[str, Any]) -> torch.optim.AdamW:
    decay: list[torch.Tensor] = []
    no_decay: list[torch.Tensor] = []
    seen: set[int] = set()
    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(optimization["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(optimization["learning_rate"]),
        betas=(float(optimization["beta1"]), float(optimization["beta2"])),
        eps=float(optimization["adam_epsilon"]),
        foreach=False,
        fused=False,
    )


def _learning_rate(step: int, *, total_steps: int, optimization: Mapping[str, Any]) -> float:
    if not 1 <= step <= total_steps:
        _fail("optimizer step is outside the frozen recovery schedule")
    base = float(optimization["learning_rate"])
    warmup = int(optimization["warmup_steps"])
    if step <= warmup:
        return base * step / warmup
    progress = (step - warmup) / (total_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = float(optimization["min_learning_rate_ratio"])
    return base * (floor + (1.0 - floor) * cosine)


def _set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _append_metric(handle: Any, payload: Mapping[str, Any]) -> None:
    handle.write(_canonical_json(payload) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def train_continual_recovery(
    *,
    run_id: str,
    checkpoint_dir: str | Path,
    checkpoint_sha256: Mapping[str, str],
    combined_manifest: str | Path,
    combined_manifest_sha256: str,
    mobile_ids: Sequence[str],
    replay_ids: Sequence[str],
    optimization: Mapping[str, Any],
    output_dir: str | Path,
    device_name: str,
) -> dict[str, Any]:
    """Train exactly one fixed recipe and save only its final inference checkpoint."""

    if device_name != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        _fail("continual recovery requires an explicitly available bfloat16 CUDA device")
    output = Path(output_dir)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise RecoveryError(f"refusing to overwrite recovery training output {output}") from error

    model, tokenizer, verified_hashes = load_verified_model(
        checkpoint_dir, expected_sha256=checkpoint_sha256
    )
    if model.config.mtp_loss_weight != 0:
        _fail("response-only continual recovery requires mtp_loss_weight=0")
    examples = load_manifest(
        combined_manifest,
        expected_sha256=combined_manifest_sha256,
        expected_derived_split="train",
    )
    expected_ids = set(mobile_ids) | set(replay_ids)
    if len(expected_ids) != len(mobile_ids) + len(replay_ids):
        _fail("Mobile and PRESTO recovery IDs overlap")
    if {row.example_id for row in examples} != expected_ids:
        _fail("combined recovery manifest membership differs from its materialization audit")

    eos_token_id = tokenizer.token_to_id("<eos>")
    pad_token_id = tokenizer.token_to_id("<pad>")
    if eos_token_id is None or pad_token_id is None or eos_token_id == pad_token_id:
        _fail("recovery tokenizer lacks distinct <eos> and <pad> IDs")
    tokenized, rejected = tokenize_examples(
        examples,
        tokenizer,
        eos_token_id=eos_token_id,
        max_seq_len=int(optimization["max_seq_len"]),
    )
    if rejected:
        _atomic_json(
            output / "overlength-rejections.json",
            {"rows": [row.to_dict() for row in rejected]},
        )
        _fail(f"recovery refuses {len(rejected)} overlength examples")
    if len(tokenized) != len(examples):
        _fail("recovery tokenization silently changed the training population")

    seed = int(optimization["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda", 0)
    model.to(device=device, dtype=torch.float32).train()
    optimizer = _optimizer(model, optimization)
    batches = list(
        deterministic_batches(
            tokenized,
            batch_size=int(optimization["batch_size"]),
            seed=seed,
            epoch=0,
            shuffle=True,
        )
    )
    total_steps = len(batches)
    _expect(total_steps, int(optimization["expected_optimizer_steps"]), "optimizer steps")

    metrics_path = output / "metrics.jsonl"
    observed_ids: list[str] = []
    started = time.monotonic()
    with metrics_path.open("x", encoding="utf-8", newline="\n") as metrics:
        for step, rows in enumerate(batches, start=1):
            batch = collate_sft(rows, pad_token_id=pad_token_id).to(device)
            optimizer.zero_grad(set_to_none=True)
            learning_rate = _learning_rate(
                step, total_steps=total_steps, optimization=optimization
            )
            _set_learning_rate(optimizer, learning_rate)
            with _autocast(device):
                model_output = model(
                    batch.input_ids,
                    labels=batch.labels,
                    attention_mask=batch.attention_mask,
                )
            loss = model_output.loss
            if loss is None or not bool(torch.isfinite(loss).item()):
                _fail(f"non-finite or missing recovery loss at step {step}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(optimization["gradient_clip_norm"]),
                error_if_nonfinite=True,
                foreach=False,
            )
            optimizer.step()
            observed_ids.extend(batch.example_ids)
            _append_metric(
                metrics,
                {
                    "elapsed_seconds": time.monotonic() - started,
                    "examples": len(batch.example_ids),
                    "gradient_norm": float(gradient_norm.detach().float()),
                    "learning_rate": learning_rate,
                    "optimizer_step": step,
                    "response_nll": float(loss.detach().float()),
                    "target_tokens": batch.target_tokens,
                },
            )
    torch.cuda.synchronize(device)
    duration = time.monotonic() - started
    observed = Counter(observed_ids)
    if set(observed) != expected_ids or any(count != 1 for count in observed.values()):
        _fail("recovery did not present every combined training ID exactly once")
    if any(observed.get(sample_id) != 1 for sample_id in mobile_ids):
        _fail("one exact Mobile train epoch invariant failed")
    if any(observed.get(sample_id) != 1 for sample_id in replay_ids):
        _fail("PRESTO replay presentation invariant failed")

    checkpoint = output / "checkpoint"
    checkpoint.mkdir()
    save_model(model, checkpoint / "model.safetensors")
    model.config.save_json(checkpoint / "barun_config.json")
    tokenizer.save(str(checkpoint / "tokenizer.json"))
    checkpoint_hashes = {
        name: sha256_file(checkpoint / name)
        for name in ("barun_config.json", "model.safetensors", "tokenizer.json")
    }
    _atomic_json(
        checkpoint / "checkpoint_manifest.json",
        {
            "schema_version": "barun-release-checkpoint-v1",
            "run_id": run_id,
            "source_checkpoint": str(Path(checkpoint_dir).resolve()),
            "input_checkpoint_sha256": dict(verified_hashes),
            "file_sha256": checkpoint_hashes,
            "lineage_stage": "continual_recovery",
        },
    )
    optimizer_path = output / "optimizer.pt"
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "optimizer_steps": total_steps,
            "seed": seed,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            "input_checkpoint_sha256": dict(verified_hashes),
            "combined_manifest_sha256": combined_manifest_sha256,
        },
        optimizer_path,
    )
    summary = {
        "schema_version": RECOVERY_TRAINING_VERSION,
        "method": "full_parameter_response_only_sft_with_train_only_rehearsal",
        "checkpoint_policy": "final_only",
        "development_evaluations_during_training": 0,
        "epochs": 1,
        "seed": seed,
        "optimizer_steps": total_steps,
        "examples_presented": len(observed_ids),
        "unique_examples_presented": len(observed),
        "mobile_examples_presented_exactly_once": len(mobile_ids),
        "presto_replay_examples_presented_exactly_once": len(replay_ids),
        "presentation_order_sha256": hashlib.sha256(
            ("\n".join(observed_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "combined_manifest": str(Path(combined_manifest).resolve()),
        "combined_manifest_sha256": combined_manifest_sha256,
        "input_checkpoint": {
            "directory": str(Path(checkpoint_dir).resolve()),
            "file_sha256": dict(verified_hashes),
        },
        "output_checkpoint": {
            "directory": str(checkpoint.resolve()),
            "file_sha256": checkpoint_hashes,
        },
        "optimizer_state_sha256": sha256_file(optimizer_path),
        "tokenization": {
            "examples": len(tokenized),
            "encoded_tokens": sum(len(row.input_ids) for row in tokenized),
            "target_tokens": sum(row.target_tokens for row in tokenized),
            "maximum_tokens": max(len(row.input_ids) for row in tokenized),
            "overlength_rejections": 0,
        },
        "parameter_counts": model.parameter_counts(),
        "optimization": dict(optimization),
        "duration_seconds": duration,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def _metric(metrics: Mapping[str, Any], *path: str) -> float:
    current: object = metrics
    for name in path:
        current = _mapping(current, ".".join(path)).get(name)
    return _number(current, ".".join(path))


def evaluate_recovery_presto_gate(
    metrics: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply every frozen PRESTO capability, safety, and corrected hard-bucket gate."""

    _expect(metrics.get("schema_version"), PRESTO_SCORER_VERSION, "PRESTO metric schema")
    _expect(metrics.get("sample_count"), 14_288, "PRESTO terminal sample count")
    exact = _metric(metrics, "ast_exact_match", "value")
    schema = _metric(metrics, "schema_valid", "value")
    abstention = _metric(metrics, "abstention", "f1")
    false_call = _metric(metrics, "false_call_on_gate", "value")
    buckets = _mapping(metrics.get("headline_phenomenon_buckets"), "PRESTO hard buckets")
    simple = _mapping(buckets.get("no_phenomenon"), "no-phenomenon bucket")
    revision = _mapping(buckets.get("revision"), "revision bucket")
    disfluency = _mapping(buckets.get("disfluency"), "disfluency bucket")
    represented = all(
        _integer(bucket.get("count"), f"{name} count", minimum=1) > 0
        for name, bucket in (
            ("no_phenomenon", simple),
            ("revision", revision),
            ("disfluency", disfluency),
        )
    )
    simple_exact = _metric(simple, "ast_exact_match", "value")
    gaps = {
        "revision": simple_exact - _metric(revision, "ast_exact_match", "value"),
        "disfluency": simple_exact - _metric(disfluency, "ast_exact_match", "value"),
    }
    per_phenomenon = _mapping(metrics.get("per_phenomenon"), "raw phenomenon metrics")
    raw_revision_metrics: dict[str, Mapping[str, Any]] = {}
    for label in sorted(PRESTO_USER_REVISION_RAW_LABELS_V2):
        raw = _mapping(per_phenomenon.get(label), f"raw revision tag {label}")
        _integer(raw.get("count"), f"raw revision tag {label} count", minimum=1)
        raw_revision_metrics[label] = raw

    maximum_gap = _number(
        thresholds.get("maximum_no_phenomenon_to_revision_or_disfluency_gap"),
        "PRESTO maximum hard gap",
    )
    checks = {
        "capability_derived_ast_exact": exact
        >= _number(thresholds.get("derived_ast_exact_at_least"), "PRESTO exact threshold"),
        "capability_schema_valid": schema
        >= _number(thresholds.get("schema_valid_at_least"), "PRESTO schema threshold"),
        "safety_abstention_f1": abstention
        >= _number(thresholds.get("abstention_f1_at_least"), "PRESTO abstention threshold"),
        "safety_false_call_on_gate": false_call
        <= _number(thresholds.get("false_call_on_gate_at_most"), "PRESTO false-call threshold"),
        "hard_buckets_represented": represented,
        "hard_bucket_gaps": represented and all(gap <= maximum_gap for gap in gaps.values()),
        "corrected_full_user_revision_family_reported": (
            set(raw_revision_metrics) == set(PRESTO_USER_REVISION_RAW_LABELS_V2)
        ),
    }
    return {
        "scorer_version": PRESTO_SCORER_VERSION,
        "taxonomy_version": PRESTO_PHENOMENON_TAXONOMY_VERSION,
        "revision_alias_set_version": PRESTO_USER_REVISION_ALIAS_SET_VERSION,
        "thresholds": dict(thresholds),
        "observed": {
            "derived_ast_exact": exact,
            "schema_valid": schema,
            "abstention_f1": abstention,
            "false_call_on_gate": false_call,
            "no_phenomenon_minus_hard_bucket_gap": gaps,
            "revision_raw_tag_metrics": {
                label: dict(raw_revision_metrics[label]) for label in sorted(raw_revision_metrics)
            },
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


__all__ = [
    "RECOVERY_CONFIG_SHA256",
    "RECOVERY_CONFIG_VERSION",
    "RECOVERY_RECIPE_ID",
    "RECOVERY_SELECTOR_VERSION",
    "RECOVERY_TRAINING_VERSION",
    "RECOVERY_VIEW_VERSION",
    "RecoveryError",
    "evaluate_recovery_presto_gate",
    "load_recovery_config",
    "materialize_recovery_view",
    "train_continual_recovery",
]
