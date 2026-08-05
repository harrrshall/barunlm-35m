"""Frozen matched-adaptation runner for Qwen2.5-0.5B on Mobile Actions.

The official Mobile Actions evaluation tail is deliberately not an input to this
runner.  It accepts only the already-derived, hash-pinned internal-train train/dev
manifests and the adapter audit which proves the 961-row tail stayed opaque.

``transformers`` is imported only inside :func:`run`, keeping the reusable
validation and rendering logic CPU-testable without an optional heavyweight
dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from barunlm.evaluation.mobile_actions import MOBILE_ACTIONS_SCORER_VERSION, write_scores
from barunlm.training.data import (
    IGNORE_INDEX,
    SFTExample,
    TokenizedExample,
    collate_sft,
    deterministic_batches,
    load_manifest,
    sha256_file,
)

MATCHED_RECIPE_SCHEMA_VERSION = "barun-mobile-matched-baseline-recipe-v1"
NATIVE_RENDER_VERSION = "barun-native-chat-action-render-v1"
RESULT_SCHEMA_VERSION = "barun-mobile-matched-baseline-result-v1"
ATTEMPT_SCHEMA_VERSION = "barun-qwen05b-matched-attempt-v1"
SOURCE_SNAPSHOT_SCHEMA_VERSION = "barun-qwen05b-source-snapshot-v1"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EXPECTED_UNIQUE_PARAMETERS = 494_032_768
MOBILE_DATASET_REVISION = "e920309bc2acbc2e99a5e3201cf37df2b9fd9151"
TRAIN_MANIFEST_SHA256 = "131473ccb5bfb51cac0439b42159e72ec4c598025e50364a52b122b056c2e84e"
DEV_MANIFEST_SHA256 = "988bdce5874d1f1a775feeb5ba2b58cd2bdc128f57e73cb9a63d535fae7c1d55"
AUDIT_SHA256 = "dc756f97c0a7ef706ec8ffefe2d57cf7e16d75a6ccd932f906d16b6a6ee2f83c"
TRAIN_MEMBERSHIP_SHA256 = "4cdfc3649c21a9f0d3dc4d8d9cffcae3cb5c6abc4414a9fb09636e9220afe96f"
DEV_MEMBERSHIP_SHA256 = "31e0170c1de49d9c4591064587c405f0cb9f981e7026ac457f6fa2902b3647de"
TRAIN_ROWS = 7_937
DEV_ROWS = 756
OFFICIAL_EVAL_ROWS = 961
OFFICIAL_SOURCE_SHA256 = "91d251ee958cfd295af6c4504c236a3a1ad19517de240c3bc680bacfcbf7e7d9"
EXPECTED_SNAPSHOT_HASHES = {
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "README.md": "b19c806a904db6dc878a0462e70b551f6b7ac78dfbb88c2eb966ca2b9109ae15",
    "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "generation_config.json": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
}
EXPECTED_MODEL_SAFETENSORS_BYTES = 988_097_824

RECIPE_PATH = Path(__file__).resolve().parents[3] / "configs" / "mobile_qwen05b_matched_v1.json"
# Frozen before any baseline download, training, or model scoring.
RECIPE_SHA256 = "6855eaf52c34089d802b8584c566b95619f9ff2780d4f130f333126bc33357e1"

_RUN_ID = re.compile(r"\d{8}-\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*-s\d+")
_PROMPT_PREFIX = "<bos><system>\n"
_USER_MARKER = "\n<user>\n"
_PROMPT_SUFFIX = "\n<assistant>\n"
_SNAPSHOT_ALLOW_PATTERNS = (
    "LICENSE*",
    "README.md",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


class MatchedBaselineError(RuntimeError):
    """A frozen matched-comparison invariant was violated."""


class ExistingMatchedRunError(FileExistsError):
    """An immutable run directory already exists."""


class NativeChatTokenizer(Protocol):
    """Subset of a Transformers tokenizer used by the renderer."""

    chat_template: str | None
    eos_token_id: int | None
    pad_token_id: int | None

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any: ...

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool) -> str: ...


@dataclass(frozen=True, slots=True)
class ParsedActionPrompt:
    """Semantic content recovered losslessly from the Barun role-token wrapper."""

    system_content: str
    user_content: str

    def messages(self) -> tuple[dict[str, str], dict[str, str]]:
        return (
            {"role": "system", "content": self.system_content},
            {"role": "user", "content": self.user_content},
        )


@dataclass(frozen=True, slots=True)
class NativeChatExample:
    """One example after native-template rendering, before tensor collation."""

    example_id: str
    prompt_ids: tuple[int, ...]
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_sha256: str
    prompt_tokens: int
    response_tokens: int
    content_sha256: str

    def to_training_example(self) -> TokenizedExample:
        return TokenizedExample(
            example_id=self.example_id,
            input_ids=self.input_ids,
            labels=self.labels,
            prompt_tokens=self.prompt_tokens,
            target_tokens=self.response_tokens,
            content_sha256=self.content_sha256,
        )


@dataclass(frozen=True, slots=True)
class FrozenMobileInputs:
    train: tuple[SFTExample, ...]
    dev: tuple[SFTExample, ...]
    audit: Mapping[str, Any]
    train_membership_sha256: str
    dev_membership_sha256: str


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _membership_sha256(rows: Sequence[SFTExample]) -> str:
    payload = "\n".join(sorted(row.example_id for row in rows)) + "\n"
    return _sha256_bytes(payload.encode("utf-8"))


def _as_ids(value: Any, *, operation: str) -> tuple[int, ...]:
    if isinstance(value, Tensor):
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1:
            raise MatchedBaselineError(f"{operation} returned a non-vector tensor")
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MatchedBaselineError(f"{operation} did not return a token-ID sequence")
    ids: list[int] = []
    for token in value:
        if type(token) is not int or token < 0:
            raise MatchedBaselineError(f"{operation} returned an invalid token ID")
        ids.append(token)
    if not ids:
        raise MatchedBaselineError(f"{operation} returned no token IDs")
    return tuple(ids)


def parse_action_prompt(prompt: str) -> ParsedActionPrompt:
    """Remove only the frozen Barun role wrapper; preserve all semantic bytes.

    Baselines receive exactly the same ``ACTION_IR_V1`` system content, timestamp,
    ordered tools, descriptions, types, required flags, and user request.  Their
    native chat template supplies role-control tokens and nothing else.
    """

    if not isinstance(prompt, str) or not prompt.startswith(_PROMPT_PREFIX):
        raise MatchedBaselineError("prompt does not start with the frozen system-role wrapper")
    if not prompt.endswith(_PROMPT_SUFFIX):
        raise MatchedBaselineError("prompt does not end with the frozen assistant-role wrapper")
    semantic = prompt[len(_PROMPT_PREFIX) : -len(_PROMPT_SUFFIX)]
    parts = semantic.split(_USER_MARKER)
    if len(parts) != 2:
        raise MatchedBaselineError("prompt must contain exactly one frozen user-role boundary")
    system_content, user_content = parts
    if not system_content.startswith("ACTION_IR_V1\nNOW "):
        raise MatchedBaselineError("system content does not start with ACTION_IR_V1 and NOW")
    if "\nTOOLS\n" not in system_content:
        raise MatchedBaselineError("system content lacks the frozen TOOLS section")
    tool_block = system_content.split("\nTOOLS\n", 1)[1]
    if not tool_block or not user_content:
        raise MatchedBaselineError("tool block and user request must both be non-empty")
    if any(marker in system_content or marker in user_content for marker in (_PROMPT_PREFIX,)):
        raise MatchedBaselineError("nested frozen role wrappers are forbidden")
    return ParsedActionPrompt(system_content=system_content, user_content=user_content)


def tokenize_native_chat_example(
    example: SFTExample,
    tokenizer: NativeChatTokenizer,
    *,
    max_seq_len: int,
) -> NativeChatExample:
    """Render one row solely with the baseline's native chat template.

    The full-message encoding must have the generation prompt as an exact token
    prefix.  Only the assistant suffix, including the template's native closing
    token(s), is supervised.  Overlength examples abort the run; none are dropped or
    truncated.
    """

    if max_seq_len < 2:
        raise ValueError("max_seq_len must be at least two")
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise MatchedBaselineError("the pinned baseline tokenizer has no native chat template")
    if type(tokenizer.eos_token_id) is not int or tokenizer.eos_token_id < 0:
        raise MatchedBaselineError("the native tokenizer has no valid EOS token ID")
    if type(tokenizer.pad_token_id) is not int or tokenizer.pad_token_id < 0:
        raise MatchedBaselineError("the native tokenizer has no valid padding token ID")

    parsed = parse_action_prompt(example.prompt)
    messages = list(parsed.messages())
    prompt_ids = _as_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        ),
        operation="native generation-prompt rendering",
    )
    full_ids = _as_ids(
        tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": example.target}],
            tokenize=True,
            add_generation_prompt=False,
        ),
        operation="native full-message rendering",
    )
    if len(full_ids) <= len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        raise MatchedBaselineError(
            f"example {example.example_id!r}: full native chat is not prefixed by its "
            "generation prompt"
        )
    response_ids = full_ids[len(prompt_ids) :]
    if tokenizer.eos_token_id not in response_ids:
        raise MatchedBaselineError(
            f"example {example.example_id!r}: native assistant suffix has no EOS token"
        )
    if len(full_ids) > max_seq_len:
        raise MatchedBaselineError(
            f"example {example.example_id!r}: native chat has {len(full_ids)} tokens, "
            f"above the frozen {max_seq_len}-token limit"
        )
    labels = (IGNORE_INDEX,) * len(prompt_ids) + response_ids
    if len(labels) != len(full_ids):  # pragma: no cover - construction invariant
        raise MatchedBaselineError("response-only label alignment failed")
    return NativeChatExample(
        example_id=example.example_id,
        prompt_ids=prompt_ids,
        input_ids=full_ids,
        labels=labels,
        prompt_sha256=_sha256_bytes(_canonical_json(prompt_ids).encode("utf-8")),
        prompt_tokens=len(prompt_ids),
        response_tokens=len(response_ids),
        content_sha256=example.content_sha256,
    )


def load_frozen_recipe(path: str | Path = RECIPE_PATH) -> dict[str, Any]:
    recipe_path = Path(path)
    actual_hash = sha256_file(recipe_path)
    if actual_hash != RECIPE_SHA256:
        raise MatchedBaselineError(
            f"matched recipe SHA-256 changed: expected {RECIPE_SHA256}, got {actual_hash}"
        )
    try:
        payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MatchedBaselineError("matched recipe is not valid JSON") from error
    if not isinstance(payload, dict):
        raise MatchedBaselineError("matched recipe must be a JSON object")
    if payload.get("schema_version") != MATCHED_RECIPE_SCHEMA_VERSION:
        raise MatchedBaselineError("matched recipe schema version changed")
    if _RUN_ID.fullmatch(str(payload.get("run_id", ""))) is None:
        raise MatchedBaselineError("matched recipe run_id is not immutable-ID shaped")

    model = payload.get("model")
    data = payload.get("data")
    optimization = payload.get("optimization")
    evaluation = payload.get("evaluation")
    if not all(isinstance(value, dict) for value in (model, data, optimization, evaluation)):
        raise MatchedBaselineError(
            "matched recipe lacks model/data/optimization/evaluation objects"
        )
    assert isinstance(model, dict) and isinstance(data, dict)
    assert isinstance(optimization, dict) and isinstance(evaluation, dict)
    frozen_values = {
        "model.id": (model.get("id"), MODEL_ID),
        "model.revision": (model.get("revision"), MODEL_REVISION),
        "model.expected_unique_parameters": (
            model.get("expected_unique_parameters"),
            EXPECTED_UNIQUE_PARAMETERS,
        ),
        "data.dataset_revision": (data.get("dataset_revision"), MOBILE_DATASET_REVISION),
        "data.train_manifest_sha256": (
            data.get("train_manifest_sha256"),
            TRAIN_MANIFEST_SHA256,
        ),
        "data.dev_manifest_sha256": (data.get("dev_manifest_sha256"), DEV_MANIFEST_SHA256),
        "data.audit_sha256": (data.get("audit_sha256"), AUDIT_SHA256),
        "data.train_rows": (data.get("train_rows"), TRAIN_ROWS),
        "data.dev_rows": (data.get("dev_rows"), DEV_ROWS),
        "data.official_eval_rows": (data.get("official_eval_rows"), OFFICIAL_EVAL_ROWS),
        "optimization.seed": (optimization.get("seed"), 17),
        "optimization.epochs": (optimization.get("epochs"), 1),
        "optimization.per_device_batch_size": (
            optimization.get("per_device_batch_size"),
            21,
        ),
        "optimization.gradient_accumulation_steps": (
            optimization.get("gradient_accumulation_steps"),
            3,
        ),
        "optimization.expected_optimizer_steps": (
            optimization.get("expected_optimizer_steps"),
            126,
        ),
        "evaluation.scorer_version": (
            evaluation.get("scorer_version"),
            MOBILE_ACTIONS_SCORER_VERSION,
        ),
        "evaluation.max_new_tokens": (evaluation.get("max_new_tokens"), 192),
        "evaluation.generation_batch_size": (
            evaluation.get("generation_batch_size"),
            64,
        ),
    }
    for label, (actual, expected) in frozen_values.items():
        if actual != expected:
            raise MatchedBaselineError(f"{label} changed: expected {expected!r}, got {actual!r}")
    if model.get("license") != "Apache-2.0" or model.get("trust_remote_code") is not False:
        raise MatchedBaselineError("baseline license or trust_remote_code policy changed")
    if evaluation.get("decoding") != "unconstrained_deterministic_greedy":
        raise MatchedBaselineError("primary decoding must stay unconstrained and greedy")
    expected_generation_overrides = {
        "do_sample": False,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "temperature": None,
        "top_k": None,
        "top_p": None,
    }
    if evaluation.get("generation_config_overrides") != expected_generation_overrides:
        raise MatchedBaselineError("Qwen vendor generation settings were not fully neutralized")
    return payload


def _audit_manifest_identity(
    audit: Mapping[str, Any], *, split: str, artifact_hash: str, membership_hash: str
) -> None:
    try:
        identity = audit["artifacts"]["manifests"][split]
    except (KeyError, TypeError) as error:
        raise MatchedBaselineError(f"adapter audit lacks {split} manifest identity") from error
    if not isinstance(identity, Mapping):
        raise MatchedBaselineError(f"adapter audit {split} manifest identity is not an object")
    if identity.get("sha256") != artifact_hash:
        raise MatchedBaselineError(f"adapter audit {split} artifact SHA-256 changed")
    if identity.get("membership_sha256") != membership_hash:
        raise MatchedBaselineError(f"adapter audit {split} membership SHA-256 changed")


def verify_frozen_mobile_inputs(
    *,
    train_manifest: str | Path,
    dev_manifest: str | Path,
    audit_path: str | Path,
) -> FrozenMobileInputs:
    """Verify identical train/dev rows and the official-test opacity evidence."""

    train = tuple(
        load_manifest(
            train_manifest,
            expected_sha256=TRAIN_MANIFEST_SHA256,
            expected_derived_split="train",
        )
    )
    dev = tuple(
        load_manifest(
            dev_manifest,
            expected_sha256=DEV_MANIFEST_SHA256,
            expected_derived_split="dev",
        )
    )
    if len(train) != TRAIN_ROWS or len(dev) != DEV_ROWS:
        raise MatchedBaselineError("frozen Mobile Actions train/dev row counts changed")
    train_ids = {row.example_id for row in train}
    dev_ids = {row.example_id for row in dev}
    if train_ids.intersection(dev_ids):
        raise MatchedBaselineError("Mobile Actions train and development IDs overlap")
    train_membership = _membership_sha256(train)
    dev_membership = _membership_sha256(dev)
    if train_membership != TRAIN_MEMBERSHIP_SHA256:
        raise MatchedBaselineError("frozen Mobile Actions train membership changed")
    if dev_membership != DEV_MEMBERSHIP_SHA256:
        raise MatchedBaselineError("frozen Mobile Actions development membership changed")

    audit_file = Path(audit_path)
    if sha256_file(audit_file) != AUDIT_SHA256:
        raise MatchedBaselineError("frozen Mobile Actions adapter audit SHA-256 changed")
    try:
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MatchedBaselineError("Mobile Actions adapter audit is invalid JSON") from error
    if not isinstance(audit, dict):
        raise MatchedBaselineError("Mobile Actions adapter audit is not an object")
    _audit_manifest_identity(
        audit,
        split="train",
        artifact_hash=TRAIN_MANIFEST_SHA256,
        membership_hash=TRAIN_MEMBERSHIP_SHA256,
    )
    _audit_manifest_identity(
        audit,
        split="dev",
        artifact_hash=DEV_MANIFEST_SHA256,
        membership_hash=DEV_MEMBERSHIP_SHA256,
    )
    expected_official = {
        "labels_parsed": False,
        "lengths_computed": False,
        "materialized": False,
        "opaque_unparsed": True,
        "overlaps_computed": False,
        "prompts_parsed": False,
        "rows": OFFICIAL_EVAL_ROWS,
        "source_sha256": OFFICIAL_SOURCE_SHA256,
        "summaries_computed": False,
        "targets_parsed": False,
        "tool_schemas_parsed": False,
    }
    if audit.get("official_eval") != expected_official:
        raise MatchedBaselineError("official Mobile Actions evaluation tail did not stay opaque")
    artifacts = audit.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"manifests"}:
        raise MatchedBaselineError("adapter audit contains an unexpected official-eval artifact")
    counts = audit.get("counts")
    try:
        derived = counts["derived"]
        opaque_count = counts["source"]["official_eval_opaque_unparsed"]
    except (KeyError, TypeError) as error:
        raise MatchedBaselineError("adapter audit lacks frozen count evidence") from error
    if derived != {"dev": DEV_ROWS, "train": TRAIN_ROWS} or opaque_count != OFFICIAL_EVAL_ROWS:
        raise MatchedBaselineError("adapter audit frozen counts changed")
    return FrozenMobileInputs(
        train=train,
        dev=dev,
        audit=audit,
        train_membership_sha256=train_membership,
        dev_membership_sha256=dev_membership,
    )


def count_unique_parameters(model: nn.Module) -> int:
    """Count parameter elements while deduplicating exact tied-storage views."""

    seen: set[tuple[Any, ...]] = set()
    total = 0
    try:
        named = model.named_parameters(remove_duplicate=False)
    except TypeError:  # pragma: no cover - PyTorch 3.10 support floor still has this argument
        named = model.named_parameters()
    for _name, parameter in named:
        if parameter.device.type == "meta":
            key = ("meta", id(parameter))
        else:
            storage = parameter.untyped_storage()
            key = (
                parameter.device.type,
                parameter.device.index,
                storage.data_ptr(),
                parameter.storage_offset(),
                tuple(parameter.shape),
                tuple(parameter.stride()),
                str(parameter.dtype),
            )
        if key in seen:
            continue
        seen.add(key)
        total += parameter.numel()
    return total


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
    temporary.replace(path)


def _tree_sha256(root: Path, *, excluded_names: frozenset[str] = frozenset()) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded_names
    }


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MatchedBaselineError(f"{label} is not readable strict JSON") from error
    if not isinstance(payload, dict):
        raise MatchedBaselineError(f"{label} must be a JSON object")
    return payload


def _manifest_tree_sha256(files: Mapping[str, str]) -> str:
    lines = "".join(f"{digest}  {path}\n" for path, digest in sorted(files.items()))
    return _sha256_bytes(lines.encode("utf-8"))


def _verify_attempt_provenance(
    *,
    attempt_path: Path,
    source_snapshot_path: Path,
    run_id: str,
    jarvis_machine_id: int,
) -> dict[str, Any]:
    """Verify the preregistration after exact-ID binding and the staged source tree."""

    attempt = _load_json_object(attempt_path, label="attempt preregistration")
    if attempt.get("schema_version") != ATTEMPT_SCHEMA_VERSION:
        raise MatchedBaselineError("attempt preregistration schema changed")
    if attempt.get("run_id") != run_id:
        raise MatchedBaselineError("attempt preregistration run ID changed")
    if attempt.get("status") != "frozen_before_model_download_or_development_scoring":
        raise MatchedBaselineError("attempt preregistration was not frozen before scoring")
    if attempt.get("baseline_downloaded") is not False:
        raise MatchedBaselineError("attempt preregistration does not prove a pre-download freeze")
    if attempt.get("development_predictions_or_scores_observed") is not False:
        raise MatchedBaselineError("attempt preregistration was created after development scoring")
    if attempt.get("official_evaluation_rows_read_or_materialized") != 0:
        raise MatchedBaselineError("attempt preregistration violates the official-eval firewall")

    recipe_identity = attempt.get("scientific_recipe")
    if not isinstance(recipe_identity, Mapping):
        raise MatchedBaselineError("attempt preregistration lacks scientific recipe identity")
    if recipe_identity.get("path") != "configs/mobile_qwen05b_matched_v1.json":
        raise MatchedBaselineError("attempt preregistration recipe path changed")
    if recipe_identity.get("sha256") != RECIPE_SHA256:
        raise MatchedBaselineError("attempt preregistration recipe hash changed")

    inventory_block = attempt.get("prelaunch_inventory")
    if not isinstance(inventory_block, Mapping):
        raise MatchedBaselineError("attempt preregistration lacks bound prelaunch inventory")
    if inventory_block.get("captured_before_project_instance_creation") is not True:
        raise MatchedBaselineError("prelaunch inventory was not captured before creation")
    if inventory_block.get("fresh_project_instance") is not True:
        raise MatchedBaselineError("attempt did not require a fresh project instance")
    if inventory_block.get("project_machine_id") != jarvis_machine_id:
        raise MatchedBaselineError("attempt project machine ID does not match the controller")
    protected = inventory_block.get("protected_machine_ids")
    if (
        not isinstance(protected, list)
        or not protected
        or any(type(value) is not int or value < 1 for value in protected)
        or protected != sorted(set(protected))
        or 463058 not in protected
        or jarvis_machine_id in protected
    ):
        raise MatchedBaselineError("attempt preregistration has an invalid protected-ID denylist")

    snapshot = _load_json_object(source_snapshot_path, label="source snapshot")
    if snapshot.get("schema_version") != SOURCE_SNAPSHOT_SCHEMA_VERSION:
        raise MatchedBaselineError("source snapshot schema changed")
    if snapshot.get("run_id") != run_id:
        raise MatchedBaselineError("source snapshot run ID changed")
    if snapshot.get("created_before_model_download_or_development_scoring") is not True:
        raise MatchedBaselineError("source snapshot timing assertion changed")
    files = snapshot.get("files_sha256")
    if not isinstance(files, Mapping) or not files:
        raise MatchedBaselineError("source snapshot has no file manifest")
    normalized_files: dict[str, str] = {}
    stage_root = RECIPE_PATH.parents[1].resolve()
    for relative, expected_hash in files.items():
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        ):
            raise MatchedBaselineError("source snapshot contains an invalid file identity")
        candidate = stage_root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise MatchedBaselineError(f"source snapshot file is absent or unsafe: {relative}")
        if sha256_file(candidate) != expected_hash:
            raise MatchedBaselineError(f"source snapshot file changed: {relative}")
        normalized_files[relative] = expected_hash
    if snapshot.get("file_count") != len(normalized_files):
        raise MatchedBaselineError("source snapshot file count changed")
    if snapshot.get("content_tree_sha256") != _manifest_tree_sha256(normalized_files):
        raise MatchedBaselineError("source snapshot tree hash changed")

    snapshot_identity = attempt.get("source_snapshot")
    if not isinstance(snapshot_identity, Mapping):
        raise MatchedBaselineError("attempt preregistration lacks source snapshot identity")
    if snapshot_identity.get("path") != "source-snapshot.json":
        raise MatchedBaselineError("attempt preregistration source snapshot path changed")
    if snapshot_identity.get("sha256") != sha256_file(source_snapshot_path):
        raise MatchedBaselineError("attempt preregistration source snapshot hash changed")
    if snapshot_identity.get("content_tree_sha256") != snapshot.get("content_tree_sha256"):
        raise MatchedBaselineError("attempt and source snapshot tree identities differ")
    return {
        "attempt_sha256": sha256_file(attempt_path),
        "source_snapshot_sha256": sha256_file(source_snapshot_path),
        "content_tree_sha256": snapshot["content_tree_sha256"],
        "protected_machine_ids": protected,
    }


def _snapshot_manifest(snapshot: Path) -> dict[str, Any]:
    files = _tree_sha256(snapshot)
    if "config.json" not in files or "tokenizer_config.json" not in files:
        raise MatchedBaselineError("pinned baseline snapshot lacks config or tokenizer config")
    if not any(name.endswith(".safetensors") for name in files):
        raise MatchedBaselineError("pinned baseline snapshot lacks safetensors weights")
    for name, expected_hash in EXPECTED_SNAPSHOT_HASHES.items():
        if files.get(name) != expected_hash:
            raise MatchedBaselineError(f"pinned Qwen snapshot hash changed: {name}")
    if (snapshot / "model.safetensors").stat().st_size != EXPECTED_MODEL_SAFETENSORS_BYTES:
        raise MatchedBaselineError("pinned Qwen safetensors byte length changed")
    return {
        "schema_version": "barun-pinned-hf-snapshot-v1",
        "repo_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "resolved_directory": str(snapshot),
        "files_sha256": files,
        "files_bytes": {
            str(path.relative_to(snapshot)): path.stat().st_size
            for path in sorted(snapshot.rglob("*"))
            if path.is_file()
        },
    }


def _verify_model_identity(
    model: nn.Module, tokenizer: Any, recipe: Mapping[str, Any]
) -> dict[str, Any]:
    model_recipe = recipe["model"]
    architecture = model_recipe["architecture"]
    config = model.config
    for field, expected in architecture.items():
        actual = getattr(config, field, None)
        if actual != expected:
            raise MatchedBaselineError(
                f"pinned baseline architecture {field} changed: expected {expected!r}, "
                f"got {actual!r}"
            )
    commit_hash = getattr(config, "_commit_hash", None)
    if commit_hash not in (None, MODEL_REVISION):
        raise MatchedBaselineError(f"loaded model resolved unexpected commit {commit_hash!r}")
    if not getattr(tokenizer, "is_fast", False):
        raise MatchedBaselineError("pinned baseline must use its fast native tokenizer")
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise MatchedBaselineError("pinned baseline tokenizer lacks its native chat template")
    unique_parameters = count_unique_parameters(model)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if unique_parameters != EXPECTED_UNIQUE_PARAMETERS:
        raise MatchedBaselineError(
            f"pinned baseline unique parameter count changed: expected "
            f"{EXPECTED_UNIQUE_PARAMETERS}, got {unique_parameters}"
        )
    if trainable_parameters != unique_parameters:
        raise MatchedBaselineError("matched lane must full-fine-tune every unique parameter")
    input_embeddings = model.get_input_embeddings().weight
    output_embeddings_module = model.get_output_embeddings()
    if output_embeddings_module is None:
        raise MatchedBaselineError("pinned baseline has no output embedding module")
    output_embeddings = output_embeddings_module.weight
    if input_embeddings.data_ptr() != output_embeddings.data_ptr():
        raise MatchedBaselineError("expected tied input/output embeddings are not storage-tied")
    vendor_generation = {
        "do_sample": getattr(model.generation_config, "do_sample", None),
        "eos_token_id": getattr(model.generation_config, "eos_token_id", None),
        "pad_token_id": getattr(model.generation_config, "pad_token_id", None),
        "repetition_penalty": getattr(model.generation_config, "repetition_penalty", None),
        "temperature": getattr(model.generation_config, "temperature", None),
        "top_k": getattr(model.generation_config, "top_k", None),
        "top_p": getattr(model.generation_config, "top_p", None),
    }
    expected_vendor_generation = {
        "do_sample": True,
        "eos_token_id": [151645, 151643],
        "pad_token_id": 151643,
        "repetition_penalty": 1.1,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
    }
    if vendor_generation != expected_vendor_generation:
        raise MatchedBaselineError("pinned Qwen vendor generation configuration changed")
    if tokenizer.eos_token_id != 151645 or tokenizer.pad_token_id != 151643:
        raise MatchedBaselineError("pinned Qwen tokenizer EOS or padding identity changed")
    return {
        "unique_parameters": unique_parameters,
        "trainable_unique_parameters": trainable_parameters,
        "parameter_count_policy": "exact storage-view deduplication; tied embeddings counted once",
        "native_chat_template_sha256": _sha256_bytes(tokenizer.chat_template.encode("utf-8")),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "config_commit_hash": commit_hash,
        "vendor_generation_config": vendor_generation,
    }


def _configure_determinism(seed: int) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise MatchedBaselineError("matched baseline requires CUDA with bfloat16 support")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise MatchedBaselineError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before torch import")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def _environment(determinism: Mapping[str, Any]) -> dict[str, Any]:
    package_versions: dict[str, str | None] = {}
    for package in ("huggingface-hub", "safetensors", "tokenizers", "transformers"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "determinism": dict(determinism),
        "packages": package_versions,
    }


def _run_repository_tests(output_path: Path) -> None:
    relevant_tests = (
        "tests/test_mobile_qwen05b_matched_baseline.py",
        "tests/test_qwen05b_bundle_verification.py",
        "tests/test_mobile_actions_scoring.py",
        "tests/test_evaluation.py",
        "tests/test_training.py",
    )
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *relevant_tests],
        cwd=Path(__file__).resolve().parents[3],
        check=False,
        capture_output=True,
        text=True,
    )
    output_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise MatchedBaselineError("repository tests failed before baseline download or training")


def _tokenize_rows(
    rows: Sequence[SFTExample], tokenizer: NativeChatTokenizer, *, max_seq_len: int
) -> tuple[list[TokenizedExample], list[NativeChatExample]]:
    rendered = [
        tokenize_native_chat_example(row, tokenizer, max_seq_len=max_seq_len) for row in rows
    ]
    return [row.to_training_example() for row in rendered], rendered


def _render_audit(
    *,
    train: Sequence[NativeChatExample],
    dev: Sequence[NativeChatExample],
    chat_template_sha256: str,
) -> dict[str, Any]:
    def stats(rows: Sequence[NativeChatExample]) -> dict[str, Any]:
        full = [len(row.input_ids) for row in rows]
        prompts = [row.prompt_tokens for row in rows]
        responses = [row.response_tokens for row in rows]
        return {
            "rows": len(rows),
            "full_tokens": {"min": min(full), "max": max(full), "sum": sum(full)},
            "prompt_tokens": {
                "min": min(prompts),
                "max": max(prompts),
                "sum": sum(prompts),
            },
            "response_tokens": {
                "min": min(responses),
                "max": max(responses),
                "sum": sum(responses),
            },
            "overlength_or_dropped": 0,
        }

    return {
        "schema_version": NATIVE_RENDER_VERSION,
        "mapping": (
            "remove only frozen Barun role wrappers; pass identical system and user semantic "
            "content to tokenizer.apply_chat_template; append the unchanged Action IR target "
            "as the assistant message"
        ),
        "native_chat_template_sha256": chat_template_sha256,
        "assistant_output_only_labels": True,
        "extra_semantic_instructions": False,
        "packing": False,
        "truncation": False,
        "splits": {"train": stats(train), "dev": stats(dev)},
    }


def _render_records(rows: Sequence[NativeChatExample]) -> list[dict[str, Any]]:
    return [
        {
            "id": row.example_id,
            "source_record_sha256": row.content_sha256,
            "native_prompt_sha256": row.prompt_sha256,
            "prompt_tokens": row.prompt_tokens,
            "response_tokens": row.response_tokens,
            "full_tokens": len(row.input_ids),
            "truncated_or_dropped": False,
        }
        for row in rows
    ]


def _lr_multiplier(
    step: int, *, total_steps: int, warmup_steps: int, minimum_ratio: float
) -> float:
    if not 1 <= step <= total_steps:
        raise ValueError("step is outside the frozen schedule")
    if step <= warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _train_one_recipe(
    *,
    model: nn.Module,
    examples: Sequence[TokenizedExample],
    tokenizer: NativeChatTokenizer,
    optimization: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    device = torch.device("cuda", 0)
    model.train()
    if bool(optimization["gradient_checkpointing"]):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
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
    if total_steps != int(optimization["expected_optimizer_steps"]):
        raise MatchedBaselineError(
            f"optimizer-step budget changed: expected "
            f"{optimization['expected_optimizer_steps']}, got {total_steps}"
        )
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
                batch = collate_sft(microbatch, pad_token_id=int(tokenizer.pad_token_id)).to(device)
                step_example_ids.extend(batch.example_ids)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        labels=batch.labels,
                        use_cache=False,
                    )
                if not bool(torch.isfinite(output.loss).item()):
                    raise MatchedBaselineError(
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
            multiplier = _lr_multiplier(
                optimizer_step,
                total_steps=total_steps,
                warmup_steps=int(optimization["warmup_steps"]),
                minimum_ratio=float(optimization["min_learning_rate_ratio"]),
            )
            learning_rate = float(optimization["learning_rate"]) * multiplier
            _set_learning_rate(optimizer, learning_rate)
            optimizer.step()
            observed_ids.extend(step_example_ids)
            record = {
                "optimizer_step": optimizer_step,
                "examples": len(step_example_ids),
                "target_tokens": total_target_tokens,
                "response_nll": weighted_loss / total_target_tokens,
                "gradient_norm": float(gradient_norm.detach().float()),
                "learning_rate": learning_rate,
                "elapsed_seconds": time.monotonic() - started,
            }
            metrics.write(_canonical_json(record) + "\n")
            metrics.flush()
    if len(observed_ids) != TRAIN_ROWS or set(observed_ids) != {row.example_id for row in examples}:
        raise MatchedBaselineError("training did not present every frozen train ID exactly once")

    checkpoint = output_dir / "checkpoint"
    checkpoint.mkdir(exist_ok=False)
    model.config.use_cache = True
    model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(checkpoint)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "optimizer_steps": total_steps,
            "seed": int(optimization["seed"]),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        },
        output_dir / "optimizer.pt",
    )
    summary = {
        "schema_version": "barun-mobile-matched-training-v1",
        "method": "full_parameter_response_only_sft",
        "epochs": 1,
        "examples_presented": len(observed_ids),
        "unique_examples_presented": len(set(observed_ids)),
        "presentation_order_sha256": _sha256_bytes(
            ("\n".join(observed_ids) + "\n").encode("utf-8")
        ),
        "optimizer_steps": total_steps,
        "final_checkpoint_policy": "final_only",
        "final_checkpoint": str(checkpoint),
        "checkpoint_files_sha256": _tree_sha256(checkpoint),
        "optimizer_state_sha256": sha256_file(output_dir / "optimizer.pt"),
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _left_padded(
    rows: Sequence[NativeChatExample], *, pad_token_id: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    width = max(len(row.prompt_ids) for row in rows)
    input_ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        length = len(row.prompt_ids)
        input_ids[index, width - length :] = torch.tensor(
            row.prompt_ids, dtype=torch.long, device=device
        )
        attention_mask[index, width - length :] = True
    return input_ids, attention_mask


def _prediction_record(
    row: NativeChatExample,
    *,
    prediction_raw: str | None,
    truncated: bool,
    generation_failure: str | None,
    generated_tokens: int,
) -> dict[str, Any]:
    return {
        "id": row.example_id,
        "prediction_raw": prediction_raw,
        "truncated": truncated,
        "generation_failure": generation_failure,
        "prompt_tokens": row.prompt_tokens,
        "generated_tokens": generated_tokens,
    }


def _greedy_generation_kwargs(
    tokenizer: NativeChatTokenizer,
    *,
    max_new_tokens: int,
    generation_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct the complete Qwen-neutral greedy configuration."""

    return {
        "do_sample": generation_overrides["do_sample"],
        "length_penalty": generation_overrides["length_penalty"],
        "max_new_tokens": max_new_tokens,
        "no_repeat_ngram_size": generation_overrides["no_repeat_ngram_size"],
        "num_beams": generation_overrides["num_beams"],
        "repetition_penalty": generation_overrides["repetition_penalty"],
        "temperature": generation_overrides["temperature"],
        "top_k": generation_overrides["top_k"],
        "top_p": generation_overrides["top_p"],
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
        "use_cache": True,
    }


def _generate_development(
    *,
    model: nn.Module,
    tokenizer: NativeChatTokenizer,
    rows: Sequence[NativeChatExample],
    max_seq_len: int,
    max_new_tokens: int,
    batch_size: int,
    generation_overrides: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    device = torch.device("cuda", 0)
    model.eval()
    model.config.use_cache = True
    records_by_id: dict[str, dict[str, Any]] = {}
    eligible: list[NativeChatExample] = []
    for row in rows:
        if row.prompt_tokens + max_new_tokens > max_seq_len:
            records_by_id[row.example_id] = _prediction_record(
                row,
                prediction_raw=None,
                truncated=False,
                generation_failure="context_overflow",
                generated_tokens=0,
            )
        else:
            eligible.append(row)
    eligible.sort(key=lambda row: (row.prompt_tokens, row.example_id))
    started = time.monotonic()
    for start in range(0, len(eligible), batch_size):
        batch = eligible[start : start + batch_size]
        input_ids, attention_mask = _left_padded(
            batch, pad_token_id=int(tokenizer.pad_token_id), device=device
        )
        prompt_width = input_ids.shape[1]
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **_greedy_generation_kwargs(
                        tokenizer,
                        max_new_tokens=max_new_tokens,
                        generation_overrides=generation_overrides,
                    ),
                )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            for row in batch:
                records_by_id[row.example_id] = _prediction_record(
                    row,
                    prediction_raw=None,
                    truncated=False,
                    generation_failure="oom",
                    generated_tokens=0,
                )
            continue
        for batch_index, row in enumerate(batch):
            continuation = output_ids[batch_index, prompt_width:].tolist()
            eos_position = next(
                (
                    position
                    for position, token_id in enumerate(continuation)
                    if token_id == tokenizer.eos_token_id
                ),
                None,
            )
            truncated = eos_position is None
            content_ids = continuation if truncated else continuation[:eos_position]
            generated_tokens = len(continuation) if truncated else eos_position + 1
            raw = tokenizer.decode(content_ids, skip_special_tokens=False)
            records_by_id[row.example_id] = _prediction_record(
                row,
                prediction_raw=raw,
                truncated=truncated,
                generation_failure=None,
                generated_tokens=generated_tokens,
            )
    ordered = [records_by_id[row.example_id] for row in rows]
    _write_jsonl(output_path, ordered)
    summary = {
        "schema_version": "barun-native-chat-generation-v1",
        "decoding": "unconstrained_deterministic_greedy",
        "grammar_constrained": False,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "generation_config_overrides": dict(generation_overrides),
        "examples": len(rows),
        "generated": sum(row["generation_failure"] is None for row in ordered),
        "failed": sum(row["generation_failure"] is not None for row in ordered),
        "truncated": sum(bool(row["truncated"]) for row in ordered),
        "predictions_sha256": sha256_file(output_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), summary)
    return summary


def _copy_essential(export: Path, essential: Path) -> None:
    essential.mkdir(exist_ok=False)
    for name in (
        "attempt-preregistration.json",
        "environment.json",
        "model-identity.json",
        "preregistration.json",
        "repository-tests.log",
        "result.json",
        "snapshot-manifest.json",
        "source-snapshot.json",
    ):
        shutil.copy2(export / name, essential / name)
    shutil.copytree(export / "training" / "checkpoint", essential / "checkpoint")
    training_evidence = essential / "training"
    training_evidence.mkdir()
    for name in (
        "metrics.jsonl",
        "render-audit.json",
        "rendered-dev.jsonl",
        "rendered-train.jsonl",
        "summary.json",
    ):
        shutil.copy2(export / "training" / name, training_evidence / name)
    shutil.copytree(export / "evaluation", essential / "evaluation")
    data_evidence = essential / "data"
    data_evidence.mkdir()
    shutil.copy2(export / "data" / "audit.json", data_evidence / "audit.json")
    shutil.copy2(export / "data" / "identity.json", data_evidence / "identity.json")
    if any(path.name == "optimizer.pt" for path in essential.rglob("*")):
        raise MatchedBaselineError("optimizer state entered the compact essential bundle")
    _write_json(
        essential / "bundle-manifest.json",
        {
            "schema_version": "barun-mobile-matched-essential-v1",
            "official_eval_artifacts_included": False,
            "optimizer_state_included": False,
            "files_sha256": _tree_sha256(essential),
        },
    )
    _write_json(
        essential / "artifact-sha256.json",
        _tree_sha256(essential, excluded_names=frozenset({"artifact-sha256.json"})),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute exactly one frozen full-SFT recipe and final dev evaluation."""

    started = time.monotonic()
    recipe = load_frozen_recipe(args.config)
    if args.run_id != recipe["run_id"]:
        raise MatchedBaselineError("CLI run_id differs from the frozen preregistration")
    run_root = args.artifact_root / args.run_id
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    try:
        run_root.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ExistingMatchedRunError(f"refusing to overwrite {run_root}") from error
    export = run_root / "export"
    export.mkdir()
    shutil.copy2(args.config, export / "preregistration.json")

    provenance = _verify_attempt_provenance(
        attempt_path=args.attempt_preregistration,
        source_snapshot_path=args.source_snapshot,
        run_id=args.run_id,
        jarvis_machine_id=args.jarvis_machine_id,
    )
    shutil.copy2(args.attempt_preregistration, export / "attempt-preregistration.json")
    shutil.copy2(args.source_snapshot, export / "source-snapshot.json")

    determinism = _configure_determinism(int(recipe["optimization"]["seed"]))
    _write_json(export / "environment.json", _environment(determinism))
    _run_repository_tests(export / "repository-tests.log")

    frozen = verify_frozen_mobile_inputs(
        train_manifest=args.train_manifest,
        dev_manifest=args.dev_manifest,
        audit_path=args.audit,
    )
    data_dir = export / "data"
    data_dir.mkdir()
    shutil.copy2(args.audit, data_dir / "audit.json")
    _write_json(
        data_dir / "identity.json",
        {
            "schema_version": "barun-mobile-matched-input-identity-v1",
            "dataset": "google/mobile-actions",
            "revision": MOBILE_DATASET_REVISION,
            "train_manifest_sha256": TRAIN_MANIFEST_SHA256,
            "train_membership_sha256": frozen.train_membership_sha256,
            "train_rows": len(frozen.train),
            "dev_manifest_sha256": DEV_MANIFEST_SHA256,
            "dev_membership_sha256": frozen.dev_membership_sha256,
            "dev_rows": len(frozen.dev),
            "official_eval_rows": OFFICIAL_EVAL_ROWS,
            "official_eval_rows_available_to_runner": 0,
            "official_eval_materialized": False,
        },
    )

    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - exercised on remote with frozen requirements
        raise MatchedBaselineError(
            "install requirements/matched-baseline.txt before launching this recipe"
        ) from error

    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(_SNAPSHOT_ALLOW_PATTERNS),
            cache_dir=args.cache_dir,
        )
    ).resolve()
    if snapshot.name != MODEL_REVISION:
        raise MatchedBaselineError(
            f"snapshot resolved {snapshot.name!r}, expected exact commit {MODEL_REVISION!r}"
        )
    snapshot_manifest = _snapshot_manifest(snapshot)
    _write_json(export / "snapshot-manifest.json", snapshot_manifest)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(torch.device("cuda", 0))
    identity = _verify_model_identity(model, tokenizer, recipe)
    _write_json(export / "model-identity.json", identity)

    max_seq_len = int(recipe["optimization"]["max_seq_len"])
    train_examples, rendered_train = _tokenize_rows(
        frozen.train, tokenizer, max_seq_len=max_seq_len
    )
    _dev_training_examples, rendered_dev = _tokenize_rows(
        frozen.dev, tokenizer, max_seq_len=max_seq_len
    )
    render_audit = _render_audit(
        train=rendered_train,
        dev=rendered_dev,
        chat_template_sha256=identity["native_chat_template_sha256"],
    )
    training_dir = export / "training"
    training_dir.mkdir()
    rendered_train_path = training_dir / "rendered-train.jsonl"
    rendered_dev_path = training_dir / "rendered-dev.jsonl"
    _write_jsonl(rendered_train_path, _render_records(rendered_train))
    _write_jsonl(rendered_dev_path, _render_records(rendered_dev))
    render_audit["sample_artifacts"] = {
        "train": {
            "path": rendered_train_path.name,
            "sha256": sha256_file(rendered_train_path),
        },
        "dev": {
            "path": rendered_dev_path.name,
            "sha256": sha256_file(rendered_dev_path),
        },
    }
    _write_json(training_dir / "render-audit.json", render_audit)
    training_summary = _train_one_recipe(
        model=model,
        examples=train_examples,
        tokenizer=tokenizer,
        optimization=recipe["optimization"],
        output_dir=training_dir,
    )

    evaluation_dir = export / "evaluation"
    evaluation_dir.mkdir()
    predictions_path = evaluation_dir / "predictions.jsonl"
    generation = _generate_development(
        model=model,
        tokenizer=tokenizer,
        rows=rendered_dev,
        max_seq_len=max_seq_len,
        max_new_tokens=int(recipe["evaluation"]["max_new_tokens"]),
        batch_size=int(recipe["evaluation"]["generation_batch_size"]),
        generation_overrides=recipe["evaluation"]["generation_config_overrides"],
        output_path=predictions_path,
    )
    write_scores(args.dev_manifest, predictions_path, evaluation_dir / "scores")
    aggregate = json.loads(
        (evaluation_dir / "scores" / "aggregate.json").read_text(encoding="utf-8")
    )
    elapsed = time.monotonic() - started
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "jarvis_machine_id": args.jarvis_machine_id,
        "status": "completed",
        "preregistration_sha256": RECIPE_SHA256,
        "attempt_provenance": provenance,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, **identity},
        "data": json.loads((data_dir / "identity.json").read_text(encoding="utf-8")),
        "training": training_summary,
        "generation": generation,
        "evaluation": {
            "scorer_version": MOBILE_ACTIONS_SCORER_VERSION,
            "aggregate": aggregate,
            "sample_scores_sha256": sha256_file(evaluation_dir / "scores" / "sample_scores.jsonl"),
            "checkpoint_policy": "final_only",
            "dev_trials_consumed": 1,
        },
        "claim_limit": recipe["claim_limit"],
        "elapsed_seconds": elapsed,
        "estimated_cost_inr": elapsed * args.hourly_cost_inr / 3600.0,
    }
    _write_json(export / "result.json", result)
    _copy_essential(export, export / "essential")
    _write_json(
        export / "artifact-sha256.json",
        _tree_sha256(export, excluded_names=frozenset({"artifact-sha256.json"})),
    )
    return result


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _run_id(value: str) -> str:
    if _RUN_ID.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must match YYYYMMDD-HHMM-lowercase-name-sN")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=_run_id, required=True)
    parser.add_argument("--jarvis-machine-id", type=_positive_int, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--attempt-preregistration", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=RECIPE_PATH)
    parser.add_argument("--artifact-root", type=Path, default=Path("/home/barun-artifacts"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/hf-cache"))
    parser.add_argument("--hourly-cost-inr", type=_nonnegative_float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    failure_root = args.artifact_root / args.run_id / "export"
    try:
        result = run(args)
    except ExistingMatchedRunError:
        raise
    except BaseException as error:
        failure_root.mkdir(parents=True, exist_ok=True)
        _write_json(
            failure_root / "failure.json",
            {
                "schema_version": "barun-mobile-matched-failure-v1",
                "run_id": args.run_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "official_eval_rows_read": 0,
            },
        )
        raise
    print(json.dumps(result, allow_nan=False, sort_keys=True))


__all__ = [
    "AUDIT_SHA256",
    "DEV_MANIFEST_SHA256",
    "DEV_MEMBERSHIP_SHA256",
    "EXPECTED_UNIQUE_PARAMETERS",
    "MATCHED_RECIPE_SCHEMA_VERSION",
    "MODEL_ID",
    "MODEL_REVISION",
    "RECIPE_PATH",
    "RECIPE_SHA256",
    "TRAIN_MANIFEST_SHA256",
    "TRAIN_MEMBERSHIP_SHA256",
    "FrozenMobileInputs",
    "MatchedBaselineError",
    "NativeChatExample",
    "ParsedActionPrompt",
    "build_parser",
    "count_unique_parameters",
    "load_frozen_recipe",
    "main",
    "parse_action_prompt",
    "run",
    "tokenize_native_chat_example",
    "verify_frozen_mobile_inputs",
]
