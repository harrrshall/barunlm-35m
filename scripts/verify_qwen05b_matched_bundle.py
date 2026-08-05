"""Independently verify a downloaded Qwen matched-baseline evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from safetensors import safe_open

import barunlm.evaluation.action_ir as action_ir_module
import barunlm.evaluation.evaluator as evaluator_module
import barunlm.evaluation.mobile_actions as mobile_actions_module
from barunlm.evaluation.mobile_actions import aggregate_record, score_rows

EXPECTED_RUN_ID = "20260803-2122-mobile-qwen05b-matched-s17"
EXPECTED_INSTANCE_NAME = "barun-qwen05b-matched-2122-s17"
EXPECTED_MACHINE_ID = 463689
EXPECTED_MANAGED_RUN_ID = "r_85efbb92"
EXPECTED_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EXPECTED_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EXPECTED_TRAIN_SHA256 = "131473ccb5bfb51cac0439b42159e72ec4c598025e50364a52b122b056c2e84e"
EXPECTED_DEV_SHA256 = "988bdce5874d1f1a775feeb5ba2b58cd2bdc128f57e73cb9a63d535fae7c1d55"
EXPECTED_AUDIT_SHA256 = "dc756f97c0a7ef706ec8ffefe2d57cf7e16d75a6ccd932f906d16b6a6ee2f83c"
EXPECTED_RECIPE_SHA256 = "6855eaf52c34089d802b8584c566b95619f9ff2780d4f130f333126bc33357e1"
EXPECTED_PROVENANCE_SHA256 = "1a1c19cdfacaa25e296ab1c3f92a42ebc8c08802c0cec8047ab581d50669769e"
EXPECTED_CANDIDATE_PREDICTIONS_SHA256 = (
    "e5aea59e5090e3aa7c9dbd6b1a810c753410a159ce18dc8bb889fde3a58b8665"
)
EXPECTED_CANDIDATE_SAMPLES_SHA256 = (
    "a29c4d5c83493ec437a9f0fee362ea54cf97c2219844e5b46ba60d0f759f556c"
)
EXPECTED_CANDIDATE_AGGREGATE_SHA256 = (
    "5d7244d2fa449a7ce4b2aa3a20fc095fb118a38920b1dd9181f56669a7ddfea1"
)
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "badbb1dc91daa78a7807b998c7c3042a6b35523e4bd3024b7714ab6fdb69ba06"
)
EXPECTED_PARAMETERS = 494_032_768
EXPECTED_ROWS = 756
EXPECTED_TRAIN_ROWS = 7_937
PROTECTED_MACHINE_ID = 463058
EXPECTED_SCORER_SHA256 = "4c7fc82d546996d5ba739a24ca59abf805acf44221f65edeccee1c095babb7ac"
EXPECTED_ACTION_IR_SHA256 = "563d40afabd03bc81fdd4e2a0bf0224c5f6e9b686ee8f9f868611043d42642c8"
EXPECTED_EVALUATOR_SHA256 = "36db751c9b4d791e45ebdd5b6338f3364789aefcd33ebed271c1023dcbf0d3a5"
EXPECTED_CHECKPOINT_HEADER_SHA256 = (
    "bd808652be103f0e4e29786b9aa8829e0f38607880486926c7246ee2de577a76"
)
EXPECTED_CHECKPOINT_FILES = {
    "added_tokens.json": "58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b",
    "config.json": "ca17f5fb7fd1acdd0a7b4668680fc25ee73801c3e442e81b5e8217b5eb74ed24",
    "generation_config.json": "0b69aa62af0e081a1a5ce1400b09f77603f56ed7250e0e1e160f33be1b555ea9",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": "21f3f234ce9bfc2fa387cf396807fa35f5eb72e311f23a89cd98c98224c78793",
    "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
    "tokenizer.json": "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa",
    "tokenizer_config.json": "57fea41fa99fce34f91450643e48b3bfbb078c32c6831a3c06834f122c200510",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
EXPECTED_ARTIFACT_FILES = {
    "attempt-preregistration.json": "d569b463346304b00249588c9f42f6a784a2d7e8f55b1340d27b270603cac05c",
    "bundle-manifest.json": "72d89b9de38c41835518904608130c5619699379619c2542fd6779274c4ef1fe",
    **{f"checkpoint/{name}": digest for name, digest in EXPECTED_CHECKPOINT_FILES.items()},
    "data/audit.json": EXPECTED_AUDIT_SHA256,
    "data/identity.json": "679874c4dd928f32a99a2e58b31a9c90fd2b3e5ed9ff15098acb5852a068dd75",
    "environment.json": "5c129a8b96c581bb70ca6c14fb1ed027c31d455761afd64c66526c5c437010c6",
    "evaluation/predictions.jsonl": "836788091d08b931bf37565d197c4053e6e6f4664d379c38e7668ae9b8739b08",
    "evaluation/predictions.jsonl.manifest.json": "260f45f31b029f226018b625e803900b709b638ac79364e84a066ddefc213366",
    "evaluation/scores/aggregate.json": "2184b93d383f1136c050d95e71fb129bbe083b3fea4de86da6bdf5ec7f3e1756",
    "evaluation/scores/sample_scores.jsonl": "5ed69b469de56f8afd3b73dedf9bad2009a0fb21ce01ee91a346167ed0d47bda",
    "model-identity.json": "97bbfbdb4afc1f337cf4cb1779f1abe249de876ecf3adf17cd64c457033bd680",
    "preregistration.json": EXPECTED_RECIPE_SHA256,
    "repository-tests.log": "a202b7307875ab17d8110bc5201e2cefb4820fe80c40c9f25f068447f1dc6b35",
    "result.json": "ed733c35d54add1e956dcc1506b2434c01ebd3c707b8bcff3dac4ad760e9c17f",
    "snapshot-manifest.json": "a186a5b67e0f8db50228005a13d527bb9f5b4be9d68081505ec076982da6be3c",
    "source-snapshot.json": "f9c187539e097ac21bec8315daf75c48e74c6f3bcc8e9cfe969d3aaf568d9cb6",
    "training/metrics.jsonl": "ac7995539f39cc23502247272a8bd751b7980eaa21c9605fe453382cd308d3e5",
    "training/render-audit.json": "461e2d6da2e7da7b89c18963a9585c4fa842b0d41881ff811db91d6ab176c44e",
    "training/rendered-dev.jsonl": "b1202ee151804982fea24126a9197410157f4c42cf9968a74af279aa6e88af7d",
    "training/rendered-train.jsonl": "31d33028855b85baceb7d026cfe748fb6f56f0f74c9de9ad7bb243864d395c49",
    "training/summary.json": "b1b43823eff3350df8e5f977a38525dc1fc27a8fa0f113e4660323e5d2d92da8",
}
EXPECTED_BUNDLE_PATHS = frozenset({"artifact-sha256.json", *EXPECTED_ARTIFACT_FILES})
_SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)JL_API_KEY\s*=\s*[^\s\"']+"),
    re.compile(r"(?i)(?:api[_-]?key|password|secret)\s*[:=]\s*[\"'][^\"']{8,}"),
)
_SECRET_NAMES = {".env", ".netrc", "credentials.json", "id_ed25519", "id_rsa"}
_SECRET_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}


class VerificationError(RuntimeError):
    """Downloaded evidence failed a required independent check."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON constant {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise VerificationError(f"non-finite JSON float {value}")
    return parsed


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_strict_json(text: str, *, label: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
            object_pairs_hook=_unique_json_object,
        )
    except VerificationError:
        raise
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid strict JSON in {label}: {exc}") from exc


def strict_json(path: Path) -> dict[str, Any]:
    payload = parse_strict_json(path.read_text(encoding="utf-8"), label=str(path))
    if not isinstance(payload, dict):
        raise VerificationError(f"{path} is not a JSON object")
    return payload


def strict_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            raise VerificationError(f"blank JSONL line in {path}:{line_number}")
        payload = parse_strict_json(line, label=f"{path}:{line_number}")
        if not isinstance(payload, dict):
            raise VerificationError(f"JSONL row is not an object in {path}:{line_number}")
        rows.append(payload)
    return rows


def json_native(payload: Any) -> Any:
    """Normalize records exactly as an immutable JSON artifact would."""

    return json.loads(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def file_map(root: Path, *, excluded: set[str] | None = None) -> dict[str, str]:
    excluded = excluded or set()
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise VerificationError(f"tree contains symlink: {relative}")
        if path.is_file() and relative not in excluded:
            result[relative] = sha256_file(path)
    return result


def safe_manifest_member(root: Path, name: str, *, label: str) -> Path:
    pure = PurePosixPath(name)
    if (
        not name
        or pure.is_absolute()
        or name != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise VerificationError(f"{label} contains unsafe path: {name!r}")
    root_resolved = root.resolve()
    candidate = root / pure
    if candidate.is_symlink() or not candidate.is_file():
        raise VerificationError(f"{label} file absent or unsafe: {name}")
    try:
        candidate.resolve().relative_to(root_resolved)
    except ValueError as exc:
        raise VerificationError(f"{label} path escapes root: {name}") from exc
    return candidate


def verify_hash_map(root: Path, expected: dict[str, Any], *, label: str) -> None:
    normalized: dict[str, str] = {}
    for name, digest in expected.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise VerificationError(f"{label} contains a non-string identity")
        candidate = safe_manifest_member(root, name, label=label)
        if sha256_file(candidate) != digest:
            raise VerificationError(f"{label} hash mismatch: {name}")
        normalized[name] = digest
    if len(normalized) != len(expected):
        raise VerificationError(f"{label} contains duplicate identities")


def safetensor_inventory(path: Path) -> dict[str, dict[str, Any]]:
    """Read tensor shapes and dtypes through the stable safe_open keys interface."""

    with safe_open(path, framework="pt", device="cpu") as handle:
        names = sorted(handle.keys())
        return {
            name: {
                "shape": handle.get_slice(name).get_shape(),
                "dtype": handle.get_slice(name).get_dtype(),
            }
            for name in names
        }


def safetensor_shapes(path: Path) -> dict[str, list[int]]:
    return {name: record["shape"] for name, record in safetensor_inventory(path).items()}


def safetensor_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise VerificationError("checkpoint lacks safetensors header length")
        header_bytes = int.from_bytes(prefix, byteorder="little", signed=False)
        header = handle.read(header_bytes)
    if len(header) != header_bytes:
        raise VerificationError("checkpoint has a truncated safetensors header")
    return {
        "json_bytes": header_bytes,
        "json_sha256": hashlib.sha256(header).hexdigest(),
        "data_bytes": path.stat().st_size - 8 - header_bytes,
    }


def assert_json_equivalent(
    actual: Any,
    expected: Any,
    *,
    label: str,
    float_tolerance: float = 1e-12,
) -> float:
    """Require structural equality, tolerating only sub-picometric float drift."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise VerificationError(f"{label} object keys differ")
        return max(
            (
                assert_json_equivalent(
                    actual[key],
                    expected[key],
                    label=f"{label}.{key}",
                    float_tolerance=float_tolerance,
                )
                for key in expected
            ),
            default=0.0,
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise VerificationError(f"{label} list shape differs")
        return max(
            (
                assert_json_equivalent(
                    actual_value,
                    expected_value,
                    label=f"{label}[{index}]",
                    float_tolerance=float_tolerance,
                )
                for index, (actual_value, expected_value) in enumerate(zip(actual, expected))
            ),
            default=0.0,
        )
    if type(expected) is float:
        if type(actual) is not float:
            raise VerificationError(f"{label} changed type")
        difference = abs(actual - expected)
        if not math.isfinite(difference) or difference > float_tolerance:
            raise VerificationError(f"{label} float differs by {difference}")
        return difference
    if actual != expected or type(actual) is not type(expected):
        raise VerificationError(f"{label} differs")
    return 0.0


def verify_bundle(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    actual_paths = frozenset(file_map(bundle))
    if actual_paths != EXPECTED_BUNDLE_PATHS:
        missing = sorted(EXPECTED_BUNDLE_PATHS - actual_paths)
        extra = sorted(actual_paths - EXPECTED_BUNDLE_PATHS)
        raise VerificationError(f"essential tree differs; missing={missing}, extra={extra}")
    artifact_manifest = strict_json(bundle / "artifact-sha256.json")
    if artifact_manifest != EXPECTED_ARTIFACT_FILES:
        raise VerificationError("essential artifact manifest differs from frozen identities")
    actual = file_map(bundle, excluded={"artifact-sha256.json"})
    if artifact_manifest != actual:
        raise VerificationError("essential artifact manifest does not match the downloaded tree")
    bundle_manifest = strict_json(bundle / "bundle-manifest.json")
    if bundle_manifest.get("official_eval_artifacts_included") is not False:
        raise VerificationError("bundle does not assert official-evaluation exclusion")
    if bundle_manifest.get("optimizer_state_included") is not False:
        raise VerificationError("bundle does not assert optimizer-state exclusion")
    files_sha256 = bundle_manifest.get("files_sha256")
    if not isinstance(files_sha256, dict):
        raise VerificationError("bundle manifest lacks file hashes")
    expected_inner = {
        name: digest
        for name, digest in EXPECTED_ARTIFACT_FILES.items()
        if name != "bundle-manifest.json"
    }
    if files_sha256 != expected_inner:
        raise VerificationError("bundle manifest does not have the exact frozen inner tree")
    verify_hash_map(bundle, files_sha256, label="bundle manifest")

    training = strict_json(bundle / "training" / "summary.json")
    checkpoint_hashes = training.get("checkpoint_files_sha256")
    if checkpoint_hashes != EXPECTED_CHECKPOINT_FILES:
        raise VerificationError("training summary checkpoint identities changed")
    checkpoint_tree = file_map(bundle / "checkpoint")
    if checkpoint_tree != EXPECTED_CHECKPOINT_FILES:
        raise VerificationError("checkpoint tree does not exactly match frozen identities")
    verify_hash_map(bundle / "checkpoint", checkpoint_hashes, label="checkpoint summary")
    checkpoint = bundle / "checkpoint" / "model.safetensors"
    tensor_inventory = safetensor_inventory(checkpoint)
    parameters = sum(math.prod(record["shape"]) for record in tensor_inventory.values())
    if parameters != EXPECTED_PARAMETERS:
        raise VerificationError(
            f"checkpoint has {parameters} serialized parameters, expected {EXPECTED_PARAMETERS}"
        )
    if len(tensor_inventory) != 290:
        raise VerificationError("checkpoint tensor count changed")
    if {record["dtype"] for record in tensor_inventory.values()} != {"BF16"}:
        raise VerificationError("checkpoint is not an all-BF16 tensor inventory")
    if "lm_head.weight" in tensor_inventory:
        raise VerificationError("tied checkpoint unexpectedly serialized a separate lm_head")
    header = safetensor_header(checkpoint)
    if (
        header["json_bytes"] != 32_280
        or header["json_sha256"] != EXPECTED_CHECKPOINT_HEADER_SHA256
        or header["data_bytes"] != 988_065_536
    ):
        raise VerificationError(
            "checkpoint tensor signature/header differs from pinned architecture"
        )
    config = strict_json(bundle / "checkpoint" / "config.json")
    expected_architecture = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "vocab_size": 151936,
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "max_position_embeddings": 32768,
        "use_sliding_window": False,
        "sliding_window": None,
    }
    if any(config.get(key) != value for key, value in expected_architecture.items()):
        raise VerificationError("checkpoint architecture fields differ from frozen Qwen identity")
    config_formula = (
        int(config["vocab_size"]) * int(config["hidden_size"])
        + int(config["num_hidden_layers"])
        * (
            int(config["hidden_size"]) ** 2
            + int(config["hidden_size"])
            + 2
            * (
                int(config["num_key_value_heads"])
                * (int(config["hidden_size"]) // int(config["num_attention_heads"]))
                * int(config["hidden_size"])
                + int(config["num_key_value_heads"])
                * (int(config["hidden_size"]) // int(config["num_attention_heads"]))
            )
            + int(config["hidden_size"]) ** 2
            + 3 * int(config["hidden_size"]) * int(config["intermediate_size"])
            + 2 * int(config["hidden_size"])
        )
        + int(config["hidden_size"])
    )
    if config_formula != EXPECTED_PARAMETERS or config.get("tie_word_embeddings") is not True:
        raise VerificationError("checkpoint architecture does not derive the expected tied count")
    tokenizer_config = strict_json(bundle / "checkpoint" / "tokenizer_config.json")
    chat_template = tokenizer_config.get("chat_template")
    if (
        not isinstance(chat_template, str)
        or hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
        != "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
        or tokenizer_config.get("eos_token") != "<|im_end|>"
        or tokenizer_config.get("pad_token") != "<|endoftext|>"
    ):
        raise VerificationError("checkpoint tokenizer semantic identity changed")
    return {
        "artifact_files": len(actual),
        "checkpoint_files": len(checkpoint_hashes),
        "checkpoint_model_sha256": sha256_file(checkpoint),
        "checkpoint_header": header,
        "serialized_tensor_count": len(tensor_inventory),
        "serialized_dtype": "BF16",
        "serialized_parameters": parameters,
        "config_derived_parameters": config_formula,
        "exact_tree_verified": True,
    }


def require_fields(
    payload: dict[str, Any], expected: dict[tuple[str, ...], Any], *, label: str
) -> None:
    for path, expected_value in expected.items():
        current: Any = payload
        for key in path:
            if not isinstance(current, dict) or key not in current:
                raise VerificationError(f"{label} lacks {'.'.join(path)}")
            current = current[key]
        if current != expected_value or type(current) is not type(expected_value):
            raise VerificationError(f"{label} changed {'.'.join(path)}")


def indexed_ids(rows: list[dict[str, Any]], *, label: str) -> set[str]:
    result: set[str] = set()
    for row in rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise VerificationError(f"{label} row lacks an ID")
        if sample_id in result:
            raise VerificationError(f"{label} has duplicate ID {sample_id}")
        result.add(sample_id)
    return result


def membership_sha256(sample_ids: set[str]) -> str:
    payload = "".join(f"{sample_id}\n" for sample_id in sorted(sample_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_semantics(args: argparse.Namespace, bundle: Path) -> dict[str, Any]:
    expected_inputs = {
        args.train_manifest: EXPECTED_TRAIN_SHA256,
        args.dev_manifest: EXPECTED_DEV_SHA256,
        args.audit: EXPECTED_AUDIT_SHA256,
        args.recipe: EXPECTED_RECIPE_SHA256,
        args.provenance: EXPECTED_PROVENANCE_SHA256,
        args.candidate_predictions: EXPECTED_CANDIDATE_PREDICTIONS_SHA256,
        args.candidate_predictions_manifest: EXPECTED_CANDIDATE_MANIFEST_SHA256,
        args.candidate_sample_scores: EXPECTED_CANDIDATE_SAMPLES_SHA256,
        args.candidate_aggregate: EXPECTED_CANDIDATE_AGGREGATE_SHA256,
    }
    for path, expected_hash in expected_inputs.items():
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_hash:
            raise VerificationError(f"frozen input identity changed: {path}")

    train_rows = strict_jsonl(args.train_manifest)
    dev_rows = strict_jsonl(args.dev_manifest)
    train_ids = indexed_ids(train_rows, label="train manifest")
    dev_ids = indexed_ids(dev_rows, label="development manifest")
    if (
        len(train_ids) != EXPECTED_TRAIN_ROWS
        or len(dev_ids) != EXPECTED_ROWS
        or train_ids & dev_ids
        or membership_sha256(train_ids)
        != "4cdfc3649c21a9f0d3dc4d8d9cffcae3cb5c6abc4414a9fb09636e9220afe96f"
        or membership_sha256(dev_ids)
        != "31e0170c1de49d9c4591064587c405f0cb9f981e7026ac457f6fa2902b3647de"
    ):
        raise VerificationError("train/development population identity changed")

    audit = strict_json(args.audit)
    require_fields(
        audit,
        {
            ("schema_version",): "barun-mobile-actions-audit-v2",
            ("source", "dataset_id"): "google/mobile-actions",
            ("source", "revision"): "e920309bc2acbc2e99a5e3201cf37df2b9fd9151",
            ("license", "id"): "CC-BY-4.0",
            ("counts", "derived", "train"): EXPECTED_TRAIN_ROWS,
            ("counts", "derived", "dev"): EXPECTED_ROWS,
            ("counts", "records_dropped"): 0,
            ("counts", "records_truncated"): 0,
            ("official_eval", "rows"): 961,
            ("official_eval", "opaque_unparsed"): True,
            ("official_eval", "materialized"): False,
            ("official_eval", "labels_parsed"): False,
            ("official_eval", "prompts_parsed"): False,
            ("official_eval", "targets_parsed"): False,
            ("official_eval", "tool_schemas_parsed"): False,
            ("official_eval", "lengths_computed"): False,
            ("official_eval", "overlaps_computed"): False,
        },
        label="data audit",
    )
    if strict_json(bundle / "data" / "audit.json") != audit:
        raise VerificationError("bundled audit differs from frozen local audit")

    recipe = strict_json(args.recipe)
    require_fields(
        recipe,
        {
            ("schema_version",): "barun-mobile-matched-baseline-recipe-v1",
            ("run_id",): EXPECTED_RUN_ID,
            ("status",): "frozen_before_remote_launch_or_model_download",
            ("model", "id"): EXPECTED_MODEL_ID,
            ("model", "revision"): EXPECTED_MODEL_REVISION,
            ("model", "expected_unique_parameters"): EXPECTED_PARAMETERS,
            ("model", "trust_remote_code"): False,
            ("optimization", "seed"): 17,
            ("optimization", "epochs"): 1,
            ("optimization", "precision"): "bfloat16",
            ("optimization", "per_device_batch_size"): 21,
            ("optimization", "gradient_accumulation_steps"): 3,
            ("optimization", "effective_batch_size"): 63,
            ("optimization", "expected_optimizer_steps"): 126,
            ("optimization", "learning_rate"): 0.00002,
            ("optimization", "warmup_steps"): 12,
            ("optimization", "checkpoint_policy"): "final_only",
            ("optimization", "early_stopping"): False,
            ("information_budget", "examples_presented"): EXPECTED_TRAIN_ROWS,
            ("information_budget", "train_passes"): 1,
            ("information_budget", "augmentations"): "none",
            ("information_budget", "teacher_information"): "none",
            ("information_budget", "packing"): False,
            ("information_budget", "truncation_or_row_drop"): False,
            ("evaluation", "generation_batch_size"): 64,
            ("evaluation", "max_new_tokens"): 192,
            ("evaluation", "grammar_constrained"): False,
            ("trial_budget", "recipes"): 1,
            ("trial_budget", "training_seeds"): [17],
            ("trial_budget", "development_scores_consulted"): 1,
        },
        label="frozen recipe",
    )
    if strict_json(bundle / "preregistration.json") != recipe:
        raise VerificationError("bundled recipe differs from frozen local recipe")

    provenance = strict_json(args.provenance)
    require_fields(
        provenance,
        {
            ("model_id",): EXPECTED_MODEL_ID,
            ("revision",): EXPECTED_MODEL_REVISION,
            ("public",): True,
            ("gated",): False,
            ("license",): "Apache-2.0",
            ("architecture_class",): "Qwen2ForCausalLM",
            ("trust_remote_code",): False,
            ("exact_unique_parameters",): EXPECTED_PARAMETERS,
        },
        label="model provenance",
    )

    attempt = strict_json(bundle / "attempt-preregistration.json")
    require_fields(
        attempt,
        {
            ("schema_version",): "barun-qwen05b-matched-attempt-v1",
            ("run_id",): EXPECTED_RUN_ID,
            ("status",): "frozen_before_model_download_or_development_scoring",
            ("development_predictions_or_scores_observed",): False,
            ("qwen_model_weights_downloaded",): False,
            ("official_evaluation_rows_read_or_materialized",): 0,
            ("model", "id"): EXPECTED_MODEL_ID,
            ("model", "revision"): EXPECTED_MODEL_REVISION,
            ("model", "provenance_sha256"): EXPECTED_PROVENANCE_SHA256,
            ("scientific_recipe", "sha256"): EXPECTED_RECIPE_SHA256,
            ("prelaunch_inventory", "project_machine_id"): EXPECTED_MACHINE_ID,
            ("compute", "instance_name"): EXPECTED_INSTANCE_NAME,
        },
        label="attempt preregistration",
    )
    protected_ids = attempt["prelaunch_inventory"]["protected_machine_ids"]
    if (
        not isinstance(protected_ids, list)
        or len(protected_ids) != len(set(protected_ids))
        or PROTECTED_MACHINE_ID not in protected_ids
        or EXPECTED_MACHINE_ID in protected_ids
    ):
        raise VerificationError("attempt protected-resource denylist is invalid")

    source = strict_json(bundle / "source-snapshot.json")
    require_fields(
        source,
        {
            ("schema_version",): "barun-qwen05b-source-snapshot-v1",
            ("run_id",): EXPECTED_RUN_ID,
            ("created_before_model_download_or_development_scoring",): True,
            ("file_count",): 124,
            ("content_bytes",): 21324978,
            (
                "content_tree_sha256",
            ): "1a131cc8bc7a24c73c7c39cacc0c313f6ce7037ef5033077e0e8f4cfbbadeabb",
            ("git_base",): "fe4ee1b773160c9bfdf648f9da5bacd7a481fb16",
            (
                "dirty_tree_patch_sha256",
            ): "ee1e2b1dd5288115caaf59d2206758302bf59ef468f39ef0d2e06cb118bedbb2",
        },
        label="source snapshot",
    )

    snapshot = strict_json(bundle / "snapshot-manifest.json")
    expected_snapshot_hashes = {
        "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
        "README.md": "b19c806a904db6dc878a0462e70b551f6b7ac78dfbb88c2eb966ca2b9109ae15",
        "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
        "generation_config.json": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
        "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
        "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
        "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
        "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    }
    if (
        snapshot.get("repo_id") != EXPECTED_MODEL_ID
        or snapshot.get("revision") != EXPECTED_MODEL_REVISION
        or snapshot.get("files_sha256") != expected_snapshot_hashes
    ):
        raise VerificationError("pinned source snapshot identity changed")

    model_identity = strict_json(bundle / "model-identity.json")
    require_fields(
        model_identity,
        {
            ("config_commit_hash",): EXPECTED_MODEL_REVISION,
            ("unique_parameters",): EXPECTED_PARAMETERS,
            ("trainable_unique_parameters",): EXPECTED_PARAMETERS,
            ("eos_token_id",): 151645,
            ("pad_token_id",): 151643,
            (
                "native_chat_template_sha256",
            ): "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f",
        },
        label="model identity",
    )

    data_identity = strict_json(bundle / "data" / "identity.json")
    require_fields(
        data_identity,
        {
            ("dataset",): "google/mobile-actions",
            ("revision",): "e920309bc2acbc2e99a5e3201cf37df2b9fd9151",
            ("train_rows",): EXPECTED_TRAIN_ROWS,
            ("dev_rows",): EXPECTED_ROWS,
            ("train_manifest_sha256",): EXPECTED_TRAIN_SHA256,
            ("dev_manifest_sha256",): EXPECTED_DEV_SHA256,
            ("official_eval_rows",): 961,
            ("official_eval_rows_available_to_runner",): 0,
            ("official_eval_materialized",): False,
        },
        label="data identity",
    )

    environment = strict_json(bundle / "environment.json")
    require_fields(
        environment,
        {
            ("cuda_device_name",): "NVIDIA H200",
            ("bf16_supported",): True,
            ("determinism", "seed"): 17,
            ("determinism", "deterministic_algorithms"): True,
            ("determinism", "tf32_matmul"): False,
            ("determinism", "tf32_cudnn"): False,
            ("packages", "transformers"): "4.48.3",
            ("packages", "safetensors"): "0.8.0",
            ("packages", "tokenizers"): "0.21.4",
        },
        label="remote environment",
    )
    command = environment.get("command")
    if not isinstance(command, list) or command[-2:] != ["--jarvis-machine-id", "463689"]:
        raise VerificationError("remote command does not bind the exact owned machine")

    render_audit = strict_json(bundle / "training" / "render-audit.json")
    require_fields(
        render_audit,
        {
            ("schema_version",): "barun-native-chat-action-render-v1",
            ("assistant_output_only_labels",): True,
            ("extra_semantic_instructions",): False,
            ("packing",): False,
            ("truncation",): False,
            ("splits", "train", "rows"): EXPECTED_TRAIN_ROWS,
            ("splits", "train", "overlength_or_dropped"): 0,
            ("splits", "dev", "rows"): EXPECTED_ROWS,
            ("splits", "dev", "overlength_or_dropped"): 0,
        },
        label="render audit",
    )
    rendered_train = strict_jsonl(bundle / "training" / "rendered-train.jsonl")
    rendered_dev = strict_jsonl(bundle / "training" / "rendered-dev.jsonl")
    if indexed_ids(rendered_train, label="rendered train") != train_ids:
        raise VerificationError("rendered train IDs differ from frozen manifest")
    if indexed_ids(rendered_dev, label="rendered dev") != dev_ids:
        raise VerificationError("rendered development IDs differ from frozen manifest")
    if any(row.get("truncated_or_dropped") is not False for row in rendered_train + rendered_dev):
        raise VerificationError("render evidence contains a truncated or dropped row")

    training = strict_json(bundle / "training" / "summary.json")
    require_fields(
        training,
        {
            ("method",): "full_parameter_response_only_sft",
            ("epochs",): 1,
            ("examples_presented",): EXPECTED_TRAIN_ROWS,
            ("unique_examples_presented",): EXPECTED_TRAIN_ROWS,
            ("optimizer_steps",): 126,
            ("final_checkpoint_policy",): "final_only",
            ("checkpoint_files_sha256",): EXPECTED_CHECKPOINT_FILES,
        },
        label="training summary",
    )
    metrics = strict_jsonl(bundle / "training" / "metrics.jsonl")
    if (
        len(metrics) != 126
        or [row.get("optimizer_step") for row in metrics] != list(range(1, 127))
        or sum(row.get("examples", 0) for row in metrics) != EXPECTED_TRAIN_ROWS
    ):
        raise VerificationError("training metrics do not prove the frozen 126-step presentation")

    generation = strict_json(bundle / "evaluation" / "predictions.jsonl.manifest.json")
    neutral_overrides = {
        "do_sample": False,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "temperature": None,
        "top_k": None,
        "top_p": None,
    }
    require_fields(
        generation,
        {
            ("schema_version",): "barun-native-chat-generation-v1",
            ("decoding",): "unconstrained_deterministic_greedy",
            ("grammar_constrained",): False,
            ("batch_size",): 64,
            ("max_new_tokens",): 192,
            ("examples",): EXPECTED_ROWS,
            ("generated",): EXPECTED_ROWS,
            ("failed",): 0,
            ("truncated",): 0,
            ("generation_config_overrides",): neutral_overrides,
            ("predictions_sha256",): EXPECTED_ARTIFACT_FILES["evaluation/predictions.jsonl"],
        },
        label="generation manifest",
    )

    result = strict_json(bundle / "result.json")
    require_fields(
        result,
        {
            ("schema_version",): "barun-mobile-matched-baseline-result-v1",
            ("run_id",): EXPECTED_RUN_ID,
            ("status",): "completed",
            ("jarvis_machine_id",): EXPECTED_MACHINE_ID,
            ("preregistration_sha256",): EXPECTED_RECIPE_SHA256,
            ("model", "id"): EXPECTED_MODEL_ID,
            ("model", "revision"): EXPECTED_MODEL_REVISION,
            ("model", "unique_parameters"): EXPECTED_PARAMETERS,
            ("training", "optimizer_steps"): 126,
            ("training", "examples_presented"): EXPECTED_TRAIN_ROWS,
            ("evaluation", "dev_trials_consumed"): 1,
            ("generation", "generation_config_overrides"): neutral_overrides,
            ("generation", "failed"): 0,
            ("generation", "truncated"): 0,
            ("evaluation", "aggregate", "ast_exact_match", "numerator"): 663,
            ("evaluation", "aggregate", "ast_exact_match", "denominator"): EXPECTED_ROWS,
            ("evaluation", "aggregate", "parse_valid", "numerator"): 755,
            ("evaluation", "aggregate", "schema_valid", "numerator"): 754,
            ("evaluation", "aggregate", "catastrophic_unauthorized_actions"): 0,
        },
        label="terminal result",
    )
    if result["evaluation"]["aggregate"] != strict_json(
        bundle / "evaluation" / "scores" / "aggregate.json"
    ):
        raise VerificationError("terminal result aggregate differs from recorded aggregate")

    repository_tests = (bundle / "repository-tests.log").read_text(encoding="utf-8")
    if "73 passed" not in repository_tests or " failed" in repository_tests:
        raise VerificationError("remote repository-test evidence is not a clean 73-test pass")

    candidate_manifest = strict_json(args.candidate_predictions_manifest)
    require_fields(
        candidate_manifest,
        {
            ("schema_version",): "barun-greedy-generation-v1",
            ("manifest_sha256",): EXPECTED_DEV_SHA256,
            ("predictions_sha256",): EXPECTED_CANDIDATE_PREDICTIONS_SHA256,
            ("examples",): EXPECTED_ROWS,
            ("generated",): EXPECTED_ROWS,
            ("failed",): 0,
            ("truncated",): 0,
            ("max_new_tokens",): 192,
        },
        label="candidate generation manifest",
    )

    code_hashes = {
        "verifier": sha256_file(Path(__file__).resolve()),
        "mobile_scorer": sha256_file(Path(mobile_actions_module.__file__).resolve()),
        "action_ir": sha256_file(Path(action_ir_module.__file__).resolve()),
        "evaluator": sha256_file(Path(evaluator_module.__file__).resolve()),
    }
    if (
        code_hashes["mobile_scorer"] != EXPECTED_SCORER_SHA256
        or code_hashes["action_ir"] != EXPECTED_ACTION_IR_SHA256
        or code_hashes["evaluator"] != EXPECTED_EVALUATOR_SHA256
    ):
        raise VerificationError("local independent scoring implementation differs from frozen code")
    return {
        "input_sha256": {str(path): digest for path, digest in expected_inputs.items()},
        "code_sha256": code_hashes,
        "train_unique_ids": len(train_ids),
        "development_unique_ids": len(dev_ids),
        "train_dev_id_overlap": 0,
        "training_metric_rows": len(metrics),
        "rendered_train_rows": len(rendered_train),
        "rendered_development_rows": len(rendered_dev),
        "official_mobile_evaluation_rows_read": 0,
        "frozen_fields_cross_bound": True,
    }


def score_again(
    *,
    manifest: Path,
    predictions: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate, samples = score_rows(strict_jsonl(manifest), strict_jsonl(predictions))
    return json_native(aggregate_record(aggregate)), json_native(
        [sample.to_record() for sample in samples]
    )


def indexed_samples(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise VerificationError(f"{label} has a row without sample_id")
        if sample_id in indexed:
            raise VerificationError(f"{label} has duplicate ID {sample_id}")
        indexed[sample_id] = row
    if len(indexed) != EXPECTED_ROWS:
        raise VerificationError(f"{label} has {len(indexed)} rows, expected {EXPECTED_ROWS}")
    return indexed


def parse_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise VerificationError(f"{label} is not a timestamp string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise VerificationError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise VerificationError(f"{label} is not timezone-aware")
    return parsed


def verify_jarvis(record_path: Path, pause_proof_path: Path, bundle: Path) -> dict[str, Any]:
    record = strict_json(record_path)
    pause_proof = strict_json(pause_proof_path)
    attempt = strict_json(bundle / "attempt-preregistration.json")
    result = strict_json(bundle / "result.json")
    environment = strict_json(bundle / "environment.json")
    machine_id = record.get("machine_id")
    preexisting = {
        row.get("machine_id")
        for row in record.get("preexisting_resources", [])
        if isinstance(row, dict)
    }
    final_instance = record.get("final_instance")
    latest = record.get("latest_run_status")
    download = record.get("artifact_download")
    if (
        record.get("created_by_safe_run") is not True
        or record.get("schema_version") != 1
        or machine_id != EXPECTED_MACHINE_ID
        or machine_id in preexisting
        or machine_id == PROTECTED_MACHINE_ID
        or record.get("instance_name") != EXPECTED_INSTANCE_NAME
        or record.get("run_id") != EXPECTED_MANAGED_RUN_ID
        or record.get("remote_exit_code") != 0
        or record.get("protected_ids_missing_from_live_inventory") != []
        or not isinstance(final_instance, dict)
        or final_instance.get("machine_id") != machine_id
        or final_instance.get("name") != EXPECTED_INSTANCE_NAME
        or final_instance.get("status") != "Paused"
        or not isinstance(latest, dict)
        or latest.get("machine_id") != machine_id
        or latest.get("run_id") != EXPECTED_MANAGED_RUN_ID
        or latest.get("state") != "succeeded"
        or latest.get("exit_code") != 0
        or not isinstance(download, dict)
        or download.get("machine_id") != machine_id
        or download.get("exit_code") != 0
        or download.get("recursive") is not True
        or download.get("direction") != "download"
    ):
        raise VerificationError(
            "Jarvis controller record does not prove successful exact-ID cleanup"
        )
    protected_attempt = set(attempt["prelaunch_inventory"]["protected_machine_ids"])
    protected_binding = set(record["attempt_inventory_binding"]["protected_machine_ids"])
    if preexisting != protected_attempt or protected_attempt != protected_binding:
        raise VerificationError("Jarvis protected inventory differs from bound preregistration")
    if (
        record["attempt_inventory_binding"].get("machine_id") != machine_id
        or record["attempt_inventory_binding"].get("bound_before_attached_run") is not True
        or record["attempt_inventory_binding"].get("before_sha256")
        != "365d65f75c7c0695cddfd78c068a5e92ff7c6b94d6ebc42999ab73ada74df653"
        or record["attempt_inventory_binding"].get("after_sha256")
        != EXPECTED_ARTIFACT_FILES["attempt-preregistration.json"]
        or result.get("jarvis_machine_id") != machine_id
        or attempt["prelaunch_inventory"].get("project_machine_id") != machine_id
        or environment.get("command", [])[-2:] != ["--jarvis-machine-id", str(machine_id)]
    ):
        raise VerificationError("Jarvis machine ID is not cross-bound through frozen evidence")
    expected_remote = f"/home/barun-artifacts/{EXPECTED_RUN_ID}/export/essential"
    if (
        download.get("source") != expected_remote
        or Path(str(download.get("dest"))).resolve() != bundle.resolve()
    ):
        raise VerificationError("artifact download does not bind the exact essential bundle")
    if (
        pause_proof.get("schema_version") != "barun-jarvis-pause-proof-v1"
        or pause_proof.get("query") != f"jl get {machine_id} --json"
        or pause_proof.get("machine_id") != machine_id
        or pause_proof.get("name") != EXPECTED_INSTANCE_NAME
        or pause_proof.get("status") != "Paused"
        or pause_proof.get("filtered") is not True
    ):
        raise VerificationError("fresh pause proof does not match the exact owned machine")
    started = parse_timestamp(record.get("controller_started_at"), label="controller start")
    finished = parse_timestamp(record.get("controller_finished_at"), label="controller finish")
    proof_time = parse_timestamp(pause_proof.get("queried_at_utc"), label="pause proof query")
    event_times = [
        parse_timestamp(event.get("at"), label=f"Jarvis event {index}")
        for index, event in enumerate(record.get("events", []))
        if isinstance(event, dict)
    ]
    if (
        finished <= started
        or proof_time < finished
        or not event_times
        or event_times != sorted(event_times)
        or record["events"][-1].get("event") != "pause_verified"
        or record["events"][-1].get("status") != "Paused"
    ):
        raise VerificationError("Jarvis lifecycle timestamps or terminal event are inconsistent")
    return {
        "machine_id": machine_id,
        "instance_name": record.get("instance_name"),
        "final_state": "Paused",
        "preexisting_machine_count": len(preexisting),
        "lifecycle_seconds": (finished - started).total_seconds(),
        "run_id": latest.get("run_id"),
        "controller_record_sha256": sha256_file(record_path),
        "pause_proof_sha256": sha256_file(pause_proof_path),
        "protected_inventory_exact": True,
        "artifact_download_exact": True,
    }


def secret_and_firewall_audit(bundle: Path) -> dict[str, Any]:
    scanned = 0
    for path in sorted(bundle.rglob("*")):
        if path.is_symlink():
            raise VerificationError(f"bundle contains symlink: {path}")
        if not path.is_file():
            continue
        lowered = path.name.lower()
        relative = path.relative_to(bundle).as_posix().lower()
        if lowered in _SECRET_NAMES or path.suffix.lower() in _SECRET_SUFFIXES:
            raise VerificationError(f"bundle contains secret-like filename: {relative}")
        if (
            lowered in {"dataset.jsonl", "test.jsonl", "official_eval.jsonl"}
            or "official-eval" in relative
            or "official_eval" in relative
        ):
            raise VerificationError(f"bundle contains forbidden evaluation artifact: {relative}")
        if path.suffix.lower() in {".json", ".jsonl", ".log", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            scanned += 1
            for pattern in _SECRET_PATTERNS:
                if pattern.search(text):
                    raise VerificationError(f"bundle contains a secret-like marker: {relative}")
    return {"text_files_scanned": scanned, "symlinks": 0, "official_eval_artifacts": 0}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    bundle = args.bundle.resolve()
    if output.exists():
        raise VerificationError(f"refusing to overwrite {args.output}")
    if output == bundle or output.is_relative_to(bundle):
        raise VerificationError("verification output cannot be inside the immutable bundle")
    input_files = {
        args.train_manifest,
        args.dev_manifest,
        args.audit,
        args.recipe,
        args.provenance,
        args.candidate_predictions,
        args.candidate_predictions_manifest,
        args.candidate_sample_scores,
        args.candidate_aggregate,
        args.jarvis_record,
        args.pause_proof,
    }
    if output in {path.resolve() for path in input_files}:
        raise VerificationError("verification output aliases an input file")

    bundle_evidence = verify_bundle(bundle)
    semantic_evidence = verify_semantics(args, bundle)
    firewall = secret_and_firewall_audit(bundle)
    jarvis = verify_jarvis(args.jarvis_record, args.pause_proof, bundle)
    baseline_predictions = bundle / "evaluation" / "predictions.jsonl"
    baseline_aggregate, baseline_samples = score_again(
        manifest=args.dev_manifest,
        predictions=baseline_predictions,
    )
    candidate_aggregate, candidate_samples = score_again(
        manifest=args.dev_manifest,
        predictions=args.candidate_predictions,
    )
    recorded_baseline = strict_json(bundle / "evaluation" / "scores" / "aggregate.json")
    recorded_candidate = strict_json(args.candidate_aggregate)
    baseline_float_drift = assert_json_equivalent(
        baseline_aggregate,
        recorded_baseline,
        label="baseline aggregate",
    )
    candidate_float_drift = assert_json_equivalent(
        candidate_aggregate,
        recorded_candidate,
        label="candidate aggregate",
    )
    recorded_baseline_samples = strict_jsonl(
        bundle / "evaluation" / "scores" / "sample_scores.jsonl"
    )
    if baseline_samples != recorded_baseline_samples:
        raise VerificationError("recomputed baseline samples differ from downloaded evidence")
    frozen_candidate_samples = strict_jsonl(args.candidate_sample_scores)
    if candidate_samples != frozen_candidate_samples:
        raise VerificationError("recomputed candidate samples differ from frozen reference")

    baseline_by_id = indexed_samples(baseline_samples, label="baseline samples")
    candidate_by_id = indexed_samples(candidate_samples, label="candidate samples")
    if set(baseline_by_id) != set(candidate_by_id):
        raise VerificationError("candidate and baseline sample IDs differ")
    paired: list[dict[str, Any]] = []
    counts = {
        "candidate_only_win": 0,
        "baseline_only_win": 0,
        "both_correct": 0,
        "both_wrong": 0,
    }
    for sample_id in sorted(baseline_by_id):
        baseline_exact = bool(baseline_by_id[sample_id]["ast_exact"])
        candidate_exact = bool(candidate_by_id[sample_id]["ast_exact"])
        if candidate_exact and not baseline_exact:
            outcome = "candidate_only_win"
        elif baseline_exact and not candidate_exact:
            outcome = "baseline_only_win"
        elif candidate_exact:
            outcome = "both_correct"
        else:
            outcome = "both_wrong"
        counts[outcome] += 1
        paired.append(
            {
                "id": sample_id,
                "baseline_ast_exact": baseline_exact,
                "candidate_ast_exact": candidate_exact,
                "outcome": outcome,
            }
        )
    baseline_exact = int(baseline_aggregate["ast_exact_match"]["numerator"])
    candidate_exact = int(candidate_aggregate["ast_exact_match"]["numerator"])
    comparison = {
        "candidate_minus_baseline_exact_matches": candidate_exact - baseline_exact,
        "candidate_minus_baseline_percentage_points": (
            (candidate_exact - baseline_exact) * 100.0 / EXPECTED_ROWS
        ),
        "candidate_exact": candidate_exact,
        "baseline_exact": baseline_exact,
        "denominator": EXPECTED_ROWS,
        "candidate_wins": counts["candidate_only_win"],
        "baseline_wins": counts["baseline_only_win"],
        "ties": counts["both_correct"] + counts["both_wrong"],
        **counts,
    }
    result = {
        "schema_version": "barun-qwen05b-independent-verification-v2",
        "status": "verified",
        "run_id": EXPECTED_RUN_ID,
        "baseline": f"{EXPECTED_MODEL_ID}@{EXPECTED_MODEL_REVISION}",
        "candidate": "BarunAction-35M candidate-v2 batch63 final checkpoint",
        "bundle": bundle_evidence,
        "frozen_evidence": semantic_evidence,
        "firewall_and_secret_audit": firewall,
        "jarvis": jarvis,
        "comparison": comparison,
        "baseline_metrics": {
            "parse_valid": baseline_aggregate["parse_valid"],
            "schema_valid": baseline_aggregate["schema_valid"],
            "truncation": baseline_aggregate["truncation"],
            "argument_key_micro": baseline_aggregate["argument_key_micro"],
            "argument_value_micro": baseline_aggregate["argument_value_micro"],
            "tool_macro_f1": baseline_aggregate["tool_macro_f1"],
            "catastrophic_unauthorized_actions": baseline_aggregate[
                "catastrophic_unauthorized_actions"
            ],
            "false_action": baseline_aggregate["false_action"],
        },
        "official_mobile_evaluation_rows_read": 0,
        "cross_python_float_reproducibility": {
            "comparison_tolerance": 1e-12,
            "baseline_max_absolute_drift": baseline_float_drift,
            "candidate_max_absolute_drift": candidate_float_drift,
            "sample_records_required_exact": True,
            "reason": (
                "Python 3.12 changed built-in float summation; only derived aggregate floats "
                "may differ within tolerance, while sample evidence and integer numerators stay exact."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_jsonl(temporary / "paired-samples.jsonl", paired)
        write_json(temporary / "baseline-aggregate.recomputed.json", baseline_aggregate)
        write_json(temporary / "candidate-aggregate.recomputed.json", candidate_aggregate)
        result["paired_samples_sha256"] = sha256_file(temporary / "paired-samples.jsonl")
        write_json(temporary / "independent-verification.json", result)
        write_json(
            temporary / "artifact-sha256.json",
            file_map(temporary, excluded={"artifact-sha256.json"}),
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions-manifest", type=Path, required=True)
    parser.add_argument("--candidate-sample-scores", type=Path, required=True)
    parser.add_argument("--candidate-aggregate", type=Path, required=True)
    parser.add_argument("--jarvis-record", type=Path, required=True)
    parser.add_argument("--pause-proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
