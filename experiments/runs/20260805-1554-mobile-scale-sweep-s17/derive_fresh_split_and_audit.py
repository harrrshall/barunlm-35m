"""CPU build step for run 20260805-1554-mobile-scale-sweep-s17.

1. Rebuilds the frozen Mobile Actions train/dev manifests from the pinned public
   source and the pinned BarunLM-35M tokenizer.json (text/JSON files only; no
   model weights), verifying every frozen SHA-256.
2. Derives the fresh grouped selection split from the verified train manifest.
3. Audits gold token lengths per roster tokenizer plus the Barun reference
   tokenizer, and applies the frozen max_new_tokens rule.

Artifacts written into this run directory contain only IDs, hashes, and
aggregate statistics; no dataset content is committed.  The 961 official
evaluation rows stay opaque inside prepare_mobile_actions exactly as in every
prior run.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from barunlm.baselines.mobile_scale_sweep import (
    AUDIT_SHA256,
    TRAIN_MANIFEST_SHA256,
    TRAIN_ROWS,
    gold_token_length_audit,
    max_new_tokens_from_audit,
    partition_frozen_train_manifest,
    verify_context_fit,
    write_partition_artifacts,
)
from barunlm.datasets.mobile_actions import (
    TokenizerIdentity,
    download_pinned_source,
    prepare_mobile_actions,
)
from barunlm.training.data import load_manifest, sha256_file

RUN_DIR = Path(__file__).resolve().parent
DEV_MANIFEST_SHA256 = "988bdce5874d1f1a775feeb5ba2b58cd2bdc128f57e73cb9a63d535fae7c1d55"

BARUN_REPO = "harrrshall/BarunLM-35M"
BARUN_REVISION = "ef3e483a9fd7d906ecf2a7929babeffaf82d1d16"
BARUN_TOKENIZER_SHA256 = "70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6"

ROSTER_TOKENIZERS = {
    "gptneox-50304": {
        "repo_id": "EleutherAI/pythia-70m-deduped",
        "revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
        "tokenizer_json_sha256": (
            "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624"
        ),
        "context_length": 2048,
    },
    "smollm2-49152": {
        "repo_id": "HuggingFaceTB/SmolLM2-135M",
        "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        "tokenizer_json_sha256": (
            "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
        ),
        "context_length": 8192,
    },
    "barun-16384": {
        "repo_id": BARUN_REPO,
        "revision": BARUN_REVISION,
        "tokenizer_json_sha256": BARUN_TOKENIZER_SHA256,
        "context_length": 2048,
    },
}
MAX_SEQ_LEN = 2048


def fetch_tokenizer(repo_id: str, revision: str, expected_sha256: str) -> Tokenizer:
    local = Path(hf_hub_download(repo_id, "tokenizer.json", revision=revision))
    actual = sha256_file(local)
    if actual != expected_sha256:
        raise RuntimeError(f"{repo_id} tokenizer.json hash mismatch: {actual}")
    return Tokenizer.from_file(str(local))


class AuditTokenizer:
    """Frozen audit encoding rule: tokenizers encode with add_special_tokens=False."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str, add_special_tokens: bool = False) -> object:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="barun-scale-sweep-build-"))
    print(f"working directory (not committed): {work}")

    barun_tokenizer = fetch_tokenizer(BARUN_REPO, BARUN_REVISION, BARUN_TOKENIZER_SHA256)
    source = download_pinned_source(work / "source-cache")
    prepared = prepare_mobile_actions(
        source,
        work / "data",
        tokenizer=barun_tokenizer,
        tokenizer_identity=TokenizerIdentity(
            identifier=BARUN_REPO,
            revision=BARUN_REVISION,
            sha256=BARUN_TOKENIZER_SHA256,
        ),
    )
    checks = {
        "train": (prepared.hashes["train"], TRAIN_MANIFEST_SHA256),
        "dev": (prepared.hashes["dev"], DEV_MANIFEST_SHA256),
        "audit": (prepared.hashes["audit"], AUDIT_SHA256),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise RuntimeError(f"rebuilt {name} hash {actual} != frozen {expected}")
    print("frozen train/dev/audit hashes reproduced byte-identically")
    if prepared.train_rows != TRAIN_ROWS or prepared.dev_rows != 756:
        raise RuntimeError("rebuilt row counts changed")

    partition = partition_frozen_train_manifest(prepared.train_manifest)
    artifacts = write_partition_artifacts(partition, work / "fresh-split")
    receipt = artifacts["receipt"]
    (RUN_DIR / "split-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (RUN_DIR / "sweep-train-membership.txt").write_text(
        "\n".join(sorted(partition.sweep_train_ids)) + "\n", encoding="utf-8"
    )
    (RUN_DIR / "selection-membership.txt").write_text(
        "\n".join(sorted(partition.selection_ids)) + "\n", encoding="utf-8"
    )

    sweep_rows = load_manifest(
        Path(artifacts["paths"]["sweep_train_manifest"]),
        expected_sha256=receipt["sweep_train_manifest_sha256"],
        expected_derived_split="train",
    )
    selection_rows = load_manifest(
        Path(artifacts["paths"]["selection_manifest"]),
        expected_sha256=receipt["selection_manifest_sha256"],
        expected_derived_split="train",
    )

    per_tokenizer: dict[str, dict] = {}
    rule_inputs: list[int] = []
    for audit_key, spec in ROSTER_TOKENIZERS.items():
        tokenizer = AuditTokenizer(
            fetch_tokenizer(spec["repo_id"], spec["revision"], spec["tokenizer_json_sha256"])
        )
        entry = {
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "tokenizer_json_sha256": spec["tokenizer_json_sha256"],
            "context_length": spec["context_length"],
            "sweep_train": gold_token_length_audit(sweep_rows, tokenizer),
            "selection": gold_token_length_audit(selection_rows, tokenizer),
        }
        per_tokenizer[audit_key] = entry
        rule_inputs.append(
            max(
                entry["sweep_train"]["max_target_tokens_with_eos"],
                entry["selection"]["max_target_tokens_with_eos"],
            )
        )
    global_max_gold = max(rule_inputs)
    max_new_tokens = max_new_tokens_from_audit(global_max_gold)
    for audit_key, entry in per_tokenizer.items():
        context = min(MAX_SEQ_LEN, entry["context_length"])
        for split_name in ("sweep_train", "selection"):
            verify_context_fit(
                prompt_tokens_max=entry[split_name]["prompt_tokens"]["max"],
                total_with_eos_max=entry[split_name]["total_with_eos"]["max"],
                max_new_tokens=max_new_tokens,
                context_limit=context,
            )
    audit_payload = {
        "schema_version": "barun-mobile-scale-sweep-gold-token-audit-v1",
        "run_id": "20260805-1554-mobile-scale-sweep-s17",
        "encoding_rule": "tokenizers.Tokenizer.encode(text, add_special_tokens=False) on the "
        "pinned tokenizer.json bytes; prompt and target encoded separately",
        "max_new_tokens_rule": "smallest multiple of 64 at or above "
        "(max gold target tokens + 1 EOS + 32 margin) across all audited tokenizers",
        "global_max_target_tokens_with_eos": global_max_gold,
        "max_new_tokens": max_new_tokens,
        "max_seq_len": MAX_SEQ_LEN,
        "context_fit": "verified for every audited tokenizer on both splits",
        "per_tokenizer": per_tokenizer,
    }
    (RUN_DIR / "gold-token-audit.json").write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    import math

    microbatches = math.ceil(len(sweep_rows) / 21)
    expected_steps = math.ceil(microbatches / 3)
    summary = {
        "sweep_train_rows": receipt["sweep_train_rows"],
        "selection_rows": receipt["selection_rows"],
        "sweep_train_clusters": receipt["sweep_train_clusters"],
        "selection_clusters": receipt["selection_clusters"],
        "sweep_train_manifest_sha256": receipt["sweep_train_manifest_sha256"],
        "selection_manifest_sha256": receipt["selection_manifest_sha256"],
        "sweep_train_membership_sha256": receipt["sweep_train_membership_sha256"],
        "selection_membership_sha256": receipt["selection_membership_sha256"],
        "expected_optimizer_steps": expected_steps,
        "max_new_tokens": max_new_tokens,
        "global_max_target_tokens_with_eos": global_max_gold,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
