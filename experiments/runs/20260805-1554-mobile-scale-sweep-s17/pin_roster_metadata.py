"""One-shot read-only roster pinning for run 20260805-1554-mobile-scale-sweep-s17.

Queries public Hugging Face Hub metadata for the three sweep base checkpoints,
pins exact revision SHAs, and downloads ONLY small text/JSON metadata files
(config.json + tokenizer files). No model weights are downloaded. No credential
is read from or written to this repository.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

RUN_DIR = Path(__file__).resolve().parent
ROSTER = (
    "EleutherAI/pythia-70m-deduped",
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M",
)
METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
FORBIDDEN_SUFFIXES = (".bin", ".safetensors", ".pt", ".gguf", ".onnx", ".msgpack", ".h5")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    api = HfApi()
    receipt: dict = {
        "schema_version": 1,
        "run_id": "20260805-1554-mobile-scale-sweep-s17",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "network_scope": "read-only public Hugging Face metadata and small text/JSON files; no weights",
        "models": [],
    }
    for repo_id in ROSTER:
        info = api.model_info(repo_id, files_metadata=False)
        revision = info.sha
        safetensors = getattr(info, "safetensors", None)
        param_total = None
        if safetensors is not None:
            param_total = int(safetensors.total)
        card_data = info.card_data.to_dict() if info.card_data is not None else {}
        entry: dict = {
            "repo_id": repo_id,
            "pinned_revision": revision,
            "safetensors_total_parameters": param_total,
            "license": card_data.get("license"),
            "downloaded_metadata_files": {},
        }
        for filename in METADATA_FILES:
            if filename.endswith(FORBIDDEN_SUFFIXES):
                raise RuntimeError(f"forbidden file requested: {filename}")
            try:
                local = hf_hub_download(repo_id, filename, revision=revision)
            except Exception as exc:  # noqa: BLE001 - file may be absent for some repos
                entry["downloaded_metadata_files"][filename] = {"absent": True, "reason": type(exc).__name__}
                continue
            local_path = Path(local)
            entry["downloaded_metadata_files"][filename] = {
                "sha256": sha256_file(local_path),
                "bytes": local_path.stat().st_size,
            }
            if filename == "config.json":
                config = json.loads(local_path.read_text())
                entry["architecture"] = config.get("architectures")
                entry["max_position_embeddings"] = config.get("max_position_embeddings")
                entry["vocab_size"] = config.get("vocab_size")
                entry["hidden_size"] = config.get("hidden_size")
                entry["num_hidden_layers"] = config.get("num_hidden_layers")
                entry["eos_token_id_config"] = config.get("eos_token_id")
                entry["bos_token_id_config"] = config.get("bos_token_id")
            if filename == "tokenizer_config.json":
                tok_config = json.loads(local_path.read_text())
                entry["tokenizer_class"] = tok_config.get("tokenizer_class")
                entry["eos_token"] = tok_config.get("eos_token")
                entry["bos_token"] = tok_config.get("bos_token")
                entry["chat_template_present"] = "chat_template" in tok_config
        receipt["models"].append(entry)
    out_path = RUN_DIR / "roster-metadata.json"
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    print(f"receipt sha256 {sha256_file(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
