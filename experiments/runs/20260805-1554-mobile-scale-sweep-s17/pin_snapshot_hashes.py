"""One-shot attempt-3 pinning of complete challenger snapshot file hashes.

Read-only Hugging Face access only:
- ``model_info(..., files_metadata=True)`` exposes the LFS ``sha256`` OID for
  ``model.safetensors`` per pinned revision, so weight hashes are pinned with
  **no weight download**.
- The small non-LFS JSON files matching the runner's frozen allow patterns
  (config.json, generation_config.json, special_tokens_map.json,
  tokenizer.json, tokenizer_config.json) are downloaded as metadata text and
  hashed locally, exactly like the attempt-1 roster pinning.

The receipt cross-checks every hash that attempt 1 already pinned
(roster-metadata.json and the v2 config) and records the exact per-arm file
set so the runner can enforce set equality (a file's pinned absence, e.g.
pythia's missing generation_config.json, is proven by the frozen set).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

RUN_DIR = Path(__file__).resolve().parent
RUN_ID = RUN_DIR.name
OUTPUT = RUN_DIR / "snapshot-pins.json"

# Must stay byte-identical to the runner's snapshot_download allow_patterns.
ALLOW_PATTERNS = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

ARMS = (
    (
        "pythia-70m-deduped",
        "EleutherAI/pythia-70m-deduped",
        "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
    ),
    ("smollm2-135m", "HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
    ("smollm2-360m", "HuggingFaceTB/SmolLM2-360M", "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"),
)


def matches_allow_patterns(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in ALLOW_PATTERNS)


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    api = HfApi()
    arms = []
    for arm_id, repo_id, revision in ARMS:
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise SystemExit(f"{repo_id}: resolved revision {info.sha} != pinned {revision}")
        files: dict[str, dict[str, object]] = {}
        for sibling in info.siblings:
            name = sibling.rfilename
            if not matches_allow_patterns(name):
                continue
            if sibling.lfs is not None:
                files[name] = {
                    "sha256": sibling.lfs.sha256,
                    "bytes": sibling.lfs.size,
                    "hash_source": "hf_lfs_metadata_no_download",
                }
            else:
                local = hf_hub_download(repo_id, name, revision=revision)
                payload = Path(local).read_bytes()
                files[name] = {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "hash_source": "small_metadata_file_downloaded_and_hashed_locally",
                }
        if "model.safetensors" not in files:
            raise SystemExit(f"{repo_id}: no model.safetensors at the pinned revision")
        arms.append(
            {
                "arm_id": arm_id,
                "repo_id": repo_id,
                "revision": revision,
                "allow_patterns": list(ALLOW_PATTERNS),
                "files": files,
            }
        )
    receipt = {
        "schema_version": "barun-mobile-scale-sweep-snapshot-pins-v1",
        "run_id": RUN_ID,
        "attempt": 3,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "network_scope": (
            "read-only Hugging Face metadata (files_metadata LFS sha256 for weights; no "
            "weight blob downloaded) plus small non-LFS JSON metadata files hashed locally"
        ),
        "arms": arms,
    }
    OUTPUT.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
