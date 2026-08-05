"""Create the exact prelaunch source-tree manifest for the isolated Qwen lane."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "barun-qwen05b-source-snapshot-v1"
EXCLUDED_FILES = {"attempt-preregistration.json", "source-snapshot.json"}
EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}
EXCLUDED_SUFFIXES = {".egg-info"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_excluded(path: Path, *, root: Path) -> bool:
    relative = path.relative_to(root)
    return (
        relative.as_posix() in EXCLUDED_FILES
        or any(part in EXCLUDED_DIRECTORIES for part in relative.parts)
        or any(part.endswith(tuple(EXCLUDED_SUFFIXES)) for part in relative.parts)
    )


def content_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if is_excluded(path, root=root):
            continue
        if path.is_symlink():
            raise RuntimeError(f"source snapshot refuses symlink: {path}")
        if path.is_file():
            files.append(path)
    return files


def tree_sha256(files: dict[str, str]) -> str:
    lines = "".join(f"{digest}  {name}\n" for name, digest in sorted(files.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    output = args.output.resolve(strict=False)
    if output.parent != root:
        raise ValueError("source snapshot must be written at the staged project root")
    files = content_files(root)
    identities = {path.relative_to(root).as_posix(): sha256_file(path) for path in files}
    sizes = {path.relative_to(root).as_posix(): path.stat().st_size for path in files}
    required = {
        "configs/mobile_qwen05b_matched_v1.json",
        "data/audit.json",
        "data/dev.jsonl",
        "data/train.jsonl",
        "docs/mobile-qwen05b-matched-baseline.md",
        "infra/jarvis/safe_run.py",
        "requirements/qwen05b-matched.txt",
        "scripts/run_mobile_qwen05b_matched_baseline.py",
        "scripts/verify_qwen05b_matched_bundle.py",
        "src/barunlm/baselines/mobile_qwen05b_matched.py",
        "tests/test_mobile_qwen05b_matched_baseline.py",
    }
    missing = sorted(required.difference(identities))
    if missing:
        raise RuntimeError(f"source snapshot lacks required files: {missing}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "created_before_model_download_or_development_scoring": True,
        "git_base": args.git_base,
        "dirty_tree_patch_sha256": args.dirty_patch_sha256,
        "root_directory_name": root.name,
        "file_count": len(files),
        "content_bytes": sum(sizes.values()),
        "content_tree_sha256": tree_sha256(identities),
        "content_tree_hash_method": (
            "SHA-256 of UTF-8 LC_ALL=C-style sorted '<file-sha256>  <relative-path>\\n' lines"
        ),
        "files_sha256": identities,
        "files_bytes": sizes,
        "exact_exclusions": sorted(EXCLUDED_FILES),
        "excluded_directory_names": sorted(EXCLUDED_DIRECTORIES),
        "staging_policy": {
            "included": (
                "Explicit regular-file source allowlist, Qwen-specific runner/config/tests/docs, "
                "safe exact-ID controller, frozen Mobile train/dev/audit manifests, and no "
                "candidate checkpoint or optimizer state."
            ),
            "excluded": [
                "Git metadata",
                "credentials, keys, tokens, and environment files",
                "virtual environments, caches, bytecode, and egg-info",
                "W&B output",
                "model weights and tokenizers before the remote pinned download",
                "Mobile Actions combined source and all 961 official evaluation rows",
                "PRESTO official test or combined artifacts",
                "unrelated checkpoints, experiments, and optimizer states",
                "symlinks and .DS_Store",
            ],
        },
    }
    write_json(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-base", required=True)
    parser.add_argument("--dirty-patch-sha256", required=True)
    return parser


def main() -> None:
    payload = run(build_parser().parse_args())
    print(json.dumps(payload, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
