"""Materialize the frozen screening and conditional full-refit temporal views."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from barunlm.datasets.mobile_temporal_counterfactual import (
    PINNED_MOBILE_TOKENIZER_IDENTIFIER,
    PINNED_MOBILE_TOKENIZER_REVISION,
    PINNED_MOBILE_TOKENIZER_SHA256,
    TemporalTokenizerIdentity,
    materialize_full_counterfactual_view,
    materialize_temporal_views,
)
from barunlm.training.data import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def run(*, source_train: Path, tokenizer_path: Path, output_root: Path) -> dict[str, Any]:
    """Verify the tokenizer, then create both immutable views under a new root."""

    if output_root.exists():
        raise FileExistsError(f"refusing to reuse output root {output_root}")
    actual_tokenizer_sha256 = sha256_file(tokenizer_path)
    if actual_tokenizer_sha256 != PINNED_MOBILE_TOKENIZER_SHA256:
        raise ValueError(
            "tokenizer SHA-256 mismatch: "
            f"{actual_tokenizer_sha256} != {PINNED_MOBILE_TOKENIZER_SHA256}"
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    identity = TemporalTokenizerIdentity(
        identifier=PINNED_MOBILE_TOKENIZER_IDENTIFIER,
        revision=PINNED_MOBILE_TOKENIZER_REVISION,
        sha256=PINNED_MOBILE_TOKENIZER_SHA256,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    screening = materialize_temporal_views(
        source_train_path=source_train,
        tokenizer=tokenizer,
        tokenizer_identity=identity,
        output_dir=output_root / "materialized-screening-v1",
    )
    full = materialize_full_counterfactual_view(
        source_train_path=source_train,
        tokenizer=tokenizer,
        tokenizer_identity=identity,
        output_dir=output_root / "materialized-full-refit-v1",
    )
    return {
        "schema_version": "barun-mobile-temporal-materialization-summary-v1",
        "source_train": str(source_train),
        "tokenizer": {
            "path": str(tokenizer_path),
            "identifier": identity.identifier,
            "revision": identity.revision,
            "sha256": identity.sha256,
        },
        "screening": {
            "counts": dict(screening.counts),
            "hashes": dict(screening.hashes),
        },
        "conditional_full_refit": {
            "source_rows": full.source_rows,
            "variant_rows": full.variant_rows,
            "hashes": dict(full.hashes),
        },
        "development_rows_read": 0,
        "official_evaluation_rows_read": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        source_train=args.source_train,
        tokenizer_path=args.tokenizer,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
