"""Create the frozen candidate-v2 ARM64 int8 Mobile retention receipt."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from barunlm.evaluation.mobile_int8_retention import (
    MOBILE_INT8_RETENTION_PROTOCOL_SHA256,
    create_retention_receipt,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT / "configs" / "barunaction" / "candidate-v2-arm64-int8-retention-v1.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--protocol-sha256",
        required=True,
        help="Required out-of-band SHA-256 for the frozen protocol.",
    )
    parser.add_argument(
        "--int8-evaluation",
        type=Path,
        required=True,
        help="Output directory produced by scripts/evaluate_mobile_actions.py.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.protocol_sha256 != MOBILE_INT8_RETENTION_PROTOCOL_SHA256:
        parser.error("--protocol-sha256 differs from the compiled frozen protocol identity")
    result = create_retention_receipt(
        repository_root=args.repository_root,
        protocol_path=args.protocol,
        int8_evaluation_dir=args.int8_evaluation,
        output_dir=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0 if result["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
