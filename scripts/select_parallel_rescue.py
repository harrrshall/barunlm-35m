"""Verify downloaded rescue evidence and apply the frozen four-trial selector offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from barunlm.evaluation.parallel_rescue_selection import (
    PARALLEL_RESCUE_SELECTION_RECEIPT_VERSION,
    PARALLEL_RESCUE_SELECTION_SHA256,
    ParallelRescueSelectionError,
    select_parallel_rescue,
    write_exclusive_receipt,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTER_ATTEMPT = REPOSITORY_ROOT / "configs" / "parallel_rescue_selection_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--continual-bundle",
        type=Path,
        required=True,
        help="Downloaded continual-recovery export/essential directory.",
    )
    parser.add_argument(
        "--interpolation-bundle",
        type=Path,
        required=True,
        help="Downloaded interpolation-rescue export/essential directory.",
    )
    parser.add_argument(
        "--outer-attempt",
        type=Path,
        default=DEFAULT_OUTER_ATTEMPT,
        help="Frozen shared four-trial selection attempt/protocol JSON.",
    )
    parser.add_argument(
        "--outer-attempt-sha256",
        default=PARALLEL_RESCUE_SELECTION_SHA256,
        help="Out-of-band SHA-256 for the frozen shared selection attempt.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        help="Optional exclusive-create JSON receipt path outside both bundles.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = select_parallel_rescue(
            continual_bundle_root=args.continual_bundle,
            interpolation_bundle_root=args.interpolation_bundle,
            outer_attempt_path=args.outer_attempt,
            outer_attempt_sha256=args.outer_attempt_sha256,
        )
        if args.receipt is not None:
            write_exclusive_receipt(
                args.receipt,
                receipt,
                forbidden_roots=(args.continual_bundle, args.interpolation_bundle),
            )
    except (OSError, ParallelRescueSelectionError) as error:
        print(
            json.dumps(
                {
                    "schema_version": PARALLEL_RESCUE_SELECTION_RECEIPT_VERSION,
                    "selection_completed": False,
                    "error": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
