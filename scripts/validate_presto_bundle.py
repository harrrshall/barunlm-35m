"""Validate a downloaded PRESTO essential bundle against local trust anchors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from barunlm.presto_bundle_audit import PrestoBundleAuditError, validate_presto_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Downloaded export/essential directory.",
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        required=True,
        help="Local preregistration frozen before launch; must be outside the bundle.",
    )
    parser.add_argument(
        "--jarvis-record",
        type=Path,
        required=True,
        help="Completed local safe-run record with download and pause evidence.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new JSON receipt path; an existing path is never overwritten.",
    )
    return parser


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = validate_presto_bundle(
            args.bundle,
            preregistration_path=args.preregistration,
            jarvis_record_path=args.jarvis_record,
        ).to_dict()
        if args.output is not None:
            _write_receipt(args.output, receipt)
    except (OSError, PrestoBundleAuditError) as error:
        print(
            json.dumps(
                {
                    "schema_version": "barun-presto-independent-bundle-audit-v1",
                    "passed": False,
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
