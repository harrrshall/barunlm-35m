"""Run the non-retroactive PRESTO user-revision taxonomy correction audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from barunlm.evaluation.presto_taxonomy_correction import (
    CORRECTION_SCHEMA_VERSION,
    PROTOCOL_PATH,
    PrestoTaxonomyCorrectionError,
    audit_presto_taxonomy_correction,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--jarvis-record", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new receipt path; existing files are never overwritten.",
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
        receipt = audit_presto_taxonomy_correction(
            args.bundle,
            preregistration_path=args.preregistration,
            jarvis_record_path=args.jarvis_record,
            protocol_path=args.protocol,
        ).to_dict()
        if args.output is not None:
            _write_receipt(args.output, receipt)
    except (OSError, PrestoTaxonomyCorrectionError) as error:
        print(
            json.dumps(
                {
                    "schema_version": CORRECTION_SCHEMA_VERSION,
                    "audit_passed": False,
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
