from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from barunlm.evaluation.generation import generate_manifest
from barunlm.evaluation.mobile_actions import write_scores


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _hash_map(value: str | None) -> dict[str, str] | None:
    if value is None:
        return None
    payload = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("checkpoint hash file must contain an object")
    if "file_sha256" in payload:
        payload = payload["file_sha256"]
    if not isinstance(payload, dict):
        raise TypeError("checkpoint file_sha256 must contain an object")
    return {str(name): str(digest) for name, digest in payload.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and strictly score a Mobile Actions development manifest."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-format", choices=("float", "int8"), default="float")
    parser.add_argument("--checkpoint-hashes", help="JSON hash map for a base checkpoint")
    parser.add_argument(
        "--int8-manifest-sha256",
        help="Required out-of-band manifest SHA-256 for an int8 checkpoint.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.checkpoint_format == "float" and args.int8_manifest_sha256 is not None:
        parser.error("--int8-manifest-sha256 is valid only with --checkpoint-format int8")
    if args.checkpoint_format == "int8":
        if args.checkpoint_hashes is not None:
            parser.error("--checkpoint-hashes is valid only with --checkpoint-format float")
        if args.int8_manifest_sha256 is None:
            parser.error("--checkpoint-format int8 requires --int8-manifest-sha256")
        if not _is_sha256(args.int8_manifest_sha256):
            parser.error("--int8-manifest-sha256 must be a lowercase SHA-256")
        if args.device != "cpu":
            parser.error("--checkpoint-format int8 requires --device cpu")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_hashes = _hash_map(args.checkpoint_hashes)
    args.output.mkdir(parents=True, exist_ok=False)
    predictions = args.output / "predictions.jsonl"
    generation = generate_manifest(
        checkpoint_dir=args.checkpoint,
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        predictions_path=predictions,
        device_name=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        expected_checkpoint_sha256=checkpoint_hashes,
        checkpoint_format=args.checkpoint_format,
        expected_int8_manifest_sha256=args.int8_manifest_sha256,
    )
    score_paths = write_scores(args.manifest, predictions, args.output / "scores")
    print(
        json.dumps(
            {
                "generation": generation.to_dict(),
                "scores": {name: str(path) for name, path in score_paths.items()},
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
