from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import TrainingRunConfig
from .trainer import train_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-parameter, response-only SFT for a pinned BarunLM checkpoint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a strict barun-sft-config-v1 JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingRunConfig.from_json(args.config)
    summary = train_sft(config)
    print(json.dumps(summary.to_dict(), allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
