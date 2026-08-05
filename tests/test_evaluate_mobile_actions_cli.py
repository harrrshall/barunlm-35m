from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "barun_evaluate_mobile_actions_cli",
    ROOT / "scripts" / "evaluate_mobile_actions.py",
)
if _SCRIPT_SPEC is None or _SCRIPT_SPEC.loader is None:  # pragma: no cover - import invariant
    raise RuntimeError("cannot load evaluate_mobile_actions.py for tests")
evaluate_mobile_actions = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(evaluate_mobile_actions)


def _base_args(tmp_path: Path) -> list[str]:
    return [
        "--checkpoint",
        str(tmp_path / "checkpoint"),
        "--manifest",
        str(tmp_path / "dev.jsonl"),
        "--manifest-sha256",
        "a" * 64,
        "--output",
        str(tmp_path / "output"),
        "--device",
        "cpu",
    ]


def test_mobile_cli_accepts_explicit_int8_contract(tmp_path: Path) -> None:
    args = evaluate_mobile_actions.parse_args(
        [
            *_base_args(tmp_path),
            "--checkpoint-format",
            "int8",
            "--int8-manifest-sha256",
            "b" * 64,
        ]
    )
    assert args.checkpoint_format == "int8"
    assert args.device == "cpu"
    assert args.checkpoint_hashes is None


@pytest.mark.parametrize(
    "extra",
    [
        ["--checkpoint-format", "int8"],
        [
            "--checkpoint-format",
            "int8",
            "--int8-manifest-sha256",
            "b" * 64,
            "--checkpoint-hashes",
            "hashes.json",
        ],
        ["--checkpoint-format", "float", "--int8-manifest-sha256", "b" * 64],
        ["--checkpoint-format", "int8", "--int8-manifest-sha256", "not-a-sha256"],
    ],
)
def test_mobile_cli_rejects_incompatible_checkpoint_arguments(
    tmp_path: Path, extra: list[str]
) -> None:
    with pytest.raises(SystemExit):
        evaluate_mobile_actions.parse_args([*_base_args(tmp_path), *extra])


def test_mobile_cli_rejects_cuda_for_int8(tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    args[-1] = "cuda"
    with pytest.raises(SystemExit):
        evaluate_mobile_actions.parse_args(
            [
                *args,
                "--checkpoint-format",
                "int8",
                "--int8-manifest-sha256",
                "b" * 64,
            ]
        )


def test_checkpoint_hash_reader_accepts_candidate_provenance_wrapper(tmp_path: Path) -> None:
    hashes = {
        "barun_config.json": "a" * 64,
        "model.safetensors": "b" * 64,
        "tokenizer.json": "c" * 64,
    }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps({"candidate_id": "candidate-v2", "file_sha256": hashes}))
    assert evaluate_mobile_actions._hash_map(str(path)) == hashes
