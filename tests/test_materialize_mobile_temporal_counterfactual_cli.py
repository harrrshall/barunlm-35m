from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script() -> ModuleType:
    path = ROOT / "scripts" / "materialize_mobile_temporal_counterfactual.py"
    spec = importlib.util.spec_from_file_location("barun_temporal_materializer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_materializer_rejects_unpinned_tokenizer_before_writing(tmp_path: Path) -> None:
    module = load_script()
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="tokenizer SHA-256 mismatch"):
        module.run(
            source_train=tmp_path / "unused-train.jsonl",
            tokenizer_path=tokenizer,
            output_root=output,
        )

    assert not output.exists()


def test_materializer_refuses_an_existing_output_before_reading_inputs(tmp_path: Path) -> None:
    module = load_script()
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to reuse output root"):
        module.run(
            source_train=tmp_path / "unused-train.jsonl",
            tokenizer_path=tmp_path / "unused-tokenizer.json",
            output_root=output,
        )
