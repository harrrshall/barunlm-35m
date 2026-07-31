from __future__ import annotations

import json
from pathlib import Path

import torch

from barunlm import BarunConfig, BarunLM

ROOT = Path(__file__).resolve().parents[1]


def test_release_config_and_parameter_count() -> None:
    config = BarunConfig.from_json(ROOT / "barun_config.json")
    model = BarunLM(config)

    assert config.ffn_dim == 1_228
    assert config.mtp_loss_weight == 0.0
    assert model.parameter_counts() == {
        "total": 35_072_768,
        "embedding": 7_340_032,
        "non_embedding": 27_732_736,
    }
    assert model.lm_head.weight is model.embedding.weight


def test_default_config_matches_release_json() -> None:
    payload = json.loads((ROOT / "barun_config.json").read_text())
    assert BarunConfig() == BarunConfig(**payload)


def test_forward_shape_and_finite_logits() -> None:
    model = BarunLM(BarunConfig()).eval()
    with torch.inference_mode():
        output = model(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
    assert output.logits.shape == (1, 4, 16_384)
    assert torch.isfinite(output.logits).all()
