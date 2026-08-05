"""Replaceable identity manifest for the current BarunAction-35M candidate."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

MODEL_NAME = "BarunAction-35M"
BASE_MODEL_NAME = "BarunLM-35M"
CANDIDATE_ID = "candidate-v2"
CANDIDATE_RUN_ID = "20260803-1845-mob-batch63-s17"
CANDIDATE_SELECTION_RUN_ID = "20260803-1845-mobile-followup-retry-s17"
CANDIDATE_ARM_ID = "batch63"
CANDIDATE_STEP = 126
CANDIDATE_RELATIVE_CHECKPOINT_PATH = (
    "experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/arms/batch63/checkpoint"
)
CANDIDATE_CHECKPOINT_SHA256 = MappingProxyType(
    {
        "barun_config.json": "9b3a1d71baa95a198744d250f9629231738d942570b8685c44307fd83dd33565",
        "model.safetensors": "fdb95ccf58a095e0d321be998924318b35ee59a334f6dd97d8726d2cf80021d3",
        "tokenizer.json": "70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6",
    }
)
CANDIDATE_MANIFEST_SHA256 = "c743ab7c4d33ae75c6b0aa4547458a961b92766da8fcf85fd148fda2ebb5530a"


def candidate_identity(checkpoint_sha256: Mapping[str, str]) -> tuple[str | None, str | None]:
    """Label only the exact hash-pinned candidate; custom checkpoints remain unlabeled."""

    if dict(checkpoint_sha256) == dict(CANDIDATE_CHECKPOINT_SHA256):
        return CANDIDATE_ID, CANDIDATE_RUN_ID
    return None, None


__all__ = [
    "BASE_MODEL_NAME",
    "CANDIDATE_ARM_ID",
    "CANDIDATE_CHECKPOINT_SHA256",
    "CANDIDATE_ID",
    "CANDIDATE_MANIFEST_SHA256",
    "CANDIDATE_RELATIVE_CHECKPOINT_PATH",
    "CANDIDATE_RUN_ID",
    "CANDIDATE_SELECTION_RUN_ID",
    "CANDIDATE_STEP",
    "MODEL_NAME",
    "candidate_identity",
]
