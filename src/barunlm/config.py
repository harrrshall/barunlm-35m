from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class BarunConfig:
    vocab_size: int = 16_384
    dim: int = 448
    n_layers: int = 12
    n_heads: int = 7
    n_kv_heads: int = 1
    ffn_dim: int = 1_228
    max_seq_len: int = 2_048
    rope_theta: float = 10_000.0
    rope_fraction: float = 0.5
    local_window: int = 256
    full_attention_every: int = 4
    attention_gate: bool = True
    qk_norm: bool = True
    residual_select_every: int = 4
    activation_clip: float = 10.0
    dropout: float = 0.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    mtp_offset: int = 2
    mtp_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        head_dim = self.dim // self.n_heads
        rope_dim = int(head_dim * self.rope_fraction)
        if rope_dim < 2 or rope_dim % 2:
            raise ValueError("rope_fraction must yield a positive, even rotary dimension")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.local_window < 1:
            raise ValueError("local_window must be positive")
        if self.full_attention_every < 1:
            raise ValueError("full_attention_every must be positive")
        if self.residual_select_every < 0:
            raise ValueError("residual_select_every cannot be negative")
        if self.mtp_offset < 1:
            raise ValueError("mtp_offset must be positive")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def rope_dim(self) -> int:
        value = int(self.head_dim * self.rope_fraction)
        return value - value % 2

    @classmethod
    def from_json(cls, path: str | Path) -> BarunConfig:
        with Path(path).open() as handle:
            return cls(**json.load(handle))

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w") as handle:
            json.dump(asdict(self), handle, indent=2, sort_keys=True)
            handle.write("\n")
