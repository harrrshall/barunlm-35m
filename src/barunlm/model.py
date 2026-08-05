from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import BarunConfig

try:
    from flash_attn import flash_attn_func
except ImportError:  # pragma: no cover - exercised on GPU installations only
    flash_attn_func = None

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:  # pragma: no cover - compatibility fallback for older PyTorch
    create_block_mask = None
    flex_attention = None

compiled_flex_attention = (
    torch.compile(flex_attention, dynamic=False) if flex_attention is not None else None
)


@dataclass
class BarunOutput:
    logits: Tensor
    loss: Tensor | None = None
    causal_loss: Tensor | None = None
    mtp_loss: Tensor | None = None
    past_key_values: list[tuple[Tensor, Tensor]] | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        y = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        return y.to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class PartialRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        inv_freq = theta ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        angles = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", angles.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None, :, :], persistent=False)
        self.dim = dim

    def forward(self, q: Tensor, k: Tensor, position_offset: int = 0) -> tuple[Tensor, Tensor]:
        seq_len = q.shape[-2]
        end = position_offset + seq_len
        if end > self.cos.shape[-2]:
            raise ValueError(
                f"rotary position {end} exceeds configured maximum {self.cos.shape[-2]}"
            )
        cos = self.cos[:, :, position_offset:end].to(device=q.device, dtype=q.dtype)
        sin = self.sin[:, :, position_offset:end].to(device=q.device, dtype=q.dtype)
        q_rot, q_pass = q[..., : self.dim], q[..., self.dim :]
        k_rot, k_pass = k[..., : self.dim], k[..., self.dim :]
        q = torch.cat((q_rot * cos + _rotate_half(q_rot) * sin, q_pass), dim=-1)
        k = torch.cat((k_rot * cos + _rotate_half(k_rot) * sin, k_pass), dim=-1)
        return q, k


@lru_cache(maxsize=32)
def _local_causal_mask(seq_len: int, window: int, device_name: str) -> Tensor:
    """Return a reusable normal tensor, even if first called under inference mode."""
    # The complete device name is part of the cache key so cuda:0 and cuda:1 can
    # never share a device-local tensor.  Disabling inference mode while creating
    # the cached constant prevents an inference tensor from later being reused by
    # a gradient-enabled forward pass.
    with torch.inference_mode(False):
        device = torch.device(device_name)
        row = torch.arange(seq_len, device=device)[:, None]
        col = torch.arange(seq_len, device=device)[None, :]
        allowed = (col <= row) & (col > row - window)
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.float32)
        return mask.masked_fill(~allowed, float("-inf"))


@lru_cache(maxsize=32)
def _local_bidirectional_mask(seq_len: int, window: int, device_name: str) -> Tensor:
    """Window constraint for custom masks such as bidirectional PrefixLM prompts."""
    with torch.inference_mode(False):
        device = torch.device(device_name)
        row = torch.arange(seq_len, device=device)[:, None]
        col = torch.arange(seq_len, device=device)[None, :]
        allowed = (col - row).abs() < window
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.float32)
        return mask.masked_fill(~allowed, float("-inf"))


@lru_cache(maxsize=32)
def _local_block_mask(seq_len: int, window: int, device: str):
    if create_block_mask is None:
        return None

    def causal_window(batch, head, query_index, key_index):
        del batch, head
        return (query_index >= key_index) & (query_index - key_index < window)

    with torch.inference_mode(False):
        return create_block_mask(
            causal_window,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
            _compile=True,
        )


class GroupedAttention(nn.Module):
    def __init__(self, config: BarunConfig, *, is_full: bool) -> None:
        super().__init__()
        self.config = config
        self.is_full = is_full
        self.q_proj = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
        self.g_proj = (
            nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
            if config.attention_gate
            else None
        )
        self.q_norm = RMSNorm(config.head_dim, config.norm_eps) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(config.head_dim, config.norm_eps) if config.qk_norm else nn.Identity()
        self.rope = PartialRotaryEmbedding(config.rope_dim, config.max_seq_len, config.rope_theta)

    @staticmethod
    def _is_binary_mask(mask: Tensor) -> bool:
        if mask.dtype == torch.bool:
            return True
        return bool(torch.all((mask == 0) | (mask == 1)).item())

    def _is_key_mask(self, mask: Tensor, batch: int, query_len: int) -> bool:
        """Distinguish a standard [B, K] key mask from a [Q, K] policy mask.

        The shapes are inherently ambiguous when B == Q.  Boolean/integer and
        0/1 floating masks use the standard padding-mask interpretation in that
        case; a floating square matrix containing additive values such as -inf
        retains the custom PrefixLM-policy interpretation.
        """
        if mask.ndim != 2 or mask.shape[0] != batch:
            return False
        return batch != query_len or self._is_binary_mask(mask)

    def _causal_mask(
        self,
        q: Tensor,
        key_len: int,
        *,
        query_start: int,
        key_start: int,
    ) -> Tensor:
        query_len = q.shape[-2]
        if query_len == key_len and query_start == key_start:
            window = key_len if self.is_full else self.config.local_window
            return _local_causal_mask(query_len, window, str(q.device)).to(dtype=q.dtype)

        query_positions = query_start + torch.arange(query_len, device=q.device)[:, None]
        key_positions = key_start + torch.arange(key_len, device=q.device)[None, :]
        allowed = key_positions <= query_positions
        if not self.is_full:
            allowed &= query_positions - key_positions < self.config.local_window
        mask = torch.zeros((query_len, key_len), device=q.device, dtype=q.dtype)
        return mask.masked_fill(~allowed, float("-inf"))

    def _local_policy_mask(
        self,
        q: Tensor,
        key_len: int,
        *,
        query_start: int,
        key_start: int,
    ) -> Tensor:
        query_len = q.shape[-2]
        if query_len == key_len and query_start == key_start:
            return _local_bidirectional_mask(query_len, self.config.local_window, str(q.device)).to(
                dtype=q.dtype
            )

        query_positions = query_start + torch.arange(query_len, device=q.device)[:, None]
        key_positions = key_start + torch.arange(key_len, device=q.device)[None, :]
        allowed = (key_positions - query_positions).abs() < self.config.local_window
        mask = torch.zeros((query_len, key_len), device=q.device, dtype=q.dtype)
        return mask.masked_fill(~allowed, float("-inf"))

    def _key_mask_to_additive(self, mask: Tensor, q: Tensor) -> Tensor:
        mask = mask.to(device=q.device)
        if mask.dtype == torch.bool:
            valid = mask.to(dtype=torch.bool)
            additive = torch.zeros(valid.shape, device=q.device, dtype=q.dtype)
            additive = additive.masked_fill(~valid, float("-inf"))
        elif not torch.is_floating_point(mask):
            if not self._is_binary_mask(mask):
                raise ValueError("integer attention masks must contain only 0 and 1")
            valid = mask.to(dtype=torch.bool)
            additive = torch.zeros(valid.shape, device=q.device, dtype=q.dtype)
            additive = additive.masked_fill(~valid, float("-inf"))
        else:
            # Avoid a device synchronization in the common training path while
            # accepting either floating 0/1 key masks or additive key masks.
            valid = mask.to(dtype=torch.bool)
            binary_additive = torch.zeros(valid.shape, device=q.device, dtype=q.dtype)
            binary_additive = binary_additive.masked_fill(~valid, float("-inf"))
            is_binary = torch.all((mask == 0) | (mask == 1))
            additive = torch.where(is_binary, binary_additive, mask.to(dtype=q.dtype))
        return additive[:, None, None, :]

    @staticmethod
    def _ensure_nonempty_rows(
        mask: Tensor,
        *,
        query_start: int,
        key_start: int,
    ) -> Tensor:
        """Give otherwise fully masked padding queries one harmless self key.

        PyTorch 2.4's math SDPA backend returns NaNs for an all-``-inf`` row.
        Padding-query outputs are not semantically used, but those NaNs can enter
        later K/V projections before the key mask is applied.  Letting only such
        rows attend their own position keeps them finite without exposing a pad
        key to any valid query.
        """
        missing = ~torch.isfinite(mask).any(dim=-1, keepdim=True)
        query_len = mask.shape[-2]
        key_len = mask.shape[-1]
        fallback = query_start + torch.arange(query_len, device=mask.device) - key_start
        fallback = fallback.clamp(0, key_len - 1)
        indices = fallback[None, None, :, None].expand(mask.shape[0], 1, query_len, 1)
        current = mask.gather(dim=-1, index=indices)
        values = torch.where(missing, torch.zeros_like(current), current)
        return mask.scatter(dim=-1, index=indices, src=values)

    def _prepare_custom_mask(self, mask: Tensor, q: Tensor, key_len: int) -> Tensor:
        batch = q.shape[0]
        query_len = q.shape[-2]
        mask = mask.to(device=q.device)
        if mask.ndim == 2:
            if mask.shape != (query_len, key_len):
                raise ValueError(
                    "2D additive attention masks must have shape [query_length, key_length]"
                )
        elif mask.ndim == 3:
            if mask.shape[0] not in (1, batch) or mask.shape[-2:] != (query_len, key_len):
                raise ValueError(
                    "3D additive attention masks must have shape [batch, query_length, key_length]"
                )
            mask = mask[:, None, :, :]
        elif mask.ndim == 4:
            if (
                mask.shape[0] not in (1, batch)
                or mask.shape[1] not in (1, self.config.n_heads)
                or mask.shape[-2:] != (query_len, key_len)
            ):
                raise ValueError(
                    "4D additive attention masks must broadcast to "
                    "[batch, heads, query_length, key_length]"
                )
        else:
            raise ValueError("attention_mask must be a 2D, 3D, or 4D tensor")

        if mask.dtype == torch.bool:
            return mask
        if not torch.is_floating_point(mask):
            if not self._is_binary_mask(mask):
                raise ValueError("integer attention masks must contain only 0 and 1")
            return mask.to(dtype=torch.bool)
        return mask.to(dtype=q.dtype)

    def _combine_custom_mask(
        self,
        mask: Tensor,
        q: Tensor,
        key_len: int,
        *,
        query_start: int,
        key_start: int,
    ) -> Tensor:
        if self.is_full:
            return mask
        local_mask = self._local_policy_mask(
            q, key_len, query_start=query_start, key_start=key_start
        )
        if mask.dtype == torch.bool:
            return mask & torch.isfinite(local_mask)
        return mask + local_mask

    def _torch_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None) -> Tensor:
        use_flex = (
            not self.is_full
            and mask is None
            and compiled_flex_attention is not None
            and q.is_cuda
            and self.config.dropout == 0
        )
        if use_flex:
            block_mask = _local_block_mask(q.shape[-2], self.config.local_window, str(q.device))
            return compiled_flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
        repeat = self.config.n_heads // self.config.n_kv_heads
        if repeat > 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        if mask is not None:
            batch = q.shape[0]
            query_len = q.shape[-2]
            key_len = k.shape[-2]
            if self._is_key_mask(mask, batch, query_len):
                if mask.shape[-1] != key_len:
                    raise ValueError(
                        "2D padding attention masks must have shape [batch, sequence_length]"
                    )
                causal_mask = self._causal_mask(q, key_len, query_start=0, key_start=0)
                mask = causal_mask[None, None, :, :] + self._key_mask_to_additive(mask, q)
                mask = self._ensure_nonempty_rows(mask, query_start=0, key_start=0)
            else:
                mask = self._prepare_custom_mask(mask, q, key_len)
                mask = self._combine_custom_mask(mask, q, key_len, query_start=0, key_start=0)
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.config.dropout if self.training else 0.0
            )
        if self.is_full:
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.config.dropout if self.training else 0.0
            )
        local_mask = _local_causal_mask(q.shape[-2], self.config.local_window, str(q.device)).to(
            dtype=q.dtype
        )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=local_mask,
            dropout_p=self.config.dropout if self.training else 0.0,
        )

    def _project(self, x: Tensor, position_offset: int) -> tuple[Tensor, Tensor, Tensor]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.config.n_heads, self.config.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.config.n_kv_heads, self.config.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.config.n_kv_heads, self.config.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self.rope(q, k, position_offset)
        return q, k, v

    def _finish(self, x: Tensor, out: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        if self.g_proj is not None:
            out = out * torch.sigmoid(self.g_proj(x))
        return self.o_proj(out)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        q, k, v = self._project(x, position_offset=0)

        use_flash = flash_attn_func is not None and x.is_cuda and attention_mask is None
        if use_flash:
            window = (-1, -1) if self.is_full else (self.config.local_window - 1, 0)
            out = flash_attn_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=self.config.dropout if self.training else 0.0,
                causal=True,
                window_size=window,
            ).transpose(1, 2)
        else:
            out = self._torch_attention(q, k, v, attention_mask)
        return self._finish(x, out)

    def forward_cached(
        self,
        x: Tensor,
        cache: tuple[Tensor, Tensor] | None,
        position_offset: int,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        q, new_k, new_v = self._project(x, position_offset)
        if cache is None:
            k, v = new_k, new_v
            out = self._torch_attention(q, k, v, mask=attention_mask)
        else:
            past_len = cache[0].shape[-2]
            if position_offset < past_len:
                raise ValueError("position_offset cannot be smaller than the cached sequence")
            key_start = position_offset - past_len
            k = torch.cat((cache[0], new_k), dim=-2)
            v = torch.cat((cache[1], new_v), dim=-2)
            repeat = self.config.n_heads // self.config.n_kv_heads
            expanded_k = k.repeat_interleave(repeat, dim=1) if repeat > 1 else k
            expanded_v = v.repeat_interleave(repeat, dim=1) if repeat > 1 else v
            key_len = k.shape[-2]
            query_len = q.shape[-2]
            prepared_mask = None
            if attention_mask is not None:
                if self._is_key_mask(attention_mask, q.shape[0], query_len):
                    mask_len = attention_mask.shape[-1]
                    if mask_len == key_len:
                        key_mask = attention_mask
                    elif mask_len >= key_start + key_len:
                        key_mask = attention_mask[:, key_start : key_start + key_len]
                    else:
                        raise ValueError(
                            "cached padding mask does not cover all cached and current keys"
                        )
                    causal_mask = self._causal_mask(
                        q,
                        key_len,
                        query_start=position_offset,
                        key_start=key_start,
                    )
                    prepared_mask = causal_mask[None, None, :, :] + self._key_mask_to_additive(
                        key_mask, q
                    )
                    prepared_mask = self._ensure_nonempty_rows(
                        prepared_mask,
                        query_start=position_offset,
                        key_start=key_start,
                    )
                else:
                    custom_mask = attention_mask
                    expected_shape = (query_len, key_len)
                    if custom_mask.shape[-2:] != expected_shape:
                        query_end = position_offset + query_len
                        key_end = key_start + key_len
                        if custom_mask.shape[-2] < query_end or custom_mask.shape[-1] < key_end:
                            raise ValueError(
                                "cached additive mask does not cover the requested positions"
                            )
                        custom_mask = custom_mask[..., position_offset:query_end, key_start:key_end]
                    prepared_mask = self._prepare_custom_mask(custom_mask, q, key_len)
                    prepared_mask = self._combine_custom_mask(
                        prepared_mask,
                        q,
                        key_len,
                        query_start=position_offset,
                        key_start=key_start,
                    )
            elif not self.is_full or query_len > 1:
                prepared_mask = self._causal_mask(
                    q,
                    key_len,
                    query_start=position_offset,
                    key_start=key_start,
                )
            out = F.scaled_dot_product_attention(
                q,
                expanded_k,
                expanded_v,
                attn_mask=prepared_mask,
                dropout_p=self.config.dropout if self.training else 0.0,
            )
        if not self.is_full:
            k = k[:, :, -self.config.local_window :]
            v = v[:, :, -self.config.local_window :]
        return self._finish(x, out), (k, v)


class BoundedSwiGLU(nn.Module):
    def __init__(self, config: BarunConfig) -> None:
        super().__init__()
        self.gate_up = nn.Linear(config.dim, 2 * config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.clip = config.activation_clip

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        if self.clip > 0:
            gate = gate.clamp(max=self.clip)
            up = up.clamp(min=-self.clip, max=self.clip)
        return self.down(F.silu(gate) * up)


class BarunBlock(nn.Module):
    def __init__(self, config: BarunConfig, *, is_full: bool) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = GroupedAttention(config, is_full=is_full)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn = BoundedSwiGLU(config)
        self.dropout = config.dropout

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        x = x + F.dropout(self.attn(self.attn_norm(x), attention_mask), self.dropout, self.training)
        x = x + F.dropout(self.ffn(self.ffn_norm(x)), self.dropout, self.training)
        return x

    def forward_cached(
        self,
        x: Tensor,
        cache: tuple[Tensor, Tensor] | None,
        position_offset: int,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attention, new_cache = self.attn.forward_cached(
            self.attn_norm(x), cache, position_offset, attention_mask
        )
        x = x + attention
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class ResidualSelector(nn.Module):
    """Convexly select between a group's input checkpoint and its transformed output."""

    def __init__(self, config: BarunConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.dim, config.norm_eps, affine=False)
        self.score = nn.Linear(config.dim, 1, bias=False)
        self.last_mean_weights: Tensor | None = None

    def forward(self, checkpoint: Tensor, current: Tensor) -> Tensor:
        candidates = torch.stack((checkpoint, current), dim=-2)
        weights = self.score(self.norm(candidates)).softmax(dim=-2)
        if not torch.compiler.is_compiling():
            self.last_mean_weights = weights.detach().float().mean(dim=(0, 1, 3))
        return (weights * candidates).sum(dim=-2)


class BarunLM(nn.Module):
    def __init__(self, config: BarunConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            BarunBlock(config, is_full=(index + 1) % config.full_attention_every == 0)
            for index in range(config.n_layers)
        )
        selector_count = (
            config.n_layers // config.residual_select_every if config.residual_select_every else 0
        )
        self.selectors = nn.ModuleList(ResidualSelector(config) for _ in range(selector_count))
        self.final_norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.mtp_loss_weight > 0:
            self.mtp_norm = RMSNorm(config.dim, config.norm_eps)
            self.mtp_proj = nn.Linear(config.dim, config.dim, bias=False)
        else:
            self.mtp_norm = None
            self.mtp_proj = None
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        for name, parameter in module.named_parameters(recurse=False):
            if name == "weight" and isinstance(module, RMSNorm) and parameter is not None:
                nn.init.ones_(parameter)

    @staticmethod
    def _safe_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
        """Cross entropy that remains differentiable when every target is ignored."""
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1)
        if flat_targets.numel() == 0:
            return flat_logits.sum() * 0.0
        loss_sum = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100, reduction="sum")
        valid_targets = (flat_targets != -100).sum().clamp_min(1)
        return loss_sum / valid_targets

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        past_key_values: list[tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> BarunOutput:
        if position_offset + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(f"sequence length exceeds max_seq_len={self.config.max_seq_len}")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must have one entry per layer")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        x = self.embedding(input_ids)
        checkpoint = x
        selector_index = 0
        new_past_key_values = [] if use_cache else None
        for index, layer in enumerate(self.layers):
            if use_cache:
                layer_cache = past_key_values[index] if past_key_values is not None else None
                x, new_cache = layer.forward_cached(x, layer_cache, position_offset, attention_mask)
                new_past_key_values.append(new_cache)
            else:
                x = layer(x, attention_mask)
            stride = self.config.residual_select_every
            if stride and (index + 1) % stride == 0:
                x = self.selectors[selector_index](checkpoint, x)
                checkpoint = x
                selector_index += 1
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)
        if labels is None:
            return BarunOutput(logits=logits, past_key_values=new_past_key_values)

        # The state at position t predicts the label at t + 1.  This lets callers
        # use the conventional labels=input_ids contract and mask prompt tokens by
        # replacing their label positions with -100.
        causal_loss = self._safe_cross_entropy(logits[:, :-1], labels[:, 1:])
        mtp_loss = None
        loss = causal_loss
        offset = self.config.mtp_offset
        if self.mtp_proj is not None and hidden.shape[1] > offset:
            mtp_hidden = self.mtp_norm(hidden[:, :-offset] + self.mtp_proj(hidden[:, :-offset]))
            mtp_logits = self.lm_head(mtp_hidden)
            # MTP at position t predicts exactly the caller-supplied target at
            # t + offset.  Using labels directly preserves completion-only masks.
            mtp_labels = labels[:, offset:]
            mtp_loss = self._safe_cross_entropy(mtp_logits, mtp_labels)
            loss = loss + self.config.mtp_loss_weight * mtp_loss
        return BarunOutput(
            logits=logits,
            loss=loss,
            causal_loss=causal_loss,
            mtp_loss=mtp_loss,
            past_key_values=new_past_key_values,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        attention_mask: Tensor | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> Tensor:
        """Generate from unpadded or left-padded prompts with an optional EOS stop."""
        self.eval()
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, sequence] with sequence > 0")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if input_ids.shape[1] + max_new_tokens > self.config.max_seq_len:
            raise ValueError("prompt plus generation exceeds max_seq_len")
        for name, token_id in (("eos_token_id", eos_token_id), ("pad_token_id", pad_token_id)):
            if token_id is not None and not 0 <= token_id < self.config.vocab_size:
                raise ValueError(f"{name} must be within the model vocabulary")

        if attention_mask is None:
            generation_mask = torch.ones_like(input_ids, dtype=torch.bool)
            prefill_mask = None
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("generation attention_mask must have shape [batch, sequence]")
            attention_mask = attention_mask.to(device=input_ids.device)
            if (
                attention_mask.dtype != torch.bool
                and not torch.all((attention_mask == 0) | (attention_mask == 1)).item()
            ):
                raise ValueError("generation attention_mask must contain only 0 and 1")
            generation_mask = attention_mask.to(dtype=torch.bool)
            if not torch.all(generation_mask.any(dim=1)).item():
                raise ValueError("each generation prompt must contain at least one unmasked token")
            if torch.any(generation_mask[:, :-1] & ~generation_mask[:, 1:]).item():
                raise ValueError("batched generation supports left padding, not right padding")
            prefill_mask = attention_mask

        if max_new_tokens == 0:
            return input_ids

        output = self(input_ids, attention_mask=prefill_mask, use_cache=True)
        past_key_values = output.past_key_values
        if past_key_values is None:  # pragma: no cover - internal invariant
            raise RuntimeError("cache-enabled model forward did not return a cache")
        generated_ids = input_ids
        finished = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.bool)
        fill_token_id = pad_token_id if pad_token_id is not None else eos_token_id
        for generated in range(max_new_tokens):
            logits = output.logits[:, -1]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = (logits / temperature).softmax(dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            previously_finished = finished.clone()
            if fill_token_id is not None:
                filler = torch.full_like(next_token, fill_token_id)
                next_token = torch.where(previously_finished[:, None], filler, next_token)
            if eos_token_id is not None:
                finished |= (~previously_finished) & next_token.squeeze(-1).eq(eos_token_id)

            generated_ids = torch.cat((generated_ids, next_token), dim=1)
            # EOS itself is a valid key.  Tokens appended after a sequence has
            # already finished are padding and stay invisible to active rows.
            step_mask = ~previously_finished
            generation_mask = torch.cat((generation_mask, step_mask[:, None]), dim=1)
            if torch.all(finished).item():
                break
            if generated + 1 < max_new_tokens:
                output = self(
                    next_token,
                    attention_mask=generation_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_offset=generated_ids.shape[1] - 1,
                )
                past_key_values = output.past_key_values
                if past_key_values is None:  # pragma: no cover - internal invariant
                    raise RuntimeError("cache-enabled model forward did not return a cache")
        return generated_ids

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        embedding = self.embedding.weight.numel()
        return {"total": total, "non_embedding": total - embedding, "embedding": embedding}
