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
def _local_causal_mask(seq_len: int, window: int, device_type: str) -> Tensor:
    device = torch.device(device_type)
    row = torch.arange(seq_len, device=device)[:, None]
    col = torch.arange(seq_len, device=device)[None, :]
    allowed = (col <= row) & (col > row - window)
    mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.float32)
    return mask.masked_fill(~allowed, float("-inf"))


@lru_cache(maxsize=32)
def _local_bidirectional_mask(seq_len: int, window: int, device_type: str) -> Tensor:
    """Window constraint for custom masks such as bidirectional PrefixLM prompts."""
    device = torch.device(device_type)
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
            mask = mask.to(dtype=q.dtype)
            if not self.is_full:
                local_mask = _local_bidirectional_mask(
                    q.shape[-2], self.config.local_window, q.device.type
                ).to(dtype=q.dtype)
                mask = mask + local_mask
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.config.dropout if self.training else 0.0
            )
        if self.is_full:
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.config.dropout if self.training else 0.0
            )
        local_mask = _local_causal_mask(q.shape[-2], self.config.local_window, q.device.type)
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
            k = torch.cat((cache[0], new_k), dim=-2)
            v = torch.cat((cache[1], new_v), dim=-2)
            if not self.is_full:
                k = k[:, :, -self.config.local_window :]
                v = v[:, :, -self.config.local_window :]
            repeat = self.config.n_heads // self.config.n_kv_heads
            expanded_k = k.repeat_interleave(repeat, dim=1) if repeat > 1 else k
            expanded_v = v.repeat_interleave(repeat, dim=1) if repeat > 1 else v
            if attention_mask is not None:
                attention_mask = attention_mask[..., -k.shape[-2] :].to(dtype=q.dtype)
            out = F.scaled_dot_product_attention(
                q,
                expanded_k,
                expanded_v,
                attn_mask=attention_mask,
                is_causal=False,
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
        x = self.embedding(input_ids)
        checkpoint = x
        selector_index = 0
        new_past_key_values = [] if use_cache else None
        for index, layer in enumerate(self.layers):
            if use_cache:
                layer_cache = past_key_values[index] if past_key_values is not None else None
                x, new_cache = layer.forward_cached(
                    x, layer_cache, position_offset, attention_mask
                )
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

        causal_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        mtp_loss = None
        loss = causal_loss
        offset = self.config.mtp_offset
        if self.mtp_proj is not None and hidden.shape[1] > offset:
            mtp_hidden = self.mtp_norm(hidden[:, :-offset] + self.mtp_proj(hidden[:, :-offset]))
            mtp_logits = self.lm_head(mtp_hidden)
            mtp_labels = input_ids[:, offset:].clone()
            target_mask = labels[:, offset - 1 : -1] == -100
            mtp_labels.masked_fill_(target_mask, -100)
            mtp_loss = F.cross_entropy(
                mtp_logits.reshape(-1, mtp_logits.shape[-1]), mtp_labels.reshape(-1)
            )
            loss = loss + self.config.mtp_loss_weight * mtp_loss
        return BarunOutput(
            logits=logits,
            loss=loss,
            causal_loss=causal_loss,
            mtp_loss=mtp_loss,
            past_key_values=new_past_key_values,
        )

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int, temperature: float = 0.8) -> Tensor:
        self.eval()
        if input_ids.shape[1] + max_new_tokens > self.config.max_seq_len:
            raise ValueError("prompt plus generation exceeds max_seq_len")
        output = self(input_ids, use_cache=True)
        past_key_values = output.past_key_values
        for generated in range(max_new_tokens):
            logits = output.logits[:, -1]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = (logits / temperature).softmax(dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if generated + 1 < max_new_tokens:
                output = self(
                    next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_offset=input_ids.shape[1] - 1,
                )
                past_key_values = output.past_key_values
        return input_ids

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        embedding = self.embedding.weight.numel()
        return {"total": total, "non_embedding": total - embedding, "embedding": embedding}
