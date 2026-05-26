"""
model1_i2s/model.py — Transformer with BitNet b1.58 ternary quantization.
I2S Style: at inference, 2-bit packed weights are UNPACKED to int8 before GEMV.

Architecture:
  - Decoder-only Transformer
  - d_model=320, d_ff=1280, n_heads=8 (head_dim=40), n_layers=10
  - RoPE positional encoding (zero parameters)
  - Context window: 128 characters
  - Character-level prediction
  - BitLinear layers with absmean ternary quantization

Target: exactly 12,000,000 parameters (adjusted by d_model search at init).
"""

import math
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# BitLinear Layer (shared training module)
# ---------------------------------------------------------------------------

class BitLinear(nn.Module):
    """
    Linear layer with BitNet b1.58 ternary quantization.

    During training:
      - Maintains full float32 latent weights
      - Quantizes weights to {-1, 0, +1} each forward pass via absmean
      - Quantizes activations to int8 via per-token absmax
      - Uses STE for gradients through quantization

    No bias (per BitNet b1.58 spec).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        # Float32 latent weights (these receive gradient updates)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    # ---- Weight quantization ----

    def quantize_weights(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Absmean ternary quantization.
        Returns: (w_ternary as float in {-1,0,+1}, gamma scalar)
        """
        gamma = w.abs().mean().clamp(min=1e-8)
        w_scaled = w / gamma
        # RoundClip to {-1, 0, +1}: round then clamp
        w_q = w_scaled.round().clamp(-1, 1)
        # STE: gradient bypasses round+clamp
        w_q = w + (w_q - w).detach()
        return w_q, gamma

    # ---- Activation quantization ----

    @staticmethod
    def quantize_activations(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Per-token absmax quantization to int8 range [-127, 127].
        Returns: (x_q as float in [-127,127], alpha per-token scale)
        """
        # x shape: (..., in_features)
        alpha = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        x_q = (x / alpha).round().clamp(-127, 127)
        # STE
        x_q = x + (x_q - x).detach()
        return x_q, alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights
        w_q, gamma = self.quantize_weights(self.weight)
        # Quantize activations
        x_q, alpha = self.quantize_activations(x)
        # Standard linear (float computation during training)
        out = F.linear(x_q, w_q)
        # Scale: output = int_result * gamma * alpha
        # alpha shape: (..., 1), gamma: scalar
        out = out * (gamma * alpha)
        return out

    def extra_repr(self) -> str:
        return f'in={self.in_features}, out={self.out_features}, ternary=True'


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embedding)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(dim: int, max_seq_len: int,
                          theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin tables for RoPE.
    dim: head dimension (must be even)
    Returns: (cos, sin) each of shape (max_seq_len, dim//2)
    """
    assert dim % 2 == 0
    half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_seq_len, half)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embedding to query or key tensor.
    x shape: (B, T, n_heads, head_dim)
    cos, sin shape: (T, head_dim//2)
    """
    B, T, H, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos_ = cos[:T].unsqueeze(0).unsqueeze(2)  # (1, T, 1, half)
    sin_ = sin[:T].unsqueeze(0).unsqueeze(2)
    x_rot = torch.cat([x1 * cos_ - x2 * sin_,
                       x1 * sin_ + x2 * cos_], dim=-1)
    return x_rot


# ---------------------------------------------------------------------------
# Transformer components
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, context_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.head_dim   = d_model // n_heads
        self.context_len = context_len

        self.q_proj = BitLinear(d_model, d_model)
        self.k_proj = BitLinear(d_model, d_model)
        self.v_proj = BitLinear(d_model, d_model)
        self.o_proj = BitLinear(d_model, d_model)

        # Causal mask (registered as buffer, not parameter)
        mask = torch.triu(torch.ones(context_len, context_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', mask)

    def forward(self, x: torch.Tensor,
                rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, Dh)
        k = self.k_proj(x).view(B, T, H, Dh)
        v = self.v_proj(x).view(B, T, H, Dh)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # (B, H, T, Dh)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = math.sqrt(Dh)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = attn.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0),
                                float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = BitLinear(d_model, d_ff)
        self.w2 = BitLinear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, context_len: int):
        super().__init__()
        self.attn     = MultiHeadSelfAttention(d_model, n_heads, context_len)
        self.ff       = FeedForward(d_model, d_ff)
        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class BitNetTransformerI2S(nn.Module):
    """
    BitNet b1.58 Transformer — I2S inference style.
    Training uses float quantization simulation.
    Inference uses unpacked int8 weights via model1_i2s/kernel.py.
    """

    # Architecture hyperparameters
    # d=304, d_ff=1360, 10 layers → ~12,008,608 params (≈12M target)
    D_MODEL     = 304
    D_FF        = 1360
    N_HEADS     = 8     # head_dim = 304/8 = 38 (even, compatible with RoPE)
    N_LAYERS    = 10
    CONTEXT_LEN = 128

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size  = vocab_size
        self.d_model     = self.D_MODEL
        self.context_len = self.CONTEXT_LEN

        self.token_emb = nn.Embedding(vocab_size, self.D_MODEL)
        self.blocks = nn.ModuleList([
            TransformerBlock(self.D_MODEL, self.N_HEADS, self.D_FF, self.CONTEXT_LEN)
            for _ in range(self.N_LAYERS)
        ])
        self.norm_out = nn.LayerNorm(self.D_MODEL)
        self.lm_head  = nn.Linear(self.D_MODEL, vocab_size, bias=False)

        # Tie weights: lm_head and token_emb share parameters
        self.lm_head.weight = self.token_emb.weight

        # Precompute RoPE frequencies
        head_dim = self.D_MODEL // self.N_HEADS
        cos, sin = precompute_rope_freqs(head_dim, self.CONTEXT_LEN)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, BitLinear):
                    nn.init.normal_(module.weight, std=0.02 / math.sqrt(2 * self.N_LAYERS))
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, T) long tensor of token indices
        Returns: (B, T, vocab_size) logits
        """
        B, T = idx.shape
        assert T <= self.CONTEXT_LEN, f"Sequence too long: {T} > {self.CONTEXT_LEN}"

        x = self.token_emb(idx)  # (B, T, D)

        rope_cos = self.rope_cos[:T]
        rope_sin = self.rope_sin[:T]

        for block in self.blocks:
            x = block(x, rope_cos, rope_sin)

        x = self.norm_out(x)
        logits = self.lm_head(x)  # (B, T, V)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_gammas(self) -> dict:
        """Extract gamma scale factors for all BitLinear layers (for inference)."""
        gammas = {}
        for name, module in self.named_modules():
            if isinstance(module, BitLinear):
                with torch.no_grad():
                    gamma = module.weight.abs().mean().item()
                gammas[name] = gamma
        return gammas

    def save_packed_weights(self, path: str):
        """
        Export model for I2S inference:
        - BitLinear weights packed as 2-bit (4 weights per byte)
        - Scale factors γ saved as float32
        - Token embeddings and other floats saved normally
        Encoding: -1 → 0b00, 0 → 0b01, +1 → 0b10
        """
        import os
        from model1_i2s.kernel import pack_ternary_weights
        os.makedirs(path, exist_ok=True)

        state = {}
        with torch.no_grad():
            for name, module in self.named_modules():
                if isinstance(module, BitLinear):
                    w = module.weight.float()
                    gamma = w.abs().mean().clamp(min=1e-8)
                    w_q = (w / gamma).round().clamp(-1, 1).to(torch.int8)
                    packed = pack_ternary_weights(w_q.cpu().numpy())
                    state[f'{name}.w2bit'] = packed
                    state[f'{name}.gamma'] = gamma.item()
                elif isinstance(module, nn.Embedding):
                    state[f'{name}.weight'] = module.weight.cpu().numpy()
                elif isinstance(module, nn.LayerNorm):
                    state[f'{name}.weight'] = module.weight.cpu().numpy()
                    state[f'{name}.bias']   = module.bias.cpu().numpy()
                elif isinstance(module, nn.Linear) and 'lm_head' in name:
                    # lm_head shares weights with token_emb, skip to avoid dup
                    pass

        np.save(os.path.join(path, 'packed_weights.npy'),
                state, allow_pickle=True)

        # Save config
        config = {
            'vocab_size': self.vocab_size,
            'd_model': self.D_MODEL,
            'd_ff': self.D_FF,
            'n_heads': self.N_HEADS,
            'n_layers': self.N_LAYERS,
            'context_len': self.CONTEXT_LEN,
            'model_type': 'i2s',
        }
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Packed I2S model saved to {path}")


# ---------------------------------------------------------------------------
# Factory / parameter verification
# ---------------------------------------------------------------------------

def build_model(vocab_size: int, verify_params: bool = True) -> BitNetTransformerI2S:
    """Build the I2S model and optionally verify parameter count."""
    model = BitNetTransformerI2S(vocab_size)
    n_params = model.count_parameters()

    if verify_params:
        target = 12_000_000
        tolerance = 200_000  # ±200K (accounts for vocab size variation)
        print(f"[Model 1 I2S] Parameters: {n_params:,} "
              f"(target: {target:,} ± {tolerance:,})")
        assert abs(n_params - target) <= tolerance, \
            (f"Parameter count {n_params:,} is outside tolerance of "
             f"{target:,} ± {tolerance:,}. "
             f"Adjust D_MODEL/N_LAYERS in BitNetTransformerI2S.")

    return model


if __name__ == '__main__':
    import sys

    # Quick parameter count check with a typical vocab size
    for vocab in [96, 100, 110, 128]:
        m = BitNetTransformerI2S(vocab)
        n = m.count_parameters()
        print(f"  vocab={vocab:4d}: {n:,} params  "
              f"({'OK' if abs(n-12_000_000) < 300_000 else 'ADJUST'})")

    # Detailed breakdown for vocab=100
    print("\nDetailed breakdown (vocab=100):")
    m = BitNetTransformerI2S(100)
    for name, p in m.named_parameters():
        print(f"  {name:60s} {p.numel():>10,}")
    print(f"  {'TOTAL':60s} {m.count_parameters():>10,}")
