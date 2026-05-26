"""
model2_true2bit/model.py — Transformer with BitNet b1.58 ternary quantization.
True 2-bit Style: at inference, packed 2-bit weights are NEVER unpacked.
Computation uses lookup-table GEMV operating directly on packed uint8 bytes.

Identical architecture to Model 1 — only the inference kernel differs.
"""

import math
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import shared training components from model1
# (Both models use identical training-time quantization)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from model1_i2s.model import (
    BitLinear,
    MultiHeadSelfAttention,
    FeedForward,
    TransformerBlock,
    precompute_rope_freqs,
    apply_rope,
)


# ---------------------------------------------------------------------------
# Main Model (identical architecture to Model 1)
# ---------------------------------------------------------------------------

class BitNetTransformerTrue2Bit(nn.Module):
    """
    BitNet b1.58 Transformer — True 2-bit inference style.
    Training is identical to Model 1.
    Inference uses lookup-table GEMV on packed uint8 bytes (kernel.py).
    Weights are NEVER unpacked to int8 at inference time.
    """

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

        # Tie weights
        self.lm_head.weight = self.token_emb.weight

        # RoPE buffers
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
        B, T = idx.shape
        assert T <= self.CONTEXT_LEN

        x = self.token_emb(idx)
        rope_cos = self.rope_cos[:T]
        rope_sin = self.rope_sin[:T]

        for block in self.blocks:
            x = block(x, rope_cos, rope_sin)

        x = self.norm_out(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_packed_weights(self, path: str):
        """
        Export model for True 2-bit inference:
        - BitLinear weights packed as 2-bit (uint8, 4 per byte)
        - Scale factors γ saved as float32
        - Embeddings and norms saved as float32
        """
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
                    # CRITICAL: store as uint8 — never convert to int8 at inference
                    state[f'{name}.w2bit'] = packed          # uint8
                    state[f'{name}.gamma'] = gamma.item()    # float32 scalar
                    state[f'{name}.shape'] = w_q.shape       # (out, in)
                elif isinstance(module, nn.Embedding):
                    state[f'{name}.weight'] = module.weight.cpu().numpy()
                elif isinstance(module, nn.LayerNorm):
                    state[f'{name}.weight'] = module.weight.cpu().numpy()
                    state[f'{name}.bias']   = module.bias.cpu().numpy()

        np.save(os.path.join(path, 'packed_weights.npy'),
                state, allow_pickle=True)

        config = {
            'vocab_size': self.vocab_size,
            'd_model': self.D_MODEL,
            'd_ff': self.D_FF,
            'n_heads': self.N_HEADS,
            'n_layers': self.N_LAYERS,
            'context_len': self.CONTEXT_LEN,
            'model_type': 'true2bit',
        }
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Packed True 2-bit model saved to {path}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(vocab_size: int, verify_params: bool = True) -> BitNetTransformerTrue2Bit:
    model = BitNetTransformerTrue2Bit(vocab_size)
    n_params = model.count_parameters()

    if verify_params:
        target = 12_000_000
        tolerance = 200_000
        print(f"[Model 2 True2Bit] Parameters: {n_params:,} "
              f"(target: {target:,} ± {tolerance:,})")
        assert abs(n_params - target) <= tolerance, \
            (f"Parameter count {n_params:,} is outside tolerance. "
             f"Adjust D_MODEL/N_LAYERS.")

    return model


if __name__ == '__main__':
    for vocab in [96, 100, 110, 128]:
        m = BitNetTransformerTrue2Bit(vocab)
        n = m.count_parameters()
        print(f"  vocab={vocab:4d}: {n:,} params  "
              f"({'OK' if abs(n-12_000_000) < 300_000 else 'ADJUST'})")
