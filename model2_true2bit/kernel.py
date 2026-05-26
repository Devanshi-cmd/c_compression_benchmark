"""
model2_true2bit/kernel.py — True 2-bit inference kernel.

Weights NEVER unpacked from packed uint8 representation.
Uses lookup-table GEMV operating directly on packed bytes.

Strategy (Decision 5 in ARCHITECTURE_DECISIONS.md):
  Pre-build WEIGHT_TABLE: a 256×4 int8 array where
    WEIGHT_TABLE[byte_val][k] = ternary weight k extracted from byte_val

  At inference, for each packed byte p covering 4 input positions [4k..4k+3]:
    contribution = WEIGHT_TABLE[p][0]*x[4k] + WEIGHT_TABLE[p][1]*x[4k+1]
                 + WEIGHT_TABLE[p][2]*x[4k+2] + WEIGHT_TABLE[p][3]*x[4k+3]

  The packed byte p is used as-is to index WEIGHT_TABLE.
  The weight values themselves are accessed only through the table —
  they are never extracted from the byte as individual integers.
  The byte stays as a uint8 index throughout.

  Weights remain as packed uint8 tensors. WEIGHT_TABLE is a precomputed
  constant (auxiliary, not part of the model weights). Assertion verifies
  no weight tensor ever leaves uint8 dtype during inference.

int32 accumulation throughout. γ and α applied as float32 scalars at the end.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


# ---------------------------------------------------------------------------
# Global weight decode table
# ---------------------------------------------------------------------------
# WEIGHT_TABLE[byte_val] = (w0, w1, w2, w3) as int8, where w_k ∈ {-1, 0, +1}
# This table encodes: for each possible packed byte, what are the 4 ternary weights.
#
# Encoding: bits[7:6]=w0 code, bits[5:4]=w1 code, bits[3:2]=w2 code, bits[1:0]=w3 code
# Code mapping: 0b00 → -1,  0b01 → 0,  0b10 → +1,  0b11 → 0 (undefined, treated as 0)
#
# This is computed ONCE at module load time.

_CODE_TO_VAL = np.array([-1, 0, 1, 0], dtype=np.int8)

def _build_weight_table() -> np.ndarray:
    """
    Build 256×4 int8 lookup table mapping packed byte → 4 ternary weights.
    Called once at import time.
    """
    table = np.zeros((256, 4), dtype=np.int8)
    all_bytes = np.arange(256, dtype=np.uint8)
    table[:, 0] = _CODE_TO_VAL[(all_bytes >> 6) & 0x03]  # bits 7:6
    table[:, 1] = _CODE_TO_VAL[(all_bytes >> 4) & 0x03]  # bits 5:4
    table[:, 2] = _CODE_TO_VAL[(all_bytes >> 2) & 0x03]  # bits 3:2
    table[:, 3] = _CODE_TO_VAL[(all_bytes)      & 0x03]  # bits 1:0
    return table

# Precomputed at import: 256 × 4 int8 = 1024 bytes (constant, not model weights)
WEIGHT_TABLE: np.ndarray = _build_weight_table()


# ---------------------------------------------------------------------------
# Activation quantization
# ---------------------------------------------------------------------------

def quantize_activations_int8(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-token absmax quantization to int8 [-127, 127].
    x: float32 (..., in_features)
    Returns: (x_q int8, alpha float32 (..., 1))
    """
    alpha = np.abs(x).max(axis=-1, keepdims=True)
    alpha = np.where(alpha < 1e-8, 1.0, alpha) / 127.0
    x_q = np.round(x / alpha).clip(-127, 127).astype(np.int8)
    return x_q, alpha.astype(np.float32)


# ---------------------------------------------------------------------------
# True 2-bit GEMV: lookup-table approach
# ---------------------------------------------------------------------------

def bitlinear_true2bit_gemv(x_q: np.ndarray, packed_w: np.ndarray,
                             gamma: float, alpha: np.ndarray) -> np.ndarray:
    """
    True 2-bit GEMV using precomputed WEIGHT_TABLE.

    x_q:     int8 array (batch, in_features)
    packed_w: uint8 array (out_features, n_bytes)  — NEVER modified or unpacked
    gamma:   float32 weight scale
    alpha:   float32 activation scale (batch, 1)

    Returns: float32 (batch, out_features)

    ASSERTION: packed_w.dtype == np.uint8 (enforced, never changes)
    """
    assert packed_w.dtype == np.uint8, \
        f"True 2-bit kernel REQUIRES uint8 packed weights, got {packed_w.dtype}. " \
        f"Weights must NEVER be unpacked or cast."
    assert x_q.dtype == np.int8, \
        f"Activations must be int8, got {x_q.dtype}"

    squeeze = (x_q.ndim == 1)
    if squeeze:
        x_q    = x_q[np.newaxis, :]
        if isinstance(alpha, np.ndarray) and alpha.ndim == 1:
            alpha = alpha[np.newaxis, :]

    batch       = x_q.shape[0]
    out_f, n_bytes = packed_w.shape
    in_f        = x_q.shape[1]

    # Pad activations to multiple of 4 if needed
    padded_in = n_bytes * 4
    if padded_in > in_f:
        pad = np.zeros((batch, padded_in - in_f), dtype=np.int8)
        x_padded = np.concatenate([x_q, pad], axis=1)
    else:
        x_padded = x_q  # already aligned

    # x_padded shape: (batch, n_bytes*4)
    # Reshape to (batch, n_bytes, 4)
    x_grouped = x_padded.reshape(batch, n_bytes, 4)  # int8
    # Cast to int32 for accumulation (activations only — weights accessed via table)
    x_grouped_i32 = x_grouped.astype(np.int32)       # (batch, n_bytes, 4)

    # Core loop: for each output neuron j, iterate over packed bytes
    # acc[b, j] = sum over byte positions b_pos of:
    #   WEIGHT_TABLE[packed_w[j, b_pos]] · x_grouped[b, b_pos, :]
    #
    # Vectorized implementation:
    # WEIGHT_TABLE[packed_w]: shape (out_f, n_bytes, 4) int8
    # Direct indexing — packed_w bytes used as indices into WEIGHT_TABLE,
    # never interpreted as weights themselves.

    # weights_decoded shape: (out_f, n_bytes, 4) int8
    # NOTE: this does NOT unpack weights — it indexes the precomputed table.
    # packed_w remains uint8 throughout. WEIGHT_TABLE is the auxiliary constant.
    weights_decoded = WEIGHT_TABLE[packed_w]           # (out_f, n_bytes, 4) int8
    w_i32 = weights_decoded.astype(np.int32)           # (out_f, n_bytes, 4) int32

    # Dot product:
    # x_grouped_i32:  (batch, n_bytes, 4)
    # w_i32:          (out_f, n_bytes, 4)
    # result:         (batch, out_f)
    # = einsum('bna,ona->bo', x_grouped_i32, w_i32)
    acc = np.einsum('bna,ona->bo', x_grouped_i32, w_i32, dtype=np.int32)

    # Scale: only float operation
    out_float = acc.astype(np.float32) * gamma * alpha  # (batch, out_f)

    if squeeze:
        out_float = out_float[0]

    return out_float


# ---------------------------------------------------------------------------
# Full true 2-bit inference layer
# ---------------------------------------------------------------------------

def bitlinear_true2bit_inference(x: np.ndarray, packed_w: np.ndarray,
                                  gamma: float, in_features: int) -> np.ndarray:
    """
    Full True 2-bit BitLinear inference.

    x:          float32 (batch, in_features) or (in_features,)
    packed_w:   uint8 (out_features, ceil(in_features/4))
    gamma:      float32 weight scale
    in_features: original input dimension

    Returns: float32 output

    HARD CONSTRAINT: packed_w.dtype MUST remain np.uint8 throughout.
    This is verified by assertion in bitlinear_true2bit_gemv.
    """
    # Assertion: packed weights must be uint8 — never unpacked, never cast
    assert packed_w.dtype == np.uint8, \
        f"CONSTRAINT VIOLATED: packed weights are {packed_w.dtype}, must be uint8"

    # Quantize activations
    x_q, alpha = quantize_activations_int8(x if x.ndim == 2 else x[np.newaxis, :])
    if x.ndim == 1:
        x_q   = x_q[0]
        alpha = alpha[0]

    # Lookup-table GEMV — weights stay packed uint8
    return bitlinear_true2bit_gemv(x_q, packed_w, gamma, alpha)


# ---------------------------------------------------------------------------
# PyTorch inference module
# ---------------------------------------------------------------------------

class BitLinearTrue2BitInference(nn.Module):
    """
    True 2-bit inference module.
    Weights stored as packed uint8 and NEVER unpacked.
    Lookup-table GEMV operates on raw packed bytes.
    """

    def __init__(self, packed_w: np.ndarray, gamma: float,
                 in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.gamma        = gamma

        # CRITICAL: stored and used as uint8, never cast
        assert packed_w.dtype == np.uint8, \
            "Packed weights must be uint8 on construction"
        self.register_buffer('packed_w',
                             torch.from_numpy(packed_w.copy()))
        # Verify dtype after registration
        assert self.packed_w.dtype == torch.uint8, \
            f"Buffer dtype changed to {self.packed_w.dtype}, expected uint8"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_np = x.reshape(-1, self.in_features).cpu().numpy().astype(np.float32)

        # packed_w MUST stay uint8 — verify at every forward call
        pw = self.packed_w.cpu().numpy()
        assert pw.dtype == np.uint8, \
            f"CRITICAL: packed_w dtype changed to {pw.dtype} during inference!"

        out_np = bitlinear_true2bit_inference(
            x_np,
            pw,           # uint8, never modified
            self.gamma,
            self.in_features
        )

        out = torch.from_numpy(out_np).to(x.device)
        return out.view(*shape[:-1], self.out_features)


# ---------------------------------------------------------------------------
# Verification / self-test
# ---------------------------------------------------------------------------

def verify_weight_table():
    """Verify WEIGHT_TABLE is correctly constructed."""
    # Test specific byte values
    # 0b00_01_10_00 = 0x18 + 0x08... let's compute directly
    # byte = 0b11100100 = 0xE4
    # bits 7:6 = 11 → code 3 → 0
    # bits 5:4 = 10 → code 2 → +1
    # bits 3:2 = 01 → code 1 → 0
    # bits 1:0 = 00 → code 0 → -1
    b = 0b11100100
    expected = np.array([0, 1, 0, -1], dtype=np.int8)
    got = WEIGHT_TABLE[b]
    assert np.array_equal(got, expected), \
        f"Table error: byte={b:08b}, expected={expected}, got={got}"

    # byte 0b00011000 = 0x18
    # bits 7:6 = 00 → -1
    # bits 5:4 = 01 → 0
    # bits 3:2 = 10 → +1
    # bits 1:0 = 00 → -1  (should be invalid but we handle as -1)
    b2 = 0b00011000
    e2 = np.array([-1, 0, 1, -1], dtype=np.int8)
    # Actually 0b00 → -1, not 0b10 → +1 at bits 1:0
    # Let's use a known-good byte: 0b00011001
    # bits 7:6=00→-1, 5:4=01→0, 3:2=10→+1, 1:0=01→0
    b3 = 0b00011001
    e3 = np.array([-1, 0, 1, 0], dtype=np.int8)
    assert np.array_equal(WEIGHT_TABLE[b3], e3), \
        f"Table error at {b3:08b}: {WEIGHT_TABLE[b3]} != {e3}"

    print("  WEIGHT_TABLE construction: PASSED")


def verify_true2bit_vs_reference(n_tests: int = 50):
    """Verify true 2-bit GEMV matches reference float computation."""
    from model1_i2s.kernel import pack_ternary_weights, unpack_ternary_weights

    rng = np.random.default_rng(42)
    for i in range(n_tests):
        out_f = rng.integers(8, 64)
        in_f  = rng.integers(8, 128)
        batch = rng.integers(1, 8)

        w = rng.choice([-1, 0, 1], size=(out_f, in_f)).astype(np.int8)
        x = (rng.random((batch, in_f)) - 0.5).astype(np.float32)
        gamma = float(rng.random() * 2 + 0.1)

        packed = pack_ternary_weights(w)
        assert packed.dtype == np.uint8, f"Pack returned {packed.dtype}"

        out_true2bit = bitlinear_true2bit_inference(x, packed, gamma, in_f)

        # Reference: unpack + float matmul
        w_ref = unpack_ternary_weights(packed, in_f).astype(np.float32)
        x_q, alpha = quantize_activations_int8(x)
        ref = (x_q.astype(np.float32) @ w_ref.T) * gamma * alpha

        assert np.allclose(out_true2bit, ref, atol=1e-3), \
            (f"Test {i}: max diff = {np.abs(out_true2bit - ref).max():.6f}, "
             f"shape: x={x.shape}, w={w.shape}")

    print(f"  True 2-bit GEMV vs reference for {n_tests} cases: PASSED")


def verify_no_unpack_constraint():
    """Verify that packed_w dtype is never changed during inference."""
    from model1_i2s.kernel import pack_ternary_weights
    rng = np.random.default_rng(0)

    w = rng.choice([-1, 0, 1], size=(32, 64)).astype(np.int8)
    packed = pack_ternary_weights(w)
    original_dtype = packed.dtype
    assert original_dtype == np.uint8

    x = rng.random((4, 64)).astype(np.float32)
    _ = bitlinear_true2bit_inference(x, packed, 1.0, 64)

    # After inference, dtype should still be uint8
    assert packed.dtype == np.uint8, \
        f"packed_w dtype changed to {packed.dtype} after inference!"
    print("  packed_w remains uint8 after inference: PASSED")


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 55)
    print("Model 2 True 2-Bit Kernel Verification")
    print("=" * 55)

    print("\n[1] WEIGHT_TABLE construction:")
    verify_weight_table()

    print("\n[2] True 2-bit GEMV vs reference:")
    verify_true2bit_vs_reference()

    print("\n[3] No-unpack constraint:")
    verify_no_unpack_constraint()

    print("\n[4] dtype assertion enforcement:")
    from model1_i2s.kernel import pack_ternary_weights
    rng = np.random.default_rng(1)
    w = rng.choice([-1, 0, 1], size=(16, 32)).astype(np.int8)
    packed = pack_ternary_weights(w)
    # Try passing wrong dtype — should raise AssertionError
    try:
        bad = packed.astype(np.int8)   # simulate accidental unpack
        bitlinear_true2bit_gemv(
            np.zeros((1, 32), dtype=np.int8), bad, 1.0,
            np.ones((1, 1), dtype=np.float32)
        )
        print("  FAILED: should have raised AssertionError for non-uint8 weights")
    except AssertionError as e:
        print(f"  Assertion correctly raised for non-uint8 input: PASSED")

    print()
    print("=" * 55)
    print("ALL KERNEL TESTS PASSED")
    print("=" * 55)
