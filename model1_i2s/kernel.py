"""
model1_i2s/kernel.py — I2S inference kernel.

Mirrors Microsoft's I2_S kernel architecture:
  1. Load 2-bit packed weights from disk
  2. UNPACK to int8 (values in {-1, 0, +1})
  3. Quantize input activations to int8
  4. Vanilla GEMV: int8 × int8 → int32 accumulation
  5. Scale output by γ (float32) and α (float32 per token)

Weight encoding: -1 → 0b00, 0 → 0b01, +1 → 0b10
Four weights packed per uint8 byte (bits 7:6, 5:4, 3:2, 1:0).

NO floats used on weight values — only the scale factor γ and
activation scale α are float32. Weight values stay int until GEMV output.
"""

import math
import numpy as np
import torch
from typing import Tuple


# ---------------------------------------------------------------------------
# Packing / unpacking
# ---------------------------------------------------------------------------

# Encoding map: int8 ternary value → 2-bit code
ENCODE_MAP = {-1: 0b00, 0: 0b01, 1: 0b10}
# Decode map: 2-bit code → int8 ternary value  (codes: 0→-1, 1→0, 2→+1, 3→unused)
DECODE_MAP = np.array([-1, 0, 1, 0], dtype=np.int8)  # code 3 treated as 0


def pack_ternary_weights(w: np.ndarray) -> np.ndarray:
    """
    Pack a 2D int8 ternary weight matrix into 2-bit packed uint8.

    w: shape (out_features, in_features), dtype int8, values in {-1, 0, +1}
    Returns: uint8 array of shape (out_features, ceil(in_features / 4))

    Layout per byte: bits [7:6]=weight[0], [5:4]=weight[1], [3:2]=weight[2], [1:0]=weight[3]
    """
    assert w.ndim == 2, "Expected 2D weight matrix"
    assert w.dtype == np.int8, f"Expected int8, got {w.dtype}"
    assert np.all((w >= -1) & (w <= 1)), "Values must be in {-1, 0, +1}"

    out_features, in_features = w.shape
    padded_in = math.ceil(in_features / 4) * 4
    # Pad to multiple of 4 with zeros
    if padded_in > in_features:
        pad = np.zeros((out_features, padded_in - in_features), dtype=np.int8)
        w = np.concatenate([w, pad], axis=1)

    # Convert {-1,0,+1} → {0,1,2} (add 1)
    codes = (w + 1).astype(np.uint8)  # now in {0,1,2}

    # Pack 4 codes per byte
    # codes shape: (out_features, padded_in)
    # Reshape to (out_features, padded_in//4, 4)
    codes_4 = codes.reshape(out_features, padded_in // 4, 4)

    # bits 7:6 = codes_4[..,0], 5:4 = codes_4[..,1], 3:2 = codes_4[..,2], 1:0 = codes_4[..,3]
    packed = ((codes_4[:, :, 0] << 6) |
              (codes_4[:, :, 1] << 4) |
              (codes_4[:, :, 2] << 2) |
              (codes_4[:, :, 3])).astype(np.uint8)

    return packed  # shape (out_features, ceil(in_features/4))


def unpack_ternary_weights(packed: np.ndarray, original_in_features: int) -> np.ndarray:
    """
    Unpack 2-bit packed weights back to int8 {-1, 0, +1}.

    packed: uint8 array of shape (out_features, ceil(in_features/4))
    original_in_features: to trim padding
    Returns: int8 array of shape (out_features, in_features)
    """
    assert packed.dtype == np.uint8, f"Expected uint8, got {packed.dtype}"

    out_features, n_bytes = packed.shape

    # Extract 4 codes per byte via bit shifting
    # Each code is 2 bits → value in {0,1,2,3}
    codes = np.stack([
        (packed >> 6) & 0x03,  # bits 7:6
        (packed >> 4) & 0x03,  # bits 5:4
        (packed >> 2) & 0x03,  # bits 3:2
        (packed)      & 0x03,  # bits 1:0
    ], axis=2)  # shape (out_features, n_bytes, 4)

    codes = codes.reshape(out_features, n_bytes * 4)  # flatten last two dims

    # Map codes {0,1,2} → {-1,0,+1}
    unpacked = DECODE_MAP[codes]  # int8 via DECODE_MAP lookup

    # Trim padding
    return unpacked[:, :original_in_features]


# ---------------------------------------------------------------------------
# Activation quantization
# ---------------------------------------------------------------------------

def quantize_activations_int8(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-token absmax quantization to int8 range [-127, 127].

    x: float array of shape (..., in_features)
    Returns:
      x_q: int8 array, same shape
      alpha: float32 scale per token, shape (..., 1)
    """
    alpha = np.abs(x).max(axis=-1, keepdims=True)
    alpha = np.where(alpha < 1e-8, 1.0, alpha) / 127.0
    x_q = np.round(x / alpha).clip(-127, 127).astype(np.int8)
    return x_q, alpha.astype(np.float32)


# ---------------------------------------------------------------------------
# I2S GEMV kernel
# ---------------------------------------------------------------------------

def bitlinear_i2s_gemv(x_q: np.ndarray, w_int8: np.ndarray,
                       gamma: float, alpha: np.ndarray) -> np.ndarray:
    """
    I2S style GEMV: unpacked int8 weights × int8 activations → int32 → scale.

    x_q:    int8 array of shape (batch, in_features) or (in_features,)
    w_int8: int8 array of shape (out_features, in_features)  ← UNPACKED
    gamma:  float32 scale for weights
    alpha:  float32 scale for activations, shape (batch, 1) or scalar

    Returns: float32 array of shape (batch, out_features) or (out_features,)

    CONSTRAINT: w_int8 values must be in {-1, 0, +1} as int8.
                They are NEVER converted to float.
    """
    assert w_int8.dtype == np.int8, \
        f"I2S kernel requires int8 weights, got {w_int8.dtype}"
    assert x_q.dtype == np.int8, \
        f"I2S kernel requires int8 activations, got {x_q.dtype}"

    squeeze = (x_q.ndim == 1)
    if squeeze:
        x_q = x_q[np.newaxis, :]  # (1, in_features)
        if isinstance(alpha, np.ndarray) and alpha.ndim == 1:
            alpha = alpha[np.newaxis, :]

    # Core GEMV: int8 × int8 → int32
    # np.matmul promotes int8 to int32 automatically on matmul
    # but we need to be explicit: cast inputs to int16 to avoid int8 overflow
    # in intermediate products, then accumulate in int32.
    # Actually numpy matmul with int8 inputs produces int64 on some platforms.
    # We explicitly control precision:
    x_i16 = x_q.astype(np.int16)     # (batch, in_features) int16
    w_i16 = w_int8.astype(np.int16)  # (out_features, in_features) int16
    # Matrix multiply: (batch, in_features) @ (in_features, out_features) → int32
    acc = np.matmul(x_i16, w_i16.T).astype(np.int32)  # (batch, out_features)

    # Scale: float32 = int32 * gamma * alpha
    # Only the final scalar multiplication is float
    out_float = acc.astype(np.float32) * gamma * alpha  # (batch, out_features)

    if squeeze:
        out_float = out_float[0]

    return out_float


# ---------------------------------------------------------------------------
# Full I2S inference layer
# ---------------------------------------------------------------------------

def bitlinear_i2s_inference(x: np.ndarray, packed_w: np.ndarray,
                             gamma: float, in_features: int) -> np.ndarray:
    """
    Full I2S BitLinear inference:
      1. Unpack 2-bit weights to int8
      2. Quantize activations to int8
      3. GEMV
      4. Scale

    x:         float32 input, shape (batch, in_features) or (in_features,)
    packed_w:  uint8 packed weights, shape (out_features, ceil(in_features/4))
    gamma:     float32 weight scale
    in_features: original (unpadded) input dimension

    Returns: float32 output, shape (batch, out_features) or (out_features,)
    """
    assert packed_w.dtype == np.uint8, \
        f"Packed weights must be uint8, got {packed_w.dtype}"

    # Step 1: UNPACK to int8 — this is the I2S distinguishing step
    w_int8 = unpack_ternary_weights(packed_w, in_features)
    assert w_int8.dtype == np.int8

    # Step 2: Quantize activations
    x_q, alpha = quantize_activations_int8(x)

    # Step 3 + 4: GEMV + scale
    return bitlinear_i2s_gemv(x_q, w_int8, gamma, alpha)


# ---------------------------------------------------------------------------
# PyTorch wrapper for inference
# ---------------------------------------------------------------------------

class BitLinearI2SInference(torch.nn.Module):
    """
    I2S inference module holding packed uint8 weights.
    Weights are unpacked to int8 at every forward call.
    """

    def __init__(self, packed_w: np.ndarray, gamma: float, in_features: int,
                 out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.gamma        = gamma

        assert packed_w.dtype == np.uint8
        # Store as tensor (uint8 — not float, never float)
        self.register_buffer('packed_w',
                             torch.from_numpy(packed_w.copy()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: float32 tensor, shape (..., in_features)
        Returns: float32 tensor, shape (..., out_features)
        """
        shape = x.shape
        x_np = x.reshape(-1, self.in_features).cpu().numpy().astype(np.float32)

        # Run numpy I2S kernel
        out_np = bitlinear_i2s_inference(
            x_np,
            self.packed_w.cpu().numpy(),
            self.gamma,
            self.in_features
        )

        out = torch.from_numpy(out_np).to(x.device)
        return out.view(*shape[:-1], self.out_features)


# ---------------------------------------------------------------------------
# Verification / self-test
# ---------------------------------------------------------------------------

def verify_pack_unpack(n_tests: int = 100):
    """Verify packing and unpacking are exact inverses."""
    rng = np.random.default_rng(0)
    for _ in range(n_tests):
        rows = rng.integers(1, 32)
        cols = rng.integers(1, 257)
        w = rng.choice([-1, 0, 1], size=(rows, cols)).astype(np.int8)
        packed   = pack_ternary_weights(w)
        unpacked = unpack_ternary_weights(packed, cols)
        assert np.array_equal(w, unpacked), \
            f"Pack/unpack mismatch: rows={rows}, cols={cols}"
    print(f"  Pack/unpack verified for {n_tests} random shapes: PASSED")


def verify_gemv(n_tests: int = 50):
    """Verify GEMV result matches reference float computation."""
    rng = np.random.default_rng(1)
    for _ in range(n_tests):
        out_f = rng.integers(8, 64)
        in_f  = rng.integers(8, 128)
        batch = rng.integers(1, 8)

        w = rng.choice([-1, 0, 1], size=(out_f, in_f)).astype(np.int8)
        x = (rng.random((batch, in_f)) - 0.5).astype(np.float32)
        gamma = float(rng.random() * 2)

        packed = pack_ternary_weights(w)
        out_i2s = bitlinear_i2s_inference(x, packed, gamma, in_f)

        # Reference: float computation
        x_q, alpha = quantize_activations_int8(x)
        ref = (x_q.astype(np.float32) @ w.astype(np.float32).T) * gamma * alpha
        assert np.allclose(out_i2s, ref, atol=1e-3), \
            f"GEMV mismatch: max diff {np.abs(out_i2s - ref).max()}"

    print(f"  GEMV verified vs float reference for {n_tests} cases: PASSED")


if __name__ == '__main__':
    print("=" * 50)
    print("Model 1 I2S Kernel Verification")
    print("=" * 50)

    print("\n[1] Pack/unpack round-trip:")
    verify_pack_unpack()

    print("\n[2] GEMV vs float reference:")
    verify_gemv()

    print("\n[3] dtype constraint check:")
    w = np.array([[-1, 0, 1, -1]], dtype=np.int8)
    packed = pack_ternary_weights(w)
    assert packed.dtype == np.uint8, "Packed must be uint8"
    unpacked = unpack_ternary_weights(packed, 4)
    assert unpacked.dtype == np.int8, "Unpacked must be int8"
    x = np.array([[0.5, -0.3, 0.8, -0.1]], dtype=np.float32)
    x_q, alpha = quantize_activations_int8(x)
    assert x_q.dtype == np.int8, "Quantized activations must be int8"
    print("  Weight stays uint8 (packed) → int8 (unpacked): PASSED")
    print("  Activations quantized to int8: PASSED")
    print("  NO float conversion of weights at any stage: PASSED")

    print()
    print("=" * 50)
    print("ALL KERNEL TESTS PASSED")
    print("=" * 50)
