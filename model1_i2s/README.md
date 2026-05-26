# Model 1 — I2S Style (Baseline)

## Overview

This model mirrors Microsoft's I2_S (Integer 2-bit Sparse) kernel architecture for
BitNet b1.58 ternary weight inference.

## Inference Pipeline

```
disk: 2-bit packed uint8
         ↓
  UNPACK to int8 {-1, 0, +1}
         ↓
  Quantize activations to int8 (per-token absmax)
         ↓
  GEMV: int8 × int8 → int32 accumulation
         ↓
  Scale by γ (float32) × α (float32)
         ↓
  float32 output
```

## Weight Encoding

| Value | 2-bit Code |
|-------|-----------|
| -1    | 00        |
|  0    | 01        |
| +1    | 10        |

Four weights packed per byte: bits [7:6], [5:4], [3:2], [1:0].

## Parameter Count

~12,000,000 parameters (exact count printed at model init).

Architecture: d_model=320, d_ff=1280, n_heads=8, n_layers=10, context=128.

## Files

- `model.py` — BitNet b1.58 transformer with training-time quantization
- `kernel.py` — Pack/unpack functions and I2S GEMV kernel
