# Model 2 — True 2-bit (Experimental)

## Overview

Experimental model where packed 2-bit weights are **never** unpacked to int8
or any wider type during inference. Computation uses lookup-table GEMV operating
directly on packed uint8 bytes.

## Inference Pipeline

```
disk: 2-bit packed uint8
         ↓
  (NO UNPACK — stays as uint8)
         ↓
  Quantize activations to int8 (per-token absmax)
         ↓
  Lookup-table GEMV: WEIGHT_TABLE[packed_byte] → int8 factors
  accumulated in int32 without ever extracting individual weights
         ↓
  Scale by γ (float32) × α (float32)
         ↓
  float32 output
```

## Key Difference from Model 1

| Aspect            | Model 1 (I2S)        | Model 2 (True 2-bit)           |
|-------------------|----------------------|--------------------------------|
| Weight dtype      | uint8 → **int8**     | uint8 (unchanged)              |
| Weight access     | Individual extract   | Table index by byte value      |
| Runtime size      | 4× packed (int8)     | 1× packed (uint8)              |
| Auxiliary data    | None                 | WEIGHT_TABLE (1024 bytes, const)|

## Lookup Table

`WEIGHT_TABLE[256][4]` — precomputed at module load, maps each possible byte
value to 4 int8 ternary values. Used as a constant; not part of model weights.

## Assertion

`assert packed_w.dtype == np.uint8` is enforced at every inference call.
Any violation causes an immediate AssertionError, not a silent data type cast.

## Files

- `model.py` — Identical transformer architecture to Model 1
- `kernel.py` — Lookup-table GEMV kernel, WEIGHT_TABLE constant
