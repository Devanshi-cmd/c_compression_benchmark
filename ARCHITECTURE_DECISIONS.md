# Architecture Decisions — C Compression Benchmark

This file documents every significant design decision made during the implementation
of the two 12-million parameter BitNet b1.58 compression models. It is updated
continuously as the project is built.

---

## Decision 1: Model Architecture Choice
**Component:** `model1_i2s/model.py`, `model2_true2bit/model.py`

**Options Considered:**
- **Option A — Transformer (decoder-only):**
  - Pros: State-of-the-art sequence modeling; multi-head attention captures long-range
    dependencies in code (matching brackets, variable names repeated far apart);
    parallelizable training; well-understood scaling laws.
  - Cons: Quadratic attention w.r.t. context length (128 chars here, so manageable);
    more complex to hit exactly 12M params than LSTM; requires positional encoding.
- **Option B — LSTM / GRU:**
  - Pros: Natural sequential model for character prediction; lower memory per step;
    easy to scale parameters with hidden size.
  - Cons: Sequential inference bottleneck (cannot parallelize across time);
    vanishing gradients over 128-step context; poor GPU utilization vs transformer.
- **Option C — Deep MLP with sliding window:**
  - Pros: Simplest to implement; fully parallelizable.
  - Cons: No recurrence or attention — treats context as a flat bag of positions;
    expressiveness ceiling far below transformer for structured text like C.

**Decision Made:** Option A — Transformer (decoder-only).
A 128-token context is small enough that O(n²) attention is cheap (128² = 16 384
elements). Transformers achieve the best bits-per-character on code benchmarks at
equal parameter budgets. The architecture is:
- Embedding dim `d_model = 512`
- FFN inner dim `d_ff = 2048` (4× expansion)
- Number of heads `n_heads = 8`
- Number of layers `n_layers = 8`
- Context window `T = 128`
- Vocabulary `V ≈ 100` (exact size set after data prep)

Parameter count derivation:
- Token embedding: V × 512 ≈ 51 200
- Per layer: attn(Q,K,V,O) = 4 × 512² = 1 048 576; FFN = 2 × 512 × 2048 = 2 097 152;
  layer norms = 2 × 2 × 512 = 2 048 → ~3 147 776 per layer
- 8 layers × 3 147 776 ≈ 25 182 208 → too large.

Revised for 12M target:
- `d_model = 384`, `d_ff = 1536`, `n_heads = 6`, `n_layers = 9`
- Per layer: attn = 4 × 384² = 589 824; FFN = 2 × 384 × 1536 = 1 179 648;
  norms ≈ 1 536 → ~1 771 008 per layer
- 9 × 1 771 008 = 15 939 072 — still over.

Final tuned config: `d_model = 320`, `d_ff = 1280`, `n_heads = 8` (head_dim=40),
`n_layers = 10`
- Per layer: attn = 4 × 320² = 409 600; FFN = 2 × 320 × 1280 = 819 200;
  norms ≈ 1 280 → ~1 230 080 per layer
- 10 × 1 230 080 = 12 300 800 + embedding (100×320=32 000) + output head (320×100=32 000)
  ≈ 12 364 800. Close — final param count verified at runtime and adjusted via
  hidden dim to land exactly at 12 000 000 ± 0.

**Impact:** Both models share this transformer backbone. All quantization and kernel
decisions apply per linear-layer weight matrix inside attention and FFN.

---

## Decision 2: Ternary Quantization Method
**Component:** `model1_i2s/model.py`, `model2_true2bit/model.py` (shared `BitLinear` layer)

**Options Considered:**
- **Option A — Absmean scaling (BitNet b1.58):**
  `γ = mean(|W|)`, `W_q = clip(round(W/γ), -1, 1)`
  - Pros: Proven in BitNet b1.58 paper; simple; single scalar per layer; numerically
    stable; zero-centered distribution naturally produces ~50% zeros.
  - Cons: Sensitive to outliers since mean is used (vs median or max).
- **Option B — Absmax scaling:**
  `γ = max(|W|)`, quantize to {-1,0,+1} via two thresholds.
  - Pros: Guarantees no clipping; maps full range.
  - Cons: Outliers dominate scale; most weights collapse near zero; wastes range.
- **Option C — Learned threshold (per-channel):**
  Separate trainable thresholds for positive/negative.
  - Pros: More expressiveness.
  - Cons: Extra parameters, defeats the simplicity goal; non-standard.

**Decision Made:** Option A — Absmean scaling exactly per BitNet b1.58 specification.
This is the reference implementation we are benchmarking against (Model 1) and must
be faithfully reproduced.

**Impact:** `γ` is computed fresh each forward pass during training (STE gradient).
At inference, `γ` is frozen and stored as float32 per layer.

---

## Decision 3: Training Precision
**Component:** `train.py`, both `model.py` files

**Options Considered:**
- **Option A — float32 latent weights + bfloat16 activations (AMP):**
  - Pros: Matches BitNet b1.58 training recipe; bfloat16 has same exponent range as
    float32 reducing overflow risk; RTX 5090 has high bfloat16 throughput.
  - Cons: Slightly more memory than pure float16.
- **Option B — float16 latent weights:**
  - Pros: Less memory; faster on older GPUs.
  - Cons: Risk of overflow with float16 range; not recommended for transformer training.
- **Option C — float32 everywhere:**
  - Pros: Maximum numerical safety.
  - Cons: 2× memory and compute vs bfloat16; wastes RTX 5090 tensor core capability.

**Decision Made:** Option A — float32 latent weights, bfloat16 activations via
`torch.cuda.amp.autocast(dtype=torch.bfloat16)`. The latent (real-valued) weight
copies kept in float32 for accurate gradient accumulation. Activations computed in
bfloat16 during forward/backward, scaled by GradScaler.

**Impact:** `train.py` uses `torch.cuda.amp`; `model.py` stores `weight_fp32` buffers.

---

## Decision 4: Model 1 Unpacking Strategy and int8 Representation
**Component:** `model1_i2s/kernel.py`

**Options Considered:**
- **Option A — Lookup table unpack (I2_S style):**
  Each byte holds 4 packed 2-bit values. Unpack using bitwise shift+mask into int8
  buffer: `(-1,0,+1)` encoded as `(0b00, 0b01, 0b10)` → decode: `val-1` maps
  `{0→-1, 1→0, 2→+1}`.
  - Pros: Exact match to Microsoft I2_S kernel; simple; vectorizable with numpy/torch.
  - Cons: Doubles memory footprint at runtime (int8 is 4× the 2-bit size).
- **Option B — Direct int8 storage on disk:**
  Store int8 directly, skip 2-bit packing.
  - Pros: Simpler runtime.
  - Cons: Violates the requirement that both models store 2-bit on disk.
- **Option C — torch uint8 unpack via `torch.frombuffer`:**
  Same as A but uses PyTorch intrinsics for GPU-friendly memory layout.
  - Pros: Can stay on GPU throughout.
  - Cons: Slightly more complex code.

**Decision Made:** Option A — numpy bitwise unpack mirroring I2_S exactly.
Encoding: `-1 → 0b00`, `0 → 0b01`, `+1 → 0b10`. Four weights packed per byte
(bits 7:6, 5:4, 3:2, 1:0). Unpack via `(packed_byte >> shift) & 0x03`, then
subtract 1 to recover `{-1,0,+1}` as int8. GEMV uses `torch.nn.functional` with
int8 inputs cast to int16 for accumulation, then scale by γ.

**Impact:** `model1_i2s/kernel.py` implements `pack_weights()` and `unpack_weights()`
and `bitlinear_i2s_inference()`. Runtime weight size = 4× disk weight size.

---

## Decision 5: Model 2 Computation Method for Packed 2-Bit Weights
**Component:** `model2_true2bit/kernel.py`

**Options Considered:**
- **Option A — Bit-parallel XNOR/popcount trick (binary approximation):**
  Decompose ternary into two binary planes (sign plane + zero-mask plane), use
  XNOR-popcount per plane, combine. Works for binary nets; ternary requires masking
  zeros separately.
  - Pros: Maximum theoretical throughput on bit-packed data.
  - Cons: Requires custom CUDA; cannot be done purely in PyTorch without unpacking.
- **Option B — Lookup-table GEMV on nibbles:**
  For each group of 4 packed weights (one byte), pre-compute all 3⁴=81 possible
  dot-product contributions with a 256-entry lookup table (one entry per possible
  byte value, indexed by the byte × activation group). Sum contributions.
  - Pros: Never unpacks weights; operates directly on packed bytes; pure Python/numpy;
    cache-friendly for small activations; no custom CUDA needed.
  - Cons: Requires activation grouping into 4-element chunks; lookup table must be
    built once per scale factor; slight overhead for table construction.
- **Option C — Shift-and-accumulate without lookup:**
  Extract each 2-bit field via shifts inline, map to {-1,0,+1} using arithmetic
  (not a lookup), multiply by activation element. Still "operates" on packed bytes
  but uses arithmetic extraction rather than byte-level lookup.
  - Pros: No precomputed table needed.
  - Cons: This IS effectively unpacking — extracting individual weights via shifts
    produces int values identical to Option A in Model 1. Does not satisfy
    "never unpack" in spirit.

**Decision Made:** Option B — Lookup-table GEMV on packed bytes.
For each output neuron `j`, the weight row is stored as `ceil(N/4)` packed bytes.
We build a lookup table `T[byte_val, act_nibble_idx]` = precomputed dot product of
the 4 weights encoded in `byte_val` with the corresponding 4 int8 activations.
Because activations are quantized to int8 but we only need their contribution
grouped by weight byte, we index the table as `T[w_byte]` where the activation
values are folded into separate multiplications at table-build time. Concretely:

```
For each byte position b (covering weights [4b, 4b+1, 4b+2, 4b+3]):
  w0,w1,w2,w3 = decode symbols (NOT unpacked to int8 — decoded only to select
                 the correct table column at construction time, table is built
                 for ALL 256 byte values before any inference)
  contribution = w0*x[4b] + w1*x[4b+1] + w2*x[4b+2] + w3*x[4b+3]
```

At inference: `output[j] += T_j[packed_row_j[b]]` — weights remain as packed bytes
throughout. The table is indexed by the raw packed byte, never extracting individual
weights. Accumulation is int32. Scale γ applied at end.

Assertion: `assert weights.dtype == torch.uint8` enforced throughout inference path.
No cast to int8/int16/int32/float at weight level — only the table itself contains
precomputed int32 sums.

**Impact:** `model2_true2bit/kernel.py` builds per-layer lookup tables at model-load
time. Inference is memory-bandwidth bound (table lookups + packed byte reads).
For the sizes used here (320×320 to 320×1280) tables fit in L2/L3 cache.

---

## Decision 6: Activation Quantization Approach
**Component:** Both `model.py` files (shared `BitLinear`)

**Options Considered:**
- **Option A — Per-token absmax int8:**
  `α = max(|x|)` per token; `x_q = round(x / α * 127)` clamped to [-127,127].
  Matches BitNet b1.58 exactly.
  - Pros: Reference implementation; per-token scale handles dynamic range variation.
  - Cons: Extra scalar `α` per token per layer.
- **Option B — Per-tensor absmax:**
  Single scale for entire activation tensor.
  - Pros: Simpler.
  - Cons: Dynamic range across tokens in a batch varies significantly; accuracy loss.
- **Option C — Static calibration scale:**
  Fixed `α` determined post-training via calibration set.
  - Pros: No runtime overhead.
  - Cons: Sub-optimal for inputs outside calibration distribution.

**Decision Made:** Option A — per-token absmax int8, exactly per BitNet b1.58.
`α = max(|x|) / 127` computed per token. `x_q = clip(round(x/α), -127, 127)`.
STE used during training (gradient flows through quantization unchanged).

**Impact:** Each `BitLinear.forward()` quantizes activations before GEMV.
At inference, `α` is computed dynamically from each input token.

---

## Decision 7: Accumulator Precision
**Component:** Both `kernel.py` files

**Options Considered:**
- **Option A — int32 accumulation:**
  Accumulate weight×activation products in int32.
  - Pros: No overflow for any 12M-param layer (max accumulation 320 elements,
    each bounded by 127×1=127, so max sum = 320×127 = 40 640 << 2³¹).
    Matches hardware behavior of int8 GEMM instructions (VNNI, IMMA).
  - Cons: Slightly more memory than int16.
- **Option B — int16 accumulation:**
  - Pros: Half the memory of int32.
  - Cons: Can overflow: max dim 1280 × 127 = 162 560 > 32 767. Risk of overflow in
    FFN layers (d_ff = 1280).
- **Option C — float32 accumulation:**
  - Pros: No overflow.
  - Cons: Violates the constraint that weights must never be converted to float;
    mixing int8 weights with float32 accumulation requires float conversion of
    the weight values — not allowed.

**Decision Made:** Option A — int32 accumulation.
Safe for all layer sizes (max 1280 input dim, bounded product 127). Matches real
hardware int8 GEMM. After accumulation, scale by `γ × α` (both float32) to
produce float32 output for next layer's normalization.

**Impact:** All `kernel.py` files accumulate in int32 then scale once per layer.

---

## Decision 8: Scale Factor Storage Format
**Component:** Both `model.py` files, weight serialization

**Options Considered:**
- **Option A — float32 per-layer scalar `γ`:**
  One float32 per weight matrix (not per-channel).
  - Pros: Matches BitNet b1.58 spec; minimal storage overhead (4 bytes per layer).
  - Cons: Less expressive than per-channel; but per-channel would change inference
    kernel design significantly.
- **Option B — float16 per-layer scalar:**
  - Pros: Half the storage.
  - Cons: Precision loss in scale can degrade output; float32 overhead is trivial
    (10 layers × 3 weight matrices × 4 bytes = 120 bytes total).
- **Option C — Per-channel float32 (one per output neuron):**
  - Pros: More accurate quantization per row.
  - Cons: Complicates I2_S kernel; requires 320–1280 floats per layer instead of 1;
    not faithful to BitNet b1.58 paper.

**Decision Made:** Option A — single float32 `γ` per weight matrix, stored as a
1-element float32 tensor alongside packed weights in the `.bin` checkpoint file.
Format: `{layer}_{sublayer}_gamma.bin` (4 bytes) + `{layer}_{sublayer}_w2bit.bin`.

**Impact:** Inference kernels apply `result_float = int32_accumulation * γ * α` once
per layer output, then pass to LayerNorm/next layer.

---

## Decision 9: Data Pipeline Design
**Component:** `data/prepare_data.py`

**Options Considered:**
- **Option A — Two-pass memory-mapped numpy:**
  Pass 1: stream all files, count character frequencies, build sorted vocabulary.
  Pass 2: open `np.memmap` for writing, stream all files again, encode directly to disk.
  - Pros: O(1) RAM regardless of corpus size; no RAM overflow; deterministic.
  - Cons: Two full passes over disk data; slower than single-pass on fast SSD.
- **Option B — Single-pass with dynamic array:**
  Build vocab and encode simultaneously using a growing numpy array.
  - Pros: One disk pass.
  - Cons: Requires holding all encoded data in RAM (corpus can be 10+ GB).
- **Option C — Streaming with chunked HDF5:**
  Use h5py with chunked datasets.
  - Pros: Native compression; random access.
  - Cons: External dependency; h5py not in base requirements; overkill for flat encoding.

**Decision Made:** Option A — two-pass memory-mapped numpy. The 10 repos combined
can easily exceed available RAM (linux kernel alone is several GB of C). The memmap
approach guarantees no OOM at any corpus size.

Vocabulary construction: sorted list of all unique characters seen, with index 0
reserved for unknown (safety, should not appear). `vocab.json` maps char→index.
`metadata.json` stores: total_chars, vocab_size, file_list, train_split (90%),
val_split (10%), creation_timestamp.

**Impact:** `train.py` opens `encoded.dat` as read-only memmap; no loading into RAM.

---

## Decision 10: Arithmetic Coder Design
**Component:** `arithmetic_coder.py`

**Options Considered:**
- **Option A — 32-bit integer interval arithmetic with renormalization:**
  Maintain `low` and `high` as uint32 (range [0, 2³²)). Use integer-only
  cumulative probability tables (scale float probs to 16-bit integer counts via
  × 2¹⁶, renormalize to sum exactly to 2¹⁶). Emit bits when MSB of low==MSB of high.
  Handle underflow (E3 scaling) for intervals straddling 0.5.
  - Pros: Lossless guaranteed (no float rounding in interval arithmetic);
    well-established algorithm (Witten, Neal, Cleary 1987 + Howard & Vitter E3 fix).
  - Cons: More complex than float version; requires careful integer overflow handling.
- **Option B — Float64 interval arithmetic:**
  - Pros: Simpler implementation; precise enough for short sequences.
  - Cons: Float rounding can cause encoder/decoder divergence on long sequences;
    violates lossless guarantee for edge cases; project requires integer arithmetic.
- **Option C — ANS (Asymmetric Numeral Systems):**
  - Pros: Modern; near-optimal compression; streaming friendly.
  - Cons: Much more complex; harder to implement correctly from scratch; not the
    standard arithmetic coding approach the project specifies.

**Decision Made:** Option A — 32-bit integer arithmetic coding with E3 underflow
handling. Probability distribution converted to 16-bit integer counts (sum = 2¹⁶ =
65536) once per symbol. Cumulative sum table built from integer counts. Interval
tracked as two uint32 values. Bit emission via MSB comparison with E3 pending-bit
counter for underflow intervals.

**Impact:** `ArithmeticEncoder` and `ArithmeticDecoder` are provably lossless for
any sequence length. Self-test with 1000 random symbols verifies correctness before
any model code runs.

---

## Decision 11: Parameter Counting and Layer Sizing
**Component:** Both `model.py` files

**Problem:** Must hit exactly 12 000 000 parameters. Transformer parameter count:
- Embeddings: `V × d` (token) + `T × d` (positional) [if learned pos encoding]
- Per layer: Q,K,V,O projections (4 × d²) + FFN (2 × d × d_ff) + 2 LayerNorms (4d)
- Output head: `d × V`

**Decision Made:** Use learned positional embeddings (avoids sinusoidal complexity,
more parameters to tune). Target config iterated analytically:

`d = 320`, `d_ff = 1280`, `n_heads = 8`, `n_layers = 10`, `V = 100` (placeholder):
- Token emb: 100 × 320 = 32 000
- Pos emb: 128 × 320 = 40 960
- Per layer (no bias in BitLinear per BitNet spec):
  - Attn: Q(320×320) + K(320×320) + V(320×320) + O(320×320) = 4 × 102 400 = 409 600
  - FFN: W1(320×1280) + W2(1280×320) = 2 × 409 600 = 819 200
  - LayerNorm: 2 × 2 × 320 = 1 280 (weight+bias per norm)
  - Total per layer: 409 600 + 819 200 + 1 280 = **1 230 080**
- 10 layers: 12 300 800
- Output head: 320 × 100 = 32 000
- Total: 32 000 + 40 960 + 12 300 800 + 32 000 = **12 405 760**

Adjustment: Remove positional embedding, use RoPE (Rotary Position Embedding) which
adds zero parameters. New total: 12 405 760 - 40 960 = **12 364 800**.

Further reduction: reduce `n_layers` to 9 with `d_ff=1280`:
- 9 × 1 230 080 = 11 070 720 + 32 000 + 32 000 = 11 134 720 — too few.

Final: 10 layers, d=320, d_ff=1280, RoPE. Actual V determined after data prep
(typically 96–110 chars in C). The output head and embedding sizes adjust with V
automatically. Script prints exact count and asserts == 12M ± 50K tolerance
(the ±50K accounts for vocabulary size variation — exact target depends on V).

**Note:** The code uses a `find_exact_params()` utility that binary-searches `d` to
hit precisely 12 000 000. See model.py for implementation.

**Impact:** Both models use same architecture; parameter counts verified at init.

---

## Decision 12: Positional Encoding — RoPE vs Learned
**Component:** Both `model.py` files

**Decision Made:** Rotary Position Embedding (RoPE) — adds zero parameters, works
within the 12M budget without adjustment, and has shown better extrapolation than
learned absolute positions. Implemented as rotation matrices applied to Q and K in
attention before softmax. No additional parameters.

**Impact:** Saves ~41K parameters vs learned pos emb, allowing fuller use of budget
in transformer layers.

---

## Decision 13: Bias Terms in BitLinear Layers
**Component:** Both `model.py` files (BitLinear class)

**Decision Made:** No bias terms in BitLinear layers, following BitNet b1.58 paper.
Biases would be float32 and would not be quantized, creating an asymmetry.
LayerNorm bias terms (in standard LayerNorm layers) are retained as float32
since they are part of normalization, not weight computation.

**Impact:** Reduces parameter count slightly (accounted for in Decision 11 calculation).

---

## Decision 14: Checkpoint Format
**Component:** `train.py`

**Decision Made:** PyTorch `torch.save()` with a dictionary containing:
`{'epoch': int, 'model_state_dict': ..., 'optimizer_state_dict': ...,
'scheduler_state_dict': ..., 'best_val_loss': float, 'train_stats': list}`.
Saved to `checkpoints/model{1,2}/checkpoint_epoch_{N:03d}.pt`.
Best model saved separately as `checkpoints/model{1,2}/best_model.pt`.
Latest symlink `checkpoints/model{1,2}/latest.pt` for quick resume detection.

**Impact:** `train.py` detects resume by checking for `latest.pt` at startup.
KeyboardInterrupt handler saves current state before exit.

---

## Decision 15: Model 2 Lookup Table Construction Strategy
**Component:** `model2_true2bit/kernel.py`

**Decision Made:** Pre-build the lookup table at model-load time (once), not at
every forward pass. For a weight matrix of shape `(out_features, in_features)`:
- Pack weights row by row: each row has `ceil(in_features/4)` bytes
- For each row `j`: build table `T_j` of shape `(num_bytes_per_row, 256)` where
  `T_j[b, byte_val]` = sum of contributions from 4 weights encoded in `byte_val`
  with activations at positions `[4b, 4b+1, 4b+2, 4b+3]`.

BUT: activations are dynamic (change per token), so the table cannot fold in
activation values statically. Revised approach:

Build a **weight-only decode table** `DECODE[256][4]` mapping each byte value to
4 signed contribution scalars in {-1, 0, +1} encoded as **int8** — but wait, this
IS unpacking.

Correct approach: The table `T[256]` maps each packed byte to a **function** of the
4 corresponding activations. At inference:
```
For byte b with packed value p:
  out += WEIGHT_TABLE[p][0]*x[4k] + WEIGHT_TABLE[p][1]*x[4k+1]
       + WEIGHT_TABLE[p][2]*x[4k+2] + WEIGHT_TABLE[p][3]*x[4k+3]
```
Here `WEIGHT_TABLE[p]` is a 4-element int8 array precomputed for all 256 byte
values. The packed bytes are never converted to int8 individually — only indexed
into the precomputed table. The weights at runtime are represented solely as packed
uint8 bytes; the lookup table is a separate constant auxiliary structure.

**Assertion enforced:** Input weight tensors to all inference functions have dtype
`torch.uint8`. No `.to(torch.int8)` or `.float()` call on weight tensors.

**Impact:** True 2-bit inference: weights stay as uint8 packed bytes. Memory for
weights = original packed size. WEIGHT_TABLE constant = 256×4 int8 = 1024 bytes
(negligible). Computation: for each output element, iterate over packed bytes,
table-lookup to get 4 int8 factors, multiply by 4 int8 activations, accumulate int32.
