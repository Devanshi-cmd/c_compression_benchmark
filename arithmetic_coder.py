#!/usr/bin/env python3
"""
arithmetic_coder.py — Self-contained arithmetic coder using 32-bit integer arithmetic.

No floating point is used internally for interval arithmetic.
Probability distributions (float arrays) are converted to integer counts (uint32)
scaled to 2^16 = 65536 before any interval operations begin.

Algorithm: Range coding with E3 underflow handling
  (Witten, Neal, Cleary 1987 + Howard & Vitter pending-bit technique)

Encoding:
  - low, high: uint32 tracking the current interval [low, high)
  - Bit emission when MSB of low == MSB of high (interval fully in [0,0.5) or [0.5,1))
  - E3 scaling when interval straddles [0.25, 0.75) — count pending bits

Decoding:
  - Mirror of encoding: maintain value (32-bit int read from bitstream)
  - Determine which sub-interval value falls in → decoded symbol
  - Perform same rescaling steps as encoder

Fixed precision:
  - TOP = 2^32 = 4294967296  (full range)
  - HALF = 2^31              (midpoint)
  - QUARTER = 2^30           (quarter point)
  - PROB_SCALE = 2^16 = 65536 (probability integer scale)

Usage:
  enc = ArithmeticEncoder()
  for symbol, probs in zip(symbols, prob_distributions):
      enc.encode(symbol, probs)
  enc.flush()
  compressed = enc.get_compressed_bytes()

  dec = ArithmeticDecoder(compressed)
  for probs in prob_distributions:
      symbol = dec.decode(probs)
"""

import numpy as np
from typing import List


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOP       = (1 << 32)           # 2^32, exclusive upper bound of range
HALF      = (1 << 31)           # 2^31
QUARTER   = (1 << 30)           # 2^30
THREE_QUARTER = 3 * (1 << 30)  # 3 * 2^30
PROB_SCALE = (1 << 16)          # scale for integer probability counts: 65536
UINT32_MASK = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Probability conversion
# ---------------------------------------------------------------------------

def probs_to_counts(probs: np.ndarray) -> np.ndarray:
    """
    Convert float probability array to integer counts summing to PROB_SCALE.
    Uses largest-remainder method to ensure exact sum.

    Args:
        probs: float array, values >= 0, should sum to 1.0

    Returns:
        uint32 array of counts summing exactly to PROB_SCALE (65536)
    """
    probs = np.asarray(probs, dtype=np.float64)
    # Normalize
    total = probs.sum()
    if total <= 0:
        raise ValueError("Probability distribution must have positive sum")
    probs = probs / total

    # Ensure no probability is zero (floor of 1 count for each symbol)
    probs = np.maximum(probs, 1e-10)
    probs = probs / probs.sum()

    # Scale to PROB_SCALE
    raw = probs * PROB_SCALE
    floored = np.floor(raw).astype(np.int64)
    remainder = raw - floored

    # Distribute remaining counts to symbols with largest remainders
    deficit = PROB_SCALE - int(floored.sum())
    if deficit > 0:
        indices = np.argsort(-remainder)[:deficit]
        floored[indices] += 1
    elif deficit < 0:
        # Over-allocated; remove from smallest remainders
        indices = np.argsort(remainder)[:-deficit]
        floored[indices] -= 1
        floored = np.maximum(floored, 1)
        # Re-adjust
        deficit2 = PROB_SCALE - int(floored.sum())
        if deficit2 != 0:
            floored[0] += deficit2

    counts = floored.astype(np.int64)
    # Final guarantee: all counts >= 1
    counts = np.maximum(counts, 1)
    # Re-normalize to exact sum
    overage = int(counts.sum()) - PROB_SCALE
    if overage > 0:
        # Remove from largest counts
        idx = np.argsort(-counts)[:overage]
        counts[idx] -= 1
    elif overage < 0:
        idx = np.argsort(-counts)[:-overage]
        counts[idx] += 1

    assert int(counts.sum()) == PROB_SCALE, \
        f"Count sum {counts.sum()} != {PROB_SCALE}"

    return counts.astype(np.int64)


def counts_to_cumulative(counts: np.ndarray):
    """
    Build cumulative count array.
    Returns cum_low[i], cum_high[i] for symbol i:
      cum_low[i]  = sum(counts[0..i-1])
      cum_high[i] = sum(counts[0..i])
    """
    cum = np.zeros(len(counts) + 1, dtype=np.int64)
    cum[1:] = np.cumsum(counts)
    return cum  # cum[i] = lower bound for symbol i, cum[i+1] = upper bound


# ---------------------------------------------------------------------------
# Bit I/O helpers
# ---------------------------------------------------------------------------

class BitWriter:
    """Write bits to a bytearray, MSB first."""

    def __init__(self):
        self._buf = bytearray()
        self._byte = 0
        self._bit_pos = 7  # next bit position to write (7=MSB)

    def write_bit(self, bit: int):
        if bit:
            self._byte |= (1 << self._bit_pos)
        self._bit_pos -= 1
        if self._bit_pos < 0:
            self._buf.append(self._byte)
            self._byte = 0
            self._bit_pos = 7

    def flush(self):
        """Flush any partial byte (pad with zeros)."""
        if self._bit_pos < 7:
            self._buf.append(self._byte)
            self._byte = 0
            self._bit_pos = 7

    def get_bytes(self) -> bytes:
        return bytes(self._buf)

    @property
    def bit_count(self) -> int:
        return len(self._buf) * 8 + (7 - self._bit_pos)


class BitReader:
    """Read bits from bytes, MSB first."""

    def __init__(self, data: bytes):
        self._data = data
        self._byte_pos = 0
        self._bit_pos = 7  # next bit to read from current byte
        self._current_byte = data[0] if data else 0

    def read_bit(self) -> int:
        if self._byte_pos >= len(self._data):
            return 0  # pad with zeros past end of stream
        bit = (self._current_byte >> self._bit_pos) & 1
        self._bit_pos -= 1
        if self._bit_pos < 0:
            self._byte_pos += 1
            self._bit_pos = 7
            if self._byte_pos < len(self._data):
                self._current_byte = self._data[self._byte_pos]
            else:
                self._current_byte = 0
        return bit


# ---------------------------------------------------------------------------
# Arithmetic Encoder
# ---------------------------------------------------------------------------

class ArithmeticEncoder:
    """
    Arithmetic encoder using 32-bit integer interval arithmetic.
    Never uses floating point for interval operations.
    """

    def __init__(self):
        self._low = 0
        self._high = UINT32_MASK  # [0, 2^32 - 1] inclusive
        self._pending_bits = 0    # E3 underflow counter
        self._writer = BitWriter()
        self._n_symbols = 0

    def encode(self, symbol: int, prob_distribution: np.ndarray):
        """
        Encode one symbol given a probability distribution.

        Args:
            symbol: integer index into prob_distribution
            prob_distribution: float array summing to 1.0
        """
        counts = probs_to_counts(prob_distribution)
        cum = counts_to_cumulative(counts)

        total = PROB_SCALE  # == cum[-1]
        assert 0 <= symbol < len(counts), \
            f"Symbol {symbol} out of range [0, {len(counts)})"

        # Narrow interval
        rng = self._high - self._low + 1

        # Use integer arithmetic: no floating point
        # new_high = low + floor(rng * cum[symbol+1] / total) - 1
        # new_low  = low + floor(rng * cum[symbol]   / total)
        # Use Python big integers for the intermediate products
        new_high = self._low + (rng * int(cum[symbol + 1])) // total - 1
        new_low  = self._low + (rng * int(cum[symbol]))     // total

        self._high = new_high & UINT32_MASK
        self._low  = new_low  & UINT32_MASK

        self._normalize()
        self._n_symbols += 1

    def _normalize(self):
        """Emit bits and rescale interval until high and low differ in MSB."""
        while True:
            if self._high < HALF:
                # Both in [0, HALF): emit 0, then pending 1s
                self._emit_bit(0)
            elif self._low >= HALF:
                # Both in [HALF, TOP): emit 1, then pending 0s
                self._emit_bit(1)
                self._low  = (self._low  - HALF) & UINT32_MASK
                self._high = (self._high - HALF) & UINT32_MASK
            elif self._low >= QUARTER and self._high < THREE_QUARTER:
                # E3: interval straddles midpoint [QUARTER, THREE_QUARTER)
                self._pending_bits += 1
                self._low  = (self._low  - QUARTER) & UINT32_MASK
                self._high = (self._high - QUARTER) & UINT32_MASK
            else:
                break
            # Scale up by 2
            self._low  = (self._low  << 1) & UINT32_MASK
            self._high = ((self._high << 1) | 1) & UINT32_MASK

    def _emit_bit(self, bit: int):
        self._writer.write_bit(bit)
        # Emit pending bits (opposite of emitted bit for E3)
        opposite = 1 - bit
        for _ in range(self._pending_bits):
            self._writer.write_bit(opposite)
        self._pending_bits = 0

    def flush(self):
        """Emit enough bits to unambiguously identify the final interval."""
        self._pending_bits += 1
        if self._low < QUARTER:
            self._emit_bit(0)
        else:
            self._emit_bit(1)
        self._writer.flush()

    def get_compressed_bytes(self) -> bytes:
        return self._writer.get_bytes()

    @property
    def compressed_bits(self) -> int:
        return self._writer.bit_count


# ---------------------------------------------------------------------------
# Arithmetic Decoder
# ---------------------------------------------------------------------------

class ArithmeticDecoder:
    """
    Arithmetic decoder. Must receive probability distributions in the exact
    same order as the encoder.
    """

    def __init__(self, compressed_data: bytes):
        self._reader = BitReader(compressed_data)
        self._low  = 0
        self._high = UINT32_MASK
        # Fill value register with first 32 bits from stream
        self._value = 0
        for _ in range(32):
            self._value = ((self._value << 1) | self._reader.read_bit()) & UINT32_MASK
        self._n_symbols = 0

    def decode(self, prob_distribution: np.ndarray) -> int:
        """
        Decode one symbol given probability distribution.

        Args:
            prob_distribution: float array summing to 1.0 (same as encoder used)

        Returns:
            Decoded symbol index (int)
        """
        counts = probs_to_counts(prob_distribution)
        cum = counts_to_cumulative(counts)
        total = PROB_SCALE

        # Determine which sub-interval value falls in
        rng = self._high - self._low + 1

        # scaled_value: where value sits within [0, total)
        # scaled = floor((value - low + 1) * total / rng) - 1 ... but be careful
        # We compute: offset = value - low
        offset = (self._value - self._low) & UINT32_MASK
        # Find symbol such that cum[s] <= scaled_value < cum[s+1]
        # scaled_value = floor(offset * total / rng)  — integer arithmetic
        scaled = (int(offset) * total) // int(rng)
        scaled = min(int(scaled), total - 1)  # clamp to valid range

        # Binary search for symbol
        symbol = int(np.searchsorted(cum[1:], scaled, side='right'))
        symbol = min(symbol, len(counts) - 1)

        # Verify (should always hold)
        assert cum[symbol] <= scaled < cum[symbol + 1], \
            f"Decode error: scaled={scaled}, cum[{symbol}]={cum[symbol]}, " \
            f"cum[{symbol+1}]={cum[symbol+1]}"

        # Update interval (mirror encoder)
        new_high = self._low + (rng * int(cum[symbol + 1])) // total - 1
        new_low  = self._low + (rng * int(cum[symbol]))     // total

        self._high = new_high & UINT32_MASK
        self._low  = new_low  & UINT32_MASK

        self._normalize()
        self._n_symbols += 1
        return symbol

    def _normalize(self):
        """Mirror of encoder's normalize — rescale and read bits from stream."""
        while True:
            if self._high < HALF:
                pass  # emit 0 side
            elif self._low >= HALF:
                self._value = (self._value - HALF) & UINT32_MASK
                self._low   = (self._low   - HALF) & UINT32_MASK
                self._high  = (self._high  - HALF) & UINT32_MASK
            elif self._low >= QUARTER and self._high < THREE_QUARTER:
                self._value = (self._value - QUARTER) & UINT32_MASK
                self._low   = (self._low   - QUARTER) & UINT32_MASK
                self._high  = (self._high  - QUARTER) & UINT32_MASK
            else:
                break
            self._low   = (self._low  << 1) & UINT32_MASK
            self._high  = ((self._high << 1) | 1) & UINT32_MASK
            self._value = ((self._value << 1) | self._reader.read_bit()) & UINT32_MASK


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import random
    import sys

    print("=" * 50)
    print("Arithmetic Coder Self-Test")
    print("=" * 50)

    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    # Test 1: fixed 5-symbol distribution
    print("\nTest 1: Fixed distribution, 1000 symbols")
    vocab_size = 5
    base_probs = np.array([0.5, 0.2, 0.15, 0.1, 0.05])

    symbols = [rng.choices(range(vocab_size), weights=base_probs)[0]
               for _ in range(1000)]

    enc = ArithmeticEncoder()
    for s in symbols:
        enc.encode(s, base_probs)
    enc.flush()
    compressed = enc.get_compressed_bytes()

    dec = ArithmeticDecoder(compressed)
    decoded = []
    for _ in range(1000):
        decoded.append(dec.decode(base_probs))

    assert symbols == decoded, f"MISMATCH at positions: {[i for i,(a,b) in enumerate(zip(symbols,decoded)) if a!=b]}"
    entropy_bits = -sum(base_probs[s] * np.log2(base_probs[s]) for s in symbols
                        if base_probs[s] > 0)
    compressed_bits = len(compressed) * 8
    print(f"  Symbols    : 1000")
    print(f"  Original   : {1000 * 8} bits ({1000} bytes)")
    print(f"  Compressed : {compressed_bits} bits ({len(compressed)} bytes)")
    print(f"  Entropy    : {entropy_bits:.1f} bits")
    print(f"  Overhead   : {compressed_bits - entropy_bits:.1f} bits")
    print(f"  Status     : PASSED")

    # Test 2: varying distributions
    print("\nTest 2: Varying distributions, 1000 symbols")
    vocab_size2 = 100

    symbols2 = []
    probs_list = []
    for _ in range(1000):
        probs = np_rng.dirichlet(np.ones(vocab_size2) * 0.5)
        sym = int(np_rng.choice(vocab_size2, p=probs))
        symbols2.append(sym)
        probs_list.append(probs)

    enc2 = ArithmeticEncoder()
    for s, p in zip(symbols2, probs_list):
        enc2.encode(s, p)
    enc2.flush()
    compressed2 = enc2.get_compressed_bytes()

    dec2 = ArithmeticDecoder(compressed2)
    decoded2 = []
    for p in probs_list:
        decoded2.append(dec2.decode(p))

    assert symbols2 == decoded2, \
        f"MISMATCH: {[(i,a,b) for i,(a,b) in enumerate(zip(symbols2,decoded2)) if a!=b][:5]}"
    print(f"  Symbols    : 1000 (varying 100-symbol distributions)")
    print(f"  Compressed : {len(compressed2)} bytes")
    print(f"  Status     : PASSED")

    # Test 3: edge cases — single symbol alphabet (prob=1.0)
    print("\nTest 3: Edge case — near-degenerate distribution")
    probs3 = np.zeros(10)
    probs3[3] = 0.999
    probs3[7] = 0.001
    symbols3 = [3] * 500 + [7] * 5
    rng.shuffle(symbols3)

    enc3 = ArithmeticEncoder()
    for s in symbols3:
        enc3.encode(s, probs3)
    enc3.flush()
    compressed3 = enc3.get_compressed_bytes()

    dec3 = ArithmeticDecoder(compressed3)
    decoded3 = [dec3.decode(probs3) for _ in symbols3]

    assert symbols3 == decoded3, "Edge case FAILED"
    print(f"  Symbols    : {len(symbols3)}")
    print(f"  Compressed : {len(compressed3)} bytes vs {len(symbols3)} bytes raw")
    print(f"  Status     : PASSED")

    # Test 4: long sequence stress test
    print("\nTest 4: Stress test — 10000 symbols, 256-symbol alphabet")
    vocab4 = 256
    probs4 = np_rng.dirichlet(np.ones(vocab4))
    symbols4 = [int(np_rng.choice(vocab4, p=probs4)) for _ in range(10000)]

    enc4 = ArithmeticEncoder()
    for s in symbols4:
        enc4.encode(s, probs4)
    enc4.flush()
    compressed4 = enc4.get_compressed_bytes()

    dec4 = ArithmeticDecoder(compressed4)
    decoded4 = [dec4.decode(probs4) for _ in symbols4]

    assert symbols4 == decoded4, "Stress test FAILED"
    ratio = len(compressed4) / len(symbols4)
    print(f"  Symbols    : 10000")
    print(f"  Compressed : {len(compressed4)} bytes (ratio {ratio:.3f})")
    print(f"  Status     : PASSED")

    print()
    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
    sys.exit(0)
