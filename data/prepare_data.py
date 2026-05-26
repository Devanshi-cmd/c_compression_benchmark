#!/usr/bin/env python3
"""
prepare_data.py — Build character-level training corpus from C source files.

Two-pass approach:
  Pass 1: Count all characters, build vocabulary
  Pass 2: Encode files directly to memory-mapped encoded.dat

No RAM overflow — streams file by file in both passes.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
REPOS_DIR = SCRIPT_DIR / "repos"
ENCODED_DAT = SCRIPT_DIR / "encoded.dat"
VOCAB_JSON = SCRIPT_DIR / "vocab.json"
METADATA_JSON = SCRIPT_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_c_files(repos_dir: Path):
    """Yield (path, size_bytes) for every .c and .h file under repos_dir."""
    for root, dirs, files in os.walk(repos_dir):
        # Skip hidden dirs and build artifacts
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in
                   ('build', 'BUILD', 'out', 'obj', '__pycache__', 'node_modules')]
        for fname in files:
            if fname.endswith(('.c', '.h')):
                fpath = Path(root) / fname
                try:
                    size = fpath.stat().st_size
                    if size > 0:
                        yield fpath, size
                except OSError:
                    continue


def read_file_safe(fpath: Path) -> bytes:
    """Read a file, returning empty bytes on any error."""
    try:
        return fpath.read_bytes()
    except OSError:
        return b''


# ---------------------------------------------------------------------------
# Pass 1: Build vocabulary
# ---------------------------------------------------------------------------

def build_vocabulary(repos_dir: Path):
    """
    Stream all C/H files, count unique characters.
    Returns: (char_counts Counter, total_char_count int, file_list list)
    """
    print("=== Pass 1: Building vocabulary ===")
    char_counts = Counter()
    total_chars = 0
    file_list = []
    n_files = 0
    t0 = time.time()

    for fpath, size_hint in iter_c_files(repos_dir):
        data = read_file_safe(fpath)
        if not data:
            continue
        try:
            text = data.decode('utf-8', errors='replace')
        except Exception:
            continue

        char_counts.update(text)
        total_chars += len(text)
        file_list.append(str(fpath))
        n_files += 1

        if n_files % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  {n_files:6d} files | {total_chars:12,d} chars | "
                  f"{len(char_counts):4d} unique | {elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0
    print(f"Pass 1 done: {n_files} files, {total_chars:,} chars, "
          f"{len(char_counts)} unique chars in {elapsed:.1f}s")

    return char_counts, total_chars, file_list, n_files


def build_vocab_mapping(char_counts: Counter):
    """
    Build char→index and index→char mappings.
    Index 0 is reserved for <UNK>.
    Characters sorted by frequency descending (most common get lowest index).
    """
    # Sort by frequency descending, then by character for determinism
    sorted_chars = sorted(char_counts.keys(), key=lambda c: (-char_counts[c], c))
    # Index 0 = <UNK>
    idx_to_char = ['<UNK>'] + sorted_chars
    char_to_idx = {c: i for i, c in enumerate(idx_to_char)}
    return char_to_idx, idx_to_char


# ---------------------------------------------------------------------------
# Pass 2: Encode to disk
# ---------------------------------------------------------------------------

def encode_corpus(repos_dir: Path, char_to_idx: dict, total_chars: int,
                  encoded_dat_path: Path):
    """
    Stream all files again, encode each character to uint16, write to memmap.
    uint16 supports vocab up to 65535 — more than enough for ~100 unique chars.
    We use uint16 to ensure alignment and future-proofing.
    """
    print("\n=== Pass 2: Encoding corpus to disk ===")

    # Create memory-mapped file for writing
    dtype = np.uint16
    mm = np.memmap(encoded_dat_path, dtype=dtype, mode='w+', shape=(total_chars,))

    offset = 0
    n_files = 0
    unk_idx = char_to_idx.get('<UNK>', 0)
    t0 = time.time()

    file_offsets = {}  # fpath → (start, end) for metadata

    for fpath, _ in iter_c_files(repos_dir):
        data = read_file_safe(fpath)
        if not data:
            continue
        try:
            text = data.decode('utf-8', errors='replace')
        except Exception:
            continue

        encoded = np.array([char_to_idx.get(c, unk_idx) for c in text],
                           dtype=np.uint16)
        n = len(encoded)

        if offset + n > total_chars:
            # Safety: corpus size might differ slightly between passes (race condition)
            n = total_chars - offset
            encoded = encoded[:n]

        mm[offset:offset + n] = encoded
        file_offsets[str(fpath)] = (int(offset), int(offset + n))
        offset += n
        n_files += 1

        if n_files % 1000 == 0:
            elapsed = time.time() - t0
            pct = offset / total_chars * 100
            print(f"  {n_files:6d} files | {offset:12,d}/{total_chars:12,d} chars "
                  f"({pct:.1f}%) | {elapsed:.1f}s", flush=True)

    # Flush to disk
    mm.flush()
    del mm

    elapsed = time.time() - t0
    actual_chars = offset
    print(f"Pass 2 done: {n_files} files, {actual_chars:,} chars encoded "
          f"in {elapsed:.1f}s")

    return actual_chars, file_offsets


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def compute_splits(total_chars: int, train_frac: float = 0.9):
    train_end = int(total_chars * train_frac)
    val_end = total_chars
    return train_end, val_end


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("prepare_data.py — C Source Corpus Builder")
    print("=" * 60)
    print(f"Repos directory : {REPOS_DIR}")
    print(f"Output directory: {SCRIPT_DIR}")
    print()

    if not REPOS_DIR.exists():
        print(f"ERROR: {REPOS_DIR} does not exist.")
        print("Run: bash data/fetch_data.sh first.")
        sys.exit(1)

    # Check if any C files exist
    first = next(iter_c_files(REPOS_DIR), None)
    if first is None:
        print(f"ERROR: No .c or .h files found under {REPOS_DIR}")
        sys.exit(1)

    # ---- Pass 1 ----
    char_counts, total_chars, file_list, n_files = build_vocabulary(REPOS_DIR)

    if total_chars == 0:
        print("ERROR: Zero characters found.")
        sys.exit(1)

    char_to_idx, idx_to_char = build_vocab_mapping(char_counts)
    vocab_size = len(idx_to_char)
    print(f"\nVocabulary size: {vocab_size} (including <UNK>)")
    print(f"Top 20 chars by frequency:")
    for i, ch in enumerate(idx_to_char[1:21], 1):
        print(f"  [{i:3d}] {repr(ch):8s} count={char_counts.get(ch,0):,}")

    # Save vocab
    vocab_data = {
        'char_to_idx': {repr(c)[1:-1] if len(c) == 1 else c: v
                        for c, v in char_to_idx.items()},
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size,
        'unk_index': 0,
    }
    # Use raw character keys (JSON-safe)
    vocab_save = {
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size,
        'unk_index': 0,
    }
    VOCAB_JSON.write_text(json.dumps(vocab_save, ensure_ascii=False, indent=2))
    print(f"\nVocab saved to {VOCAB_JSON}")

    # ---- Pass 2 ----
    actual_chars, file_offsets = encode_corpus(
        REPOS_DIR, char_to_idx, total_chars, ENCODED_DAT)

    # ---- Splits ----
    train_end, val_end = compute_splits(actual_chars)

    # ---- Metadata ----
    metadata = {
        'total_files': n_files,
        'total_chars': actual_chars,
        'vocab_size': vocab_size,
        'dtype': 'uint16',
        'encoded_dat': str(ENCODED_DAT),
        'train_end': train_end,
        'val_start': train_end,
        'val_end': val_end,
        'context_window': 128,
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'repos': list(set(
            str(Path(f).relative_to(REPOS_DIR).parts[0])
            for f in file_list
            if Path(f).is_relative_to(REPOS_DIR)
        )),
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2))
    print(f"\nMetadata saved to {METADATA_JSON}")

    # ---- Summary ----
    encoded_size_mb = ENCODED_DAT.stat().st_size / 1024 / 1024
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total .c/.h files   : {n_files:,}")
    print(f"  Total characters    : {actual_chars:,}")
    print(f"  Vocabulary size     : {vocab_size}")
    print(f"  encoded.dat size    : {encoded_size_mb:.1f} MB")
    print(f"  Train chars         : {train_end:,} ({train_end/actual_chars*100:.1f}%)")
    print(f"  Val chars           : {val_end - train_end:,} ({(val_end-train_end)/actual_chars*100:.1f}%)")
    print("=" * 60)


if __name__ == '__main__':
    main()
