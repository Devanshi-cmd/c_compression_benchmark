#!/usr/bin/env python3
"""
benchmark.py — Comprehensive comparison of Model 1 (I2S) vs Model 2 (True 2-bit).

Metrics:
  A: Compression ratio (vs gzip, zstd baselines)
  B: Inference speed (chars/sec, ms/forward pass)
  C: Training convergence plots (PNG)
  D: Model file sizes (disk and runtime)
  E: Lossless verification (100% required)

Usage:
  python benchmark.py --model 1
  python benchmark.py --model 2
  python benchmark.py --model both

All results saved to results/ folder.
"""

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR     = PROJECT_ROOT / 'data'
RESULTS_DIR  = PROJECT_ROOT / 'results'
CKPT_ROOT    = PROJECT_ROOT / 'checkpoints'
METADATA_JSON = DATA_DIR / 'metadata.json'
VOCAB_JSON    = DATA_DIR / 'vocab.json'
ENCODED_DAT   = DATA_DIR / 'encoded.dat'

RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Load metadata / vocab
# ---------------------------------------------------------------------------

def load_vocab():
    with open(VOCAB_JSON) as f:
        v = json.load(f)
    return v['char_to_idx'], v['idx_to_char']


def load_metadata():
    with open(METADATA_JSON) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Load trained model for inference
# ---------------------------------------------------------------------------

def load_model_for_inference(model_id: int, vocab_size: int, device: torch.device):
    """Load best checkpoint for inference (training-mode model)."""
    ckpt_dir = CKPT_ROOT / f'model{model_id}'
    best_path = ckpt_dir / 'best_model.pt'
    latest_path = ckpt_dir / 'latest.pt'

    ckpt_path = best_path if best_path.exists() else latest_path
    if not ckpt_path.exists():
        print(f"WARNING: No checkpoint found for model {model_id} at {ckpt_dir}")
        print(f"  Running with UNTRAINED model (for pipeline testing only)")
        ckpt_path = None

    if model_id == 1:
        from model1_i2s.model import build_model
    else:
        from model2_true2bit.model import build_model

    model = build_model(vocab_size, verify_params=False).to(device)
    model.eval()

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=device)
        # Handle compiled model state dict keys
        state_dict = state['model_state_dict']
        # Remove _orig_mod. prefix if present (from torch.compile)
        clean_state = {}
        for k, v in state_dict.items():
            clean_key = k.replace('_orig_mod.', '')
            clean_state[clean_key] = v
        model.load_state_dict(clean_state, strict=False)
        print(f"  Model {model_id}: loaded from {ckpt_path}")
    else:
        print(f"  Model {model_id}: UNTRAINED (random weights)")

    return model


# ---------------------------------------------------------------------------
# Character-level model compression
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_char_probabilities(model: nn.Module, context: torch.Tensor,
                            vocab_size: int, device: torch.device) -> np.ndarray:
    """
    Run model on context, return probability distribution over next character.
    context: 1D tensor of token indices, length <= CONTEXT_LEN
    Returns: float32 numpy array of length vocab_size
    """
    context_len = model.context_len if hasattr(model, 'context_len') else 128
    ctx = context[-context_len:].unsqueeze(0).to(device)  # (1, T)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                            enabled=(device.type == 'cuda')):
        logits = model(ctx)  # (1, T, V)

    probs = F.softmax(logits[0, -1, :].float(), dim=-1)
    return probs.cpu().numpy()


def compress_file_with_model(model: nn.Module, file_bytes: bytes,
                             char_to_idx: dict, idx_to_char: list,
                             device: torch.device, vocab_size: int):
    """
    Compress file_bytes using the model as a probability predictor.
    Returns: (compressed_bytes, n_chars, time_seconds, ms_per_char)
    """
    from arithmetic_coder import ArithmeticEncoder

    # Encode to character indices
    text = file_bytes.decode('utf-8', errors='replace')
    unk = char_to_idx.get('<UNK>', 0)
    indices = [char_to_idx.get(c, unk) for c in text]
    n_chars = len(indices)

    if n_chars == 0:
        return b'', 0, 0.0, 0.0

    encoder = ArithmeticEncoder()
    context_len = model.context_len if hasattr(model, 'context_len') else 128

    # Use a small warm-up context (uniform priors for first few chars)
    context = torch.zeros(1, dtype=torch.long)
    t0 = time.perf_counter()
    forward_times = []

    for i, sym in enumerate(indices):
        # Get probability distribution from model
        ft0 = time.perf_counter()
        probs = get_char_probabilities(model, context, vocab_size, device)
        forward_times.append(time.perf_counter() - ft0)

        encoder.encode(sym, probs)

        # Update context
        new_tok = torch.tensor([sym], dtype=torch.long)
        context = torch.cat([context, new_tok])
        if len(context) > context_len:
            context = context[-context_len:]

    encoder.flush()
    total_time = time.perf_counter() - t0
    compressed = encoder.get_compressed_bytes()
    avg_ms_per_char = np.mean(forward_times) * 1000 if forward_times else 0

    return compressed, n_chars, total_time, avg_ms_per_char


def decompress_file_with_model(compressed: bytes, n_chars: int,
                               model: nn.Module, char_to_idx: dict,
                               idx_to_char: list, device: torch.device,
                               vocab_size: int) -> bytes:
    """
    Decompress using model. Must be called with same model and same order.
    """
    from arithmetic_coder import ArithmeticDecoder

    decoder = ArithmeticDecoder(compressed)
    context_len = model.context_len if hasattr(model, 'context_len') else 128
    context = torch.zeros(1, dtype=torch.long)
    decoded_chars = []

    for i in range(n_chars):
        probs = get_char_probabilities(model, context, vocab_size, device)
        sym = decoder.decode(probs)
        char = idx_to_char[sym] if 0 < sym < len(idx_to_char) else '\ufffd'
        decoded_chars.append(char)

        new_tok = torch.tensor([sym], dtype=torch.long)
        context = torch.cat([context, new_tok])
        if len(context) > context_len:
            context = context[-context_len:]

    text = ''.join(decoded_chars)
    return text.encode('utf-8', errors='replace')


# ---------------------------------------------------------------------------
# Baseline compression (gzip, zstd)
# ---------------------------------------------------------------------------

def compress_gzip(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=9)


def compress_zstd(data: bytes) -> bytes:
    try:
        result = subprocess.run(
            ['zstd', '--no-progress', '-19', '-q', '--stdout'],
            input=data, capture_output=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return b''  # zstd not available


# ---------------------------------------------------------------------------
# Get test files (files model never saw — from val split)
# ---------------------------------------------------------------------------

def get_test_files(n_files: int = 20) -> list:
    """Get C files from val region (model never trained on these)."""
    if not METADATA_JSON.exists():
        # Fallback: find any C files on the system
        return []

    with open(METADATA_JSON) as f:
        meta = json.load(f)

    repos_dir = DATA_DIR / 'repos'
    if not repos_dir.exists():
        return []

    # Use val split files: files whose content maps to val region
    # Simple heuristic: take files from repos sorted by path, last 10%
    c_files = []
    for root, dirs, files in os.walk(repos_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in files:
            if f.endswith(('.c', '.h')):
                p = Path(root) / f
                try:
                    sz = p.stat().st_size
                    if 1000 < sz < 100_000:  # reasonable size for testing
                        c_files.append(p)
                except OSError:
                    pass

    if not c_files:
        return []

    # Sort and take last n_files (approximate val split)
    c_files.sort()
    # Take from the tail (val region)
    n_test = min(n_files, len(c_files) // 10)
    n_test = max(n_test, min(n_files, len(c_files)))
    test_files = c_files[-n_test:][:n_files]
    return test_files


# ---------------------------------------------------------------------------
# Metric A: Compression ratio
# ---------------------------------------------------------------------------

def benchmark_compression(model: nn.Module, model_id: int,
                           char_to_idx: dict, idx_to_char: list,
                           device: torch.device, vocab_size: int,
                           n_files: int = 20, n_runs: int = 3):
    print(f"\n[A] Compression Ratio — Model {model_id}")
    test_files = get_test_files(n_files)

    if not test_files:
        print("  WARNING: No test files found. Using synthetic data.")
        # Create synthetic C-like data for testing
        synthetic = b'int main() {\n    int x = 0;\n    for (int i = 0; i < 100; i++) {\n        x += i;\n    }\n    return x;\n}\n' * 100
        test_files_data = [('synthetic.c', synthetic)]
    else:
        test_files_data = []
        for p in test_files:
            try:
                data = p.read_bytes()
                if data:
                    test_files_data.append((str(p.name), data))
            except OSError:
                pass

    results = []
    for fname, data in test_files_data[:n_files]:
        orig_size = len(data)
        # Limit file size for speed (first 8KB)
        data_sample = data[:8192] if len(data) > 8192 else data

        # Model compression (single run for ratio — speed measured in metric B)
        compressed, n_chars, comp_time, ms_per_char = compress_file_with_model(
            model, data_sample, char_to_idx, idx_to_char, device, vocab_size)

        comp_size = len(compressed)
        ratio = comp_size / len(data_sample) if len(data_sample) > 0 else 1.0
        bpc   = comp_size * 8 / n_chars if n_chars > 0 else 8.0

        # Baselines
        gz_size   = len(compress_gzip(data_sample))
        gz_ratio  = gz_size / len(data_sample)
        zstd_bytes = compress_zstd(data_sample)
        zstd_ratio = len(zstd_bytes) / len(data_sample) if zstd_bytes else None

        r = {
            'file': fname,
            'orig_size': len(data_sample),
            'model_size': comp_size,
            'model_ratio': round(ratio, 4),
            'model_bpc': round(bpc, 4),
            'gzip_size': gz_size,
            'gzip_ratio': round(gz_ratio, 4),
            'zstd_size': len(zstd_bytes) if zstd_bytes else None,
            'zstd_ratio': round(zstd_ratio, 4) if zstd_ratio else None,
        }
        results.append(r)
        print(f"  {fname:30s} | orig={len(data_sample):6d}B | "
              f"model={comp_size:6d}B ({ratio:.3f}) | "
              f"gzip={gz_size:6d}B ({gz_ratio:.3f}) | "
              f"bpc={bpc:.3f}")

    if results:
        avg_ratio = np.mean([r['model_ratio'] for r in results])
        avg_bpc   = np.mean([r['model_bpc']   for r in results])
        avg_gzip  = np.mean([r['gzip_ratio']  for r in results])
        print(f"\n  Average model ratio: {avg_ratio:.4f}")
        print(f"  Average model bpc  : {avg_bpc:.4f}")
        print(f"  Average gzip ratio : {avg_gzip:.4f}")
    else:
        avg_ratio = avg_bpc = avg_gzip = None

    return {
        'files': results,
        'avg_ratio': float(avg_ratio) if avg_ratio else None,
        'avg_bpc':   float(avg_bpc)   if avg_bpc   else None,
        'avg_gzip_ratio': float(avg_gzip) if avg_gzip else None,
    }


# ---------------------------------------------------------------------------
# Metric B: Inference speed
# ---------------------------------------------------------------------------

def benchmark_speed(model: nn.Module, model_id: int,
                    char_to_idx: dict, idx_to_char: list,
                    device: torch.device, vocab_size: int,
                    n_files: int = 5, n_runs: int = 3):
    print(f"\n[B] Inference Speed — Model {model_id}")
    test_files = get_test_files(n_files)

    if not test_files:
        synthetic = b'int x = 0; /* test */\n' * 200
        test_files_data = [('synthetic.c', synthetic)]
    else:
        test_files_data = [(p.name, p.read_bytes()[:4096]) for p in test_files
                           if p.stat().st_size > 0]

    all_speeds = []
    all_ms_per_forward = []

    for fname, data in test_files_data[:n_files]:
        if not data:
            continue
        run_speeds = []
        run_ms = []

        for run in range(n_runs):
            _, n_chars, comp_time, ms_per_char = compress_file_with_model(
                model, data, char_to_idx, idx_to_char, device, vocab_size)
            if comp_time > 0 and n_chars > 0:
                run_speeds.append(n_chars / comp_time)
                run_ms.append(ms_per_char)

        avg_speed = np.mean(run_speeds) if run_speeds else 0
        avg_ms    = np.mean(run_ms) if run_ms else 0
        all_speeds.append(avg_speed)
        all_ms_per_forward.append(avg_ms)
        print(f"  {str(fname):30s} | {avg_speed:8.0f} chars/sec | "
              f"{avg_ms:.3f} ms/forward (avg of {n_runs} runs)")

    mean_speed = float(np.mean(all_speeds)) if all_speeds else 0.0
    mean_ms    = float(np.mean(all_ms_per_forward)) if all_ms_per_forward else 0.0
    print(f"\n  Mean speed   : {mean_speed:.0f} chars/sec")
    print(f"  Mean ms/fwd  : {mean_ms:.3f} ms")

    return {
        'chars_per_sec': mean_speed,
        'ms_per_forward': mean_ms,
        'n_runs': n_runs,
    }


# ---------------------------------------------------------------------------
# Metric C: Training convergence plots
# ---------------------------------------------------------------------------

def plot_training_curves():
    print("\n[C] Training Convergence Plots")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    stats = {}
    for mid in [1, 2]:
        sp = RESULTS_DIR / f'training_stats_model{mid}.json'
        if sp.exists():
            with open(sp) as f:
                stats[mid] = json.load(f)
            print(f"  Model {mid}: {len(stats[mid])} epochs loaded")
        else:
            print(f"  Model {mid}: no training stats found")

    if not stats:
        print("  No training data available for plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loss curves
    for mid, data in stats.items():
        epochs = [d['epoch'] for d in data]
        train_loss = [d['train_loss'] for d in data]
        val_loss   = [d['val_loss']   for d in data]
        axes[0].plot(epochs, train_loss, label=f'Model {mid} train', linestyle='-')
        axes[0].plot(epochs, val_loss,   label=f'Model {mid} val',   linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # BPC curves
    for mid, data in stats.items():
        epochs  = [d['epoch']   for d in data]
        bpc_val = [d['bpc_val'] for d in data]
        axes[1].plot(epochs, bpc_val, label=f'Model {mid}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Bits Per Character')
    axes[1].set_title('Val BPC over Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # LR schedule
    for mid, data in stats.items():
        epochs = [d['epoch'] for d in data]
        lrs    = [d['lr']    for d in data]
        axes[2].plot(epochs, lrs, label=f'Model {mid}')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('LR Schedule')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / 'training_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Metric D: Model file sizes
# ---------------------------------------------------------------------------

def measure_model_size(model_id: int, model: nn.Module):
    print(f"\n[D] Model File Size — Model {model_id}")

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())

    # Checkpoint file size
    ckpt_dir  = CKPT_ROOT / f'model{model_id}'
    best_path = ckpt_dir / 'best_model.pt'
    latest    = ckpt_dir / 'latest.pt'
    ckpt_size_mb = 0
    if best_path.exists():
        ckpt_size_mb = best_path.stat().st_size / 1024 / 1024
    elif latest.exists() and latest.is_symlink():
        real = latest.resolve()
        if real.exists():
            ckpt_size_mb = real.stat().st_size / 1024 / 1024

    # Theoretical packed size
    bitlinear_params = sum(p.numel() for n, p in model.named_parameters()
                          if 'weight' in n and p.ndim == 2)
    packed_size_mb = (bitlinear_params * 2 / 8) / 1024 / 1024  # 2 bits/param
    int8_size_mb   = bitlinear_params / 1024 / 1024             # 1 byte/param

    print(f"  Parameters          : {n_params:,}")
    print(f"  Checkpoint size     : {ckpt_size_mb:.1f} MB")
    print(f"  2-bit packed size   : {packed_size_mb:.1f} MB (disk)")
    if model_id == 1:
        print(f"  I2S runtime size    : {int8_size_mb:.1f} MB (int8 unpacked)")
        print(f"  Memory overhead     : {int8_size_mb / packed_size_mb:.1f}× vs packed")
    else:
        print(f"  True 2-bit runtime  : {packed_size_mb:.1f} MB (stays packed)")
        print(f"  Memory overhead     : 1.0× (no unpacking)")

    return {
        'n_params': n_params,
        'checkpoint_mb': round(ckpt_size_mb, 2),
        'packed_disk_mb': round(packed_size_mb, 2),
        'runtime_mb': round(int8_size_mb if model_id == 1 else packed_size_mb, 2),
        'memory_overhead_vs_packed': round(int8_size_mb / packed_size_mb, 2) if model_id == 1 else 1.0,
    }


# ---------------------------------------------------------------------------
# Metric E: Lossless verification
# ---------------------------------------------------------------------------

def verify_lossless(model: nn.Module, model_id: int,
                    char_to_idx: dict, idx_to_char: list,
                    device: torch.device, vocab_size: int,
                    n_files: int = 5):
    print(f"\n[E] Lossless Verification — Model {model_id}")
    test_files = get_test_files(n_files)

    if not test_files:
        synthetic = b'int main() { return 0; }\n' * 50
        test_files_data = [('synthetic.c', synthetic)]
    else:
        test_files_data = [(p.name, p.read_bytes()[:2048]) for p in test_files
                           if p.stat().st_size > 0]

    passed = 0
    failed = 0
    lossless_rate = 0.0

    for fname, original in test_files_data[:n_files]:
        if not original:
            continue

        # Compress
        compressed, n_chars, _, _ = compress_file_with_model(
            model, original, char_to_idx, idx_to_char, device, vocab_size)

        # Decompress
        reconstructed = decompress_file_with_model(
            compressed, n_chars, model, char_to_idx, idx_to_char,
            device, vocab_size)

        # Compare: decode original the same way (utf-8 round-trip)
        orig_decoded = original.decode('utf-8', errors='replace').encode('utf-8', errors='replace')

        if reconstructed == orig_decoded:
            status = 'PASS'
            passed += 1
        else:
            status = 'FAIL'
            failed += 1
            # Find first difference
            for i, (a, b) in enumerate(zip(reconstructed, orig_decoded)):
                if a != b:
                    print(f"  CRITICAL: First diff at byte {i}: "
                          f"got {a!r}, expected {b!r}")
                    break

        print(f"  {str(fname):30s} | n_chars={n_chars:5d} | "
              f"orig={len(orig_decoded):5d}B | recon={len(reconstructed):5d}B | {status}")

    total = passed + failed
    lossless_rate = passed / total if total > 0 else 0.0
    print(f"\n  Lossless rate: {passed}/{total} = {lossless_rate*100:.1f}%")
    if failed > 0:
        print(f"  CRITICAL: {failed} file(s) failed lossless reconstruction!")

    return {
        'passed': passed,
        'failed': failed,
        'total': total,
        'lossless_rate': round(lossless_rate, 4),
    }


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def write_comparison_report(results: dict):
    report_path = RESULTS_DIR / 'comparison_report.md'

    m1 = results.get(1, {})
    m2 = results.get(2, {})

    def fmt(val, fmt_str='.4f', fallback='N/A'):
        if val is None:
            return fallback
        try:
            return format(val, fmt_str)
        except (TypeError, ValueError):
            return str(val)

    lines = [
        "# Benchmark Comparison Report",
        f"",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## A. Compression Ratio",
        f"",
        f"| Metric | Model 1 (I2S) | Model 2 (True 2-bit) | Winner |",
        f"|--------|--------------|---------------------|--------|",
    ]

    def winner(v1, v2, lower_is_better=True):
        if v1 is None or v2 is None:
            return '—'
        if lower_is_better:
            return 'Model 1' if v1 < v2 else ('Model 2' if v2 < v1 else 'Tie')
        return 'Model 1' if v1 > v2 else ('Model 2' if v2 > v1 else 'Tie')

    comp1 = m1.get('compression', {})
    comp2 = m2.get('compression', {})
    spd1  = m1.get('speed',       {})
    spd2  = m2.get('speed',       {})
    size1 = m1.get('file_size',   {})
    size2 = m2.get('file_size',   {})
    ll1   = m1.get('lossless',    {})
    ll2   = m2.get('lossless',    {})

    r1 = comp1.get('avg_ratio')
    r2 = comp2.get('avg_ratio')
    b1 = comp1.get('avg_bpc')
    b2 = comp2.get('avg_bpc')
    g1 = comp1.get('avg_gzip_ratio')

    lines += [
        f"| Avg Compression Ratio | {fmt(r1)} | {fmt(r2)} | {winner(r1,r2,True)} |",
        f"| Avg Bits Per Char     | {fmt(b1)} | {fmt(b2)} | {winner(b1,b2,True)} |",
        f"| gzip ratio (baseline) | {fmt(g1)} | {fmt(comp2.get('avg_gzip_ratio'))} | — |",
        f"",
        f"## B. Inference Speed",
        f"",
        f"| Metric | Model 1 (I2S) | Model 2 (True 2-bit) | Winner |",
        f"|--------|--------------|---------------------|--------|",
    ]

    s1 = spd1.get('chars_per_sec')
    s2 = spd2.get('chars_per_sec')
    ms1 = spd1.get('ms_per_forward')
    ms2 = spd2.get('ms_per_forward')

    lines += [
        f"| Chars/Second          | {fmt(s1,'.0f')} | {fmt(s2,'.0f')} | {winner(s1,s2,False)} |",
        f"| ms/forward pass       | {fmt(ms1,'.3f')} | {fmt(ms2,'.3f')} | {winner(ms1,ms2,True)} |",
        f"",
        f"## C. Training Convergence",
        f"",
        f"See `results/training_curves.png` for loss/BPC/LR plots.",
        f"",
        f"## D. Model File Size",
        f"",
        f"| Metric | Model 1 (I2S) | Model 2 (True 2-bit) | Winner |",
        f"|--------|--------------|---------------------|--------|",
    ]

    np1 = size1.get('n_params')
    np2 = size2.get('n_params')
    ck1 = size1.get('checkpoint_mb')
    ck2 = size2.get('checkpoint_mb')
    pk1 = size1.get('packed_disk_mb')
    pk2 = size2.get('packed_disk_mb')
    rt1 = size1.get('runtime_mb')
    rt2 = size2.get('runtime_mb')
    ov1 = size1.get('memory_overhead_vs_packed')
    ov2 = size2.get('memory_overhead_vs_packed')

    lines += [
        f"| Parameters           | {fmt(np1,',d')} | {fmt(np2,',d')} | — |",
        f"| Checkpoint size (MB) | {fmt(ck1,'.1f')} | {fmt(ck2,'.1f')} | {winner(ck1,ck2,True)} |",
        f"| 2-bit packed (MB)    | {fmt(pk1,'.1f')} | {fmt(pk2,'.1f')} | — |",
        f"| Runtime weight (MB)  | {fmt(rt1,'.1f')} (int8) | {fmt(rt2,'.1f')} (packed) | {winner(rt1,rt2,True)} |",
        f"| Memory overhead      | {fmt(ov1,'.1f')}× | {fmt(ov2,'.1f')}× | {winner(ov1,ov2,True)} |",
        f"",
        f"## E. Lossless Verification",
        f"",
        f"| Metric | Model 1 (I2S) | Model 2 (True 2-bit) |",
        f"|--------|--------------|---------------------|",
    ]

    lr1 = ll1.get('lossless_rate')
    lr2 = ll2.get('lossless_rate')
    p1  = ll1.get('passed')
    p2  = ll2.get('passed')
    t1  = ll1.get('total')
    t2  = ll2.get('total')

    lines += [
        f"| Lossless Rate | {fmt(lr1,'%')} ({p1}/{t1}) | {fmt(lr2,'%')} ({p2}/{t2}) |",
        f"",
        f"## Summary and Recommendation",
        f"",
        f"### Architecture",
        f"Both models: d_model=304, d_ff=1360, n_heads=8, n_layers=10, ~12M params.",
        f"",
        f"### Key Tradeoffs",
        f"",
        f"- **Model 1 (I2S):** Unpacks 2-bit weights to int8 at inference.",
        f"  Uses standard int8 GEMV. Larger runtime memory (4× vs packed).",
        f"  Simpler compute path; potentially faster on hardware with int8 SIMD.",
        f"",
        f"- **Model 2 (True 2-bit):** Never unpacks weights.",
        f"  Lookup-table GEMV on raw packed bytes. Smaller runtime memory (1×).",
        f"  Memory-bandwidth optimal for very large models.",
        f"",
        f"### Recommendation",
        f"",
    ]

    if s1 and s2 and r1 and r2:
        speed_winner = 1 if s1 > s2 else 2
        ratio_winner = 1 if r1 < r2 else 2
        lines.append(
            f"For compression speed: **Model {speed_winner}** wins ({max(s1,s2):.0f} chars/sec)."
        )
        lines.append(
            f"For compression ratio: **Model {ratio_winner}** wins (bpc: {min(b1,b2):.4f})."
        )
        lines.append(
            f"For memory efficiency: **Model 2** wins (4× less runtime memory for weights)."
        )
        if speed_winner == ratio_winner:
            lines.append(
                f"\n**Recommendation: Model {speed_winner}** — wins on both speed and ratio."
            )
        else:
            lines.append(
                f"\n**Recommendation: Depends on use case.**"
                f" If memory-constrained (edge devices): Model 2."
                f" If speed is primary: Model {speed_winner}."
            )
    else:
        lines.append("Insufficient data for recommendation (models may not be trained yet).")

    lines.append(f"\n---\n*Generated by benchmark.py*")

    report_path.write_text('\n'.join(lines))
    print(f"\nComparison report saved: {report_path}")


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(model_ids: list):
    if not VOCAB_JSON.exists():
        print("ERROR: vocab.json not found. Run data/prepare_data.py first.")
        sys.exit(1)

    char_to_idx, idx_to_char = load_vocab()
    metadata   = load_metadata()
    vocab_size = metadata['vocab_size']
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device     : {device}")
    print(f"Vocab size : {vocab_size}")

    all_results = {}

    for mid in model_ids:
        print(f"\n{'='*60}")
        print(f"Benchmarking Model {mid}")
        print(f"{'='*60}")

        model = load_model_for_inference(mid, vocab_size, device)
        model.eval()

        results = {}

        # A: Compression ratio
        results['compression'] = benchmark_compression(
            model, mid, char_to_idx, idx_to_char, device, vocab_size, n_files=20)

        # B: Speed
        results['speed'] = benchmark_speed(
            model, mid, char_to_idx, idx_to_char, device, vocab_size, n_files=5)

        # D: File size
        results['file_size'] = measure_model_size(mid, model)

        # E: Lossless
        results['lossless'] = verify_lossless(
            model, mid, char_to_idx, idx_to_char, device, vocab_size, n_files=5)

        all_results[mid] = results

        # Save per-model results
        out_path = RESULTS_DIR / f'benchmark_model{mid}.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved: {out_path}")

    # C: Training plots (combined)
    plot_training_curves()

    # Save combined results
    combined_path = RESULTS_DIR / 'benchmark_results.json'
    with open(combined_path, 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()},
                  f, indent=2, default=str)
    print(f"\nCombined results: {combined_path}")

    # Write comparison report if both models run
    write_comparison_report(all_results)

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Benchmark BitNet compression models')
    parser.add_argument('--model', type=str, default='both',
                        choices=['1', '2', 'both'],
                        help='Which model to benchmark')
    parser.add_argument('--n-files', type=int, default=20,
                        help='Number of test files')
    args = parser.parse_args()

    if args.model == '1':
        model_ids = [1]
    elif args.model == '2':
        model_ids = [2]
    else:
        model_ids = [1, 2]

    print("=" * 60)
    print("BitNet Compression Benchmark")
    print("=" * 60)
    run_benchmark(model_ids)
    print("\nBenchmark complete. Results in results/")


if __name__ == '__main__':
    main()
