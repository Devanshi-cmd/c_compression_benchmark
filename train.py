#!/usr/bin/env python3
"""
train.py — Shared training loop for both BitNet b1.58 compression models.

Usage:
  python train.py --model 1     (train Model 1 I2S only)
  python train.py --model 2     (train Model 2 True2Bit only)
  python train.py --model both  (train both sequentially)

Features:
  - AdamW optimizer, cosine LR schedule with linear warmup
  - Mixed precision training (bfloat16 activations, float32 weights)
  - Resumable from latest checkpoint (Ctrl-C safe + R2 restore)
  - Epoch-level logging to JSON and training.log
  - RTX 5090 optimizations: torch.compile, pin_memory, non_blocking transfers
  - R2: downloads training data at start, uploads checkpoints after each epoch
  - Auto-shutdown after all epochs complete (exit 0 for Salad restart=on-failure)
"""

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Load .env before anything else
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
except ImportError:
    pass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR     = PROJECT_ROOT / 'data'
ENCODED_DAT  = DATA_DIR / 'encoded.dat'
VOCAB_JSON   = DATA_DIR / 'vocab.json'
METADATA_JSON = DATA_DIR / 'metadata.json'
RESULTS_DIR  = PROJECT_ROOT / 'results'
CKPT_ROOT    = PROJECT_ROOT / 'checkpoints'

RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
CONTEXT_LEN   = 128
BATCH_SIZE    = 512
EPOCHS        = int(os.environ.get('EPOCHS', 20))
CHUNK_SIZE    = 250_000_000   # chars per epoch chunk
LR            = 3e-4
BETAS         = (0.9, 0.95)
WEIGHT_DECAY  = 0.1
GRAD_CLIP     = 1.0
WARMUP_STEPS  = 500
VAL_FRAC      = 0.05          # fraction of chunk used for validation each epoch


# ---------------------------------------------------------------------------
# R2 integration
# ---------------------------------------------------------------------------

def r2_available() -> bool:
    return bool(os.environ.get('R2_ACCESS_KEY_ID') and os.environ.get('R2_BUCKET'))


def fetch_training_data_from_r2():
    """Download encoded.dat, vocab.json, metadata.json from R2 if not local."""
    if not r2_available():
        print("[R2] Credentials not set — skipping data download.")
        return
    if (ENCODED_DAT.exists() and VOCAB_JSON.exists() and METADATA_JSON.exists()):
        print("[R2] Training data already present locally.")
        return
    print("[R2] Fetching training data from R2...")
    try:
        from r2_storage import download_training_data
        ok = download_training_data(str(DATA_DIR))
        if not ok:
            print("[R2] WARNING: Some training data files missing from R2.")
    except Exception as e:
        print(f"[R2] Data fetch failed: {e}")


def fetch_checkpoint_from_r2(model_id: int, ckpt_dir: Path):
    """Download latest checkpoint from R2 if not present locally."""
    if not r2_available():
        return
    for name in ['latest.pt', 'best_model.pt']:
        local = ckpt_dir / name
        if local.exists():
            continue
        print(f"[R2] Fetching checkpoint model{model_id}/{name}...")
        try:
            from r2_storage import download_checkpoint
            download_checkpoint(model_id, name, str(local))
        except Exception as e:
            print(f"[R2] Checkpoint fetch failed: {e}")


def push_checkpoint_to_r2(model_id: int, ckpt_path: Path, is_best: bool):
    """Upload checkpoint to R2 after each epoch."""
    if not r2_available():
        return
    try:
        from r2_storage import upload_checkpoint
        upload_checkpoint(str(ckpt_path), model_id, 'latest.pt')
        if is_best:
            upload_checkpoint(str(ckpt_path), model_id, 'best_model.pt')
    except Exception as e:
        print(f"[R2] Checkpoint upload failed: {e}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Memory-mapped character dataset.
    Returns (context, target) pairs from a slice of encoded.dat.
    """

    def __init__(self, data: np.memmap, context_len: int):
        self.data        = data
        self.context_len = context_len
        self.n_samples   = len(data) - context_len

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.context_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def make_dataloaders(metadata: dict, chunk_offset: int, batch_size: int,
                     num_workers: int = 4):
    """
    Create train and val DataLoaders for a given chunk offset.
    Chunks rotate through the training corpus across epochs.
    """
    total_chars = metadata['total_chars']
    train_end   = metadata['train_end']
    val_start   = metadata['val_start']
    val_end_meta = metadata['val_end']

    # Training chunk: rotate through training data
    chunk_start = chunk_offset % max(1, train_end - CHUNK_SIZE)
    chunk_end   = min(chunk_start + CHUNK_SIZE, train_end)
    # Val: fixed last 5% of val split (or val_frac of chunk)
    val_len     = min(int(CHUNK_SIZE * VAL_FRAC), val_end_meta - val_start)
    val_start_actual = val_end_meta - val_len

    # Open memmaps
    full_data = np.memmap(str(ENCODED_DAT), dtype=np.uint16, mode='r',
                          shape=(total_chars,))

    train_data = full_data[chunk_start:chunk_end]
    val_data   = full_data[val_start_actual:val_end_meta]

    train_ds = CharDataset(train_data, CONTEXT_LEN)
    val_ds   = CharDataset(val_data,   CONTEXT_LEN)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, drop_last=False,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader, chunk_start


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, total_steps: int) -> float:
    """Linear warmup then cosine decay to 0."""
    if step < WARMUP_STEPS:
        return LR * step / max(1, WARMUP_STEPS)
    # Cosine decay
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return LR * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def get_checkpoint_dir(model_id: int) -> Path:
    d = CKPT_ROOT / f'model{model_id}'
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_checkpoint(state: dict, ckpt_dir: Path, epoch: int, is_best: bool):
    ckpt_path = ckpt_dir / f'checkpoint_epoch_{epoch:03d}.pt'
    torch.save(state, ckpt_path)
    # Update latest symlink
    latest = ckpt_dir / 'latest.pt'
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(ckpt_path.name)
    if is_best:
        best_path = ckpt_dir / 'best_model.pt'
        torch.save(state, best_path)
    return ckpt_path


def load_checkpoint(ckpt_dir: Path, model: nn.Module, optimizer, scheduler,
                    device: torch.device):
    """Load latest checkpoint if it exists. Returns (start_epoch, best_val_loss, stats)."""
    latest = ckpt_dir / 'latest.pt'
    if not latest.exists():
        return 0, float('inf'), []

    print(f"  Resuming from {latest.resolve()}")
    state = torch.load(latest, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in state:
        scheduler.load_state_dict(state['scheduler_state_dict'])
    start_epoch  = state.get('epoch', 0) + 1
    best_val     = state.get('best_val_loss', float('inf'))
    stats        = state.get('train_stats', [])
    print(f"  Resumed at epoch {start_epoch}, best_val={best_val:.4f}")
    return start_epoch, best_val, stats


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_epoch(model: nn.Module, loader: DataLoader, optimizer,
                scaler: torch.cuda.amp.GradScaler, device: torch.device,
                global_step: int, total_steps: int, log_interval: int = 200):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    t0         = time.time()

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Update LR
        lr = get_lr(global_step, total_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                enabled=device.type == 'cuda'):
            logits = model(x)                         # (B, T, V)
            loss   = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches  += 1
        global_step += 1

        if batch_idx % log_interval == 0:
            elapsed = time.time() - t0
            bpc = total_loss / n_batches / math.log(2)
            print(f"    step {global_step:6d} | loss {total_loss/n_batches:.4f} | "
                  f"bpc {bpc:.4f} | lr {lr:.2e} | {elapsed:.0f}s elapsed",
                  flush=True)

    avg_loss = total_loss / max(1, n_batches)
    return avg_loss, global_step


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                enabled=device.type == 'cuda'):
            logits = model(x)
            loss   = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(1, n_batches)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model(model_id: int):
    print(f"\n{'='*60}")
    print(f"Training Model {model_id}")
    print(f"{'='*60}")

    # ---- Step 0: pull training data from R2 if not local ----
    fetch_training_data_from_r2()

    # Load metadata
    if not METADATA_JSON.exists():
        print(f"ERROR: {METADATA_JSON} not found. Run data/prepare_data.py first.")
        sys.exit(1)
    with open(METADATA_JSON) as f:
        metadata = json.load(f)
    vocab_size  = metadata['vocab_size']
    total_chars = metadata['total_chars']

    print(f"Vocab size  : {vocab_size}")
    print(f"Total chars : {total_chars:,}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device      : {device}")

    # Build model
    if model_id == 1:
        from model1_i2s.model import build_model
    else:
        from model2_true2bit.model import build_model

    model = build_model(vocab_size).to(device)
    n_params = model.count_parameters()
    print(f"Parameters  : {n_params:,}")

    # torch.compile (RTX 5090 optimization)
    if device.type == 'cuda':
        print("Compiling model with torch.compile()...")
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("torch.compile() applied successfully.")
        except Exception as e:
            print(f"torch.compile() skipped: {e}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY,
        fused=(device.type == 'cuda' and torch.__version__ >= '2.0')
    )

    # GradScaler for mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # Checkpoint dir
    ckpt_dir   = get_checkpoint_dir(model_id)
    stats_path = RESULTS_DIR / f'training_stats_model{model_id}.json'
    log_path   = PROJECT_ROOT / f'training_model{model_id}.log'

    # ---- Step 1: pull latest checkpoint from R2 if not local ----
    fetch_checkpoint_from_r2(model_id, ckpt_dir)

    # Resume from local checkpoint
    start_epoch, best_val_loss, all_stats = load_checkpoint(
        ckpt_dir, model, optimizer, None, device)

    # If already finished all epochs, skip
    if start_epoch >= EPOCHS:
        print(f"[Model {model_id}] Already completed {EPOCHS} epochs. Nothing to do.")
        return all_stats

    # Estimate total steps
    samples_per_chunk = min(CHUNK_SIZE, metadata['train_end']) - CONTEXT_LEN
    steps_per_epoch   = max(1, samples_per_chunk // BATCH_SIZE)
    total_steps       = EPOCHS * steps_per_epoch
    global_step       = start_epoch * steps_per_epoch
    print(f"Epochs      : {start_epoch} → {EPOCHS} (resuming)")
    print(f"Est steps/epoch : {steps_per_epoch:,}")
    print(f"Est total steps : {total_steps:,}")

    # Ctrl-C / SIGTERM handler
    interrupted = [False]
    def handle_interrupt(sig, frame):
        print("\n[Interrupt] Saving checkpoint before exit...")
        interrupted[0] = True
    signal.signal(signal.SIGINT,  handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    log_file = open(log_path, 'a')

    def log(msg):
        print(msg, flush=True)
        print(msg, file=log_file, flush=True)

    log(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training Model {model_id} "
        f"starting at epoch {start_epoch + 1}/{EPOCHS}")

    epoch = start_epoch  # ensure defined for finally block
    try:
        for epoch in range(start_epoch, EPOCHS):
            if interrupted[0]:
                break

            epoch_t0     = time.time()
            chunk_offset = epoch * CHUNK_SIZE

            log(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

            num_workers = min(4, os.cpu_count() or 1)
            train_loader, val_loader, chunk_start = make_dataloaders(
                metadata, chunk_offset, BATCH_SIZE, num_workers)
            log(f"  Chunk offset: {chunk_start:,} chars")
            log(f"  Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}")

            train_loss, global_step = train_epoch(
                model, train_loader, optimizer, scaler, device,
                global_step, total_steps
            )

            val_loss   = eval_epoch(model, val_loader, device)
            epoch_time = time.time() - epoch_t0
            bpc_train  = train_loss / math.log(2)
            bpc_val    = val_loss   / math.log(2)
            current_lr = get_lr(global_step, total_steps)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            # ETA
            epochs_left  = EPOCHS - (epoch + 1)
            eta_seconds  = epochs_left * epoch_time
            eta_str      = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.0f}m"

            epoch_stat = {
                'epoch':        epoch + 1,
                'train_loss':   round(train_loss, 6),
                'val_loss':     round(val_loss,   6),
                'bpc_train':    round(bpc_train,  6),
                'bpc_val':      round(bpc_val,    6),
                'lr':           round(current_lr, 8),
                'epoch_time_s': round(epoch_time, 2),
                'is_best':      is_best,
                'global_step':  global_step,
                'eta':          eta_str,
            }
            all_stats.append(epoch_stat)

            log(f"  train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"bpc_val={bpc_val:.4f} | lr={current_lr:.2e} | "
                f"time={epoch_time:.0f}s | ETA {eta_str} "
                f"{'[BEST]' if is_best else ''}")

            # Save stats locally
            with open(stats_path, 'w') as f:
                json.dump(all_stats, f, indent=2)

            # Save checkpoint locally
            state = {
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss':        best_val_loss,
                'train_stats':          all_stats,
                'vocab_size':           vocab_size,
                'model_id':             model_id,
            }
            ckpt_path = save_checkpoint(state, ckpt_dir, epoch + 1, is_best)
            log(f"  Checkpoint saved: {ckpt_path.name}")

            # ---- Upload checkpoint to R2 ----
            push_checkpoint_to_r2(model_id, ckpt_path, is_best)

    except KeyboardInterrupt:
        log("\n[KeyboardInterrupt] Saving emergency checkpoint...")
        interrupted[0] = True

    finally:
        # Always save + push on exit (interrupt or normal)
        try:
            state = {
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss':        best_val_loss,
                'train_stats':          all_stats,
                'vocab_size':           vocab_size,
                'model_id':             model_id,
            }
            emergency_path = ckpt_dir / 'emergency_checkpoint.pt'
            torch.save(state, emergency_path)
            log(f"Emergency checkpoint saved: {emergency_path}")
            push_checkpoint_to_r2(model_id, emergency_path, False)
        except Exception as e:
            log(f"Emergency save failed: {e}")

        log_file.close()

    log(f"\nTraining Model {model_id} complete.")
    log(f"Best val loss : {best_val_loss:.4f} ({best_val_loss/math.log(2):.4f} bpc)")
    log(f"Stats saved   : {stats_path}")

    return all_stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train BitNet compression models')
    parser.add_argument('--model', type=str, default='both',
                        choices=['1', '2', 'both'],
                        help='Which model to train: 1, 2, or both')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile()')
    args = parser.parse_args()

    global EPOCHS, BATCH_SIZE
    if args.epochs:
        EPOCHS = args.epochs
    if args.batch_size:
        BATCH_SIZE = args.batch_size

    print(f"Training configuration:")
    print(f"  epochs     = {EPOCHS}")
    print(f"  batch_size = {BATCH_SIZE}")
    print(f"  lr         = {LR}")
    print(f"  model      = {args.model}")

    if args.model in ('1', 'both'):
        train_model(1)

    if args.model in ('2', 'both'):
        train_model(2)

    print("\nAll training complete.")


if __name__ == '__main__':
    main()
