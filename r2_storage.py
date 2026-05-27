"""
r2_storage.py — Cloudflare R2 storage utility (S3-compatible via boto3).

Loads credentials from .env or environment variables.

R2 layout:
  data/encoded.dat          ← training corpus
  data/vocab.json
  data/metadata.json
  checkpoints/model1/latest.pt
  checkpoints/model1/best_model.pt
  checkpoints/model2/latest.pt
  checkpoints/model2/best_model.pt
  results/                  ← benchmark outputs
"""

import os
import sys
import time
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
except ImportError:
    pass

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _get_client():
    """Create boto3 S3 client pointed at Cloudflare R2."""
    endpoint = os.environ.get('R2_ENDPOINT', '')
    # Strip bucket name from endpoint if present
    # e.g. https://xxx.r2.cloudflarestorage.com/bucket → https://xxx.r2.cloudflarestorage.com
    bucket = os.environ.get('R2_BUCKET', '')
    if endpoint.endswith('/' + bucket):
        endpoint = endpoint[: -(len(bucket) + 1)]

    return boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
        region_name='auto',
        config=Config(
            retries={'max_attempts': 5, 'mode': 'adaptive'},
            max_pool_connections=10,
        )
    )


def _bucket():
    return os.environ.get('R2_BUCKET', '')


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_file(local_path: str, r2_key: str, show_progress: bool = True) -> bool:
    """
    Upload a local file to R2.
    Uses multipart upload automatically for large files.
    Returns True on success.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        print(f"[R2] Upload skipped — file not found: {local_path}")
        return False

    size_mb = local_path.stat().st_size / 1024 / 1024
    print(f"[R2] Uploading {local_path.name} ({size_mb:.1f} MB) → r2://{_bucket()}/{r2_key}")

    client = _get_client()
    t0 = time.time()

    try:
        if show_progress and size_mb > 10:
            from boto3.s3.transfer import TransferConfig
            import threading

            uploaded = [0]
            total = local_path.stat().st_size

            def progress(chunk):
                uploaded[0] += chunk
                pct = uploaded[0] / total * 100
                mb = uploaded[0] / 1024 / 1024
                print(f"\r  {mb:.0f}/{size_mb:.0f} MB ({pct:.0f}%)", end='', flush=True)

            client.upload_file(
                str(local_path), _bucket(), r2_key,
                Callback=progress,
                Config=TransferConfig(
                    multipart_threshold=64 * 1024 * 1024,
                    multipart_chunksize=64 * 1024 * 1024,
                    max_concurrency=8,
                )
            )
            print()
        else:
            client.upload_file(str(local_path), _bucket(), r2_key)

        elapsed = time.time() - t0
        speed = size_mb / elapsed if elapsed > 0 else 0
        print(f"[R2] Upload done in {elapsed:.1f}s ({speed:.1f} MB/s)")
        return True

    except Exception as e:
        print(f"[R2] Upload failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(r2_key: str, local_path: str, show_progress: bool = True) -> bool:
    """
    Download a file from R2 to local path.
    Returns True on success, False if key doesn't exist.
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    client = _get_client()

    # Check if key exists
    try:
        meta = client.head_object(Bucket=_bucket(), Key=r2_key)
        size_mb = meta['ContentLength'] / 1024 / 1024
    except ClientError as e:
        if e.response['Error']['Code'] in ('404', 'NoSuchKey'):
            print(f"[R2] Not found: r2://{_bucket()}/{r2_key}")
            return False
        raise

    print(f"[R2] Downloading r2://{_bucket()}/{r2_key} ({size_mb:.1f} MB) → {local_path}")
    t0 = time.time()

    try:
        if show_progress and size_mb > 10:
            downloaded = [0]
            total = meta['ContentLength']

            def progress(chunk):
                downloaded[0] += chunk
                pct = downloaded[0] / total * 100
                mb = downloaded[0] / 1024 / 1024
                print(f"\r  {mb:.0f}/{size_mb:.0f} MB ({pct:.0f}%)", end='', flush=True)

            from boto3.s3.transfer import TransferConfig
            client.download_file(
                _bucket(), r2_key, str(local_path),
                Callback=progress,
                Config=TransferConfig(
                    multipart_threshold=64 * 1024 * 1024,
                    multipart_chunksize=64 * 1024 * 1024,
                    max_concurrency=8,
                )
            )
            print()
        else:
            client.download_file(_bucket(), r2_key, str(local_path))

        elapsed = time.time() - t0
        speed = size_mb / elapsed if elapsed > 0 else 0
        print(f"[R2] Download done in {elapsed:.1f}s ({speed:.1f} MB/s)")
        return True

    except Exception as e:
        print(f"[R2] Download failed: {e}")
        if local_path.exists():
            local_path.unlink()
        return False


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def upload_training_data(data_dir: str = 'data'):
    """Upload encoded.dat, vocab.json, metadata.json to R2."""
    data_dir = Path(data_dir)
    files = {
        data_dir / 'encoded.dat':   'data/encoded.dat',
        data_dir / 'vocab.json':    'data/vocab.json',
        data_dir / 'metadata.json': 'data/metadata.json',
    }
    results = {}
    for local, key in files.items():
        results[key] = upload_file(str(local), key)
    return results


def download_training_data(data_dir: str = 'data') -> bool:
    """Download training corpus from R2 if not present locally."""
    data_dir = Path(data_dir)
    all_ok = True
    for fname, key in [
        ('encoded.dat',   'data/encoded.dat'),
        ('vocab.json',    'data/vocab.json'),
        ('metadata.json', 'data/metadata.json'),
    ]:
        local = data_dir / fname
        if local.exists():
            print(f"[R2] {fname} already exists locally, skipping download.")
        else:
            ok = download_file(key, str(local))
            if not ok:
                all_ok = False
    return all_ok


def upload_checkpoint(local_path: str, model_id: int, name: str):
    """Upload a checkpoint file to R2."""
    key = f'checkpoints/model{model_id}/{name}'
    return upload_file(local_path, key)


def download_checkpoint(model_id: int, name: str, local_path: str) -> bool:
    """Download a checkpoint from R2."""
    key = f'checkpoints/model{model_id}/{name}'
    return download_file(key, local_path)


def upload_results(results_dir: str = 'results'):
    """Upload all files in results/ to R2."""
    results_dir = Path(results_dir)
    for f in results_dir.iterdir():
        if f.is_file():
            upload_file(str(f), f'results/{f.name}', show_progress=False)


def key_exists(r2_key: str) -> bool:
    """Check if a key exists in R2."""
    try:
        _get_client().head_object(Bucket=_bucket(), Key=r2_key)
        return True
    except ClientError:
        return False


# ---------------------------------------------------------------------------
# CLI — python r2_storage.py upload/download ...
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='R2 storage utility')
    sub = parser.add_subparsers(dest='cmd')

    p = sub.add_parser('upload-data',   help='Upload training data to R2')
    p = sub.add_parser('download-data', help='Download training data from R2')

    p2 = sub.add_parser('upload', help='Upload a file')
    p2.add_argument('local')
    p2.add_argument('key')

    p3 = sub.add_parser('download', help='Download a file')
    p3.add_argument('key')
    p3.add_argument('local')

    args = parser.parse_args()

    if args.cmd == 'upload-data':
        upload_training_data()
    elif args.cmd == 'download-data':
        download_training_data()
    elif args.cmd == 'upload':
        upload_file(args.local, args.key)
    elif args.cmd == 'download':
        download_file(args.key, args.local)
    else:
        parser.print_help()
