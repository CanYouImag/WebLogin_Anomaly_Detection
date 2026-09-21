"""
Download and prepare UNSW-NB15 for binary intrusion detection experiments.

The Balanced CSV (~300k rows) is the recommended version for CPU-only research.

Usage:
    python scripts/prepare_unsw_nb15.py
    python scripts/prepare_unsw_nb15.py --url <custom_url_to_csv>

After running, files land in data/raw/UNSW-NB15/UNSW-NB15-Balanced.csv.
"""
import argparse
import os
import sys
import urllib.request

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE, "data", "raw", "UNSW-NB15")

DEFAULT_URLS = [
    "http://205.174.165.80/CICDataset/UNSW-NB15/UNSW-NB15-Balanced.csv",
    "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
]

FILENAMES = [
    "UNSW-NB15-Balanced.csv",
    "UNSW_NB15_training-set.csv",
]


def try_download(url, out_path):
    print(f"Trying: {url}")
    try:
        urllib.request.urlretrieve(url, out_path, reporthook=_progress)
        print(f"\n  Saved to: {out_path}")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = downloaded / total_size * 100 if total_size > 0 else 0
    mb = downloaded / 1024 / 1024
    print(f"\r  {mb:.1f}MB / {total_size/1024/1024:.1f}MB  ({pct:.0f}%)", end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=None,
                        help="Direct URL to a UNSW-NB15 CSV file")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Check if already present
    for fn in FILENAMES:
        path = os.path.join(OUT_DIR, fn)
        if os.path.exists(path):
            print(f"Already exists: {path}")
            print("Done.")
            return

    if args.url:
        urls = [args.url]
    else:
        urls = DEFAULT_URLS

    # Try each default URL
    target = os.path.join(OUT_DIR, "UNSW-NB15-Balanced.csv")
    for url in urls:
        if try_download(url, target):
            break
    else:
        print("\n" + "=" * 60)
        print("AUTO-DOWNLOAD FAILED")
        print("=" * 60)
        print("Please download UNSW-NB15 manually:")
        print("  1. Visit: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
        print("  2. Download one of:")
        print("     - UNSW-NB15-Balanced.csv  (300k rows, binary, recommended)")
        print("     - UNSW_NB15_training-set.csv  (2.5M rows, binary)")
        print(f"  3. Place it in: {OUT_DIR}/")
        sys.exit(1)

    print("\nUNSW-NB15 is ready.")
    print("Run experiments with:")
    print("  python src/run_experiments.py --datasets cicids2017 unsw_nb15")


if __name__ == "__main__":
    main()