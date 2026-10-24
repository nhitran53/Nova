"""Download answerdotai/ModernBERT-base (safetensors only) into a fresh dir.

Usage: python download_base.py [target_dir]   (default: router/models/ModernBERT-base)
The target dir must be new/empty; downloaded files are untrusted data, so keep
scripts out of it and load with `python -I`.
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "answerdotai/ModernBERT-base"
DEFAULT_DIR = Path(__file__).parent / "models" / "ModernBERT-base"


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    if target.exists() and any(target.iterdir()):
        sys.exit(f"{target} is not empty; refusing to download into it")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO_ID,
        local_dir=str(target),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE*"],
    )
    print(f"downloaded {REPO_ID} -> {target}")


if __name__ == "__main__":
    main()
