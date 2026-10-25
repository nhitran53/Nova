"""Download ModernBERT weights into a new folder.

Use --revision to pin the source commit.
Record the resolved revision and file hashes for reproducibility.
Load safetensors without remote code.
"""
import argparse
import hashlib
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from router.core import base_model_dir

REPO_ID = "answerdotai/ModernBERT-base"


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", nargs="?", default=str(base_model_dir()))
    ap.add_argument("--revision", default=None, help="commit SHA to pin (default: current head)")
    args = ap.parse_args(argv)
    target = Path(args.target)
    if target.exists() and any(target.iterdir()):
        sys.exit(f"{target} is not empty; refusing to download into it")
    revision = args.revision or HfApi().model_info(REPO_ID).sha
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO_ID,
        revision=revision,
        local_dir=str(target),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE*"],
    )
    print(f"downloaded {REPO_ID} @ {revision} -> {target}")
    for path in sorted(p for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts):
        print(f"sha256 {sha256_of(path)}  {path.relative_to(target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
