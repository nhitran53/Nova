"""Shared fail-closed artifact, split and CUDA checks for offline experiments."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import torch

from .core import base_model_dir, models_dir, resolve_device


def is_v3(data_dir):
    path = Path(data_dir)
    manifest = path / 'manifest.json'
    return (path.name.startswith('v3-trip') or (manifest.is_file() and
            json.loads(manifest.read_text()).get('schema_version', '').startswith('v3-trip')))



def data_contract(data_dir):
    """Recognize explicit trip schemas; directory fallback retains v3's original behavior."""
    root = Path(data_dir)
    manifest = root / 'manifest.json'
    schema = json.loads(manifest.read_text()).get('schema_version') if manifest.is_file() else None
    # Dataset revisions share one tooling/checkpoint compatibility contract.
    if schema in ('v4-trip-1', 'v4-trip-1.1', 'v4-trip-1.2'):
        return 'v4-trip-1'
    if is_v3(root):
        return 'v3-trip-1'
    if root.name.startswith('v4-trip') or (schema and schema.startswith('v4-trip')):
        raise ValueError('v4 requires manifest schema_version v4-trip-1, v4-trip-1.1 or v4-trip-1.2')
    return 'legacy'


def is_trip(data_dir):
    return data_contract(data_dir) in ('v3-trip-1', 'v4-trip-1')


def check_checkpoint_contract(meta, data_dir):
    actual = data_contract(data_dir)
    expected = meta.get('data_contract')
    if (actual == 'v4-trip-1' or expected in ('v3-trip-1', 'v4-trip-1')) and actual != expected:
        raise ValueError(f'checkpoint/data contract mismatch: {expected} vs {actual}')


V4_CAVEAT = 'agent-authored synthetic, not safety evidence'

def calibration_input(data_dir, explicit=None):
    """Trip contracts never fall back to dev, even when calibration is missing."""
    root = Path(data_dir)
    path = Path(explicit) if explicit else root / ('calibration.jsonl' if is_trip(root) else 'dev.jsonl')
    if not path.is_file():
        raise ValueError(f'calibration input missing: {path}')
    if is_trip(root):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError('empty calibration input')
        # Match data/v3-trip/release.py's lexical leakage contract; this cannot prove semantic independence.
        def normalize(text):
            return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))

        texts = {normalize(row['text']) for row in rows}
        for split in ('train', 'dev', 'threshold', 'test'):
            other = root / f'{split}.jsonl'
            if path.resolve() == other.resolve():
                raise ValueError(f'calibration must be independent of {split}')
            other_texts = {normalize(json.loads(line)['text']) for line in other.read_text().splitlines() if line.strip()}
            if texts & other_texts:
                raise ValueError(f'calibration overlaps {split}')
            other_rows = [json.loads(line) for line in other.read_text().splitlines() if line.strip()]
            for key in ('family_id', 'template_id', 'procedure_family'):
                if {row[key] for row in rows if row.get(key)} & {row[key] for row in other_rows if row.get(key)}:
                    raise ValueError(f'calibration {key} overlaps {split}')
    return path


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reserve_checkpoint(path):
    path = Path(path)
    if path.name in ('v1', 'v2', 'default', 'ModernBERT-base') or path.resolve() in (
            models_dir().resolve(), base_model_dir().resolve()):
        raise ValueError(f'protected checkpoint destination: {path}')
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_report(path, report):
    """Exclusive create: neither another split nor a rerun may overwrite evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2) + '\n'
    with path.open('x') as stream:
        stream.write(content)
    return path


def checked_device(device, require_cuda=False):
    resolved = resolve_device(device)
    if require_cuda and (resolved.type != 'cuda' or not torch.cuda.is_available()):
        raise ValueError('--require-cuda requires available CUDA; CPU/MPS fallback is forbidden')
    return resolved


def cuda_probe(device="cuda"):
    """Actual CUDA math/backward, not a GPU-name compatibility claim."""
    if not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; smoke refuses CPU training')
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(index)
    x = torch.randn(32, 32, device=device, requires_grad=True)
    (x @ x.T).square().mean().backward()
    torch.cuda.synchronize(device)
    return {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'gpu': prop.name, 'device_index': index, 'capability': list(torch.cuda.get_device_capability(index)),
            'vram_bytes': prop.total_memory, 'compiled_arches': torch.cuda.get_arch_list(),
            'cuda_tensor_backward': True}
