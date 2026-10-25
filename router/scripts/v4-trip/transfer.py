#!/usr/bin/env python3
"""Stdlib-only dirty-worktree source transfer builder and verifier. Never follows links."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

EXACT = {'README.md', 'NOTES.md', 'router/README.md', 'router/pyproject.toml', 'router/uv.lock', 'simulator_rules.py',
         'research/trip-feasibility-and-clarification.md'}
# Explicit source scopes; no recursive whole-repository copy.
SCOPES = {'router': {'.py', '.json'}, 'router/tests': {'.py'}, 'router/docs': {'.md'},
          'router/scripts/v3-trip': {'.py', '.sh'},
          'router/scripts/v4-trip': {'.py', '.sh', '.md'},
          'router/data/v4-trip': {'.py', '.json', '.jsonl', '.md'}, 'router/data': {'.py', '.jsonl'},
          'router/data/v2': {'.jsonl'}, 'router/data/v3-trip': {'.py', '.json', '.jsonl', '.md'}}
FORBIDDEN = {'.git', '.pi', '.venv', 'venv', '__pycache__', '.pytest_cache', 'logs', 'reports',
             'credentials', 'secrets', 'node_modules'}
SECRET = re.compile(rb'(?:\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|(?i:(?:api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*[\"\x27][A-Za-z0-9_+/=-]{16,}[\"\x27]))')


def allowed(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p) != name or not p.parts:
        return False
    if any(x in FORBIDDEN or x.startswith('.env') or x.startswith('.') for x in p.parts):
        return False
    return name in EXACT or p.suffix in SCOPES.get(str(p.parent), set())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def scan(data):
    if SECRET.search(data):
        raise ValueError('credential-like content detected; transfer rejected (match suppressed)')


def checked_read(root, name):
    path = root
    for part in PurePosixPath(name).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('symlink rejected')
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('non-file or path escape rejected')
    data = path.read_bytes()
    scan(data)
    return data


def verify_manifest(manifest, files):
    if manifest.get('format') != 1 or not isinstance(manifest.get('files'), dict) or not manifest['files']:
        raise ValueError('invalid manifest')
    if set(files) != set(manifest['files']):
        raise ValueError('manifest inventory mismatch')
    for name, data in files.items():
        if not allowed(name):
            raise ValueError('excluded archive path')
        scan(data)
        if manifest['files'][name] != {'sha256': digest(data), 'bytes': len(data)}:
            raise ValueError('manifest hash/size mismatch')
    return {'files': len(files), 'bytes': sum(map(len, files.values()))}


def verify_folder(folder):
    folder = Path(folder)
    manifest = json.loads(checked_read(folder, 'TRANSFER-MANIFEST.json'))
    names = []
    for directory, dirs, files in os.walk(folder, followlinks=False):
        for item in dirs + files:
            path = Path(directory) / item
            if path.is_symlink():
                raise ValueError('symlink rejected')
        names.extend(str((Path(directory) / f).relative_to(folder)) for f in files)
    payload_names = [n for n in names if n != 'TRANSFER-MANIFEST.json']
    # Validate the entire inventory before opening even the first payload file.
    if any(not allowed(n) for n in payload_names):
        raise ValueError('excluded folder path')
    payload = {n: checked_read(folder, n) for n in payload_names}
    return verify_manifest(manifest, payload)


def verify_archive(archive):
    with tarfile.open(archive, 'r:gz') as tar:
        files = {}
        for member in tar:
            if not member.isfile() or member.name in files:
                raise ValueError('non-file/duplicate archive member rejected')
            if member.name != 'TRANSFER-MANIFEST.json' and not allowed(member.name):
                raise ValueError('excluded archive path')
            files[member.name] = tar.extractfile(member).read()
    manifest = json.loads(files.pop('TRANSFER-MANIFEST.json'))
    return verify_manifest(manifest, files)


def build(root, destination, sealed=None):
    root, destination = Path(root).resolve(), Path(destination).absolute()
    if destination.resolve().is_relative_to(root) or destination.exists() or Path(str(destination) + '.tar.gz').exists():
        raise ValueError('destination must be new and outside repository')
    sealed_hash = None
    if sealed is not None:
        sealed_path = Path(sealed).resolve()
        if sealed_path.is_relative_to(root):
            raise ValueError('sealed input must be external to repository')
        sealed_hash = digest(sealed_path.read_bytes())
    names = set()
    for parent in SCOPES:
        directory = root / parent
        if directory.is_symlink():
            raise ValueError('symlink source scope rejected')
        if directory.is_dir():
            for p in directory.iterdir():
                name = str(p.relative_to(root))
                if allowed(name):
                    names.add(name)
    names.update(n for n in EXACT if (root / n).exists() or (root / n).is_symlink())
    payload = {n: checked_read(root, n) for n in sorted(names)}
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if (root / '.git').exists():
        snapshot = {'head': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
                    'dirty_paths': git('status', '--porcelain', '--untracked-files=all').splitlines()}
    else:
        # An extracted handoff intentionally contains no git history. Preserve its origin
        # and distinguish local edits rather than claiming the old dirty paths are new.
        original = json.loads(checked_read(root, 'TRANSFER-MANIFEST.json'))
        snapshot = {'head': original['snapshot']['head'], 'branch': original['snapshot']['branch'],
                    'origin_dirty_paths': original['snapshot'].get('dirty_paths', []),
                    'extracted_source': True,
                    'dirty_paths': [n for n in sorted(set(payload) | set(original['files']))
                                    if n not in payload or n not in original['files'] or
                                    digest(payload[n]) != original['files'][n]['sha256']]}
    manifest = {'format': 1, 'snapshot': snapshot,
                'files': {n: {'sha256': digest(b), 'bytes': len(b)} for n, b in payload.items()}}
    if sealed_hash is not None:
        manifest['sealed_sha256'] = sealed_hash
    metadata = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    scan(metadata)
    destination.mkdir(parents=True)
    for name, data in payload.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (destination / 'TRANSFER-MANIFEST.json').write_bytes(metadata)
    verify_folder(destination)
    archive = Path(str(destination) + '.tar.gz')
    with tarfile.open(archive, 'x:gz') as tar:
        for path in sorted(p for p in destination.rglob('*') if p.is_file()):
            tar.add(path, arcname=str(path.relative_to(destination)), recursive=False)
    result = verify_archive(archive)
    result.update(folder=str(destination), archive=str(archive), archive_sha256=digest(archive.read_bytes()))
    if sealed_hash is not None:
        result['sealed_sha256'] = sealed_hash
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    b = sub.add_parser('build')
    b.add_argument('--root', default='.')
    b.add_argument('--destination', required=True)
    b.add_argument('--sealed', help='External file: hash only in receipt, never packaged')
    v = sub.add_parser('verify')
    v.add_argument('path')
    args = ap.parse_args()
    if args.action == 'build':
        build(args.root, args.destination, args.sealed)
    else:
        path = Path(args.path)
        print(json.dumps(verify_folder(path) if path.is_dir() else verify_archive(path)))


if __name__ == '__main__':
    main()
