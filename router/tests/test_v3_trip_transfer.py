"""Source-named transfer safety tests; stdlib-only builder, no GPU or model."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

SPEC = importlib.util.spec_from_file_location('v3_transfer', Path(__file__).parents[1] / 'scripts/v3-trip/transfer.py')
transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)


def source(tmp_path):
    root = tmp_path / 'repo'
    (root / 'router').mkdir(parents=True)
    (root / 'router/core.py').write_text('original = True\n')
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture'], check=True)
    return root


def test_dirty_source_capture_and_exclusions(tmp_path):
    root = source(tmp_path)
    (root / 'router/core.py').write_text('dirty = True\n')
    for name in ['router/.env', 'router/.env.local', 'router/reports/run.json', 'router/.venv/config.json', 'router/__pycache__/x.py', 'unrelated/private.py']:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('never copied')
    dest = tmp_path / 'transfer'
    result = transfer.build(root, dest)
    assert (dest / 'router/core.py').read_text() == 'dirty = True\n'
    assert result['files'] == 1
    assert transfer.verify_archive(result['archive'])['files'] == 1
    manifest = json.loads((dest / 'TRANSFER-MANIFEST.json').read_text())
    assert any('router/core.py' in x for x in manifest['snapshot']['dirty_paths'])


def test_extracted_snapshot_can_be_repackaged_without_git(tmp_path):
    first = transfer.build(source(tmp_path), tmp_path / 'first')
    (Path(first['folder']) / 'router/core.py').write_text('teammate edit\n')
    second = transfer.build(first['folder'], tmp_path / 'second')
    manifest = json.loads((Path(second['folder']) / 'TRANSFER-MANIFEST.json').read_text())
    assert manifest['snapshot']['extracted_source'] is True
    assert manifest['snapshot']['dirty_paths'] == ['router/core.py']
    assert transfer.verify_archive(second['archive'])['files'] == 1


def test_source_symlink_rejected(tmp_path):
    root = source(tmp_path)
    (root / 'router/link.py').symlink_to(root / 'router/core.py')
    with pytest.raises(ValueError, match='symlink'):
        transfer.build(root, tmp_path / 'transfer')


def test_manifest_corruption(tmp_path):
    result = transfer.build(source(tmp_path), tmp_path / 'transfer')
    (Path(result['folder']) / 'router/core.py').write_text('corrupted')
    with pytest.raises(ValueError, match='hash/size'):
        transfer.verify_folder(result['folder'])


@pytest.mark.parametrize('name,kind', [('../escape.py', 'file'), ('router/.env', 'file'), ('router/reports/run.json', 'file'), ('router/core.py', 'symlink')])
def test_archive_exclusions_and_links(tmp_path, name, kind):
    archive = tmp_path / 'bad.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        member = tarfile.TarInfo(name)
        if kind == 'symlink':
            member.type = tarfile.SYMTYPE
            member.linkname = '/etc/passwd'
            tar.addfile(member)
        else:
            member.size = 1
            tar.addfile(member, io.BytesIO(b'x'))
    with pytest.raises(ValueError):
        transfer.verify_archive(archive)


def test_credentials_fail_closed_without_disclosure(tmp_path):
    root = source(tmp_path)
    token = 'hf_' + 'a' * 32
    (root / 'router/core.py').write_text(token)
    with pytest.raises(ValueError) as exc:
        transfer.build(root, tmp_path / 'transfer')
    assert token not in str(exc.value)
    assert not (tmp_path / 'transfer').exists()


@pytest.mark.parametrize('excluded', ['router/.env.fixture', 'credentials/fake.json'])
def test_verify_folder_rejects_entire_inventory_before_payload_read(tmp_path, monkeypatch, excluded):
    folder = tmp_path / 'transfer'
    (folder / 'router').mkdir(parents=True)
    (folder / 'router/core.py').write_text('fixture = True\n')
    (folder / 'TRANSFER-MANIFEST.json').write_text(json.dumps({'format': 1, 'files': {}}))
    # Empty synthetic excluded files, never real credentials or environment content.
    path = folder / excluded
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    reads = []
    original = transfer.checked_read

    def spy(root, name):
        reads.append(name)
        assert name == 'TRANSFER-MANIFEST.json', 'payload read before allowlist validation'
        return original(root, name)

    monkeypatch.setattr(transfer, 'checked_read', spy)
    with pytest.raises(ValueError, match='excluded folder path'):
        transfer.verify_folder(folder)
    assert reads == ['TRANSFER-MANIFEST.json']
