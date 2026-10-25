"""Exercise shell failure propagation without running any model/training implementation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).parents[1] / 'scripts/v3-trip'
SPEC = importlib.util.spec_from_file_location('handoff_transfer', SCRIPTS / 'transfer.py')
transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)


@pytest.mark.parametrize('train_code,calibration_code,expected', [(17, 0, 17), (0, 19, 19)])
def test_run_preserves_failure_and_collects_acceptance(tmp_path, train_code, calibration_code, expected):
    run = tmp_path / 'run'
    run.mkdir()
    # Minimal independently verifiable source snapshot for the shell-level test.
    snapshot = run / 'source'
    (snapshot / 'router').mkdir(parents=True)
    data = b'fixture\n'
    (snapshot / 'router/core.py').write_bytes(data)
    (snapshot / 'TRANSFER-MANIFEST.json').write_text(json.dumps({'format': 1, 'files': {
        'router/core.py': {'sha256': transfer.digest(data), 'bytes': len(data)}}}))
    # Stub interpreter never executes modules; Python '-' validation is separately source-reviewed.
    stub = tmp_path / 'python'
    stub.write_text('#!/usr/bin/env python3\n' + f'''
import json,sys
from pathlib import Path
if sys.argv[1] == '-':
    sys.stdin.read()
    sys.exit(0)
module=sys.argv[2]
print('stub', module)
if module == 'router.train':
    sys.exit({train_code})
if module == 'router.calibrate':
    sys.exit({calibration_code})
if module == 'router.acceptance':
    report=Path(sys.argv[sys.argv.index('--report')+1])
    report.write_text(json.dumps({{'passed': False, 'failures': ['stub failure']}}))
    sys.exit(1)
''')
    stub.chmod(0o700)
    env = dict(os.environ, V3_PY=str(stub), V3_RUN=str(run), V3_BASE=str(tmp_path / 'base'),
               V3_RISK_TARGET='0.01', HOME=str(tmp_path / 'home'))
    result = subprocess.run(['bash', str(SCRIPTS / 'run.sh'), 'train'], env=env, cwd=REPO,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == expected, result.stderr
    assert json.loads((run / 'acceptance.json').read_text())['passed'] is False
    assert 'acceptance\t1' in (run / 'exit-codes.tsv').read_text()
    commands = (run / 'commands.txt').read_text()
    assert 'router.acceptance' in commands
    if train_code == 0:
        evaluations = [line for line in commands.splitlines() if 'router.evaluate' in line]
        assert len(evaluations) == 3
        for split in ('test', 'heldout_scenarios', 'regression'):
            command = next(line for line in evaluations if f'--split {split} ' in line)
            assert ('--label-field expected_label' in command) == (split != 'test')
            if split == 'test':
                assert '--label-field' not in command


def test_evidence_inventory_and_corruption(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'acceptance.json').write_text('{"passed": false}\n')
    # Deliberately forbidden content must not be opened.
    (run / '.env').write_text('not evidence')
    checkpoint = tmp_path / 'missing-candidate'
    command = ['python3', str(SCRIPTS / 'evidence.py'), '--run-dir', str(run), '--checkpoint', str(checkpoint)]
    subprocess.run(command, check=True, cwd=REPO, capture_output=True)
    manifest = json.loads((run / 'RETURN-MANIFEST.json').read_text())
    assert manifest['candidate_present'] is False
    assert set(manifest['evidence']) == {'acceptance.json'}
    subprocess.run(command + ['--verify'], check=True, cwd=REPO, capture_output=True)
    (run / 'acceptance.json').write_text('changed')
    result = subprocess.run(command + ['--verify'], cwd=REPO, capture_output=True)
    assert result.returncode != 0


@pytest.mark.parametrize('alias', ['dotdot', 'symlink-parent'])
def test_run_rejects_resolved_repository_evidence_before_writes(tmp_path, alias):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'router/scripts/v3-trip').mkdir(parents=True)
    (repo / 'router/scripts/v3-trip/run.sh').touch()
    if alias == 'dotdot':
        run = repo / '..' / 'repo' / 'new-run'
    else:
        parent = tmp_path / 'external-alias'
        parent.symlink_to(repo, target_is_directory=True)
        run = parent / 'new-run'
    env = dict(os.environ, V3_PY='/unused/python', V3_BASE='/unused/base', V3_RUN=str(run))
    result = subprocess.run(['bash', str(SCRIPTS.resolve() / 'run.sh'), 'smoke'], cwd=repo,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert 'Evidence must be outside repo' in result.stderr
    assert not (repo / 'new-run').exists()
    assert list(repo.iterdir()) == [repo / 'router']


@pytest.mark.parametrize('target,mutation', [
    ('source/router/core.py', 'corrupt'),
    ('source/router/core.py', 'delete'),
    ('source/TRANSFER-MANIFEST.json', 'delete'),
    ('source/TRANSFER-MANIFEST.json', 'corrupt'),
    ('source.tar.gz', 'corrupt'),
    ('source.tar.gz', 'delete'),
    ('source', 'delete-tree'),
])
def test_evidence_return_verifies_source_bytes(tmp_path, target, mutation):
    import shutil

    root = tmp_path / 'repo'
    (root / 'router').mkdir(parents=True)
    (root / 'router/core.py').write_text('fixture = True\n')
    # Extracted-source provenance avoids invoking any real checkout operations.
    (root / 'TRANSFER-MANIFEST.json').write_text(json.dumps({
        'snapshot': {'head': 'fixture', 'branch': 'fixture'}, 'files': {}}))
    run = tmp_path / 'run'
    run.mkdir()
    transfer.build(root, run / 'source')
    command = ['python3', str(SCRIPTS / 'evidence.py'), '--run-dir', str(run),
               '--checkpoint', str(tmp_path / 'absent-checkpoint')]
    subprocess.run(command, check=True, cwd=REPO, capture_output=True, timeout=30)
    manifest = json.loads((run / 'RETURN-MANIFEST.json').read_text())
    assert {'source/TRANSFER-MANIFEST.json', 'source.tar.gz'} <= manifest['evidence'].keys()
    subprocess.run(command + ['--verify'], check=True, cwd=REPO, capture_output=True, timeout=30)
    path = run / target
    if mutation == 'delete-tree':
        shutil.rmtree(path)
    elif mutation == 'delete':
        path.unlink()
    else:
        path.write_bytes(b'corrupted fixture')
    result = subprocess.run(command + ['--verify'], cwd=REPO, capture_output=True, timeout=30)
    assert result.returncode != 0


def test_run_requires_repository_root(tmp_path):
    result = subprocess.run(['bash', str(SCRIPTS.resolve() / 'run.sh'), 'smoke'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert 'Run from repository root' in result.stderr
    assert list(tmp_path.iterdir()) == []
