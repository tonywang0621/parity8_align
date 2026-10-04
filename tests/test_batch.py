import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import run_experiments as batch
import experiment_runner as runner
import experiment_tasks as tasks


@pytest.fixture
def project(tmp_path, monkeypatch):
    shutil.copytree(ROOT / 'src', tmp_path / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    # Only this copied subprocess entry gets a tiny test budget.
    source = tmp_path / 'src/parity8_align_20260924.py'
    text = source.read_text().replace('n_runs=40, seed_base=42, emax=2500, ex=10000',
                                    'n_runs=2, seed_base=42, emax=1, ex=2')
    source.write_text(text)
    for module in (batch, runner, tasks):
        monkeypatch.setattr(module, 'PROJECT_ROOT', tmp_path)
    return tmp_path


def spec(name='parity', task='parity'):
    return {'name': name, 'task': task, 'fams': ['2LNN'], 'p': [0], 'data_dir': 'data'}


def test_plan_defaults_and_dry_run(project, capsys):
    path = project / 'experiments.json'
    path.write_text(json.dumps({'experiments': [{'name': task, 'task': task} for task in tasks.TASKS]}))
    plan = batch.read_plan(path, 'data')
    assert len(plan) == 6 and plan[0]['p'] == list(range(9)) and len(plan[0]['fams']) == 6
    assert batch.main(['--dry-run']) == 0
    assert '新實驗' in capsys.readouterr().out
    assert not (project / 'run').exists() and not (project / 'data').exists()


@pytest.mark.parametrize('entries', [[], [{'name': 'bad', 'task': 'unknown'}],
    [{'name': '../x', 'task': 'parity'}], [{'name': 'x', 'task': 'parity', 'p': [True]}],
    [{'name': 'x', 'task': 'parity', 'fams': ['bad']}],
    [{'name': 'x', 'task': 'parity'}, {'name': 'x', 'task': 'parity'}]])
def test_invalid_plan_before_any_training(project, entries):
    path = project / 'experiments.json'
    path.write_text(json.dumps({'experiments': entries}))
    with pytest.raises(runner.RunError):
        batch.read_plan(path, 'data')
    assert not (project / 'run').exists()


def test_repeat_same_batch_and_append(project):
    plan = [spec()]
    assert batch.execute_batch(plan, 'demo') == 0
    run = batch.discover(plan[0], 'demo')
    checkpoint = run / 'checkpoints/2LNN_p0_latest.pt'
    stamp, contents = checkpoint.stat().st_mtime_ns, checkpoint.read_bytes()
    assert batch.execute_batch(plan, 'demo') == 0
    assert batch.discover(plan[0], 'demo') == run
    assert checkpoint.stat().st_mtime_ns == stamp and checkpoint.read_bytes() == contents
    assert batch.execute_batch([*plan, spec('linear', 'linearly_separable')], 'demo') == 0
    assert checkpoint.stat().st_mtime_ns == stamp
    state = json.loads((project / 'run/batches/demo/batch.json').read_text())
    assert all(x['status'] == 'completed' for x in state['experiments'].values())
    # New batch is independent.
    assert batch.execute_batch(plan, 'fresh') == 0
    assert batch.discover(plan[0], 'fresh') != run


def test_changed_settings_and_missing_run_refused(project):
    plan = [spec()]
    batch.execute_batch(plan, 'demo')
    with pytest.raises(runner.RunError, match='設定不同'):
        batch.execute_batch([dict(spec(), p=[2])], 'demo')
    shutil.rmtree(batch.discover(plan[0], 'demo'))
    with pytest.raises(runner.RunError, match='原 run 遺失'):
        batch.execute_batch(plan, 'demo')


def test_ambiguous_and_partial_runs_refused(project):
    base = project / 'run/parity/demo__parity'
    (base / 'first').mkdir(parents=True)
    with pytest.raises(runner.RunError, match='不完整 run'):
        batch.execute_batch([spec()], 'demo')
    (base / 'second').mkdir()
    with pytest.raises(runner.RunError, match='多個 run'):
        batch.execute_batch([spec()], 'demo')


def test_child_failure_stops_remaining_tasks(project, monkeypatch):
    class Child:
        def __init__(self, *args, **kwargs):
            self.returncode = None
        def wait(self):
            self.returncode = 7
            return 7
        def poll(self):
            return self.returncode
    monkeypatch.setattr(batch.subprocess, 'Popen', Child)
    assert batch.execute_batch([spec(), spec('next')], 'failure') == 7
    state = runner.read_json(project / 'run/batches/failure/batch.json')
    assert state['experiments']['parity']['status'] == 'stopped'
    assert 'next' not in state['experiments']
    assert not (project / 'run/batches/failure/writer.lock').exists()


def test_live_batch_lock_refused(project):
    directory = project / 'run/batches/demo'
    directory.mkdir(parents=True)
    with runner.run_lock(directory):
        with pytest.raises(runner.RunError, match='鎖定'):
            batch.execute_batch([spec()], 'demo')


def test_real_nohup_style_signal_forward_and_restart(project):
    source = project / 'src/parity8_align_20260924.py'
    text = source.read_text().replace('emax=1, ex=2', 'emax=10, ex=60')
    text = text.replace('if __name__ == "__main__":', '''if __name__ == "__main__":
    original_step = torch.optim.Adam.step
    def slow_step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        time.sleep(.03)
        return result
    torch.optim.Adam.step = slow_step
''')
    source.write_text(text)
    (project / 'experiments.json').write_text(json.dumps({'experiments': [{'name': 'parity', 'task': 'parity', 'fams': ['2LNN'], 'p': [0]}]}))
    cmd = [sys.executable, '-u', str(project / 'src/run_experiments.py'), '--batch', 'signal', '--checkpoint-every', '2']
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    log = project / 'test.log'
    with log.open('w') as output:
        process = subprocess.Popen(cmd, cwd=project.parent, stdout=output, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                paths = list(project.glob('run/parity/signal__parity/*/progress.json'))
                if paths:
                    state = json.loads(paths[0].read_text())
                    if state.get('cells', {}).get('2LNN_p0', {}).get('saved_epoch', 0) >= 2:
                        break
                if process.poll() is not None:
                    pytest.fail(log.read_text())
                time.sleep(.02)
            else:
                pytest.fail('timeout: ' + log.read_text())
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=20) == 143
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
    run = paths[0].parent
    saved = runner.unpack_checkpoint((run / 'checkpoints/2LNN_p0_latest.pt').read_bytes())
    assert 2 <= saved['completed_epoch'] < 60 and not saved['completed']
    assert not (run / 'writer.lock').exists()
    assert not (project / 'run/batches/signal/writer.lock').exists()
    resumed = subprocess.run(cmd, cwd=project.parent, env=env, capture_output=True, text=True, timeout=30)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert len(list((project / 'run/parity/signal__parity').iterdir())) == 1
    assert json.loads((run / 'progress.json').read_text())['complete']
