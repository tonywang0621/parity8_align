"""Checkpoint correctness tests; all shortened budgets are isolated monkeypatches."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import experiment_tasks as tasks
import experiment_runner as runner
import parity8_align_20260924 as core
import predict_weights
spec = importlib.util.spec_from_file_location('baseline_training', ROOT / 'tests/baseline_training.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(tasks, 'PROJECT_ROOT', tmp_path)
    cfg = dict(core.CFG, n_runs=3, emax=3, ex=7)
    monkeypatch.setattr(core, 'CFG', cfg)
    monkeypatch.setattr(baseline, 'CFG', dict(cfg))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield tmp_path
    finally:
        torch.set_num_threads(previous_threads)


def arguments(**values):
    defaults = dict(task='parity', tag='test', fams=['2LNN'], p=[0], data_dir='data', device='cpu', checkpoint_every=2)
    return argparse.Namespace(**(defaults | values))


def setup_run(**values):
    args = arguments(**values)
    X, y, meta = tasks.load_task(args.task)
    config = runner.new_config(core, args, meta)
    directory = runner.create_run(config, meta)
    return directory, config, torch.tensor(X), torch.tensor(y)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def comparable(row):
    return {k: v for k, v in row.items() if k not in ('wall_sec', 'task', 'n_samples', 'data_sha256')}


@pytest.mark.parametrize('fam', core.FAMS)
@pytest.mark.parametrize('p', [0, 2])
def test_baseline_gradients_adam_and_resume(project, monkeypatch, fam, p):
    directory, config, X, y = setup_run(fams=[fam], p=[p])
    original_step = torch.optim.Adam.step
    traces = []
    def step(opt, *args, **kwargs):
        gradients = [param.grad.detach().clone() for group in opt.param_groups for param in group['params']]
        result = original_step(opt, *args, **kwargs)
        traces.append((gradients, copy.deepcopy(opt.state_dict()),
                       [param.detach().clone() for group in opt.param_groups for param in group['params']]))
        return result
    monkeypatch.setattr(torch.optim.Adam, 'step', step)
    expected = baseline.run_cell(p, fam, X, y, 'cpu')
    old_trace = traces.copy()
    traces.clear()
    full = runner.CellSession(directory, fam, p, config, runner.StopFlag(), 2, 'cpu')
    actual = core.run_cell(p, fam, X, y, 'cpu', session=full)
    equal(old_trace, traces)
    equal(comparable(expected), comparable(actual))
    full_state = full.load(required=True)
    directory2 = runner.create_run(config, tasks.load_task('parity')[2])
    stop = runner.StopFlag()
    def update(name, state):
        if state['completed_epoch'] == 2:
            stop.requested = True
    partial = runner.CellSession(directory2, fam, p, config, stop, 2, 'cpu', update)
    with pytest.raises(runner.TrainingStopped):
        core.run_cell(p, fam, X, y, 'cpu', session=partial)
    restarted = runner.CellSession(directory2, fam, p, config, runner.StopFlag(), 2, 'cpu')
    saved = restarted.load(True)
    assert saved['next_epoch'] == 3 and saved['completed_epoch'] == 2
    random.random(); np.random.rand(); torch.rand(3)
    result = core.run_cell(p, fam, X, y, 'cpu', session=restarted)
    final = restarted.load(True)
    for key in ('parameters', 'optimizer', 'conv', 'acc2500', 'theta', 'completed_epoch', 'phase'):
        equal(full_state[key], final[key])
    equal(comparable(actual), comparable(result))
    assert final['completed'] and final['next_epoch'] == core.CFG['ex'] + 1


def test_rng_restoration(project):
    directory, config, X, y = setup_run()
    stop = runner.StopFlag()
    stop.requested = True
    session = runner.CellSession(directory, '2LNN', 0, config, stop, 2, 'cpu')
    with pytest.raises(runner.TrainingStopped):
        core.run_cell(0, '2LNN', X, y, 'cpu', session=session)
    state = session.load(True)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    random.seed(91); np.random.seed(91); torch.manual_seed(91)
    runner.restore_rng(state['rng'], 'cpu')
    actual = (random.random(), np.random.rand(), torch.rand(3))
    equal(expected, actual)


def test_boundary_success_semantics_and_resume(project, monkeypatch):
    # Actual 2500/2501/10000 boundary numbers, tiny two-row controlled forward.
    monkeypatch.setattr(core, 'CFG', dict(core.CFG, n_runs=4, emax=2500, ex=10000))
    directory, config, _, _ = setup_run()
    X, y = torch.zeros((2, 8)), torch.tensor([0., 1.])
    counter = {'epoch': 0}
    success_epochs = [2500, 2501, 10000, None]
    def forward(X, P, p, fam, w0):
        counter['epoch'] += 1
        values = torch.zeros((4, 2))
        for i, target in enumerate(success_epochs):
            if counter['epoch'] == target:
                values[i, 1] = 1.
        return values + P['bo'] * .001
    monkeypatch.setattr(core, 'fwd', forward)
    stop = runner.StopFlag()
    states = {}
    def update(name, state):
        if state['completed_epoch'] in (2499, 2500, 2501, 9999, 10000):
            states[state['completed_epoch']] = copy.deepcopy(state)
        if state['completed_epoch'] in (2499, 2500, 2501, 9999):
            stop.requested = True
    session = runner.CellSession(directory, '2LNN', 0, config, stop, 1, 'cpu', update)
    # Avoid thousands of fsyncs in the controlled boundary test; still serialize all requested saves.
    original_save = session.save
    def boundary_save(*args, **kwargs):
        ep = args[6]
        if ep in (0, 2499, 2500, 2501, 9999, 10000):
            return original_save(*args, **kwargs)
    session.save = boundary_save
    while True:
        try:
            result = core.run_cell(0, '2LNN', X, y, 'cpu', session=session)
            break
        except runner.TrainingStopped:
            session.load(True)
            stop.requested = False
    assert list(result['conv_by_seed'].values()) == [2500, 2501, 10000, -1]
    assert result['succ'] == 1 and result['ex_rescued'] == 2
    assert result['median_epoch'] == 2500
    assert result['acc_at_emax']['42'] == 1 and result['acc_at_emax']['43'] == .5
    equal(states[2500]['theta'][0], states[10000]['theta'][0])
    assert not torch.equal(states[2500]['parameters']['bo'][0], states[10000]['parameters']['bo'][0])
    assert states[10000]['phase'] == 'after_step'
    assert all(v['step'].item() == 10000 for v in states[10000]['optimizer']['state'].values())
    assert states[10000]['theta'][3] is None


def test_all_success_no_step_weights_inference_and_repair(project, monkeypatch):
    original_init = core.init_params
    def init(p, device):
        seeds, P = original_init(p, device)
        with torch.no_grad():
            P['Wo'].fill_(np.pi / 2)
            P['bo'].fill_(np.pi / 2)
        return seeds, P
    monkeypatch.setattr(core, 'init_params', init)
    directory, config, X, y = setup_run(fams=['2LPIP'])
    flag = runner.StopFlag()
    with runner.run_lock(directory):
        assert runner.execute(core, directory, config, X.numpy(), y.numpy(), 'cpu', 2, flag)
    session = runner.CellSession(directory, '2LPIP', 0, config, flag, 2, 'cpu')
    state = session.load(True)
    assert state['completed_epoch'] == 1 and state['phase'] == 'all_success_before_step'
    assert state['optimizer']['state'] == {}
    path = directory / 'weights/2LPIP_p0_theta_star.pt'
    labels, accuracy, epoch = predict_weights.predict(path, 42)
    assert accuracy == 1 and epoch == 1
    with pytest.raises(runner.RunError, match='沒有首次成功權重'):
        predict_weights.predict(path, 999)
    path.unlink()
    for f in (directory / 'results').iterdir():
        f.unlink()
    def fail(*a, **k):
        pytest.fail('completed cell must not retrain')
    monkeypatch.setattr(core, 'run_cell', fail)
    with runner.run_lock(directory):
        assert runner.execute(core, directory, config, X.numpy(), y.numpy(), 'cpu', 2, flag)
    assert path.exists()
    assert json.loads((directory / 'progress.json').read_text())['complete']


def test_partial_results_cell_boundary_and_recovery(project, monkeypatch):
    directory, config, X, y = setup_run(fams=['2LNN', 'SIREN'])
    flag = runner.StopFlag()
    original = core.run_cell
    def stop_between(*args, **kwargs):
        row = original(*args, **kwargs)
        flag.requested = True
        return row
    monkeypatch.setattr(core, 'run_cell', stop_between)
    assert not runner.execute(core, directory, config, X.numpy(), y.numpy(), 'cpu', 2, flag)
    summary = json.loads((directory / 'results/summary.json').read_text())
    assert summary['status'] == 'partial' and summary['completed_cells'] == 1
    assert 'False' in (directory / 'results/test_主結果.csv').read_text(encoding='utf-8-sig')
    monkeypatch.setattr(core, 'run_cell', original)
    assert runner.execute(core, directory, config, X.numpy(), y.numpy(), 'cpu', 2, runner.StopFlag())
    assert json.loads((directory / 'results/summary.json').read_text())['status'] == 'complete'


def test_completed_checkpoint_before_outputs(project, monkeypatch):
    directory, config, X, y = setup_run()
    real_write = runner.write_results
    def crash(core, directory, config, results):
        if results:
            raise OSError('simulated failure after completed checkpoint')
        real_write(core, directory, config, results)
    monkeypatch.setattr(runner, 'write_results', crash)
    with pytest.raises(OSError, match='simulated'):
        runner.execute(core, directory, config, X.numpy(), y.numpy(), 'cpu', 2, runner.StopFlag())
    monkeypatch.setattr(runner, 'write_results', real_write)
    def fail(*args, **kwargs):
        pytest.fail('must rebuild rather than retrain')
    monkeypatch.setattr(core, 'run_cell', fail)
    assert runner.execute(core, directory, config, X.numpy(), y.numpy(), 'cpu', 2, runner.StopFlag())
    assert (directory / 'weights/2LNN_p0_theta_star.pt').exists()


def test_corrupt_latest_fallback_and_both_invalid(project, capsys):
    directory, config, X, y = setup_run()
    session = runner.CellSession(directory, '2LNN', 0, config, runner.StopFlag(), 2, 'cpu')
    core.run_cell(0, '2LNN', X, y, 'cpu', session=session)
    session.latest.write_bytes(b'corrupt')
    restarted = runner.CellSession(directory, '2LNN', 0, config, runner.StopFlag(), 2, 'cpu')
    state = restarted.load(True)
    assert state['completed_epoch'] == 6 and '回退 previous' in capsys.readouterr().out
    core.run_cell(0, '2LNN', X, y, 'cpu', session=restarted)
    # previous must still be the valid fallback, never the corrupted latest.
    assert runner.unpack_checkpoint(restarted.previous.read_bytes())['completed_epoch'] == 6
    restarted.latest.write_bytes(b'bad')
    restarted.previous.write_bytes(b'bad')
    with pytest.raises(runner.RunError, match='沒有有效 checkpoint'):
        restarted.load(True)


def test_missing_checkpoint_never_restarts(project):
    directory, config, X, y = setup_run()
    runner.atomic_json(directory / 'progress.json', {'config_id': config['config_id'], 'complete': False,
                                                    'cells': {'2LNN_p0': {'status': 'running'}}})
    with pytest.raises(runner.RunError, match='不會從零重跑'):
        runner.execute(core, directory, config, X.numpy(), y.numpy(), 'cpu', 2, runner.StopFlag())


def test_config_data_conflict_and_missing_cache(project):
    directory, config, X, y = setup_run(fams=['SIREN'], p=[2], tag='custom')
    args = argparse.Namespace(task=None, fams=None, p=None, tag=None, data_dir=None)
    actual, *_ = runner.load_config(directory, core, args)
    assert actual['task'] == 'parity' and actual['fams'] == ['SIREN'] and actual['p'] == [2]
    for key, value in [('task', 'two_curves'), ('fams', ['2LNN']), ('p', [0]), ('tag', 'other'), ('data_dir', 'other')]:
        setattr(args, key, value)
        with pytest.raises(runner.RunError, match='衝突'):
            runner.load_config(directory, core, args)
        setattr(args, key, None)
    (project / 'data/parity/dataset.npz').unlink()
    with pytest.raises(runner.RunError, match='缺少快取'):
        runner.load_config(directory, core, args)
    assert not (project / 'data/parity/dataset.npz').exists()


def test_run_names_lock_atomic_writes(project, monkeypatch):
    a, config, *_ = setup_run()
    b = runner.create_run(config, tasks.load_task('parity')[2])
    assert a != b
    for bad in ('../x', 'a/b', 'a\\b', '.', 'NUL', 'a:', ''):
        with pytest.raises(runner.RunError):
            runner.safe_name(bad)
    with runner.run_lock(a):
        with pytest.raises(runner.RunError, match='鎖定'):
            with runner.run_lock(a):
                pass
        with pytest.raises(runner.RunError, match='仍存在'):
            runner.unlock(a, True)
    stale = {'host': 'other-confirmed-stopped-host', 'pid': 100, 'token': 'old'}
    runner.atomic_json(a / 'writer.lock', stale)
    with pytest.raises(runner.RunError, match='confirm-stale-lock'):
        runner.unlock(a, False)
    runner.unlock(a, True)
    dest = a / 'results/probe.json'
    runner.atomic_json(dest, {'valid': True})
    original_bytes = dest.read_bytes()
    def fail(*args):
        raise OSError('simulated replace failure')
    monkeypatch.setattr(runner.os, 'replace', fail)
    with pytest.raises(OSError):
        runner.atomic_json(dest, {'valid': False})
    assert dest.read_bytes() == original_bytes
    assert not list(dest.parent.glob('.probe.json-*'))


@pytest.mark.parametrize('task', tasks.TASKS)
def test_every_task_training_entry(project, monkeypatch, task):
    # Shared already-validated cache copied with the same project-relative layout.
    if task.startswith('mnist'):
        source = ROOT / 'data' / task
        if not source.exists():
            pytest.skip('requires prepared MNIST cache')
        (project / 'data').mkdir(exist_ok=True)
        shutil.copytree(source, project / 'data' / task)
    monkeypatch.setattr(core, 'CFG', dict(core.CFG, n_runs=2, emax=1, ex=2))
    directory, config, X, y = setup_run(task=task, fams=core.FAMS, p=[0, 2])
    assert runner.execute(core, directory, config, X.numpy(), y.numpy(), 'cpu', 1, runner.StopFlag())
    result = json.loads((directory / 'results/2LNN_p0.json').read_text())
    assert result['n_samples'] == tasks.EXPECTED_K[task]
    assert result['task'] == task
    assert result['data_sha256'] == config['data']['sha256']


def test_subprocess_signal_cli_resume_and_relocation(tmp_path):
    import os
    import time
    project = tmp_path / 'source_project'
    shutil.copytree(ROOT / 'src', project / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    # Test-only driver shortens CFG and slows each real Adam step enough to send a signal.
    driver = project / 'driver.py'
    driver.write_text("""import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
import torch
import parity8_align_20260924 as core
core.CFG.update(n_runs=2, emax=4, ex=12)
original = torch.optim.Adam.step
def step(self, *a, **kw):
    result = original(self, *a, **kw)
    time.sleep(.02)
    return result
torch.optim.Adam.step = step
raise SystemExit(core.main())
""")
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    log = tmp_path / 'signal.log'
    with log.open('w') as output:
        process = subprocess.Popen([sys.executable, str(driver), '--fams', 'SIREN', '2LPIP', '--p', '2',
                                    '--tag', 'signal', '--checkpoint-every', '2'],
                                   cwd=tmp_path, env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            progress = None
            while time.monotonic() < deadline:
                paths = list(project.glob('run/parity/signal/*/progress.json'))
                if paths:
                    try:
                        progress = json.loads(paths[0].read_text())
                    except ValueError:
                        continue
                    if progress.get('cells', {}).get('SIREN_p2', {}).get('saved_epoch', 0) >= 2:
                        break
                if process.poll() is not None:
                    pytest.fail(log.read_text())
                time.sleep(.01)
            else:
                pytest.fail('signal test timeout: ' + log.read_text())
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=20) == 128 + signal.SIGTERM
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    run = paths[0].parent
    state = runner.unpack_checkpoint((run / 'checkpoints/SIREN_p2_latest.pt').read_bytes())
    assert 2 <= state['completed_epoch'] < 12 and state['phase'] == 'after_step'
    assert not (run / 'writer.lock').exists()
    relative_run = run.relative_to(project).as_posix()
    moved = tmp_path / 'moved_project'
    shutil.copytree(project, moved)
    result = subprocess.run([sys.executable, str(moved / 'driver.py'), '--resume', relative_run],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((moved / relative_run / 'progress.json').read_text())['complete']
    # A new uninterrupted run must agree on every trainable/resumable state.
    result = subprocess.run([sys.executable, str(moved / 'driver.py'), '--fams', 'SIREN', '2LPIP', '--p', '2',
                             '--tag', 'continuous'], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    continuous = next((moved / 'run/parity/continuous').iterdir())
    for fam in ['SIREN', '2LPIP']:
        resumed = runner.unpack_checkpoint((moved / relative_run / f'checkpoints/{fam}_p2_latest.pt').read_bytes())
        full = runner.unpack_checkpoint((continuous / f'checkpoints/{fam}_p2_latest.pt').read_bytes())
        for key in ['parameters', 'optimizer', 'conv', 'theta', 'acc2500', 'phase', 'next_epoch']:
            equal(resumed[key], full[key])
    completed = subprocess.run([sys.executable, str(moved / 'driver.py'), '--resume', relative_run],
                               cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0 and '沒有重新訓練' in completed.stdout
    conflict = subprocess.run([sys.executable, str(moved / 'driver.py'), '--resume', relative_run, '--p', '0'],
                              cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30)
    assert conflict.returncode != 0 and '衝突' in conflict.stderr


def test_core_and_data_hash_mismatch_rejected(project, monkeypatch):
    directory, config, X, y = setup_run(task='linearly_separable')
    args = argparse.Namespace(task=None, fams=None, p=None, tag=None, data_dir=None)
    original_hash = runner.core_hash
    monkeypatch.setattr(runner, 'core_hash', lambda core: {'changed': 'code'})
    with pytest.raises(runner.RunError, match='不相容'):
        runner.load_config(directory, core, args)
    monkeypatch.setattr(runner, 'core_hash', original_hash)
    # A valid but different dataset must still be rejected against saved run identity.
    X = X.numpy(); y = y.numpy()
    X[0, 0] += .01
    cache = project / 'data/linearly_separable'
    np.savez_compressed(cache / 'dataset.npz', X=X, y=y)
    meta = json.loads((cache / 'metadata.json').read_text())
    meta['sha256'] = tasks.data_hash(X, y)
    runner.atomic_json(cache / 'metadata.json', meta)
    with pytest.raises(runner.RunError, match='資料 hash'):
        runner.load_config(directory, core, args)


def test_first_success_saved_before_update_and_failed_seeds_absent(project, monkeypatch):
    original_init = core.init_params
    initial = {}
    def init(p, device):
        seeds, P = original_init(p, device)
        with torch.no_grad():
            P['Wo'][0].fill_(np.pi / 2)
            P['bo'][0].fill_(np.pi / 2)
        initial.update({k: v[0].detach().clone() for k, v in P.items() if v.numel()})
        return seeds, P
    monkeypatch.setattr(core, 'init_params', init)
    directory, config, X, y = setup_run(fams=['2LPIP'])
    session = runner.CellSession(directory, '2LPIP', 0, config, runner.StopFlag(), 2, 'cpu')
    result = core.run_cell(0, '2LPIP', X, y, 'cpu', session=session)
    state = session.load(True)
    assert result['conv_by_seed']['42'] == 1
    assert state['theta'][1] is None and state['theta'][2] is None
    assert set(result['theta_star']) == {'42'}
    for key, value in initial.items():
        assert torch.equal(torch.tensor(result['theta_star']['42'][key]), value)
    assert state['optimizer']['state'][0]['step'].item() == 7
    runner.write_results(core, directory, config, {'2LPIP_p0': result})
    _, accuracy, epoch = predict_weights.predict(directory / 'weights/2LPIP_p0_theta_star.pt', 42)
    assert accuracy == 1 and epoch == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
def test_cuda_resume_and_device_transfer(project):
    directory, config, X, y = setup_run()
    stop = runner.StopFlag()
    def update(name, state):
        if state['completed_epoch'] == 2:
            stop.requested = True
    session = runner.CellSession(directory, '2LNN', 0, config, stop, 2, 'cuda', update)
    with pytest.raises(runner.TrainingStopped):
        core.run_cell(0, '2LNN', X.cuda(), y.cuda(), 'cuda', session=session)
    resumed = runner.CellSession(directory, '2LNN', 0, config, runner.StopFlag(), 2, 'cpu')
    resumed.load(True)
    core.run_cell(0, '2LNN', X, y, 'cpu', session=resumed)
    assert resumed.load(True)['completed']


def test_all_success_at_final_epoch_skips_final_step(project, monkeypatch):
    directory, config, _, _ = setup_run()
    counter = {'epoch': 0}
    X, y = torch.zeros((2, 8)), torch.tensor([0., 1.])
    def forward(X, P, p, fam, w0):
        counter['epoch'] += 1
        correct = torch.tensor([0., 1.]) if counter['epoch'] == core.CFG['ex'] else torch.zeros(2)
        return correct[None, :].expand(3, -1) + P['bo'] * .001
    monkeypatch.setattr(core, 'fwd', forward)
    session = runner.CellSession(directory, '2LNN', 0, config, runner.StopFlag(), 2, 'cpu')
    core.run_cell(0, '2LNN', X, y, 'cpu', session=session)
    state = session.load(True)
    assert state['phase'] == 'all_success_before_step' and state['completed_epoch'] == 7
    assert all(s['step'].item() == 6 for s in state['optimizer']['state'].values())


def test_fallback_invalidates_stale_derived_outputs(project):
    directory, config, X, y = setup_run()
    runner.execute(core, directory, config, X.numpy(), y.numpy(), 'cpu', 2, runner.StopFlag())
    (directory / 'checkpoints/2LNN_p0_latest.pt').write_bytes(b'broken')
    stop = runner.StopFlag()
    stop.requested = True
    assert not runner.execute(core, directory, config, X.numpy(), y.numpy(), 'cpu', 2, stop)
    assert not (directory / 'results/2LNN_p0.json').exists()
    assert not (directory / 'weights/2LNN_p0_theta_star.pt').exists()
    assert json.loads((directory / 'results/summary.json').read_text())['status'] == 'partial'


def test_checkpoint_digest_detects_payload_corruption(project):
    directory, config, X, y = setup_run()
    session = runner.CellSession(directory, '2LNN', 0, config, runner.StopFlag(), 2, 'cpu')
    core.run_cell(0, '2LNN', X, y, 'cpu', session=session)
    import io
    envelope = torch.load(io.BytesIO(session.latest.read_bytes()), weights_only=True)
    envelope['sha256'] = '0' * 64
    with pytest.raises(runner.RunError, match='SHA-256'):
        runner.unpack_checkpoint(runner.torch_bytes(envelope))


def test_actual_concurrent_writer_is_rejected(project):
    directory, *_ = setup_run()
    with runner.run_lock(directory):
        script = "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import experiment_runner as r; r.PROJECT_ROOT=Path(sys.argv[2]);\nwith r.run_lock(Path(sys.argv[3])): pass"
        result = subprocess.run([sys.executable, '-c', script, str(ROOT / 'src'), str(project), str(directory)],
                                text=True, capture_output=True, timeout=20)
        assert result.returncode != 0 and '鎖定' in result.stderr
