"""Run a JSON experiment list sequentially, automatically resuming the same batch."""
import argparse
from pathlib import Path
import shlex
import signal
import subprocess
import sys

import experiment_runner as runner
import experiment_tasks as tasks
import parity8_align_20260924 as core

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_plan(path, data_dir):
    document = runner.read_json(path)
    if not isinstance(document, dict) or set(document) != {'experiments'}:
        raise runner.RunError('清單需為 {"experiments": [...]}')
    items = document['experiments']
    if not isinstance(items, list) or not items:
        raise runner.RunError('experiments 必須是非空清單')
    seen, plan = set(), []
    for entry in items:
        if not isinstance(entry, dict) or set(entry) - {'name', 'task', 'fams', 'p'}:
            raise runner.RunError('每項只接受 name、task、fams、p')
        name, task = entry.get('name'), entry.get('task')
        if not isinstance(name, str) or not isinstance(task, str):
            raise runner.RunError('每項必須提供字串 name 與 task')
        runner.safe_name(name)
        if name in seen:
            raise runner.RunError(f'實驗 name 不可重複：{name}')
        seen.add(name)
        if task not in tasks.TASKS:
            raise runner.RunError(f'尚未支援的 task：{task}；新資料任務須先接入 experiment_tasks.py')
        fams, ps = entry.get('fams', list(core.FAMS)), entry.get('p', list(range(9)))
        if (not isinstance(fams, list) or not fams or any(not isinstance(f, str) or f not in core.FAMS for f in fams)
                or len(set(fams)) != len(fams)):
            raise runner.RunError(f'{name}: fams 必須為不重複的支援模型清單')
        if (not isinstance(ps, list) or not ps or any(type(p) is not int or p not in range(9) for p in ps)
                or len(set(ps)) != len(ps)):
            raise runner.RunError(f'{name}: p 必須為不重複的 0～8 整數清單')
        plan.append({'name': name, 'task': task, 'fams': fams, 'p': ps, 'data_dir': data_dir})
    return plan


def run_tag(batch, name):
    if '__' in batch or '__' in name:
        raise runner.RunError('batch 與實驗 name 不可包含保留分隔符 __')
    return runner.safe_name(batch + '__' + name)


def discover(spec, batch):
    parent = PROJECT_ROOT / 'run' / spec['task'] / run_tag(batch, spec['name'])
    candidates = sorted(p for p in parent.iterdir() if p.is_dir()) if parent.exists() else []
    if len(candidates) > 1:
        raise runner.RunError(f'{runner.rel(parent)} 有多個 run，無法安全自動選擇；請另取 --batch 名稱或單獨 --resume 指定 run')
    if candidates and not (candidates[0] / 'config.json').is_file():
        raise runner.RunError(f'發現不完整 run：{runner.rel(candidates[0])}；請先檢查，不會覆寫或自動重訓')
    return candidates[0] if candidates else None


def command(spec, batch, run, device, every):
    cmd = [sys.executable, '-u', str(PROJECT_ROOT / 'src/parity8_align_20260924.py')]
    if run is not None:
        cmd += ['--resume', runner.rel(run)]
    cmd += ['--task', spec['task'], '--tag', run_tag(batch, spec['name']),
            '--data-dir', spec['data_dir'], '--fams', *spec['fams'], '--p', *map(str, spec['p'])]
    if device is not None:
        cmd += ['--device', device]
    if every is not None:
        cmd += ['--checkpoint-every', str(every)]
    return cmd


def execute_batch(plan, batch, device=None, every=None, dry_run=False):
    directory = PROJECT_ROOT / 'run/batches' / batch
    state_path = directory / 'batch.json'
    state = runner.read_json(state_path) if state_path.exists() else {'format_version': 1, 'batch': batch, 'experiments': {}}
    if state.get('format_version') != 1 or state.get('batch') != batch:
        raise runner.RunError('批次狀態格式或名稱不符')
    selected = []
    # Preflight the entire list before starting any experiment.
    for spec in plan:
        record = state['experiments'].get(spec['name'])
        if record and record['spec'] != spec:
            raise runner.RunError(f"{spec['name']} 與這批次已保存的設定不同；要重新比較請換 --batch 名稱，或為新增設定使用不同 name")
        run = discover(spec, batch)
        if record and record.get('run'):
            if run is None or runner.rel(run) != record['run']:
                raise runner.RunError(f"{spec['name']} 原 run 遺失或被替換，拒絕從頭重跑")
        elif record and record.get('status') == 'starting' and run is None:
            raise runner.RunError(f"{spec['name']} 上次啟動途中異常結束且找不到 run；請先检查 batch.json 與日誌，不會默默重跑")
        selected.append((spec, run))
    if dry_run:
        for spec, run in selected:
            display = command(spec, batch, run, device, every)
            display[0], display[2] = 'python', 'src/parity8_align_20260924.py'
            print(f"[{spec['name']}] {'恢復／檢查完成狀態' if run else '新實驗'}\n{shlex.join(display)}")
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    with runner.run_lock(directory):
        # Reload and recheck inside the lock; another manager may have finished during preflight.
        current = runner.read_json(state_path) if state_path.exists() else state
        if current != state:
            raise runner.RunError('批次狀態在啟動期間改變，請重新執行')
        stop = {'signal': None, 'child': None}
        def handler(signum, frame):
            stop['signal'] = signum
            child = stop['child']
            if child is not None and child.poll() is None:
                try:
                    child.send_signal(signum)
                except ProcessLookupError:
                    pass
        previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        for s in previous:
            signal.signal(s, handler)
        try:
            for index, (spec, run) in enumerate(selected, 1):
                if stop['signal']:
                    return 128 + stop['signal']
                record = {'spec': spec, 'run': runner.rel(run) if run else None,
                          'status': 'starting', 'updated_utc': runner.utc()}
                state['experiments'][spec['name']] = record
                runner.atomic_json(state_path, state)
                print(f"\n[{index}/{len(selected)}] {spec['name']}：{'續訓／確認完成' if run else '開始新實驗'}", flush=True)
                cmd = command(spec, batch, run, device, every)
                try:
                    child = subprocess.Popen(cmd, cwd=PROJECT_ROOT, start_new_session=True)
                except OSError:
                    record.update(status='stopped', updated_utc=runner.utc())
                    runner.atomic_json(state_path, state)
                    raise
                stop['child'] = child
                # Handle a signal arriving between Popen and assigning child.
                if stop['signal'] and child.poll() is None:
                    try:
                        child.send_signal(stop['signal'])
                    except ProcessLookupError:
                        pass
                status = child.wait()
                stop['child'] = None
                run = discover(spec, batch)
                record.update(run=runner.rel(run) if run else None, returncode=status, updated_utc=runner.utc())
                if status == 0:
                    if run is None or not runner.read_json(run / 'progress.json').get('complete'):
                        record['status'] = 'stopped'
                        runner.atomic_json(state_path, state)
                        raise runner.RunError('訓練程序回報成功，但找不到已完成 run，請檢查日誌')
                record['status'] = 'completed' if status == 0 else 'stopped'
                runner.atomic_json(state_path, state)
                if status != 0 or stop['signal']:
                    print('批次已停止；使用相同命令重新啟動即可續訓，後面的任務尚未開始。', flush=True)
                    return (128 + stop['signal']) if stop['signal'] else (128 - status if status < 0 else status)
            print(f"\n批次 {batch}：清單中的 {len(plan)} 個實驗全部完成。", flush=True)
            return 0
        finally:
            child = stop['child']
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()
            for s, old in previous.items():
                signal.signal(s, old)


def main(argv=None):
    parser = argparse.ArgumentParser(description='依 JSON 清單依序執行；同批次自動續訓，新增 name 可追加實驗')
    parser.add_argument('--batch', default='full_run', help='相同名稱續跑；新名称代表一批全新實驗')
    parser.add_argument('--config', default='experiments.json')
    parser.add_argument('--device', help='新實驗預設 cpu；續訓未指定則沿用保存設定')
    parser.add_argument('--checkpoint-every', type=runner.positive)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--dry-run', action='store_true', help='只顯示命令，不準備資料、不建立 run')
    args = parser.parse_args(argv)
    try:
        runner.safe_name(args.batch)
        data_dir = runner.rel(tasks.resolve_data_dir(args.data_dir))
        plan = read_plan(runner.project_path(args.config), data_dir)
        return execute_batch(plan, args.batch, args.device, args.checkpoint_every, args.dry_run)
    except (runner.RunError, tasks.DataError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'批次錯誤：{exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
