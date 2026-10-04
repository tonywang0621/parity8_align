"""Portable run management, atomic checkpoints and recovery for the fixed protocol."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import re
import secrets
import shlex
import signal
import socket
import tempfile

import numpy as np
import torch
import experiment_tasks as tasks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORMAT = 1
PARAM_NAMES = ["Wh", "bh", "Wo", "bo"]


class RunError(RuntimeError):
    pass


class TrainingStopped(Exception):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def safe_name(value):
    if (not value or value in (".", "..") or value[-1:] in (".", " ")
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', value)
            or value.split('.')[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}):
        raise RunError(f"不安全的資料夾名稱：{value!r}")
    return value


def project_path(value):
    path = (PROJECT_ROOT / value).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise RunError("路徑必須位於專案內。") from exc
    return path


def rel(path):
    return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def atomic_bytes(path, contents):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        # Persist rename on platforms supporting directory fsync.
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def atomic_json(path, value):
    atomic_bytes(path, json_bytes(value))


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise RunError(f"無法讀取 {Path(path).name}: {exc}") from exc


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def torch_bytes(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def pack_checkpoint(value):
    payload = torch_bytes(value)
    return torch_bytes({"sha256": hashlib.sha256(payload).hexdigest(), "payload": payload})


def unpack_checkpoint(blob):
    envelope = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    payload = envelope["payload"]
    if hashlib.sha256(payload).hexdigest() != envelope["sha256"]:
        raise RunError("checkpoint SHA-256 不符")
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


def environment(device):
    return {"python": platform.python_version(), "numpy": np.__version__,
            "torch": str(torch.__version__), "device": str(device), "dtype": "float64"}


def core_hash(core):
    files = [Path(core.__file__), Path(__file__)]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def data_identity(meta):
    return {k: meta[k] for k in ("task", "sha256", "shapes", "dtype", "array_order", "hash_method")}


def capture_rng(device):
    state = np.random.get_state()
    result = {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
              "torch_cpu": torch.get_rng_state(), "cuda": None}
    if torch.device(device).type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state(device).cpu()
    return result


def restore_rng(state, device):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.device(device).type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


@contextmanager
def run_lock(directory):
    path = directory / "writer.lock"
    token = secrets.token_hex(12)
    record = {"pid": os.getpid(), "host": socket.gethostname(), "token": token, "created_utc": utc()}
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RunError(f"run 已鎖定：{rel(path)}。確認原程序已停止後，使用 --unlock {shlex.quote(rel(directory))} --confirm-stale-lock；請先閱讀 doc/EXPERIMENTS.md。") from exc
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        if path.exists() and read_json(path).get("token") == token:
            path.unlink()


def unlock(directory, confirmed):
    if not confirmed:
        raise RunError("解除鎖需要 --confirm-stale-lock，表示你已確認原程序停止。")
    path = directory / "writer.lock"
    if not path.exists():
        raise RunError("這個 run 沒有 writer.lock。")
    record = read_json(path)
    if record.get("host") == socket.gethostname():
        try:
            os.kill(int(record["pid"]), 0)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise RunError("原程序仍存在但無權查詢；拒絕解除鎖。") from exc
        else:
            raise RunError("原程序仍存在，拒絕解除鎖。")
    # No new writer can acquire this existing O_EXCL lock until it is removed.
    path.unlink()
    print("已解除確認失效的鎖：" + rel(path))


class StopFlag:
    requested = False
    signum = None

    def handler(self, signum, frame):
        self.requested = True
        self.signum = signum

    @contextmanager
    def installed(self):
        previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        for s in previous:
            signal.signal(s, self.handler)
        try:
            yield self
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)


def cell_name(fam, p):
    return f"{fam}_p{p}"


class CellSession:
    """One checkpoint always includes all seeds and their shared Adam optimizer."""
    def __init__(self, run, fam, p, config, stop, every, device, update=None):
        self.run, self.fam, self.p = Path(run), fam, p
        self.config, self.stop, self.every, self.device = config, stop, every, device
        self.update = update
        self.name = cell_name(fam, p)
        self.latest = self.run / "checkpoints" / (self.name + "_latest.pt")
        self.previous = self.run / "checkpoints" / (self.name + "_previous.pt")
        self.loaded = None
        self.last_blob = None

    def validate(self, state):
        expected = {"format_version": FORMAT, "config_id": self.config["config_id"],
                    "task": self.config["task"], "model": self.fam, "p": self.p,
                    "CFG": self.config["CFG"], "data": self.config["data"],
                    "omega0": 1.0, "parameter_names": PARAM_NAMES,
                    "seeds": self.config["seeds"], "init": self.config["init"]}
        for key, value in expected.items():
            if state.get(key) != value:
                raise RunError(f"checkpoint {key} 與設定不符")
        n = len(self.config["seeds"])
        shapes = {"Wh": (n, 8, self.p), "bh": (n, self.p),
                  "Wo": (n, self.p if self.p else 8), "bo": (n, 1)}
        for key, shape in shapes.items():
            tensor = state["parameters"][key]
            if tuple(tensor.shape) != shape or tensor.dtype != torch.float64:
                raise RunError(f"checkpoint 參數 {key} 形狀或 dtype 錯誤")
        names = [key for key in PARAM_NAMES if state["parameters"][key].numel()]
        if state["optimizer_parameter_names"] != names:
            raise RunError("Adam 參數順序不符")
        if any(len(state[k]) != n for k in ("conv", "acc2500", "theta")):
            raise RunError("checkpoint seed 狀態數量不符")
        ep, nxt = state["completed_epoch"], state["next_epoch"]
        if not (0 <= ep <= self.config["CFG"]["ex"]) or nxt != ep + 1:
            raise RunError("checkpoint epoch 邊界錯誤")
        if state["phase"] not in ("initial", "after_step", "all_success_before_step"):
            raise RunError("checkpoint phase 錯誤")
        if not state["completed"] and ep >= self.config["CFG"]["ex"]:
            raise RunError("checkpoint 超出未完成訓練邊界")
        if state["completed"] and state["result"] is None:
            raise RunError("完成 checkpoint 缺少結果")
        for conv, theta in zip(state["conv"], state["theta"]):
            if (conv == -1) != (theta is None) or (conv != -1 and not 1 <= conv <= ep):
                raise RunError("首次成功紀錄不一致")
        if state["phase"] == "all_success_before_step" and (not state["completed"] or -1 in state["conv"]):
            raise RunError("全體成功邊界不一致")
        if state["phase"] == "initial" and (ep != 0 or state["completed"]):
            raise RunError("初始邊界不一致")

    def load(self, required=False):
        errors = []
        for path in (self.latest, self.previous):
            if not path.exists():
                continue
            try:
                blob = path.read_bytes()
                state = unpack_checkpoint(blob)
                self.validate(state)
                self.loaded, self.last_blob = state, blob
                if path == self.previous:
                    print(f"警告：{self.name} latest 無效或遺失，回退 previous：epoch {state['completed_epoch']}，next_epoch={state['next_epoch']}", flush=True)
                return state
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
        if required or errors:
            raise RunError("沒有有效 checkpoint；不會從零重跑：" + self.name + "; " + "; ".join(errors))
        return None

    def restore(self, P, opt, seeds):
        state = self.loaded
        if state is None:
            return None
        if state["completed"]:
            raise RunError("已完成 cell 不應再進入訓練")
        with torch.no_grad():
            for key in PARAM_NAMES:
                P[key].copy_(state["parameters"][key].to(self.device))
        opt.load_state_dict(state["optimizer"])
        # Adam.load_state_dict handles parameter-device moments and CPU step counters.
        restore_rng(state["rng"], self.device)
        return state

    def save(self, P, opt, seeds, conv, acc2500, theta, ep, phase, elapsed, result=None):
        state = {"format_version": FORMAT, "config_id": self.config["config_id"],
                 "task": self.config["task"], "model": self.fam, "p": self.p,
                 "omega0": 1.0, "CFG": self.config["CFG"], "seeds": seeds,
                 "init": self.config["init"], "data": self.config["data"],
                 "parameter_names": PARAM_NAMES,
                 "optimizer_parameter_names": [key for key in PARAM_NAMES if P[key].numel()],
                 "parameters": {key: value.detach().cpu().clone() for key, value in P.items()},
                 "optimizer": opt.state_dict(), "completed_epoch": ep, "next_epoch": ep + 1,
                 "phase": phase, "completed": result is not None, "conv": conv,
                 "acc2500": acc2500, "theta": theta, "elapsed_sec": elapsed,
                 "rng": capture_rng(self.device), "result": result, "environment": environment(self.device)}
        self.validate(state)
        blob = pack_checkpoint(state)
        if self.last_blob is not None:
            atomic_bytes(self.previous, self.last_blob)
        atomic_bytes(self.latest, blob)
        self.last_blob = blob
        if self.update:
            self.update(self.name, state)
        return state


def new_config(core, args, meta):
    config = {"format_version": FORMAT, "task": args.task, "tag": safe_name(args.tag),
              "fams": args.fams, "p": args.p,
              "cells": [[fam, p] for fam in args.fams for p in args.p],
              "CFG": dict(core.CFG), "seeds": list(range(core.CFG['seed_base'], core.CFG['seed_base'] + core.CFG['n_runs'])),
              "init": {"distribution": "N(0,0.8^2)", "rng": "numpy.default_rng per seed", "order": PARAM_NAMES},
              "optimizer": {"name": "Adam", "lr": core.CFG['lr'], "betas": [.9, .999], "eps": 1e-8},
              "omega0": 1.0, "dtype": "float64", "data_dir": rel(tasks.resolve_data_dir(args.data_dir)),
              "data": data_identity(meta), "data_cache": meta['cache'],
              "core_hash": core_hash(core), "environment": environment(args.device),
              "device": args.device, "checkpoint_every": args.checkpoint_every, "created_utc": utc()}
    config["config_id"] = digest(config)
    return config


def create_run(config, meta):
    parent = PROJECT_ROOT / "run" / safe_name(config["task"]) / safe_name(config["tag"])
    parent.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + secrets.token_hex(4)
        directory = parent / run_id
        try:
            directory.mkdir()
            break
        except FileExistsError:
            continue
    for name in ("results", "weights", "checkpoints"):
        (directory / name).mkdir()
    atomic_json(directory / "config.json", config)
    atomic_json(directory / "data_metadata.json", meta)
    return directory


def load_config(directory, core, args):
    config = read_json(directory / "config.json")
    unhashed = dict(config)
    identifier = unhashed.pop("config_id", None)
    if identifier != digest(unhashed) or config.get("format_version") != FORMAT:
        raise RunError("config 格式或 hash 不符")
    if config["core_hash"] != core_hash(core) or config["CFG"] != core.CFG:
        raise RunError("核心訓練程式版本或 CFG 不相容，拒絕續訓。")
    for name in ("task", "tag", "fams", "p", "data_dir"):
        explicit = getattr(args, name)
        if name == "data_dir" and explicit is not None:
            explicit = rel(tasks.resolve_data_dir(explicit))
        if explicit is not None and explicit != config[name]:
            raise RunError(f"--{name.replace('_', '-')} 與保存設定衝突：指定 {explicit!r}，保存 {config[name]!r}")
    meta = read_json(directory / "data_metadata.json")
    if data_identity(meta) != config["data"]:
        raise RunError("run 資料 metadata 與 config 不符")
    # Never regenerate missing data on resume.
    cache = project_path(config["data_cache"])
    if not cache.exists() or not (cache.parent / "metadata.json").exists():
        raise RunError(f"續訓缺少快取：{config['data_cache']} 及同目錄 metadata.json；請搬入原快取。")
    X, y, actual = tasks.load_task(config["task"], config["data_dir"])
    if data_identity(actual) != config["data"]:
        raise RunError("資料 hash、shape、dtype 或順序不符，拒絕續訓。")
    return config, X, y, actual


def write_results(core, directory, config, results):
    rows = [results[cell_name(fam, p)] for fam, p in config["cells"] if cell_name(fam, p) in results]
    complete = len(rows) == len(config["cells"])
    # A fallback can turn a formerly completed cell back into a running cell.
    # Its old derived outputs must not masquerade as authoritative results.
    for fam, p in config["cells"]:
        name = cell_name(fam, p)
        if name not in results:
            for path in (directory / 'results' / (name + '.json'),
                         directory / 'weights' / (name + '_theta_star.pt')):
                if path.exists():
                    path.unlink()
    for row in rows:
        name = cell_name(row['model'], row['p'])
        atomic_json(directory / 'results' / (name + '.json'), row)
        weights = {"format_version": FORMAT, "task": config['task'], "model": row['model'], "p": row['p'],
                   "omega0": row['omega0'], "CFG": config['CFG'], "config_id": config['config_id'],
                   "core_hash": config['core_hash'], "data": config['data'], "data_dir": config['data_dir'],
                   "decision": row['decision'], "dtype": "float64", "conv_by_seed": row['conv_by_seed'],
                   "theta_star": {seed: {key: torch.tensor(value, dtype=torch.float64) for key, value in values.items()}
                                  for seed, values in row['theta_star'].items()}}
        atomic_bytes(directory / 'weights' / (name + '_theta_star.pt'), torch_bytes(weights))
    stream = io.StringIO(newline='')
    fields = core.KEYS + ['task', 'n_samples', 'data_sha256', 'run_complete']
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(dict(row, run_complete=complete) for row in rows)
    atomic_bytes(directory / 'results' / (config['tag'] + '_主結果.csv'), stream.getvalue().encode('utf-8-sig'))
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['model', 'p', 'seed', 'conv_epoch', 'success_at_2500', 'rescued_by_1e4', 'acc_at_2500',
                     'task', 'data_sha256', 'run_complete'])
    for row in rows:
        for seed, ep in row['conv_by_seed'].items():
            writer.writerow([row['model'], row['p'], seed, ep, int(ep != -1 and ep <= config['CFG']['emax']),
                             int(ep > config['CFG']['emax']), '%.6f' % row['acc_at_emax'][seed], config['task'],
                             config['data']['sha256'], complete])
    atomic_bytes(directory / 'results' / (config['tag'] + '_逐seed.csv'), stream.getvalue().encode('utf-8-sig'))
    atomic_json(directory / 'results' / 'summary.json', {
        'status': 'complete' if complete else 'partial', 'completed_cells': len(rows),
        'total_cells': len(config['cells']), 'task': config['task'], 'config_id': config['config_id']})


def execute(core, directory, config, Xn, yn, device, every, stop):
    """Called under writer lock. All derived outputs can be rebuilt from checkpoints."""
    if every <= 0:
        raise RunError("checkpoint frequency 必須為正整數")
    progress_file = directory / 'progress.json'
    progress = read_json(progress_file) if progress_file.exists() else {
        'config_id': config['config_id'], 'complete': False, 'cells': {}, 'executions': []}
    if progress['config_id'] != config['config_id']:
        raise RunError("progress 與 config 不符")
    progress.setdefault('executions', []).append({**environment(device), 'started_utc': utc(), 'checkpoint_every': every})
    results, sessions = {}, []
    def update(name, state):
        progress['cells'][name] = {'status': 'completed' if state['completed'] else 'running',
                                  'saved_epoch': state['completed_epoch'], 'next_epoch': state['next_epoch']}
        progress['complete'] = False
        atomic_json(progress_file, progress)
    # Validate every checkpoint before making any training changes.
    for fam, p in config['cells']:
        name = cell_name(fam, p)
        session = CellSession(directory, fam, p, config, stop, every, device, update)
        status = progress['cells'].get(name, {}).get('status', 'pending')
        state = session.load(required=status in ('running', 'completed'))
        sessions.append(session)
        if state:
            progress['cells'][name] = {'status': 'completed' if state['completed'] else 'running',
                                      'saved_epoch': state['completed_epoch'], 'next_epoch': state['next_epoch']}
            if state['completed']:
                results[name] = state['result']
        else:
            progress['cells'][name] = {'status': 'pending', 'saved_epoch': 0, 'next_epoch': 1}
    progress['complete'] = len(results) == len(sessions)
    atomic_json(progress_file, progress)
    write_results(core, directory, config, results)
    if progress['complete']:
        print("此 run 已完成；已檢查並補齊結果，沒有重新訓練。", flush=True)
        return True
    X = torch.tensor(Xn, dtype=torch.float64, device=device)
    y = torch.tensor(yn, dtype=torch.float64, device=device)
    print(f"task={config['task']} K={len(yn)}，共 {len(sessions)} 組，每組 {config['CFG']['n_runs']} seeds，device={device}", flush=True)
    try:
        for session in sessions:
            if session.name in results:
                continue
            if stop.requested:
                raise TrainingStopped()
            row = core.run_cell(session.p, session.fam, X, y, device, session=session)
            results[session.name] = row
            write_results(core, directory, config, results)
            print(f"{session.name}: succ={row['succ']}/{row['n_runs']}, rescued={row['ex_rescued']}, median_epoch={row['median_epoch']}", flush=True)
    except TrainingStopped:
        write_results(core, directory, config, results)
        print("已在完整 epoch 邊界保存並停止，可使用上方 resume 指令續訓。", flush=True)
        return False
    progress['complete'] = True
    atomic_json(progress_file, progress)
    write_results(core, directory, config, results)
    print(f"完成 {len(results)} 組：{rel(directory)}", flush=True)
    return True


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('必須為正整數')
    return value


def main(core, argv=None):
    parser = argparse.ArgumentParser(description='固定訓練協定的六任務實驗與續訓')
    parser.add_argument('--fams', nargs='+', choices=core.FAMS)
    parser.add_argument('--p', type=int, nargs='+', choices=range(9))
    parser.add_argument('--device')
    parser.add_argument('--tag')
    parser.add_argument('--task', choices=tasks.TASKS)
    parser.add_argument('--data-dir')
    parser.add_argument('--checkpoint-every', type=positive)
    parser.add_argument('--selfcheck', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume')
    parser.add_argument('--unlock')
    parser.add_argument('--confirm-stale-lock', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.resume and (args.selfcheck or args.prepare_only or args.unlock):
            raise RunError('--resume 不可與 --selfcheck、--prepare-only、--unlock 同時使用')
        if args.unlock:
            if args.selfcheck or args.prepare_only:
                raise RunError('--unlock 不可與資料準備或 selfcheck 同時使用')
            unlock(project_path(args.unlock), args.confirm_stale_lock)
            return
        if args.confirm_stale_lock:
            raise RunError('--confirm-stale-lock 只能配合 --unlock')
        if args.fams is not None and len(set(args.fams)) != len(args.fams):
            raise RunError('--fams 不可重複')
        if args.p is not None and len(set(args.p)) != len(args.p):
            raise RunError('--p 不可重複')
        # Independent binary truth table even for continuous-input experiments.
        check_device = torch.device(args.device or 'cpu')
        from itertools import product
        truth = torch.tensor(list(product([0., 1.], repeat=8)), dtype=torch.float64, device=check_device)
        core.selfcheck(truth, check_device)
        if args.selfcheck:
            return
        if args.resume:
            directory = project_path(args.resume)
            with run_lock(directory):
                config, X, y, meta = load_config(directory, core, args)
                device = args.device or config['device']
                every = args.checkpoint_every or config['checkpoint_every']
                print(f"run：{rel(directory)}\n續訓：python src/parity8_align_20260924.py --resume {shlex.quote(rel(directory))}", flush=True)
                flag = StopFlag()
                with flag.installed():
                    done = execute(core, directory, config, X, y, device, every, flag)
                if not done:
                    return 128 + (flag.signum or signal.SIGINT)
            return
        args.task = args.task or 'parity'
        args.data_dir = args.data_dir or 'data'
        args.device = args.device or 'cpu'
        args.tag = args.tag if args.tag is not None else '主結果'
        args.fams = args.fams if args.fams is not None else list(core.FAMS)
        args.p = args.p if args.p is not None else list(range(9))
        args.checkpoint_every = args.checkpoint_every or 100
        safe_name(args.tag)
        X, y, meta = tasks.load_task(args.task, args.data_dir)
        if args.prepare_only:
            print(f"{args.task}: X={X.shape}, y={y.shape}, float64\nSHA-256={meta['sha256']}\n快取：{meta['cache']}")
            return
        config = new_config(core, args, meta)
        directory = create_run(config, meta)
        print(f"run：{rel(directory)}\n續訓：python src/parity8_align_20260924.py --resume {shlex.quote(rel(directory))}", flush=True)
        flag = StopFlag()
        with run_lock(directory), flag.installed():
            done = execute(core, directory, config, X, y, args.device, args.checkpoint_every, flag)
        if not done:
            return 128 + (flag.signum or signal.SIGINT)
    except (RunError, tasks.DataError, OSError, ValueError) as exc:
        parser.exit(1, f"實驗錯誤：{exc}\n")
    except (torch.OutOfMemoryError, MemoryError) as exc:
        parser.exit(1, f"記憶體不足：{exc}。本程式保留完整資料與 batch；請增加可用 CPU/GPU 記憶體，並由最近有效 checkpoint 續訓。\n")
