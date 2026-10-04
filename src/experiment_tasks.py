# MNIST adaptation: Copyright 2024 Xanadu Quantum Technologies Inc.
# Adapted under Apache-2.0; see vendor/qml_benchmarks_data/LICENSE and NOTICE.
"""Reproducible training-only 8D datasets; no training or model imports.

Synthetic generators are vendored unchanged from the pinned Xanadu commit.
MNIST preprocessing adapts that commit's data/mnist.py to use training only.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
from itertools import product
import json
import os
from pathlib import Path
import platform
import random
import shutil
import tempfile
import urllib.request

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKS = ("parity", "linearly_separable", "mnist_pca", "mnist_pca_small",
         "hidden_manifold", "two_curves")
COMMIT = "95e5a07e8e9e75ba7e24e67fb32b030112a1309a"
SOURCE = "https://github.com/XanaduAI/qml-benchmarks"
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
EXPECTED_K = dict(parity=256, linearly_separable=240, mnist_pca=11552,
                  mnist_pca_small=250, hidden_manifold=240, two_curves=240)
VENDOR = Path(__file__).resolve().parent / "vendor" / "qml_benchmarks_data"
HASH_METHOD = 'SHA-256: for X then y, UTF-8 compact JSON [name,"<f8",shape], LF, little-endian float64 C-order bytes'


class DataError(RuntimeError):
    """Actionable dataset/cache errors, never replaced with synthetic fallback data."""


def resolve_data_dir(data_dir="data"):
    path = Path(data_dir)
    path = (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise DataError("--data-dir 必須位於專案內，才能保存可攜式相對路徑。") from exc
    return path


def relative(path):
    return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def data_hash(X, y):
    h = hashlib.sha256()
    for name, value in (("X", X), ("y", y)):
        header = json.dumps([name, "<f8", list(value.shape)], separators=(",", ":"))
        h.update(header.encode("utf-8") + b"\n")
        h.update(np.asarray(value, dtype="<f8", order="C").tobytes(order="C"))
    return h.hexdigest()


def validate(task, X, y):
    if X.dtype != np.float64 or y.dtype != np.float64:
        raise DataError("X 與 y 必須為 float64；拒絕默默轉換既有快取。")
    if X.shape != (EXPECTED_K[task], 8) or y.shape != (EXPECTED_K[task],):
        raise DataError(f"{task} 資料形狀錯誤：X={X.shape}, y={y.shape}")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise DataError("資料含 NaN 或 infinity。")
    if not np.array_equal(np.unique(y), [0., 1.]):
        raise DataError("標籤必須只有 0、1，且兩個類別均存在。")
    if task == "mnist_pca" and [int((y == c).sum()) for c in (0, 1)] != [6131, 5421]:
        raise DataError("MNIST 訓練集 3、5 的類別數量不正確。")
    if task == "parity":
        expected = np.array(list(product([0, 1], repeat=8)), dtype=np.float64)
        if not np.array_equal(X, expected) or not np.array_equal(y, expected.sum(1) % 2):
            raise DataError("Parity 真值表或列順序錯誤。")


@contextmanager
def numpy_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@contextmanager
def preparation_lock(path):
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise DataError(f"資料準備鎖已存在：{relative(path)}。請確認沒有其他準備程序；"
                        "若先前程序異常結束，確認後手動刪除此鎖再重試。") from exc
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        path.unlink()


def dependencies():
    result = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "scikit-learn"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            pass
    return result


def source_metadata(task):
    if task == "parity":
        return {"kind": "generated_truth_table", "commit": None}
    name = "mnist" if task.startswith("mnist") else task
    manifest = json.loads((VENDOR / "manifest.json").read_text(encoding="utf-8"))
    filename = name + ".py"
    digest = sha_file(VENDOR / filename)
    if manifest["commit"] != COMMIT or manifest["files"][filename] != digest:
        raise DataError(f"官方參考程式版本或 hash 不符：{filename}")
    return {"repository": SOURCE, "commit": COMMIT,
            "generator": f"src/qml_benchmarks/data/{filename}",
            "generator_sha256": digest,
            "script": f"paper/benchmarks/generate_{name}.py",
            "license": "Apache-2.0"}


def synthetic(task):
    from sklearn.model_selection import train_test_split
    if task == "linearly_separable":
        from vendor.qml_benchmarks_data.linearly_separable import generate_linearly_separable
        fn, seed = generate_linearly_separable, 42
        arguments = lambda d: (300, d, .02 * d)
        params = {"n_samples": 300, "n_features": 8, "margin": .16}
        preprocessing = "none"
    elif task == "hidden_manifold":
        from vendor.qml_benchmarks_data.hidden_manifold import generate_hidden_manifold_model
        fn, seed = generate_hidden_manifold_model, 3
        arguments = lambda d: (300, d, 6)
        params = {"n_samples": 300, "n_features": 8, "manifold_dimension": 6}
        preprocessing = "official tanh embedding; median teacher labels"
    else:
        from vendor.qml_benchmarks_data.two_curves import generate_two_curves
        fn, seed = generate_two_curves, 3
        arguments = lambda d: (300, d, 5, .1, .01)
        params = {"n_samples": 300, "n_features": 8, "degree": 5, "offset": .1, "noise": .01}
        preprocessing = "official StandardScaler fitted to all 300 generated rows BEFORE split"
    with numpy_seed(seed):
        for d in range(2, 9):
            X, y = fn(*arguments(d))
            X_train, _, y_train, _ = train_test_split(X, y, test_size=.2)
    return np.asarray(X_train, dtype=np.float64), (np.asarray(y_train, dtype=np.float64) + 1) / 2, {
        "parameters": params, "data_seed": {"numpy_global": seed},
        "generation_order": list(range(2, 9)),
        "split": {"used": "train", "test_size": .2, "shuffle": True,
                  "stratify": None, "random_state": "numpy global; interleaved generation and split"},
        "preprocessing": preprocessing, "label_mapping": {"-1": 0, "1": 1},
        "row_order": "official train_test_split returned order"}


def mnist_raw(data_dir):
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    target = raw / "mnist.npz"
    if target.exists():
        if sha_file(target) != MNIST_SHA256:
            raise DataError(f"{relative(target)} SHA-256 不符；請保留檔案排查或手動移走後重新下載。")
        return target
    with preparation_lock(raw / ".mnist.lock"):
        fd, temp = tempfile.mkstemp(prefix=".mnist-", suffix=".tmp", dir=raw)
        try:
            print(f"下載 MNIST → {relative(target)}", flush=True)
            with os.fdopen(fd, "wb") as out:
                with urllib.request.urlopen(MNIST_URL, timeout=60) as response:
                    shutil.copyfileobj(response, out)
                out.flush()
                os.fsync(out.fileno())
            if sha_file(temp) != MNIST_SHA256:
                raise DataError("下載的 MNIST SHA-256 不符。")
            os.replace(temp, target)
        except Exception as exc:
            raise DataError(f"MNIST 下載失敗：{exc}。可將來源網址的檔案放到 {relative(target)} 後重試。") from exc
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    return target


def mnist_full(data_dir):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    raw = mnist_raw(data_dir)
    # The test arrays in this archive are intentionally never accessed.
    with np.load(raw, allow_pickle=False) as archive:
        images, labels = archive["x_train"], archive["y_train"]
    if images.shape != (60000, 28, 28) or labels.shape != (60000,):
        raise DataError("原始 MNIST 訓練資料形狀錯誤。")
    indices = np.flatnonzero((labels == 3) | (labels == 5))
    flat = images[indices].reshape(-1, 28 * 28)
    y = (labels[indices] == 5).astype(np.float64)
    # Equivalent to the official pca branch; discarded test transforms consume no RNG.
    scaled = StandardScaler().fit_transform(flat)
    with numpy_seed(42):
        for d in range(2, 9):
            pca = PCA(n_components=d)
            X = pca.fit_transform(scaled)
    return np.asarray(X, dtype=np.float64), y, {
        "parameters": {"digits": [3, 5], "n_features": 8},
        "data_seed": {"numpy_global": 42}, "generation_order": list(range(2, 9)),
        "split": {"used": "original MNIST train", "test_used": False},
        "preprocessing": {"flatten": [28, 28], "scaler": "StandardScaler",
                          "fit_on": "all 11552 training images of digits 3 and 5",
                          "pca": {"n_components": 8, "svd_solver": "auto",
                                  "effective_solver": pca._fit_svd_solver,
                                  "random_state": "NumPy global seeded 42; replay dimensions 2..8"}},
        "label_mapping": {"3": 0, "5": 1}, "row_order": "original training order filtered to 3 and 5",
        "raw": {"path": relative(raw), "url": MNIST_URL, "sha256": MNIST_SHA256,
                "hash_source": "https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py"},
        "reproducibility_note": "Uses recorded sklearn version and its auto PCA solver; not a claim of byte equality to published paper data."}


def generate(task, data_dir):
    if task == "parity":
        X = np.array(list(product([0, 1], repeat=8)), dtype=np.float64)
        return X, X.sum(1) % 2, {
            "parameters": {"n_bits": 8}, "data_seed": None,
            "split": {"used": "all 256 rows"}, "preprocessing": "none",
            "label_mapping": {"even": 0, "odd": 1}, "row_order": "itertools.product([0,1], repeat=8)"}
    if task == "mnist_pca":
        return mnist_full(data_dir)
    if task == "mnist_pca_small":
        X, y, parent = load_task("mnist_pca", data_dir)
        indices = random.Random(42).choices(list(range(len(X))), k=250)
        return X[indices], y[indices], {
            "parameters": {"digits": [3, 5], "n_features": 8, "n_samples": 250, "replacement": True},
            "data_seed": {"numpy_global_for_parent": 42, "python_random": 42},
            "split": {"used": "sampled original training split", "test_used": False},
            "preprocessing": "reuse full mnist_pca transformed training rows; no refit",
            "label_mapping": {"3": 0, "5": 1}, "row_order": "random.choices output order",
            "parent": {"task": "mnist_pca", "sha256": parent["sha256"],
                       "cache": relative(data_dir / "mnist_pca" / "dataset.npz")},
            "sample_indices": indices, "unique_sample_count": len(set(indices)),
            "reproducibility_note": "Project supplement: Python Random(42), one 250-row draw at 8D; with replacement. Does not claim row equality to the published PCA- dataset."}
    return synthetic(task)


def read_cache(task, directory):
    try:
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        with np.load(directory / "dataset.npz", allow_pickle=False) as archive:
            if set(archive.files) != {"X", "y"}:
                raise DataError("dataset.npz 應只含 X、y。")
            X, y = archive["X"], archive["y"]
        validate(task, X, y)
        required = {"source", "parameters", "data_seed", "split", "preprocessing",
                    "label_mapping", "row_order", "dependencies", "created_utc", "cache"}
        if not required.issubset(meta):
            raise DataError(f"metadata 缺少欄位：{sorted(required - meta.keys())}")
        checks = {"format_version": 1, "task": task, "n_samples": len(y),
                  "shapes": {"X": list(X.shape), "y": list(y.shape)},
                  "dtype": "float64", "array_order": "C", "hash_method": HASH_METHOD,
                  "class_counts": {str(c): int((y == c).sum()) for c in (0, 1)},
                  "sha256": data_hash(X, y)}
        for key, value in checks.items():
            if meta.get(key) != value:
                raise DataError(f"metadata {key} 不符。")
        if meta["source"]["commit"] != (None if task == "parity" else COMMIT):
            raise DataError("資料來源 commit 不符。")
        # Cache remains relocatable when the project directory is moved.
        if meta["cache"] != relative(directory / "dataset.npz"):
            raise DataError("快取相對路徑不符；搬移時請保留專案內相對位置。")
        if task == "mnist_pca_small":
            indices = meta["sample_indices"]
            expected = random.Random(42).choices(list(range(EXPECTED_K["mnist_pca"])), k=250)
            if indices != expected or meta["unique_sample_count"] != len(set(indices)):
                raise DataError("MNIST PCA- 抽樣索引不符。")
        return X, y, meta
    except Exception as exc:
        raise DataError(f"快取無效或不完整：{relative(directory)}：{exc}。不會自動覆寫；請先檢查或移走後重新準備。") from exc


def load_task(task, data_dir="data"):
    """Return float64 X(K,8), y(K,), metadata; valid caches never overwritten."""
    if task not in TASKS:
        raise DataError(f"未知任務：{task}；可用任務：{', '.join(TASKS)}")
    root = resolve_data_dir(data_dir)
    directory = root / task
    if directory.exists():
        return read_cache(task, directory)
    root.mkdir(parents=True, exist_ok=True)
    with preparation_lock(root / f".{task}.lock"):
        if directory.exists():
            return read_cache(task, directory)
        source = source_metadata(task)
        try:
            X, y, details = generate(task, root)
        except ImportError as exc:
            raise DataError("缺少資料準備套件；請執行 python -m pip install -r requirements-data.txt") from exc
        X, y = np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(y, dtype=np.float64)
        validate(task, X, y)
        meta = {"format_version": 1, "task": task, "n_samples": len(y),
                "shapes": {"X": list(X.shape), "y": list(y.shape)},
                "dtype": "float64", "array_order": "C",
                "class_counts": {str(c): int((y == c).sum()) for c in (0, 1)},
                "source": source, "dependencies": dependencies(),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "cache": relative(directory / "dataset.npz"),
                "hash_method": HASH_METHOD, "sha256": data_hash(X, y), **details}
        # Publish both files together; an interrupted write cannot become a valid cache.
        temp = Path(tempfile.mkdtemp(prefix=f".{task}-", dir=root))
        try:
            with open(temp / "dataset.npz", "wb") as stream:
                np.savez_compressed(stream, X=X, y=y)
                stream.flush()
                os.fsync(stream.fileno())
            with open(temp / "metadata.json", "w", encoding="utf-8") as stream:
                json.dump(meta, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.rename(temp, directory)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    return read_cache(task, directory)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="只準備與驗證資料，不訓練。")
    parser.add_argument("--task", choices=TASKS, default="parity")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    try:
        X, y, meta = load_task(args.task, args.data_dir)
    except (DataError, OSError) as exc:
        parser.exit(1, f"資料準備失敗：{exc}\n")
    print(f"{args.task}: X={X.shape}, y={y.shape}, float64; SHA-256={meta['sha256']}")
    print(f"快取：{meta['cache']}")


if __name__ == "__main__":
    main()
