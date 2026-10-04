"""Offline unit tests plus integration comparisons when MNIST raw cache exists."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import random
import runpy
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import experiment_tasks as tasks


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "PROJECT_ROOT", tmp_path)
    return tmp_path


@pytest.mark.parametrize("task", ["parity", "linearly_separable", "hidden_manifold", "two_curves"])
def test_cache_roundtrip_rng_and_metadata(project, task):
    before = np.random.get_state()
    python_before = random.getstate()
    X, y, meta = tasks.load_task(task)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert python_before == random.getstate()
    files = list((project / "data" / task).iterdir())
    stamps = [f.stat().st_mtime_ns for f in files]
    X2, y2, meta2 = tasks.load_task(task)
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(y, y2)
    assert meta == meta2 and stamps == [f.stat().st_mtime_ns for f in files]
    assert X.dtype == y.dtype == np.float64
    assert set(meta["class_counts"]) == {"0", "1"}
    assert str(project) not in json.dumps(meta)


@pytest.mark.parametrize("task", ["linearly_separable", "hidden_manifold", "two_curves"])
def test_against_unmodified_official_script(project, monkeypatch, task):
    from vendor.qml_benchmarks_data.linearly_separable import generate_linearly_separable
    from vendor.qml_benchmarks_data.hidden_manifold import generate_hidden_manifold_model
    from vendor.qml_benchmarks_data.two_curves import generate_two_curves
    package = types.ModuleType("qml_benchmarks")
    package.__path__ = []
    data = types.ModuleType("qml_benchmarks.data")
    data.generate_linearly_separable = generate_linearly_separable
    data.generate_hidden_manifold_model = generate_hidden_manifold_model
    data.generate_two_curves = generate_two_curves
    monkeypatch.setitem(sys.modules, "qml_benchmarks", package)
    monkeypatch.setitem(sys.modules, "qml_benchmarks.data", data)
    output = project / "official"
    output.mkdir()
    monkeypatch.chdir(output)
    # Execute ALL loops, including later diff tasks; no rewritten generation loop in this reference.
    with tasks.numpy_seed(0):
        runpy.run_path(str(tasks.VENDOR / "scripts" / f"generate_{task}.py"))
    filenames = {"linearly_separable": "linearly_separable/linearly_separable_8d_train.csv",
                 "hidden_manifold": "hidden_manifold/hidden_manifold-6manifold-8d_train.csv",
                 "two_curves": "two_curves_diff/two_curves-5degree-0.1offset-8d_train.csv"}
    official = np.loadtxt(output / filenames[task], delimiter=",")
    X, y, _ = tasks.load_task(task)
    np.testing.assert_array_equal(X, official[:, :8])
    np.testing.assert_array_equal(y, (official[:, 8] + 1) / 2)


@pytest.mark.parametrize("damage", ["value", "dtype", "shape", "labels", "hash", "metadata", "missing"])
def test_damaged_cache_rejected_without_overwrite(project, damage):
    X, y, meta = tasks.load_task("parity")
    directory = project / "data" / "parity"
    archive = directory / "dataset.npz"
    metadata = directory / "metadata.json"
    if damage == "value":
        X[0, 0] = np.nan
    elif damage == "dtype":
        X = X.astype(np.float32)
    elif damage == "shape":
        X = X[:, :7]
    elif damage == "labels":
        y[:] = 0
    elif damage == "hash":
        meta["sha256"] = "bad"
    elif damage == "metadata":
        meta["source"]["commit"] = "bad"
    np.savez_compressed(archive, X=X, y=y)
    metadata.write_text(json.dumps(meta))
    if damage == "missing":
        archive.unlink()
    original = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(tasks.DataError, match="不會自動覆寫"):
        tasks.load_task("parity")
    assert original == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_hash_detects_finite_content_change(project):
    X, y, _ = tasks.load_task("linearly_separable")
    X[0, 0] += .01
    np.savez_compressed(project / "data/linearly_separable/dataset.npz", X=X, y=y)
    with pytest.raises(tasks.DataError, match="sha256"):
        tasks.load_task("linearly_separable")


def test_lock_and_failed_publish(project, monkeypatch):
    root = project / "data"
    root.mkdir()
    lock = root / ".parity.lock"
    lock.write_text("pid=unknown")
    with pytest.raises(tasks.DataError, match="資料準備鎖"):
        tasks.load_task("parity")
    lock.unlink()
    def failure(*args, **kwargs):
        raise OSError("simulated disk failure")
    monkeypatch.setattr(tasks.np, "savez_compressed", failure)
    with pytest.raises(OSError, match="disk failure"):
        tasks.load_task("parity")
    assert not list(root.iterdir())


def test_download_failure_no_fake_data(project, monkeypatch):
    def offline(*args, **kwargs):
        raise OSError("offline")
    monkeypatch.setattr(tasks.urllib.request, "urlopen", offline)
    with pytest.raises(tasks.DataError, match="下載失敗"):
        tasks.load_task("mnist_pca")
    assert not (project / "data/mnist_pca").exists()
    assert not list((project / "data/raw").iterdir())


def test_paths_and_relocation(project, monkeypatch, tmp_path_factory):
    monkeypatch.chdir(tmp_path_factory.mktemp("different_cwd"))
    X, y, meta = tasks.load_task("parity", "custom_data")
    assert (project / "custom_data/parity/dataset.npz").exists()
    moved = tmp_path_factory.mktemp("moved") / "project"
    shutil.copytree(project, moved)
    monkeypatch.setattr(tasks, "PROJECT_ROOT", moved)
    X2, y2, meta2 = tasks.load_task("parity", "custom_data")
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(y, y2)
    assert meta == meta2
    with pytest.raises(tasks.DataError, match="專案內"):
        tasks.load_task("parity", "../outside")
    with pytest.raises(tasks.DataError, match="未知任務"):
        tasks.load_task("../parity")


def test_cli_selfcheck_prepare_and_nonparity_guard(tmp_path):
    pytest.importorskip("torch")
    project = tmp_path / "copied_project"
    shutil.copytree(ROOT / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__"))
    entry = project / "src/parity8_align_20260924.py"
    base = [sys.executable, str(entry)]
    result = subprocess.run(base + ["--selfcheck", "--task", "mnist_pca", "--prepare-only"],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "定義自我檢查" in result.stdout and not (project / "data").exists()
    result = subprocess.run(base + ["--task", "parity", "--prepare-only"],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (project / "data/parity/dataset.npz").exists()
    assert not (project / "run").exists() and not (project / "src/結果").exists()
    result = subprocess.run(base + ["--task", "two_curves"], text=True, capture_output=True)
    assert result.returncode != 0 and "--prepare-only" in result.stderr


@pytest.mark.integration
def test_real_mnist_against_official_and_small(project, monkeypatch):
    raw = ROOT / "data/raw/mnist.npz"
    if not raw.exists():
        pytest.skip("先準備 MNIST 原始資料後可執行真實資料比較")
    target = project / "data/raw"
    target.mkdir(parents=True)
    shutil.copyfile(raw, target / "mnist.npz")
    tasks.load_task("mnist_pca_small")  # Small-first must prepare the shared full PCA cache.
    X, y, meta = tasks.load_task("mnist_pca")
    # Load the unmodified official module with lightweight data-loader adapters;
    # its scaler/PCA/label code is executed unchanged, with REAL train/test arrays.
    torchvision = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")
    keras = types.ModuleType("keras")
    datasets = types.ModuleType("keras.datasets")
    def load_data():
        with np.load(raw, allow_pickle=False) as archive:
            return ((archive['x_train'], archive['y_train']), (archive['x_test'], archive['y_test']))
    datasets.mnist = types.SimpleNamespace(load_data=load_data)
    for key, obj in {"torchvision": torchvision, "torchvision.transforms": transforms,
                     "keras": keras, "keras.datasets": datasets}.items():
        monkeypatch.setitem(sys.modules, key, obj)
    official = runpy.run_path(str(tasks.VENDOR / "mnist.py"))["generate_mnist"]
    with tasks.numpy_seed(42):
        for d in range(2, 9):
            expected, _, labels, _ = official(3, 5, "pca", n_features=d)
    np.testing.assert_array_equal(X, expected)
    np.testing.assert_array_equal(y, (labels + 1) / 2)
    assert meta["class_counts"] == {"0": 6131, "1": 5421}
    small_X, small_y, small_meta = tasks.load_task("mnist_pca_small")
    indices = random.Random(42).choices(list(range(len(X))), k=250)
    np.testing.assert_array_equal(small_X, X[indices])
    np.testing.assert_array_equal(small_y, y[indices])
    assert len(set(indices)) < 250
    assert small_meta["parent"]["sha256"] == meta["sha256"]
    # Removing the original archive must not affect reading processed caches.
    (target / "mnist.npz").unlink()
    for task in ("mnist_pca", "mnist_pca_small"):
        _, _, again = tasks.load_task(task)
        assert again["sha256"] == (meta if task == "mnist_pca" else small_meta)["sha256"]


def test_model_training_and_statistics_unchanged():
    import hashlib
    baseline = json.loads((ROOT / "tests/baseline_core.json").read_text())
    tree = ast.parse((ROOT / "src/parity8_align_20260924.py").read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name, expected in baseline["functions"].items():
        actual = hashlib.sha256(ast.dump(functions[name], include_attributes=False).encode()).hexdigest()
        assert actual == expected, f"baseline core function changed: {name}"


def test_vendored_source_matches_manifest():
    manifest = json.loads((tasks.VENDOR / "manifest.json").read_text())
    assert manifest["commit"] == tasks.COMMIT
    for path, digest in manifest["files"].items():
        assert tasks.sha_file(tasks.VENDOR / path) == digest
