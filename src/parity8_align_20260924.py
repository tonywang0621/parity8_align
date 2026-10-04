# -*- coding: utf-8 -*-
"""比較六種模型擬合8-bit parity完整真值表的成功率與收斂速度。

目前實作：
  - 256筆資料同時用於訓練與成功判定，沒有獨立測試集。
  - 預設p=0～8，每組40個種子（42～81）；參數初始化為N(0, 0.8²)。
  - 使用float64與Adam（lr=0.02）；每個epoch先判定，再更新。
  - 首次100%答對即記錄更新前參數；2500以內算成功，之後至10000算救回。
  - 已成功的seed仍隨批次更新，直到所有seed都曾成功或達到訓練上限。
  - 2LNN使用o > 0.5，其餘模型使用|o| >= 0.5分類。
  - 2LNN與SIREN-rawMSE使用原始輸出的MSE；其餘使用
    sigmoid(10(|o| - 0.5))的MSE。
  - --task / --data-dir / --prepare-only 準備六種任務資料；新任務訓練尚未接入。
  - 全部設定完成後輸出至src/結果/<tag>/；目前沒有checkpoint或resume。

用法（從專案根目錄執行）：
  python src/parity8_align_20260924.py --selfcheck
  python src/parity8_align_20260924.py --fams 2LNN SIREN 2LPIP --tag comparison --device cuda
"""
import argparse, csv, json, math, os, time
from itertools import product

from experiment_tasks import TASKS, DataError, load_task

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = dict(n_bits=8, n_runs=40, seed_base=42, emax=2500, ex=10000, lr=0.02, sharpness=10.0, tau=0.5)
FAMS = ["2LNN", "SIREN", "2LPIP", "2LPIP-Eq19", "2LPIP-coh", "SIREN-rawMSE"]
RAW_MSE = ("2LNN", "SIREN-rawMSE")          # 訓練目標為 MSE on raw o 的族
Z95 = 1.95996


# ───────────────────────────── 前向 ─────────────────────────────
def f_PI_general(V, W, wlast):
    """以複數乘積計算PIP輸出，V可為實數。

    V:(S,K,n)、W:(S,n)、wlast:(S,1)，回傳(S,K)。
    S為種子數，K為資料筆數，n為輸入維度。
    """
    c = torch.cos(math.pi * V / 2) ** 2
    d = torch.sin(math.pi * V / 2) ** 2
    fac = c + d * torch.exp(1j * W[:, None, :])
    return -(torch.exp(1j * wlast) * torch.prod(fac, dim=-1)).real


def fwd(X, P, p, fam, w0=1.0):
    """計算各seed的模型輸出，形狀為(種子數, 資料筆數)；p=0表示無隱藏層。"""
    if p == 0:
        z = torch.einsum("km,sm->sk", X, P["Wo"]) + P["bo"]
        if fam == "2LNN":
            return z                                   # Perceptron
        if fam in ("SIREN", "SIREN-rawMSE"):
            return torch.sin(w0 * z)                   # SIREN sine-out
        return -torch.cos(z)                           # 無隱藏層的PIP輸出
    z = torch.einsum("km,smp->skp", X, P["Wh"]) + P["bh"][:, None, :]
    if fam == "2LNN":
        return torch.einsum("skp,sp->sk", torch.relu(z), P["Wo"]) + P["bo"]
    if fam in ("SIREN", "SIREN-rawMSE"):
        return torch.einsum("skp,sp->sk", torch.sin(w0 * z), P["Wo"]) + P["bo"]
    if fam == "2LPIP-coh":
        q = torch.cos(z / 2) ** 2
        fac = (1 - q) + q * torch.exp(1j * P["Wo"][:, None, :])
        return -(torch.exp(1j * P["bo"]) * torch.prod(fac, dim=-1)).real
    a = (1 - torch.cos(z)) / 2                         # 此處使用cosine隱藏層公式
    if fam == "2LPIP":
        return f_PI_general(a, P["Wo"], P["bo"])      # 複數乘積輸出層
    if fam == "2LPIP-Eq19":
        return -torch.cos(torch.einsum("skp,sp->sk", a, P["Wo"]) + P["bo"])   # 負cosine輸出層
    raise ValueError(fam)


def decide(o, fam):
    if fam == "2LNN":
        return (o > CFG["tau"]).double()               # 2LNN直接對原始輸出設門檻
    return (o.abs() >= CFG["tau"]).double()            # 其餘模型對輸出絕對值設門檻


def loss_per_seed(o, y, fam):
    if fam in RAW_MSE:
        return ((o - y) ** 2).mean(dim=1)              # MSE on raw o
    return ((torch.sigmoid(CFG["sharpness"] * (o.abs() - CFG["tau"])) - y) ** 2).mean(dim=1)


# ───────────────────────────── 初始化與統計 ─────────────────────────────
def init_params(p, device):
    m, S = CFG["n_bits"], CFG["n_runs"]
    seeds = [CFG["seed_base"] + r for r in range(S)]
    A = {k: [] for k in ("Wh", "bh", "Wo", "bo")}
    for sd in seeds:
        g = np.random.default_rng(sd)
        A["Wh"].append(g.normal(0, 0.8, (m, p)))
        A["bh"].append(g.normal(0, 0.8, p))
        A["Wo"].append(g.normal(0, 0.8, p if p else m))
        A["bo"].append(g.normal(0, 0.8, 1))
    return seeds, {k: torch.tensor(np.stack(v), device=device, requires_grad=True) for k, v in A.items()}


def wilson(k, n):
    ph, d = k / n, 1 + Z95 * Z95 / n
    c = (ph + Z95 * Z95 / (2 * n)) / d
    h = Z95 * math.sqrt(ph * (1 - ph) / n + Z95 * Z95 / (4 * n * n)) / d
    return max(0.0, c - h) * 100, min(1.0, c + h) * 100


def exact_median(v):
    s = sorted(v); n = len(s)
    if not n:
        return None
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# ───────────────────────────── 訓練一組模型與p設定 ─────────────────────────────
def run_cell(p, fam, X, y, device, w0=1.0):
    S, EMAX, EX = CFG["n_runs"], CFG["emax"], CFG["ex"]
    seeds, P = init_params(p, device)
    opt = torch.optim.Adam([v for v in P.values() if v.numel() > 0],
                           lr=CFG["lr"], betas=(0.9, 0.999), eps=1e-8)
    conv, acc2500, theta = [-1] * S, [0.0] * S, [None] * S
    t0 = time.time()
    for ep in range(1, EX + 1):
        o = fwd(X, P, p, fam, w0)
        L = loss_per_seed(o, y, fam)
        with torch.no_grad():
            acc = (decide(o, fam) == y).double().mean(dim=1).cpu().tolist()
        for i in range(S):
            if conv[i] != -1:
                # 首次成功後不覆寫θ*或acc2500；此seed仍參與後續批次更新。
                continue
            if ep <= EMAX:
                acc2500[i] = acc[i]
            if acc[i] == 1.0:                                         # 先判定：更新前的參數即 θ*
                conv[i] = ep
                theta[i] = {k: P[k][i].detach().cpu().tolist() for k in P if P[k].numel() > 0}
        if all(c != -1 for c in conv):
            break
        opt.zero_grad(); L.sum().backward(); opt.step()
    ok = [c for c in conv if c != -1 and c <= EMAX]
    rescued = [c for c in conv if c > EMAX]
    lo, hi = wilson(len(ok), S)
    return dict(
        model=fam, p=p, omega0=(w0 if fam.startswith("SIREN") else None), lr=CFG["lr"], init="N(0,0.8^2)",
        decision=("o > 0.5" if fam == "2LNN" else "|o| >= 0.5"),
        loss=("MSE on o" if fam in RAW_MSE else "MSE on sigmoid(10(|o|-0.5))"),
        n_params=(9 if p == 0 else 10 * p + 1), n_runs=S, emax=EMAX, ex_emax=EX,
        succ=len(ok), wilson_lo=lo, wilson_hi=hi, median_epoch=exact_median(ok),
        epoch_min=(min(ok) if ok else None), epoch_max=(max(ok) if ok else None),
        ex_rescued=len(rescued), ex_failed_at_emax=S - len(ok),
        conv_by_seed={str(sd): c for sd, c in zip(seeds, conv)},
        acc_at_emax={str(sd): a for sd, a in zip(seeds, acc2500)},
        theta_star={str(sd): th for sd, th in zip(seeds, theta) if th is not None},
        wall_sec=round(time.time() - t0, 1), device=str(device), torch=torch.__version__)


# ───────────────────────────── 自我檢查 ─────────────────────────────
def selfcheck(X, device):
    g = np.random.default_rng(20260924)
    T = lambda sh: torch.tensor(g.normal(0, .8, sh), device=device)
    y = X.sum(1) % 2
    print("定義自我檢查")
    Xs = X[None].expand(3, -1, -1)
    W, b = T((3, 8)), T((3, 1))
    d = (f_PI_general(Xs, W, b) - (-torch.cos(torch.einsum("km,sm->sk", X, W) + b))).abs().max()
    print("  (1) Eq.(14) 通式 vs Eq.(17) binary 塌縮式，最大差 %.3e" % float(d))
    print("  (2) 2LPIP 三種輸出層定義的最大差（隨機 θ ~ N(0,0.8²)，3 組）")
    for p in (0, 1, 2, 4, 8):
        P = dict(Wh=T((3, 8, p)), bh=T((3, p)), Wo=T((3, p if p else 8)), bo=T((3, 1)))
        e14 = fwd(X, P, p, "2LPIP"); e19 = fwd(X, P, p, "2LPIP-Eq19"); coh = fwd(X, P, p, "2LPIP-coh")
        print("      p=%d  Eq14−Eq19 %.3e   Eq14−coh %.3e" % (p, (e14 - e19).abs().max(), (e14 - coh).abs().max()))
    P = dict(Wo=T((3, 8)), bo=T((3, 1)))
    d = (fwd(X, P, 0, "PIP") - fwd(X, dict(Wo=P["Wo"], bo=P["bo"] - math.pi / 2), 0, "SIREN")).abs().max()
    print("  (3) SIREN sine-out(p=0) 與 PIP(p=0)：sin(z − π/2) = −cos(z)，最大差 %.3e" % float(d))
    one = lambda v, n: torch.full((1, n), v, device=device)
    for name, fam, w, bb in (("PIP", "PIP", math.pi / 2, math.pi / 2), ("SIREN sine-out", "SIREN", math.pi / 2, 0.0)):
        o = fwd(X, dict(Wo=one(w, 8), bo=one(bb, 1)), 0, fam)
        print("  (4) %-15s p=0 構造解 w=π/2, b=%.4f，|o| ≥ 0.5 判定：256 筆正確率 %.4f"
              % (name, bb, float((decide(o, fam)[0] == y).double().mean())))


# ───────────────────────────── 輸出 ─────────────────────────────
KEYS = ["model", "p", "n_params", "omega0", "decision", "loss", "n_runs", "succ", "wilson_lo", "wilson_hi",
        "median_epoch", "epoch_min", "epoch_max", "ex_rescued", "ex_failed_at_emax", "wall_sec", "device", "torch"]


def emit(rows, out, tag):
    os.makedirs(out, exist_ok=True)
    for r in rows:
        with open(os.path.join(out, "%s_p%d.json" % (r["model"], r["p"])), "w", encoding="utf-8") as fh:
            json.dump(r, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out, tag + "_主結果.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=KEYS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out, tag + "_逐seed.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "p", "seed", "conv_epoch", "success_at_2500", "rescued_by_1e4", "acc_at_2500"])
        for r in rows:
            for sd, c in r["conv_by_seed"].items():
                w.writerow([r["model"], r["p"], sd, c, int(c != -1 and c <= CFG["emax"]),
                            int(c > CFG["emax"]), "%.6f" % r["acc_at_emax"][sd]])
    print("\n已寫入 " + out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fams", nargs="+", default=FAMS, choices=FAMS)
    ap.add_argument("--p", type=int, nargs="+", default=list(range(9)))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="主結果")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--task", choices=TASKS, default="parity")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--prepare-only", action="store_true")
    a = ap.parse_args()
    dev = torch.device(a.device)
    Xn = np.array(list(product([0, 1], repeat=CFG["n_bits"])), dtype=np.float64)
    X = torch.tensor(Xn, device=dev)
    y = torch.tensor((Xn.sum(1) % 2).astype(np.float64), device=dev)
    selfcheck(X, dev)
    if a.selfcheck:
        return
    if a.prepare_only:
        try:
            Xn, yn, metadata = load_task(a.task, a.data_dir)
        except (DataError, OSError) as exc:
            ap.exit(1, f"資料準備失敗：{exc}\n")
        print(f"{a.task}: X={Xn.shape}, y={yn.shape}, float64")
        print(f"SHA-256={metadata['sha256']}\n快取：{metadata['cache']}")
        return
    if a.task != "parity":
        ap.error("本次版本僅接入新任務的資料準備，請加 --prepare-only；新任務訓練與 run/checkpoint/resume 尚未實作。")
    print("\n2LNN：o > 0.5、MSE on o｜SIREN / PIP / 2LPIP：|o| ≥ 0.5、MSE on σ(10(|o|−0.5))｜K=256｜"
          "seeds 42–81（NumPy default_rng）｜E_max %d，EX 續訓至 %d｜Adam lr %g｜N(0,0.8²)｜%s｜torch %s"
          % (CFG["emax"], CFG["ex"], CFG["lr"], dev, torch.__version__))
    print("%-11s%4s%5s%9s%19s%9s%9s%8s" % ("Model", "p", "TP", "succ", "Wilson 95% CI", "EX", "ME", "秒"))
    rows, t0 = [], time.time()
    for fam in a.fams:
        for p in a.p:
            r = run_cell(p, fam, X, y, dev)
            rows.append(r)
            print("%-11s%4d%5d%6d/40%19s%9s%9s%8.1f" % (
                fam, p, r["n_params"], r["succ"], "[%.1f, %.1f]" % (r["wilson_lo"], r["wilson_hi"]),
                "%d/%d" % (r["ex_rescued"], r["ex_failed_at_emax"]),
                ("%g" % r["median_epoch"]) if r["median_epoch"] is not None else "—", r["wall_sec"]), flush=True)
    emit(rows, os.path.join(HERE, "結果", a.tag), a.tag)
    print("總計 %d 格，%.1f 分鐘" % (len(rows), (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
