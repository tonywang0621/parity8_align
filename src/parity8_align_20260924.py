# -*- coding: utf-8 -*-
"""8-bit parity，依 2026-09-24 課堂討論對齊（第七批）。

課堂要求與本程式的對應
  1. 判定規則（Eq. 20）
       2LNN / Perceptron：古典規則，不加絕對值，o > 0.5；訓練目標為 MSE on raw o
       SIREN、PIP、2LPIP：加絕對值，|o| ≥ 0.5；訓練目標為 MSE on σ(10(|o| − 0.5))
  2. SIREN
       p = 0 ：o = sin(ω₀·(Σ xⱼwⱼ + b))                           （sine-out，與 2-bit Table IV 相同）
       p ≥ 1 ：aⱼ = sin(ω₀·(Σ xᵢwᵢⱼ + bⱼ))，o = Σ aⱼw°ⱼ + b°        ω₀ = 1
  3. 2LPIP（主定義與 2-bit Table IV、現行 Table V 相同）
       aⱼ = (f_PI(x, w_j^H) + 1)/2 = (1 − cos(Σ xᵢwᵢⱼ + w_{j,m+1}))/2     Eq.(15)(18)
       o  = f_PI(a, w°)，以 Eq.(14) 完整分支和計算                        Eq.(16)(14)
     對照族（附表用）：
       2LPIP-Eq19：o = −cos(Σ aᵢw°ᵢ + w°_{p+1})                           Eq.(19) 字面
       2LPIP-coh ：o = −Re[e^{i b°} Π((1 − qᵢ) + qᵢe^{i w°ᵢ})]，qᵢ = cos²(φᵢ/2)
       SIREN-rawMSE：前向與判定同 SIREN（|o| ≥ 0.5），訓練目標改為 MSE on raw o（2-bit 設計文件的 SIREN loss）
  4. EX rescue（David 建議，8-bit Table V 加入）
       同一次訓練延續到 10⁴ epoch：LSR 與 ME 以 E_max = 2500 計；
       EX = 在 2500 < epoch ≤ 10⁴ 之間收斂的 run 數 / 2500 時失敗的 run 數。
       Adam 狀態與參數連續不重設，等同「失敗者續訓」。
  5. 收斂當下（更新前）的 θ* 存入每格 JSON。

固定協定（第一至六批相同）
  資料：8-bit 完整真值表 K = 256，訓練與評估同一組
  seeds：NumPy default_rng，42–81；θ⁽⁰⁾ ~ N(0, 0.8²)，抽樣序 Wh → bh → Wo → bo
  Adam(lr = 0.02, β = (0.9, 0.999), ε = 1e-8)；float64；每 epoch 先判定再更新；epoch 自 1 起算
  平台：PyTorch（autograd + torch.optim.Adam），PIP 家族以閉式計算

用法
  python parity8_align_20260924.py --selfcheck
  python parity8_align_20260924.py --fams 2LNN SIREN 2LPIP --tag 主結果 --device cuda
  python parity8_align_20260924.py --fams 2LPIP-Eq19 2LPIP-coh --tag 附表_2LPIP定義 --device cuda
"""
import argparse, csv, json, math, os, time
from itertools import product

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
    """Eq.(14) 的乘積形式，V 可為任意實數。V:(S,K,n)；W:(S,n)；wlast:(S,1)。回傳 (S,K)。"""
    c = torch.cos(math.pi * V / 2) ** 2
    d = torch.sin(math.pi * V / 2) ** 2
    fac = c + d * torch.exp(1j * W[:, None, :])
    return -(torch.exp(1j * wlast) * torch.prod(fac, dim=-1)).real


def fwd(X, P, p, fam, w0=1.0):
    """回傳輸出 o(x, w)，形狀 (S, K)。"""
    if p == 0:
        z = torch.einsum("km,sm->sk", X, P["Wo"]) + P["bo"]
        if fam == "2LNN":
            return z                                   # Perceptron
        if fam in ("SIREN", "SIREN-rawMSE"):
            return torch.sin(w0 * z)                   # SIREN sine-out
        return -torch.cos(z)                           # PIP，Eq.(17)
    z = torch.einsum("km,smp->skp", X, P["Wh"]) + P["bh"][:, None, :]
    if fam == "2LNN":
        return torch.einsum("skp,sp->sk", torch.relu(z), P["Wo"]) + P["bo"]
    if fam in ("SIREN", "SIREN-rawMSE"):
        return torch.einsum("skp,sp->sk", torch.sin(w0 * z), P["Wo"]) + P["bo"]
    if fam == "2LPIP-coh":
        q = torch.cos(z / 2) ** 2
        fac = (1 - q) + q * torch.exp(1j * P["Wo"][:, None, :])
        return -(torch.exp(1j * P["bo"]) * torch.prod(fac, dim=-1)).real
    a = (1 - torch.cos(z)) / 2                         # Eq.(15) with Eq.(18)
    if fam == "2LPIP":
        return f_PI_general(a, P["Wo"], P["bo"])      # Eq.(16) with Eq.(14)
    if fam == "2LPIP-Eq19":
        return -torch.cos(torch.einsum("skp,sp->sk", a, P["Wo"]) + P["bo"])   # Eq.(19)
    raise ValueError(fam)


def decide(o, fam):
    if fam == "2LNN":
        return (o > CFG["tau"]).double()               # 古典規則，不加絕對值
    return (o.abs() >= CFG["tau"]).double()            # Eq.(20)，加絕對值


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


# ───────────────────────────── 訓練一格 ─────────────────────────────
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
    a = ap.parse_args()
    dev = torch.device(a.device)
    Xn = np.array(list(product([0, 1], repeat=CFG["n_bits"])), dtype=np.float64)
    X = torch.tensor(Xn, device=dev)
    y = torch.tensor((Xn.sum(1) % 2).astype(np.float64), device=dev)
    selfcheck(X, dev)
    if a.selfcheck:
        return
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
