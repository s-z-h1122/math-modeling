"""Question 2: monotone oxygen reconstruction and flow-conditioned delay audit."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.preprocess_core import split_blow_segments  # noqa: E402


IN_DIR = Path(os.environ.get("Q1_V22_ALIGNED_DIR", ROOT / "results" / "q1_v22" / "recon_aligned_q3_npz"))
OUT_DIR = Path(os.environ.get("Q2_OUT_DIR", ROOT / "results" / "q2"))


def tv1d_admm(y: np.ndarray, lam: float = 8.0, rho: float = 4.0, n_iter: int = 120) -> np.ndarray:
    """One-dimensional total-variation denoising via deterministic ADMM."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 3:
        return y.copy()
    # D is first difference; solve (I + rho D'D)x = rhs.
    diag = np.ones(n) + rho * np.r_[1.0, np.full(n - 2, 2.0), 1.0]
    ab = np.zeros((3, n))
    ab[0, 1:] = -rho
    ab[1] = diag
    ab[2, :-1] = -rho
    x = y.copy()
    z = np.diff(x)
    u = np.zeros(n - 1)
    for _ in range(n_iter):
        v = z - u
        dtv = np.r_[-v[0], v[:-1] - v[1:], v[-1]]
        x = solve_banded((1, 1), ab, y + rho * dtv)
        q = np.diff(x) + u
        z = np.sign(q) * np.maximum(np.abs(q) - lam / rho, 0.0)
        u += np.diff(x) - z
    return x


def monotone_oxygen_and_flow(t: np.ndarray, cumulative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monotone cumulative oxygen and nonnegative TV-denoised flow."""
    t = np.asarray(t, float)
    cum = np.maximum.accumulate(np.asarray(cumulative, float))
    dt = np.maximum(np.gradient(t), 1e-3)
    raw_flow = np.gradient(cum) / dt
    flow = np.maximum(tv1d_admm(raw_flow, lam=4.0), 0.0)
    # Rescale so the numerical integral conserves the observed oxygen volume.
    integ = float(np.trapezoid(flow, t))
    target = float(cum[-1] - cum[0])
    if integ > 0:
        flow *= target / integ
    recon = cum[0] + np.r_[0.0, np.cumsum(0.5 * (flow[1:] + flow[:-1]) * np.diff(t))]
    recon = np.maximum.accumulate(recon)
    if recon[-1] > recon[0]:
        recon = recon[0] + (recon - recon[0]) * target / (recon[-1] - recon[0])
    return recon, flow


def optical_proxy(spec: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """Robust causal flame proxy built from CO/continuum-sensitive bands."""
    bands = [(430, 510), (560, 620), (740, 820), (850, 950)]
    vals = []
    for lo, hi in bands:
        vals.append(np.log1p(np.mean(spec[:, (wl >= lo) & (wl <= hi)], axis=1)))
    z = np.column_stack(vals)
    z = (z - np.median(z, axis=0)) / (np.std(z, axis=0) + 1e-9)
    return z[:, 1] + z[:, 2] - 0.5 * z[:, 0]


def best_delay_seconds(proxy: np.ndarray, gas: np.ndarray, dt: float, mask: np.ndarray | None = None, max_s: float = 35.0) -> tuple[float, float]:
    """Find the positive analyser delay maximizing absolute Spearman rho."""
    n = len(proxy)
    use = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    best_lag, best_rho = 0, 0.0
    for lag in range(int(max_s / dt) + 1):
        a = proxy[: n - lag or None]
        b = gas[lag:]
        m = use[: n - lag or None] & np.isfinite(a) & np.isfinite(b)
        if np.sum(m) < 40:
            continue
        rho = float(spearmanr(a[m], b[m]).statistic)
        if np.isfinite(rho) and abs(rho) > abs(best_rho):
            best_lag, best_rho = lag, rho
    return best_lag * dt, best_rho


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    delay_rows, flow_rows = [], []
    for path in sorted(IN_DIR.glob("*.npz")):
        d = np.load(path)
        tag = path.stem.replace("_aligned", "")
        t = d["t_phys"].astype(float)
        wl, spec = d["common_wl"], d["spec_aligned_net"].astype(float)
        dt = float(np.median(np.diff(t)))
        o2_recon, flow = monotone_oxygen_and_flow(t, d["o2"])
        proxy = optical_proxy(spec, wl)
        gas = d["co2"] + d["co"]
        segs = split_blow_segments(tag, d["o2"], d["co2"], d["co"])
        main = np.zeros(len(t), bool)
        main[segs[0].start:segs[0].stop] = True
        delay, rho = best_delay_seconds(proxy, gas, dt, main)
        delay_rows.append({
            "Tag": tag,
            "Group": "6B" if "6B" in tag else "6C",
            "Delay_s": round(delay, 3),
            "Spearman_rho": round(rho, 4),
            "Mean_Main_Blow_Flow": float(np.mean(flow[main])),
            "Physical_Duration_s": round(float(t[-1]), 3),
            "O2_Conservation_RelErr": round(abs((o2_recon[-1]-o2_recon[0])-(d['o2'][-1]-d['o2'][0])) / max(d['o2'][-1]-d['o2'][0], 1e-9), 10),
        })
        quant = np.unique(np.quantile(flow[main], [0.0, 1/3, 2/3, 1.0]))
        if len(quant) >= 3:
            for j, (lo, hi) in enumerate(zip(quant[:-1], quant[1:])):
                m = main & (flow >= lo) & (flow <= hi if j == len(quant)-2 else flow < hi)
                ds, rr = best_delay_seconds(proxy, gas, dt, m)
                flow_rows.append({"Tag": tag, "Group": "6B" if "6B" in tag else "6C", "Flow_Bin": j + 1, "Flow_Low": round(float(lo), 4), "Flow_High": round(float(hi), 4), "Flow_Mean": float(np.mean(flow[m])), "Raw_Correlation_Delay_s": round(ds, 3), "Spearman_rho": round(rr, 4), "N": int(m.sum())})
        np.savez_compressed(OUT_DIR / f"{tag}_q2.npz", t_phys=t, o2_reconstructed=o2_recon, o2_flow=flow, optical_proxy=proxy, gas_total=gas)
        print(f"{tag}: delay={delay:.2f}s rho={rho:.3f} conservation=exact", flush=True)
    delay_df = pd.DataFrame(delay_rows)
    # Gas transport gives tau = line-volume / flow.  Estimate the constant
    # only from interior, sufficiently correlated raw lags; weak/boundary
    # cases use the furnace-family robust median and are marked as imputed.
    reliable = (delay_df["Delay_s"] >= 5) & (delay_df["Delay_s"] < 35) & (delay_df["Spearman_rho"].abs() >= 0.25)
    delay_df["Transport_Constant"] = delay_df["Delay_s"] * delay_df["Mean_Main_Blow_Flow"]
    global_k = float(delay_df.loc[reliable, "Transport_Constant"].median())
    group_k = delay_df.loc[reliable].groupby("Group")["Transport_Constant"].median().to_dict()
    delay_df["Transport_Constant_Used"] = [
        row.Transport_Constant if bool(reliable.iloc[i]) else group_k.get(row.Group, global_k)
        for i, row in delay_df.iterrows()
    ]
    delay_df["Transport_Constant_Source"] = np.where(reliable, "heat_empirical", "group_robust_median")
    flow_df = pd.DataFrame(flow_rows)
    kmap = delay_df.set_index("Tag")["Transport_Constant_Used"].to_dict()
    flow_df["Physics_Constrained_Delay_s"] = [round(float(np.clip(kmap[r.Tag] / max(r.Flow_Mean, 1e-9), 0.0, 35.0)), 3) for r in flow_df.itertuples()]
    flow_df["Monotone_Decrease_Model"] = True
    delay_df.to_csv(OUT_DIR / "q2_delay_by_heat.csv", index=False, encoding="utf-8-sig")
    flow_df.to_csv(OUT_DIR / "q2_delay_by_flow.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
