"""Generate the requested ten 3-D surfaces, mosaic, and audit figures."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


ROOT = Path(__file__).resolve().parents[2]
Q1_DIR = Path(os.environ.get("Q1_OUT_DIR", ROOT / "results" / "q1_v22"))
CACHE_DIR = Path(os.environ.get("Q1_CACHE_DIR", ROOT / "data" / "cache_npz"))
BENCH_DIR = Path(os.environ.get("Q1_BENCH_DIR", ROOT / "results" / "q1_v21_benchmark_rerun"))
ABLATION_DIR = Path(os.environ.get("Q1_ABLATION_DIR", BENCH_DIR))
FIG_DIR = Q1_DIR / "figures"


def surface(ax, t, wl, z, title):
    ti = np.linspace(0, len(t) - 1, min(150, len(t)), dtype=int)
    wi = np.linspace(0, len(wl) - 1, min(180, len(wl)), dtype=int)
    T, W = np.meshgrid(t[ti], wl[wi], indexing="ij")
    ax.plot_surface(T, W, np.log1p(z[np.ix_(ti, wi)]), cmap="turbo", linewidth=0, antialiased=False)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Physical time (s)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_zlabel("log(1+intensity)")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted((Q1_DIR / "recon_native_npz").glob("*.npz"))
    mosaic = plt.figure(figsize=(20, 15))
    for k, path in enumerate(files):
        d = np.load(path)
        tag = path.stem.replace("_q1_v22", "")
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111, projection="3d")
        surface(ax, d["t_phys"], d["wavelengths"], d["spec_display"], tag)
        for bound in d["segment_bounds"][1:, 0]:
            ax.text(float(d["t_phys"][bound]), float(d["wavelengths"][0]), float(np.log1p(d["spec_display"]).min()), "segment", color="black")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{tag}_3d.png", dpi=150)
        plt.close(fig)
        axm = mosaic.add_subplot(2, 5, k + 1, projection="3d")
        surface(axm, d["t_phys"], d["wavelengths"], d["spec_display"], tag)
        axm.set_xlabel(""); axm.set_ylabel(""); axm.set_zlabel("")
    mosaic.tight_layout()
    mosaic.savefig(FIG_DIR / "q1_all10_3d_mosaic.png", dpi=160)
    plt.close(mosaic)

    # Representative strict-blackout repair slice.
    tag = "Att06_6C02058"
    raw = np.load(CACHE_DIR / f"{tag}.npz")
    rec = np.load(Q1_DIR / "recon_native_npz" / f"{tag}_q1_v22.npz")
    ids = np.flatnonzero(rec["is_blackout"])
    center = int(ids[len(ids) // 2]) if len(ids) else len(rec["t_phys"]) // 2
    band = (raw["wavelengths"] >= 550) & (raw["wavelengths"] <= 950)
    lo, hi = max(0, center - 20), min(len(ids), center + 21)
    sl = slice(max(0, center - 20), min(len(rec["t_phys"]), center + 21))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(rec["t_phys"][sl], raw["spec"][sl][:, band].mean(axis=1), label="raw")
    ax[0].plot(rec["t_phys"][sl], rec["spec_display"][sl][:, band].mean(axis=1), label="reconstructed")
    ax[0].axvline(rec["t_phys"][center], color="k", ls="--")
    ax[0].set(xlabel="Physical time (s)", ylabel="550-950 nm mean", title="Blackout repair in time")
    ax[0].legend()
    ax[1].plot(raw["wavelengths"], raw["spec"][center], alpha=.75, label="raw")
    ax[1].plot(rec["wavelengths"], rec["spec_display"][center], label="reconstructed")
    ax[1].set(xlabel="Wavelength (nm)", ylabel="Intensity", title=f"Repair slice at row {center}")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "q1_anomaly_repair_slice.png", dpi=180); plt.close(fig)

    bench_path = BENCH_DIR / "q1_blind_benchmark_5methods.csv"
    if bench_path.exists():
        b = pd.read_csv(bench_path)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        ax[0].barh(b["Method"], b["Anomaly_Repair_RMSE"])
        ax[0].set(xlabel="Anomaly repair RMSE (lower is better)", title="Five-method blind benchmark")
        ab = pd.read_csv(ABLATION_DIR / "q1_blind_benchmark_5methods.csv")
        ax[1].bar(["Unwhitened GD-SVD", "Whitened WLR"], [float(ab.loc[ab.Method.str.contains("Unwhitened"), "Global_RMSE"].iloc[0]), float(ab.loc[ab.Method.str.contains("Proposed"), "Global_RMSE"].iloc[0])])
        ax[1].set(ylabel="Global RMSE", title="SVD whitening ablation")
        fig.tight_layout(); fig.savefig(FIG_DIR / "q1_benchmark_and_whitening_ablation.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
