"""Publication-ready Q2/Q3 result figures from saved CSV evidence."""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
Q2 = Path(os.environ.get("Q2_OUT_DIR", ROOT / "results" / "q2"))
Q3 = Path(os.environ.get("Q3_OUT_DIR", ROOT / "results" / "q3"))
OUT = Path(os.environ.get("MODEL_FIG_DIR", ROOT / "results" / "model_figures"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    flow = pd.read_csv(Q2 / "q2_delay_by_flow.csv")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for tag, g in flow.groupby("Tag"):
        g = g.sort_values("Flow_Mean")
        ax.plot(g["Flow_Mean"], g["Physics_Constrained_Delay_s"], marker="o", alpha=.75, label=tag)
    ax.set(xlabel="Reconstructed oxygen flow", ylabel="Physics-constrained delay (s)", title="Flow-conditioned gas transport delay")
    ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "q2_delay_vs_flow.png", dpi=180); plt.close(fig)

    s = pd.read_csv(Q3 / "q3_loho_summary.csv")
    methods = list(s["Method"].drop_duplicates())
    x = np.arange(len(methods)); width = .38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    co2 = s.set_index(["Method", "Target"]).loc[[(m, "CO2") for m in methods], "MAE_mean"].to_numpy()
    co = s.set_index(["Method", "Target"]).loc[[(m, "CO") for m in methods], "MAE_mean"].to_numpy()
    ax.bar(x - width/2, co2, width, label="CO2")
    ax.bar(x + width/2, co, width, label="CO")
    ax.set_xticks(x, methods, rotation=25, ha="right")
    ax.set(ylabel="LOHO mean MAE (lower is better)", title="Fair six-model cross-heat comparison")
    ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "q3_model_comparison.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
