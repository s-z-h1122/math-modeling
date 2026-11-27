"""Question 3: fair LOHO comparison of causal gas-prediction models.

Every method sees exactly the same causal features and train/test rows.  All
scalers are fitted inside each leave-one-heat-out fold, and the last 35 s of
each heat are right-censored before target alignment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor


ROOT = Path(__file__).resolve().parents[2]
IN_DIR = Path(os.environ.get("Q1_V22_ALIGNED_DIR", ROOT / "results" / "q1_v22" / "recon_aligned_q3_npz"))
Q2_DIR = Path(os.environ.get("Q2_OUT_DIR", ROOT / "results" / "q2"))
OUT_DIR = Path(os.environ.get("Q3_OUT_DIR", ROOT / "results" / "q3"))
SEED = 2026


def _lag_indices(t: np.ndarray, lag_s: float) -> np.ndarray:
    return np.clip(np.searchsorted(t, t - lag_s, side="right") - 1, 0, len(t) - 1)


def causal_features(d: np.lib.npyio.NpzFile, q2: np.lib.npyio.NpzFile) -> tuple[np.ndarray, list[str]]:
    """Causal spectral, oxygen, level and backward-difference features."""
    wl = d["common_wl"].astype(float)
    s = d["spec_aligned_net"].astype(float)
    t = d["t_phys"].astype(float)
    bands = [(360, 420), (430, 480), (480, 540), (540, 600), (600, 680), (680, 740), (740, 800), (800, 880), (880, 960), (960, 1020)]
    base, names = [], []
    for lo, hi in bands:
        base.append(np.log1p(np.mean(s[:, (wl >= lo) & (wl < hi)], axis=1)))
        names.append(f"log_band_{lo}_{hi}")
    for line in [589.29, 766.49, 769.90]:
        m = np.abs(wl - line) <= 1.0
        base.append(np.log1p(np.max(s[:, m], axis=1)))
        names.append(f"log_peak_{line:.2f}")
    b = np.column_stack(base)
    feats = [b]
    feat_names = names.copy()
    for lag in [2.0, 5.0, 10.0]:
        ids = _lag_indices(t, lag)
        feats.append(b[ids])
        feats.append((b - b[ids]) / lag)
        feat_names += [f"{n}_lag{lag:g}" for n in names]
        feat_names += [f"{n}_slope{lag:g}" for n in names]
    flow = q2["o2_flow"].astype(float)
    cum = d["o2"].astype(float) - float(d["o2"][0])
    ids5 = _lag_indices(t, 5.0)
    feats += [flow[:, None], flow[ids5, None], cum[:, None], np.log1p(np.maximum(t, 0))[:, None]]
    feat_names += ["o2_flow", "o2_flow_lag5", "cumulative_o2", "log_elapsed_time"]
    return np.column_stack(feats), feat_names


def prepare_heat(path: Path, delay_s: float) -> dict:
    d = np.load(path)
    tag = path.stem.replace("_aligned", "")
    q2 = np.load(Q2_DIR / f"{tag}_q2.npz")
    X, names = causal_features(d, q2)
    t = d["t_phys"].astype(float)
    future = t + float(delay_s)
    # Right-censor the full maximum analyser horizon to avoid endpoint leakage.
    valid = (t <= t[-1] - 35.0) & (future <= t[-1])
    co2 = np.interp(future, t, d["co2"].astype(float))
    co = np.interp(future, t, d["co"].astype(float))
    y = np.column_stack([co2, co])
    # Deterministic 1-s approximate thinning reduces autocorrelation and keeps
    # the expensive SVR comparison tractable without changing any fold split.
    keep = np.flatnonzero(valid)[::2]
    return {"tag": tag, "group": 0 if "6B" in tag else 1, "X": X[keep], "y": y[keep], "t": t[keep], "feature_names": names, "delay_s": delay_s}


class ELM:
    def __init__(self, alpha: float, hidden: int = 128, seed: int = SEED):
        self.alpha, self.hidden, self.seed = alpha, hidden, seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0, 1 / np.sqrt(X.shape[1]), (X.shape[1], self.hidden))
        self.b = rng.uniform(-1, 1, self.hidden)
        H = np.tanh(X @ self.W + self.b)
        self.beta = np.linalg.solve(H.T @ H + self.alpha * np.eye(self.hidden), H.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X @ self.W + self.b) @ self.beta


class SoftGatedSharedMoE:
    """Shared standardized backbone with global and furnace-family experts."""
    def __init__(self, alpha: float = 4.0, temperature: float = 1.5):
        self.alpha, self.temperature = alpha, temperature

    def fit(self, X: np.ndarray, y: np.ndarray, group: np.ndarray):
        self.global_expert = Ridge(alpha=self.alpha).fit(X, y)
        self.experts, self.centers = {}, {}
        for g in np.unique(group):
            m = group == g
            self.experts[int(g)] = Ridge(alpha=self.alpha).fit(X[m], y[m])
            self.centers[int(g)] = np.mean(X[m], axis=0)
        self.scale = float(np.median(np.linalg.norm(X - np.mean(X, axis=0), axis=1)) + 1e-9)
        return self

    def predict(self, X: np.ndarray, group: np.ndarray) -> np.ndarray:
        pg = self.global_expert.predict(X)
        out = np.empty_like(pg)
        for g in np.unique(group):
            m = group == g
            if int(g) not in self.experts:
                out[m] = pg[m]
                continue
            pe = self.experts[int(g)].predict(X[m])
            dist = np.linalg.norm(X[m] - self.centers[int(g)], axis=1) / self.scale
            family_weight = 1.0 / (1.0 + np.exp((dist - 1.0) / self.temperature))
            out[m] = family_weight[:, None] * pe + (1.0 - family_weight[:, None]) * pg[m]
        return out


def model_predictions(name: str, Xtr: np.ndarray, ytr: np.ndarray, gtr: np.ndarray, Xte: np.ndarray, gte: np.ndarray) -> np.ndarray:
    if name == "Ridge":
        return Ridge(alpha=4.0).fit(Xtr, ytr).predict(Xte)
    if name == "ELM":
        return ELM(alpha=1e-6).fit(Xtr, ytr).predict(Xte)
    if name == "Regularized_ELM":
        return ELM(alpha=10.0).fit(Xtr, ytr).predict(Xte)
    if name == "SVR_RBF":
        # Same deterministic cap in every fold; no held-out rows enter it.
        ids = np.linspace(0, len(Xtr) - 1, min(4500, len(Xtr)), dtype=int)
        mdl = MultiOutputRegressor(SVR(C=8.0, epsilon=0.08, gamma="scale", cache_size=1000))
        return mdl.fit(Xtr[ids], ytr[ids]).predict(Xte)
    if name == "DNN_MLP":
        mdl = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3, batch_size=128, learning_rate_init=1e-3, max_iter=220, early_stopping=True, validation_fraction=0.15, n_iter_no_change=16, random_state=SEED)
        return mdl.fit(Xtr, ytr).predict(Xte)
    if name == "SG_MoE":
        return SoftGatedSharedMoE().fit(Xtr, ytr, gtr).predict(Xte, gte)
    raise ValueError(name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    delays = pd.read_csv(Q2_DIR / "q2_delay_by_heat.csv").set_index("Tag")["Delay_s"].to_dict()
    heats = [prepare_heat(p, delays[p.stem.replace("_aligned", "")]) for p in sorted(IN_DIR.glob("*.npz"))]
    methods = ["Ridge", "SVR_RBF", "ELM", "Regularized_ELM", "DNN_MLP", "SG_MoE"]
    rows, pred_frames = [], []
    for test_i, test in enumerate(heats):
        train = [h for i, h in enumerate(heats) if i != test_i]
        Xtr = np.vstack([h["X"] for h in train])
        ytr = np.vstack([h["y"] for h in train])
        gtr = np.concatenate([np.full(len(h["X"]), h["group"], int) for h in train])
        Xte, yte = test["X"], test["y"]
        gte = np.full(len(Xte), test["group"], int)
        xs, ys = StandardScaler().fit(Xtr), StandardScaler().fit(ytr)
        Xtr_s, Xte_s = xs.transform(Xtr), xs.transform(Xte)
        ytr_s = ys.transform(ytr)
        pf = pd.DataFrame({"Tag": test["tag"], "t_phys": test["t"], "CO2_true": yte[:, 0], "CO_true": yte[:, 1]})
        for method in methods:
            pred_s = model_predictions(method, Xtr_s, ytr_s, gtr, Xte_s, gte)
            pred = np.clip(ys.inverse_transform(pred_s), 0.0, 100.0)
            pf[f"CO2_{method}"] = pred[:, 0]
            pf[f"CO_{method}"] = pred[:, 1]
            for j, target in enumerate(["CO2", "CO"]):
                rows.append({
                    "Held_Out_Heat": test["tag"], "Group": "6B" if test["group"] == 0 else "6C", "Method": method, "Target": target,
                    "N_Train": len(Xtr), "N_Test": len(Xte), "Delay_s": test["delay_s"],
                    "MAE": mean_absolute_error(yte[:, j], pred[:, j]),
                    "RMSE": mean_squared_error(yte[:, j], pred[:, j]) ** 0.5,
                    "R2": r2_score(yte[:, j], pred[:, j]),
                })
            print(f"fold={test['tag']} method={method} done", flush=True)
        pred_frames.append(pf)
    fold = pd.DataFrame(rows)
    fold.to_csv(OUT_DIR / "q3_loho_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary = fold.groupby(["Method", "Target"], as_index=False).agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"), RMSE_mean=("RMSE", "mean"), R2_mean=("R2", "mean"))
    summary.to_csv(OUT_DIR / "q3_loho_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(pred_frames, ignore_index=True).to_csv(OUT_DIR / "q3_loho_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "q3_feature_manifest.json").write_text(json.dumps({"features": heats[0]["feature_names"], "causal": True, "right_censor_s": 35.0, "seed": SEED}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
