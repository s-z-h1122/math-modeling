"""Causal optical features and leave-one-heat-out prediction of analyzer flow."""

from dataclasses import dataclass
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from signals import DT, FLAT, GAS_LABELS, GASES, SATURATION, causal_oxygen_flow

SEEDS = (42, 123, 2026)
WINDOWS = (20, 60, 150)
LINES_NM = (422.7, 589.0, 670.8, 766.5, 769.9)
BASELINE = "elapsed-time"
MAIN_MODE = "fixed_lag"


def optical_features(spectrum, wavelength, boundaries):
    # Fill sensor-invalid values from earlier readings. A trailing median of
    # three uses no future sample and suppresses isolated one-frame spikes.
    raw = pd.DataFrame(spectrum)
    valid = np.isfinite(spectrum) & (spectrum >= 0) & (spectrum < SATURATION)
    filled = raw.where(valid).ffill().fillna(0).to_numpy()
    n = len(filled)
    index = np.arange(n)
    clean = np.median(np.stack((filled, filled[np.maximum(0, index - 1)],
                               filled[np.maximum(0, index - 2)])), axis=0)
    features = []
    for j, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mask = (wavelength >= lo) & (wavelength < hi if j < 23 else wavelength <= hi)
        values = clean[:, mask]
        features.extend((values.mean(axis=1), values.std(axis=1)))
    lines = [clean[:, np.abs(wavelength - center) <= 1.0].mean(axis=1) for center in LINES_NM]
    features.extend(lines)
    features.extend((lines[3] / (lines[1] + 1), lines[3] / (lines[0] + 1)))
    original = np.log1p(np.column_stack(features))
    frame = pd.DataFrame(original)
    windows = [original]
    for seconds in WINDOWS:
        rolling = frame.rolling(round(seconds / DT), min_periods=1)
        windows.extend((rolling.mean().to_numpy(), rolling.std(ddof=0).to_numpy()))
    return np.column_stack(windows)


@dataclass
class Prepared:
    heat: object
    features: np.ndarray
    flow: np.ndarray
    grid: int


def prepare(heats):
    # Fixed common physical band. It lies within both instrument grids.
    boundaries = np.linspace(341, 1023, 25)
    grids = []
    data = {}
    for heat in heats:
        group = next((i for i, grid in enumerate(grids)
                      if np.array_equal(grid, heat.wavelength)), None)
        if group is None:
            group = len(grids)
            grids.append(heat.wavelength)
        x = optical_features(heat.spectrum, heat.wavelength, boundaries)
        prefix = min(200, len(x))
        check = optical_features(heat.spectrum[:prefix], heat.wavelength, boundaries)
        np.testing.assert_allclose(x[:prefix], check, rtol=0, atol=1e-12)
        if x.shape != (len(heat.time), 385) or not np.isfinite(x).all():
            raise ValueError(f"{heat.name}: invalid causal features")
        data[heat.name] = Prepared(heat, x, causal_oxygen_flow(heat), group + 1)
    return data


def fit_delay(windows, training, target):
    rows = windows[windows.heat.isin(training) & (windows.target == target)
                   & windows.screened_in & ~windows.search_boundary]
    # Balance the fixed-lag estimate across heats.
    per_heat = rows.groupby("heat").lag_s.median()
    fixed = float(per_heat.median()) if len(per_heat) else 0.0
    curve = None
    if len(per_heat) >= 3 and rows.lance_flow_m3s.nunique() >= 3:
        counts = rows.heat.map(rows.heat.value_counts())
        curve = IsotonicRegression(increasing=False, out_of_bounds="clip")
        curve.fit(rows.lance_flow_m3s, rows.lag_s, sample_weight=1 / counts)
    return fixed, curve, len(rows), len(per_heat)


def alignment(prepared, mode, fixed, curve):
    indices = np.arange(len(prepared.heat.time))
    if mode == "same_time":
        return indices, indices
    delay = curve.predict(prepared.flow) if mode == "flow_lag" and curve is not None else fixed
    future = indices + np.rint(np.asarray(delay) / DT).astype(int)
    issue = indices[future < len(indices)]
    # A target timestamp can receive multiple forecasts. Keep the first issued one.
    targets, earliest = np.unique(future[issue], return_index=True)
    return issue[earliest], targets


class ELM(RegressorMixin, BaseEstimator):
    def __init__(self, seed, alpha, hidden=128):
        self.seed = seed
        self.alpha = alpha
        self.hidden = hidden

    def fit(self, x, y):
        random = np.random.default_rng(self.seed)
        self.weights_ = random.normal(0, 1 / np.sqrt(x.shape[1]), (x.shape[1], self.hidden))
        self.bias_ = random.uniform(-1, 1, self.hidden)
        hidden = np.tanh(x @ self.weights_ + self.bias_)
        self.output_ = Ridge(alpha=self.alpha).fit(hidden, y)
        return self

    def predict(self, x):
        return self.output_.predict(np.tanh(x @ self.weights_ + self.bias_))


def hgb():
    return HistGradientBoostingRegressor(max_iter=150, max_depth=6, learning_rate=.1,
                                         early_stopping=False, random_state=SEEDS[0])


def families():
    return {
        "HGB": [hgb()],
        "RBF-SVR": [SVR(C=5, epsilon=.08, gamma="scale", cache_size=256)],
        "ELM": [ELM(seed, 0.0) for seed in SEEDS],
        "RELM": [ELM(seed, 1.0) for seed in SEEDS],
        "MLP": [MLPRegressor(hidden_layer_sizes=(48, 24), alpha=.01, batch_size=256,
                             learning_rate_init=.003, max_iter=180, tol=.0002,
                             n_iter_no_change=12, early_stopping=False, random_state=seed)
                for seed in SEEDS],
    }


def fit_predict(estimator, x, y, test_x):
    # Shared target scaling makes fixed SVR and regularization settings interpretable.
    level, spread = y.mean(), y.std()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(x, (y - level) / (spread or 1))
    limit_reached = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    prediction = estimator.predict(test_x) * spread + level
    return prediction.clip(min=0), limit_reached


def score(actual, predicted):
    residuals = actual - predicted
    total = np.sum((actual - actual.mean()) ** 2)
    squared = np.sum(residuals ** 2)
    return {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(squared / len(actual))),
        "r2": float(1 - squared / total) if total > 0 else np.nan,
    }


def results_frame(held, target, mode, model, issue, times, prediction):
    return pd.DataFrame({
        "attachment": held.heat.number, "heat": held.heat.name, "grid": held.grid,
        "target": target, "mode": mode, "model": model,
        "issue_s": held.heat.time[issue], "target_s": held.heat.time[times],
        "lead_s": held.heat.time[times] - held.heat.time[issue],
        "actual_m3s": held.heat.gas[times, GASES.index(target)],
        "predicted_m3s": prediction,
        "label_status": "variable" if held.heat.valid_gas else FLAT,
    })


def predict_fold(data, reconstructed_gas, windows, test):
    from time import perf_counter

    train = {k: d for k, d in data.items() if k != test and d.heat.valid_gas}
    held = data[test]
    predictions, diagnostics, settings = [], [], []
    for target_idx, target in enumerate(GASES):
        fixed, curve, n_windows, n_heats = fit_delay(windows, train, target)
        duration = np.array([0, 3, 5, 7, 10])
        settings.append({
            "test_heat": test, "target": target, "train_heats": ",".join(train),
            "fixed_lag_s": fixed, "eligible_windows": n_windows, "eligible_heats": n_heats,
            **{f"lag_at_{v}_m3s": float(curve.predict([v])[0]) if curve is not None else fixed
               for v in duration},
        })
        for mode in ("same_time", MAIN_MODE, "flow_lag"):
            inputs, labels, groups = [], [], []
            train_series = []
            for key, x in train.items():
                issue, times = alignment(x, mode, fixed, curve)
                y = reconstructed_gas[key][times, target_idx]
                inputs.append(x.features[issue])
                labels.append(y)
                groups.extend([x.grid] * len(y))
                train_series.append((x.heat.time[times], y))

            training_x = np.vstack(inputs)
            training_y = np.concatenate(labels)
            groups = np.array(groups)
            scaler = StandardScaler().fit(training_x)
            training_x = scaler.transform(training_x)
            pca = PCA(n_components=40, svd_solver="randomized", random_state=SEEDS[0]).fit(training_x)
            issue, times = alignment(held, mode, fixed, curve)
            training_x = pca.transform(training_x)
            testing_x = pca.transform(scaler.transform(held.features[issue]))
            candidates = families() if mode == MAIN_MODE else {"HGB": [hgb()]}
            for name, estimators in candidates.items():
                runs = []
                for seed_idx, estimator in enumerate(estimators):
                    start = perf_counter()
                    prediction, limit = fit_predict(estimator, training_x, training_y, testing_x)
                    runs.append(prediction)
                    rms = score(held.heat.gas[times, target_idx], prediction)
                    diagnostics.append({
                        "test_heat": test, "target": target, "mode": mode, "model": name,
                        "seed": SEEDS[seed_idx], "fit_predict_s": perf_counter() - start,
                        "convergence_warning": limit, **rms,
                    })
                prediction = np.mean(runs, axis=0)
                predictions.append(results_frame(held, target, mode, name, issue, times, prediction))

            if mode == MAIN_MODE:
                # A simple benchmark learned from other heats, at the same target times.
                target_times = held.heat.time[times]
                profiles = np.array([np.interp(target_times, t, y, left=np.nan, right=np.nan)
                                     for t, y in train_series])
                prediction = np.full(len(times), np.median(training_y))
                present = np.isfinite(profiles).any(axis=0)
                prediction[present] = np.nanmedian(profiles[:, present], axis=0)
                predictions.append(results_frame(held, target, mode, BASELINE, issue, times, prediction))

                # Shared pooled preprocessing plus per-grid experts. Instrument
                # group is known from the header before measurements start.
                experts = {}
                for grid in np.unique(groups):
                    mask = groups == grid
                    experts[grid], _ = fit_predict(hgb(), training_x[mask], training_y[mask], testing_x)
                pooled = next(p.predicted_m3s.to_numpy() for p in reversed(predictions)
                              if p.model.iloc[0] == "HGB")
                specialist = experts.get(held.grid, np.mean(list(experts.values()), axis=0))
                mixtures = {
                    "HGB-grid-equal": np.mean(list(experts.values()), axis=0),
                    "HGB-global+grid": (pooled + specialist) / 2,
                }
                for key, p in mixtures.items():
                    predictions.append(results_frame(held, target, mode, key, issue, times, p))
    return pd.concat(predictions, ignore_index=True), diagnostics, settings


def summarize(predictions, destination, suffix=""):
    keys = ["target", "mode", "model"]
    per_heat = []
    for key, frame in predictions.groupby(keys + ["heat"], sort=True):
        metrics = score(frame.actual_m3s.to_numpy(), frame.predicted_m3s.to_numpy())
        per_heat.append(dict(zip(keys + ["heat"], key), attachment=frame.attachment.iloc[0],
                             n=len(frame), lead_s=frame.lead_s.mean(),
                             label_status=frame.label_status.iloc[0], **metrics))
    per_heat = pd.DataFrame(per_heat)
    per_heat.to_csv(destination / f"per_heat{suffix}.csv", index=False, float_format="%.6f")
    valid_heats = per_heat[per_heat.label_status != FLAT]
    summary = valid_heats.groupby(keys).agg(
        heats=("heat", "nunique"), samples=("n", "sum"), mean_lead_s=("lead_s", "mean"),
        macro_mae=("mae", "mean"), macro_rmse=("rmse", "mean"), macro_r2=("r2", "mean"),
        median_r2=("r2", "median"), worst_r2=("r2", "min"),
    ).reset_index()
    pooled = []
    for key, frame in predictions[predictions.label_status != FLAT].groupby(keys):
        metrics = score(frame.actual_m3s.to_numpy(), frame.predicted_m3s.to_numpy())
        pooled.append(dict(zip(keys, key), **{f"pooled_{k}": v for k, v in metrics.items()}))
    summary = summary.merge(pd.DataFrame(pooled), on=keys).sort_values(
        ["target", "mode", "macro_r2"], ascending=[True, True, False])
    summary.to_csv(destination / f"metrics{suffix}.csv", index=False, float_format="%.6f")
    return summary


def prediction_figures(predictions, summary, destination):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    for ax, target in zip(axes, GASES):
        rows = summary[(summary.target == target) & (summary["mode"] == MAIN_MODE)]
        rows = rows.sort_values("macro_r2")
        ax.barh(rows.model, rows.macro_r2, color="#166b83")
        ax.axvline(0, color="#555", lw=.8)
        ax.set(title=GAS_LABELS[target], xlabel="Mean held-out heat R²")
        ax.grid(axis="x", alpha=.23)
    fig.savefig(destination / "model_comparison.png", dpi=160)
    plt.close(fig)

    for heat, frame in predictions[(predictions["mode"] == MAIN_MODE)
                                  & (predictions.model == "HGB")].groupby("heat"):
        fig, axes = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True, layout="constrained")
        for ax, target in zip(axes, GASES):
            rows = frame[frame.target == target]
            ax.plot(rows.target_s, rows.actual_m3s, lw=1.05, color="#253643", label="Original analyzer")
            ax.plot(rows.target_s, rows.predicted_m3s, lw=1.05, color="#d96846", label="Forecast")
            ax.set_ylabel(f"{GAS_LABELS[target]} (m³/s)")
            ax.grid(alpha=.2)
        axes[0].legend(loc="upper right", fontsize=8)
        axes[1].set_xlabel("Analyzer target time (s)")
        fig.suptitle(f"{heat} · held-out HGB")
        fig.savefig(destination / f"{heat}_prediction.png", dpi=135)
        plt.close(fig)


def run_prediction(heats, reconstructed_gas, windows, destination):
    data = prepare(heats)
    all_predictions, diagnostics, settings = [], [], []
    for name in data:
        pred, runs, calibration = predict_fold(data, reconstructed_gas, windows, name)
        all_predictions.append(pred)
        diagnostics.extend(runs)
        settings.extend(calibration)
        print(f"Evaluated held-out heat {name}", flush=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(destination / "predictions.csv.gz", index=False, float_format="%.6f",
                       compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(diagnostics).to_csv(destination / "seed_and_runtime.csv", index=False, float_format="%.6f")
    pd.DataFrame(settings).to_csv(destination / "fold_delay_calibration.csv", index=False, float_format="%.6f")
    summary = summarize(predictions, destination)

    # Compare HGB timing methods at identical analyzer timestamps.
    hgb_rows = predictions[predictions.model == "HGB"]
    overlap = hgb_rows.groupby(["heat", "target", "target_s"])["mode"].transform("nunique") == 3
    summarize(hgb_rows[overlap], destination, suffix="_matched_timestamps")
    prediction_figures(predictions, summary, destination)
    return summary
