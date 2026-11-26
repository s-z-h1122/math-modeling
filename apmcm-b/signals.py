"""Load the original workbooks, reconstruct Q1/Q2 signals and estimate delay."""

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.stats import spearmanr

DT = 0.5
GASES = ("co2", "co")
GAS_LABELS = {"co2": "CO₂", "co": "CO"}
SATURATION = 65535
FLAT = "constant zero in both original gas columns"


@dataclass
class Heat:
    number: int
    name: str
    time: np.ndarray
    wavelength: np.ndarray
    spectrum: np.ndarray
    oxygen: np.ndarray
    gas: np.ndarray

    @property
    def valid_gas(self):
        return not np.all(self.gas == 0)


def load_heats(folder: Path):
    paths = list(folder.glob("Attachment *.xlsb"))
    manifest = []
    for path in paths:
        match = re.fullmatch(r"Attachment (\d+)\. Heat No\. (\w+)\.xlsb", path.name)
        if not match:
            raise ValueError(f"Unexpected attachment filename: {path.name}")
        manifest.append((int(match[1]), match[2], path))
    if sorted(n for n, _, _ in manifest) != list(range(1, 11)):
        raise ValueError("Expected the ten original attachments, numbered 1 to 10.")
    heats = []
    for number, name, path in sorted(manifest):
        frame = pd.read_excel(path, engine="pyxlsb", header=None)
        if frame.shape[1] != 2051:
            raise ValueError(f"{path.name}: expected 2051 columns")
        wavelengths = frame.iloc[0, 1:2048].to_numpy(float)
        data = frame.iloc[1:].to_numpy(float)
        seq = data[:, 0]
        if not np.isfinite(seq).all() or not np.all(np.diff(seq) == 1):
            raise ValueError(f"{path.name}: unexpected sample sequence")
        if not np.isfinite(wavelengths).all() or not np.all(np.diff(wavelengths) > 0):
            raise ValueError(f"{path.name}: wavelengths must increase")
        if not np.isfinite(data[:, 2049:2051]).all():
            raise ValueError(f"{path.name}: gas labels contain missing values")
        heats.append(Heat(number, name, (seq - seq[0]) * DT, wavelengths,
                          data[:, 1:2048], data[:, 2048], data[:, 2049:2051]))
        print(f"Read {number:2d}: {name} ({len(seq)} samples)", flush=True)
    return heats


def local_spikes(values, kernel, intensity_floor):
    window = (kernel,) + (1,) * (values.ndim - 1)
    level = median_filter(values, size=window, mode="nearest")
    deviation = np.abs(values - level)
    scale = median_filter(deviation, size=window, mode="nearest")
    threshold = np.maximum(8 * 1.4826 * scale, intensity_floor)
    if values.ndim == 1:
        left = np.r_[values[0], values[:-1]]
        right = np.r_[values[1:], values[-1]]
    else:
        left = np.vstack((values[0], values[:-1]))
        right = np.vstack((values[1:], values[-1]))
    return (deviation > threshold) & (np.abs(left - right) < threshold / 2)


def repair(values, invalid, time):
    valid = ~invalid & np.isfinite(values)
    if not valid.any():
        raise ValueError("Cannot reconstruct a channel without any valid sample.")
    return np.interp(time, time[valid], values[valid])


def reconstruct_spectrum(heat):
    raw = heat.spectrum
    invalid = ~np.isfinite(raw) | (raw < 0) | (raw >= SATURATION)
    provisional = np.column_stack([repair(a, m, heat.time) for a, m in zip(raw.T, invalid.T)])
    spikes = local_spikes(provisional, kernel=9, intensity_floor=30) & ~invalid
    cleaned = np.column_stack([
        repair(a, m, heat.time) for a, m in zip(provisional.T, spikes.T)
    ])
    return cleaned, int(invalid.sum()), int(spikes.sum())


def spectrum_figure(heat, cleaned, destination):
    # Evaluate a continuous interpolant on a display grid in physical coordinates.
    time = np.linspace(0, heat.time[-1], 150)
    wave = np.linspace(heat.wavelength[0], heat.wavelength[-1], 180)
    interpolated = PchipInterpolator(heat.time, cleaned, axis=0)(time)
    surface = PchipInterpolator(heat.wavelength, interpolated, axis=1)(wave)
    time_grid, wave_grid = np.meshgrid(time, wave, indexing="ij")
    fig = plt.figure(figsize=(8, 5.5))
    fig.subplots_adjust(left=.06, right=.96, bottom=.14, top=.92)
    ax = fig.add_subplot(projection="3d")
    plot = ax.plot_surface(time_grid, wave_grid, surface, cmap="viridis", linewidth=0,
                           rcount=100, ccount=100)
    ax.set(xlabel="Time (s)", ylabel="Wavelength (nm)", zlabel="Intensity (W/cm²)",
           title=f"Attachment {heat.number} · {heat.name}")
    ax.view_init(32, -132)
    fig.colorbar(plot, ax=ax, shrink=0.65, pad=0.12, label="Intensity (W/cm²)")
    fig.savefig(destination, dpi=150, bbox_inches="tight", pad_inches=.22)
    plt.close(fig)


def signal_models(heat):
    oxygen = repair(heat.oxygen, (~np.isfinite(heat.oxygen) | (heat.oxygen < 0)),
                    heat.time)
    oxygen = repair(oxygen, local_spikes(oxygen, 11, 3.0), heat.time)
    oxygen = np.maximum.accumulate(oxygen)
    # Fit monotone cumulative oxygen on coarser knots so the derivative
    # describes flow changes rather than the workbook's quantization steps.
    knots = np.unique(np.r_[np.arange(0, len(oxygen), round(5 / DT)), len(oxygen) - 1])
    cumulative = PchipInterpolator(heat.time[knots], oxygen[knots])
    gas_models = []
    counts = []
    for raw in heat.gas.T:
        outlier = (raw < 0) | local_spikes(raw, 9, 2.0)
        gas_models.append(PchipInterpolator(heat.time, repair(raw, outlier, heat.time)))
        counts.append(int(outlier.sum()))
    return cumulative, gas_models, counts


def gas_signals(heat, destination):
    cumulative, gas_models, counts = signal_models(heat)
    time = np.linspace(0, heat.time[-1], 2 * len(heat.time) - 1)
    values = {
        "lance_flow_m3s": cumulative.derivative()(time).clip(min=0),
        "cumulative_oxygen_m3": cumulative(time),
    }
    for key, model in zip(GASES, gas_models):
        integral = model.antiderivative()
        values[f"{key}_m3s"] = model(time)
        values[f"total_{key}_m3"] = integral(time) - integral(0)
    frame = pd.DataFrame({"time_s": time, **values})
    frame.to_csv(destination / f"{heat.name}_signals.csv", index=False, float_format="%.6f")
    fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex=True, layout="constrained")
    labels = ("Lance flow (m³/s)", "Cumulative oxygen (m³)", "CO₂ flow (m³/s)",
              "CO₂ total (m³)", "CO flow (m³/s)", "CO total (m³)")
    for ax, (key, data), label in zip(axes.flat, values.items(), labels):
        ax.plot(time, data, color="#166b83", lw=1.15)
        ax.set_ylabel(label)
        ax.grid(alpha=.22)
        if not heat.valid_gas and "co" in key and "oxygen" not in key:
            ax.text(.05, .75, "Original gas values are all zero", transform=ax.transAxes)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(f"Attachment {heat.number} · {heat.name}")
    fig.savefig(destination / f"{heat.name}_signals.png", dpi=145)
    plt.close(fig)
    sample = slice(None, None, 2)
    rec = {"heat": heat.name, "attachment": heat.number,
           "target_status": "variable" if heat.valid_gas else FLAT}
    o2 = values["cumulative_oxygen_m3"][sample]
    for i, key in enumerate(GASES):
        rec[f"{key}_repairs"] = counts[i]
        rec[f"{key}_total_m3"] = values[f"total_{key}_m3"][-1] if heat.valid_gas else np.nan
        for col, label in [(f"{key}_m3s", "flow"), (f"total_{key}_m3", "total")]:
            signal = values[col][sample]
            rec[f"spearman_oxygen_{key}_{label}"] = (
                spearmanr(o2, signal).statistic if heat.valid_gas and np.ptp(signal) else np.nan
            )
    return rec, values["lance_flow_m3s"][sample], np.column_stack([
        values[f"{key}_m3s"][sample] for key in GASES
    ])


def causal_oxygen_flow(heat, span_s=12):
    oxygen = pd.Series(heat.oxygen).where(lambda x: np.isfinite(x) & (x >= 0)).ffill()
    oxygen = oxygen.cummax()
    steps = int(span_s / DT)
    index = np.arange(len(oxygen))
    back = np.maximum(0, index - steps)
    elapsed = np.maximum(1, index - back) * DT
    flow = (oxygen.to_numpy() - oxygen.to_numpy()[back]) / elapsed
    return np.nan_to_num(flow)


def estimate_delay_windows(heat, flow, gas):
    """Compare changes in O₂ flow with later gas changes in disjoint windows."""
    if not heat.valid_gas:
        return []
    smoothed_flow = uniform_filter1d(flow, 11)
    smoothed_gas = uniform_filter1d(gas, 11, axis=0)
    online_flow = causal_oxygen_flow(heat)
    lookback = int(5 / DT)
    x = smoothed_flow[lookback:] - smoothed_flow[:-lookback]
    y = smoothed_gas[lookback:] - smoothed_gas[:-lookback]
    lags = np.arange(0, int(120 / DT) + 1, int(2 / DT))
    width = int(100 / DT)
    records = []
    for start in range(0, len(x) - width - int(lags[-1]) + 1, width):
        baseline = x[start:start + width]
        if baseline.std() < 0.02:
            continue
        baseline = baseline - baseline.mean()
        center_flow = np.median(online_flow[start + lookback:start + lookback + width])
        for i, target in enumerate(GASES):
            future = np.array([y[start + lag:start + lag + width, i] for lag in lags])
            future = future - future.mean(axis=1, keepdims=True)
            denominators = np.linalg.norm(future, axis=1) * np.linalg.norm(baseline)
            correlations = np.divide(future @ baseline, denominators,
                                     out=np.full(len(lags), -np.inf), where=denominators > 0)
            peak = int(np.argmax(correlations))
            if not np.isfinite(correlations[peak]):
                continue
            far = np.abs(lags - lags[peak]) * DT >= 10
            runner_up = np.max(correlations[far])
            prominence = correlations[peak] - runner_up if np.isfinite(runner_up) else np.nan
            supported = correlations[peak] >= .25 and prominence >= .05
            records.append({
                "heat": heat.name, "target": target,
                "window_start_s": (start + lookback) * DT,
                "window_end_s": (start + lookback + width - 1) * DT,
                "lance_flow_m3s": float(center_flow), "lag_s": lags[peak] * DT,
                "correlation": correlations[peak], "prominence": prominence,
                "screened_in": supported, "search_boundary": peak in (0, len(lags) - 1),
            })
    return records


def delay_by_flow(windows, destination):
    valid = windows[windows.screened_in & ~windows.search_boundary].copy()
    bins = [-np.inf, 3, 5, 7, np.inf]
    labels = ["<3", "3–<5", "5–<7", "≥7"]
    valid["flow_band_m3s"] = pd.cut(valid.lance_flow_m3s, bins, labels=labels, right=False)
    table = valid.groupby(["target", "flow_band_m3s"], observed=False).agg(
        windows=("lag_s", "size"), heats=("heat", "nunique"), median_lag_s=("lag_s", "median"),
        p25_lag_s=("lag_s", lambda s: s.quantile(.25)),
        p75_lag_s=("lag_s", lambda s: s.quantile(.75)),
    ).reset_index()
    table.to_csv(destination / "delay_by_flow.csv", index=False, float_format="%.4f")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharey=True, layout="constrained")
    for target, ax in zip(GASES, axes):
        all_rows = windows[windows.target == target]
        good = valid[valid.target == target]
        ax.scatter(all_rows.lance_flow_m3s, all_rows.lag_s, s=20, c="#a9b6ba",
                   label="All evaluated windows")
        ax.scatter(good.lance_flow_m3s, good.lag_s, s=28, c="#166b83",
                   label="Passed peak screening")
        ax.set(title=GAS_LABELS[target], xlabel="Lance flow (m³/s)", ylabel="Candidate delay (s)")
        ax.grid(alpha=.22)
    axes[0].legend(loc="best", fontsize=8)
    fig.savefig(destination / "delay_vs_flow.png", dpi=155)
    plt.close(fig)
