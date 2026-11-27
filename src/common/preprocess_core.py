"""Shared, leakage-safe preprocessing for the 2026 APMCM Problem B data.

The functions in this module are deliberately independent of any prediction
target.  They are therefore safe to reuse in the LOHO experiments in Q3.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import PchipInterpolator


NIST_LINES_NM = np.array([589.29, 766.49, 769.90], dtype=np.float64)
KNOWN_SEGMENTS = {
    "Att05_6C02057": (0, 752, 775),
    "Att07_6C02059": (0, 1435, 1555),
}


@dataclass(frozen=True)
class Segment:
    name: str
    start: int
    stop: int


def split_blow_segments(tag: str, o2: np.ndarray, co2: np.ndarray, co: np.ndarray) -> list[Segment]:
    """Split main blow/reblow using the registered physical discontinuity rule.

    A boundary requires both a cumulative-O2 jump over 15 and an absolute
    jump in CO2+CO over 15.  The two audited reblow heats retain their fixed,
    published boundaries if floating-point/source-file differences hide one
    side of the discontinuity.
    """
    o2 = np.asarray(o2, dtype=float)
    gas = np.asarray(co2, dtype=float) + np.asarray(co, dtype=float)
    cuts = (np.flatnonzero((np.diff(o2) > 15.0) & (np.abs(np.diff(gas)) > 15.0)) + 1).tolist()
    if tag in KNOWN_SEGMENTS:
        _, cut, stop = KNOWN_SEGMENTS[tag]
        if len(o2) == stop:
            cuts = [cut]
    edges = [0] + sorted(set(int(c) for c in cuts if 0 < c < len(o2))) + [len(o2)]
    out: list[Segment] = []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        out.append(Segment("main_blow" if i == 0 else f"reblow_{i}", a, b))
    return out


def oxygen_conserving_time(tag: str, o2: np.ndarray, nominal_dt: float = 0.5) -> tuple[np.ndarray, np.ndarray, float]:
    """Return nominal and O2-conserving clocks plus the inferred reference flow.

    The files normally update cumulative oxygen every two 0.5-s spectra, so a
    typical positive increment is also the typical one-second oxygen volume.
    Attachments 5 and 6 contain lost/irregular spectral timestamps.  Their
    physical clock is reconstructed from cumulative oxygen divided by this
    robust in-file flow estimate.  A PCHIP bridge keeps the clock strictly
    increasing through repeated cumulative readings without changing its
    endpoints.  Other heats keep the nominal acquisition clock.
    """
    o2 = np.asarray(o2, dtype=float)
    t_nom = np.arange(len(o2), dtype=float) * nominal_dt
    dv = np.diff(o2)
    pos = dv[(dv > 2.0) & (dv < 12.0)]
    q_ref = float(np.median(pos)) if len(pos) else 4.74
    if tag not in {"Att05_6C02057", "Att06_6C02058"}:
        return t_nom, t_nom.copy(), q_ref

    ranges = [(0, len(o2))]
    if tag in KNOWN_SEGMENTS and KNOWN_SEGMENTS[tag][2] == len(o2):
        _, cut, stop = KNOWN_SEGMENTS[tag]
        ranges = [(0, cut), (cut, stop)]
    t_phys = np.empty(len(o2), dtype=float)
    offset = 0.0
    for a, b in ranges:
        raw = (np.maximum.accumulate(o2[a:b]) - o2[a]) / max(q_ref, 1e-9)
        change = np.r_[True, np.diff(raw) > 1e-10]
        idx = np.flatnonzero(change)
        if idx[-1] != len(raw) - 1:
            idx = np.r_[idx, len(raw) - 1]
        local = PchipInterpolator(idx, raw[idx], extrapolate=True)(np.arange(len(raw)))
        local = np.maximum.accumulate(local)
        t_phys[a:b] = offset + local
        offset = float(t_phys[b - 1] + nominal_dt)
    eps = 1e-6
    for i in range(1, len(t_phys)):
        if t_phys[i] <= t_phys[i - 1]:
            t_phys[i] = t_phys[i - 1] + eps
    return t_nom, t_phys, q_ref


def gas_oxygen_ratio(o2: np.ndarray, co2: np.ndarray, co: np.ndarray, stop: int | None = None, t: np.ndarray | None = None) -> float:
    """Stoichiometric audit ratio used by the official correction check."""
    s = len(o2) if stop is None else int(stop)
    gas = np.asarray(co2[:s], float) + 0.5 * np.asarray(co[:s], float)
    blown = float(np.asarray(o2[:s], float)[-1] - np.asarray(o2[:s], float)[0])
    if t is None:
        gas_volume = float(np.sum(gas) * 0.5)
    else:
        gas_volume = float(np.trapezoid(gas, np.asarray(t[:s], float)))
    return gas_volume / max(blown, 1e-9)


def _quadratic_peak(wl: np.ndarray, y: np.ndarray, target: float, half_window: float = 2.0) -> float:
    mask = np.abs(wl - target) <= half_window
    ids = np.flatnonzero(mask)
    j = int(ids[np.argmax(y[mask])])
    if j == 0 or j == len(wl) - 1:
        return float(wl[j])
    x3, y3 = wl[j - 1:j + 2], y[j - 1:j + 2]
    coef = np.polyfit(x3, y3, 2)
    if abs(coef[0]) < 1e-12:
        return float(wl[j])
    return float(np.clip(-coef[1] / (2.0 * coef[0]), x3[0], x3[-1]))


def nist_calibration(wl: np.ndarray, spec: np.ndarray) -> dict:
    """Fit a robust affine wavelength correction to the Na/K NIST anchors."""
    wl = np.asarray(wl, float)
    y = np.nanmedian(np.asarray(spec, float)[int(0.55 * len(spec)):], axis=0)
    observed = np.array([_quadratic_peak(wl, y, x) for x in NIST_LINES_NM])
    # A second-order dispersion correction is the minimum model that can
    # honour all three independent NIST anchors.  It remains strictly
    # monotone across the instrument range for these calibrated files.
    coef = np.polyfit(observed, NIST_LINES_NM, 2)
    calibrated = np.polyval(coef, wl)
    fitted = np.polyval(coef, observed)
    if np.any(np.diff(calibrated) <= 0):
        raise ValueError("NIST calibration produced a non-monotone wavelength axis")
    return {
        "wavelengths": calibrated,
        "observed_lines_nm": observed,
        "fitted_lines_nm": fitted,
        "errors_nm": np.abs(fitted - NIST_LINES_NM),
        "quadratic_coefficients": coef,
    }


def pchip_resample(x: np.ndarray, y: np.ndarray, x_new: np.ndarray, axis: int = 0) -> np.ndarray:
    """Shape-preserving resampling with endpoint clamping and no overshoot."""
    x = np.asarray(x, float)
    x_new = np.asarray(x_new, float)
    out = PchipInterpolator(x, y, axis=axis, extrapolate=False)(np.clip(x_new, x[0], x[-1]))
    return np.asarray(out)
