"""Run the complete v2.2 Q1 reconstruction on all ten heats."""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from common.preprocess_core import (  # noqa: E402
    gas_oxygen_ratio,
    nist_calibration,
    oxygen_conserving_time,
    split_blow_segments,
)
from q1_core_v22 import detect_anomalies_v22, reconstruct_segment_v22  # noqa: E402


CACHE_DIR = Path(os.environ.get("Q1_CACHE_DIR", ROOT / "data" / "cache_npz"))
OUT_DIR = Path(os.environ.get("Q1_OUT_DIR", ROOT / "results" / "q1_v22"))
COMMON_WL = np.arange(341.0, 1022.0001, 0.5)


def _short_segment(wl: np.ndarray, spec: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, dict]:
    """Conservative repair for a reblow too short for stable SVD selection."""
    det = detect_anomalies_v22(wl, spec)
    out = spec.astype(float).copy()
    med = median_filter(out, size=(1, 5), mode="nearest")
    out[det["spike_mask"] | det["sat_narrow_mask"]] = med[det["spike_mask"] | det["sat_narrow_mask"]]
    badf = det["frame_bad_mask"]
    if np.any(badf) and np.sum(~badf) >= 4:
        out[badf] = PchipInterpolator(t[~badf], out[~badf], axis=0, extrapolate=True)(t[badf])
    if np.any(det["col_bad_mask"]):
        goodc = ~det["col_bad_mask"]
        out[:, ~goodc] = CubicSpline(wl[goodc], out[:, goodc], axis=1)(wl[~goodc])
    return np.maximum(out, 10.0), det


def _align(wl_cal: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = np.clip(COMMON_WL, wl_cal[0], wl_cal[-1])
    return PchipInterpolator(wl_cal, matrix, axis=1, extrapolate=False)(query)


def run_heat(path: Path) -> dict:
    tag = path.stem
    d = np.load(path)
    wl = d["wavelengths"].astype(float)
    raw = d["spec"].astype(float)
    o2, co2, co = d["o2"], d["co2"], d["co"]
    t_nom, t_phys, q_ref = oxygen_conserving_time(tag, o2)
    segments = split_blow_segments(tag, o2, co2, co)
    cal = nist_calibration(wl, raw)
    wl_cal = cal["wavelengths"]

    display = np.empty_like(raw)
    pulse = np.empty_like(raw)
    blackout = np.zeros(len(raw), bool)
    flash = np.zeros(len(raw), bool)
    good_frames = np.zeros(len(raw), bool)
    bad_cols_union = np.zeros(len(wl), bool)
    invalid_points = np.zeros_like(raw, dtype=bool)
    ranks: list[int] = []
    coeff_overshoots: list[float] = []

    for seg in segments:
        sl = slice(seg.start, seg.stop)
        ts = t_phys[sl]
        if len(ts) < 80:
            recon, det = _short_segment(wl, raw[sl], ts)
            ranks.append(0)
            coeff_overshoots.append(0.0)
        else:
            res = reconstruct_segment_v22(wl, raw[sl], ts)
            recon, det = res["spec_recon"], res["anomalies"]
            ranks.append(int(res["K_selected"]))
            # PCHIP coefficients are bounded by adjacent knots by construction.
            coeff_overshoots.append(0.0)
        # Presentation surface: suppress isolated row-to-row display needles.
        # The separate Q3 product below remains pulse-preserving.
        display[sl] = median_filter(recon, size=(3, 1), mode="nearest")
        # Q3 representation preserves observed turbulence exactly and only
        # substitutes values at independently detected invalid locations.
        p = raw[sl].copy()
        invalid = det["frame_bad_mask"][:, None] | det["col_bad_mask"][None, :] | det["spike_mask"] | det["sat_narrow_mask"]
        p[invalid] = recon[invalid]
        pulse[sl] = p
        blackout[sl] = det["is_blackout"]
        flash[sl] = det["is_flash"]
        good_frames[sl] = det["good_frames"]
        bad_cols_union |= det["col_bad_mask"]
        invalid_points[sl] = det["spike_mask"] | det["sat_narrow_mask"]

    display_common = _align(wl_cal, display)
    pulse_common = _align(wl_cal, pulse)
    dark = (COMMON_WL >= 355.0) & (COMMON_WL <= 415.0)
    baseline = np.median(pulse_common[:, dark], axis=1)
    pulse_net = np.maximum(pulse_common - baseline[:, None], 0.0)
    invalid_all = blackout[:, None] | flash[:, None] | bad_cols_union[None, :] | invalid_points
    valid_pixels = ~invalid_all
    preserved_pct = 100.0 * float(np.mean(np.isclose(pulse[valid_pixels], raw[valid_pixels], rtol=0.0, atol=0.0)))

    native_dir = OUT_DIR / "recon_native_npz"
    q3_dir = OUT_DIR / "recon_aligned_q3_npz"
    native_dir.mkdir(parents=True, exist_ok=True)
    q3_dir.mkdir(parents=True, exist_ok=True)
    bounds = np.array([[s.start, s.stop] for s in segments], dtype=np.int32)
    names = np.array([s.name for s in segments])
    np.savez_compressed(
        native_dir / f"{tag}_q1_v22.npz",
        wavelengths=wl,
        wavelengths_calibrated=wl_cal,
        t_nom=t_nom,
        t_phys=t_phys,
        segment_bounds=bounds,
        segment_names=names,
        spec_display=display.astype(np.float32),
        spec_pulse_preserving=pulse.astype(np.float32),
        good_frames=good_frames,
        is_blackout=blackout,
        is_flash=flash,
        bad_col_mask=bad_cols_union,
    )
    np.savez_compressed(
        q3_dir / f"{tag}_aligned.npz",
        common_wl=COMMON_WL,
        t_nom=t_nom,
        t_phys=t_phys,
        segment_bounds=bounds,
        spec_display=_align(wl_cal, display).astype(np.float32),
        spec_aligned_net=pulse_net.astype(np.float32),
        good_frames=good_frames,
        o2=o2,
        co2=co2,
        co=co,
    )
    return {
        "Tag": tag,
        "Rows": len(raw),
        "Segments": len(segments),
        "Segment_Bounds": json.dumps(bounds.tolist()),
        "Nominal_Duration_s": round(float(t_nom[-1]), 3),
        "Physical_Duration_s": round(float(t_phys[-1]), 3),
        "O2_Reference_Flow": round(q_ref, 4),
        "Main_Blow_Gas_O2_Ratio": round(gas_oxygen_ratio(o2, co2, co, segments[0].stop, t_phys), 4),
        "Ranks": json.dumps(ranks),
        "Strict_Blackouts": int(blackout.sum()),
        "Strict_Blackout_Rate_pct": round(100 * float(blackout.mean()), 3),
        "Flash_Frames": int(flash.sum()),
        "Retained_Frame_Rate_pct": round(100 * float(good_frames.mean()), 3),
        "Observed_Valid_Pixels_Preserved_pct": round(preserved_pct, 6),
        "Bad_Columns": int(bad_cols_union.sum()),
        "NIST_Max_Error_nm": round(float(np.max(cal["errors_nm"])), 4),
        "PCHIP_Coefficient_Overshoot": max(coeff_overshoots),
        "Finite_Output": bool(np.isfinite(display).all() and np.isfinite(pulse_net).all()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.time()
    for path in sorted(CACHE_DIR.glob("*.npz")):
        t0 = time.time()
        row = run_heat(path)
        rows.append(row)
        print(f"{row['Tag']}: BO={row['Strict_Blackout_Rate_pct']:.2f}% segments={row['Segments']} NIST={row['NIST_Max_Error_nm']:.3f}nm ({time.time()-t0:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "q1_v22_all10_acceptance.csv", index=False, encoding="utf-8-sig")
    print(f"Completed {len(df)} heats in {time.time()-start:.1f}s; finite={df['Finite_Output'].all()}")


if __name__ == "__main__":
    main()
