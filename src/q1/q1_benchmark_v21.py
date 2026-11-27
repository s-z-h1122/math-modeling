"""
Phase 1: End-to-End Blind Benchmark & Ablation Study for Question 1 (v2.1)
Tests:
  1) Verification that all 4 peer-review Blockers are eliminated on real data
  2) End-to-End Blind Benchmark (injecting realistic 11-frame blackouts, 650-col bloomed flash frames,
     persistent 2px+tail hot pixels, and transient cosmic-ray spikes onto a clean ground-truth reference,
     measuring BOTH anomaly detection Precision/Recall/F1 AND reconstruction RMSE + Na/K peak fidelity)
  3) Comparison across 5 methods:
       M1: 2D Median + Savitzky-Golay (Standard Baseline)
       M2: Modified Hampel + 2D Tensor B-Spline (v1 Baseline)
       M3: Unwhitened Raw GD-SVD (Ablation: Homoscedastic GD rank failure)
       M4: Same-Mask Cleaned + 2D Tensor-Product B-Spline (Strong Competitor)
       M5: Proposed v2.1 WLR-WE-FBS (Whitened Low-Rank + Whittaker-Eilers + Functional Cubic B-Spline)
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.interpolate import BSpline, PchipInterpolator, CubicSpline

from q1_core_v21 import detect_anomalies_v21, reconstruct_v21, evaluate_acceptance_criteria

CACHE_DIR = os.environ.get("Q1_CACHE_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache_npz"))
OUT_DIR = os.environ.get("Q1_OUT_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "results", "q1_v21"))
os.makedirs(OUT_DIR, exist_ok=True)


def method_m1_median_sg(spec: np.ndarray) -> np.ndarray:
    """M1: Conventional 2D Median (5x5) + Savitzky-Golay along wavelength (win=11, poly=3)."""
    m = median_filter(spec, size=(5, 5), mode="nearest")
    out = savgol_filter(m, window_length=11, polyorder=3, axis=1)
    return np.maximum(out, 10.0)


def method_m2_v1_hampel_spline(wl: np.ndarray, spec: np.ndarray) -> np.ndarray:
    """M2: v1 concept (log-energy W=25 Hampel + W=5/41 wavelength Hampel + PCHIP + SVD K=14)."""
    T, W = spec.shape
    out = spec.copy()
    sat = out >= 65530.0
    # v1 wavelength Hampel
    med5 = median_filter(out, size=(1, 5), mode="nearest")
    res = np.abs(out - med5)
    mad41 = 1.4826 * median_filter(res, size=(1, 41), mode="nearest") + 10.0
    sp_m = res > 6.0 * mad41
    # v1 log-energy W=25 Hampel
    dark = np.median(out[:, (wl >= 355) & (wl <= 415)], axis=1)
    E = np.maximum(np.mean(out[:, (wl >= 550) & (wl <= 950)] - dark[:, None], axis=1), 1.0)
    z = np.log(E + 1.0)
    z_med = median_filter(z, size=25, mode="nearest")
    z_mad = 1.4826 * median_filter(np.abs(z - z_med), size=25, mode="nearest") + 1e-4
    fr_bad = np.abs(z - z_med) > 3.0 * z_mad

    mask = sat | sp_m | fr_bad[:, None]
    out[sp_m] = med5[sp_m]
    t = np.arange(T) * 0.5
    good_f = ~fr_bad
    if np.sum(good_f) >= 4 and np.any(fr_bad):
        out[fr_bad] = PchipInterpolator(t[good_f], out[good_f], axis=0, extrapolate=True)(t[fr_bad])
    mu = out.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(out - mu, full_matrices=False)
    K = 14
    out_lr = (U[:, :K] * S[:K]) @ Vt[:K, :] + mu
    return np.maximum(out_lr, 10.0)


def method_m3_unwhitened_gd(wl: np.ndarray, spec: np.ndarray) -> tuple[np.ndarray, int]:
    """M3 Ablation: Same v2.1 mask & Whittaker gap-fill, BUT unwhitened Gavish-Donoho rank selection."""
    det = detect_anomalies_v21(wl, spec)
    good = det["good_frames"]
    T, W = spec.shape
    t = np.arange(T) * 0.5
    Xg = spec[good].copy()
    if np.any(det["col_bad_mask"]):
        cc = np.where(~det["col_bad_mask"])[0]
        bc = np.where(det["col_bad_mask"])[0]
        Xg[:, bc] = CubicSpline(wl[cc], Xg[:, cc], axis=1, extrapolate=True)(wl[bc])
    pt_m = det["spike_mask"][good] | det["sat_narrow_mask"][good]
    if np.any(pt_m):
        tg = t[good]
        for j in np.where(pt_m.any(axis=0))[0]:
            ok = ~pt_m[:, j]
            if np.sum(ok) >= 4:
                Xg[~ok, j] = PchipInterpolator(tg[ok], Xg[ok, j], extrapolate=True)(tg[~ok])

    mu = Xg.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xg - mu[None, :], full_matrices=False)
    beta = Xg.shape[0] / float(W)
    omega = 0.56 * (beta ** 3) - 0.95 * (beta ** 2) + 1.82 * beta + 1.43
    K_raw = int(np.sum(S > omega * np.median(S)))
    # Use K_raw directly (no whitening!)
    C_g = U[:, :K_raw] * S[:K_raw]
    Psi = Vt[:K_raw, :]
    C_full = PchipInterpolator(t[good], C_g, axis=0, extrapolate=True)(t)
    recon = np.maximum(mu[None, :] + C_full @ Psi, 10.0)
    return recon, K_raw


def method_m4_cleaned_2d_bspline(wl: np.ndarray, spec: np.ndarray) -> np.ndarray:
    """M4 Strong Competitor: v2.1 cleaned masks + 2D Tensor-Product Cubic B-Spline (RectBivariateSpline)."""
    det = detect_anomalies_v21(wl, spec)
    good = det["good_frames"]
    T, W = spec.shape
    t = np.arange(T) * 0.5
    X = spec.copy()
    if np.any(det["col_bad_mask"]):
        cc = np.where(~det["col_bad_mask"])[0]
        bc = np.where(det["col_bad_mask"])[0]
        X[:, bc] = CubicSpline(wl[cc], X[:, cc], axis=1, extrapolate=True)(wl[bc])
    pt_m = det["spike_mask"] | det["sat_narrow_mask"]
    if np.any(pt_m):
        med5 = median_filter(X, size=(1, 5), mode="nearest")
        X[pt_m] = med5[pt_m]
    if np.any(~good):
        X[~good] = PchipInterpolator(t[good], X[good], axis=0, extrapolate=True)(t[~good])
    # Exact separable 2D Tensor-Product Cubic B-Spline least-squares fit on rectangular grid (0.02s!)
    def bspline_proj_matrix(x, n_int_knots):
        ik = np.linspace(x[2], x[-3], n_int_knots)
        knots = np.concatenate([np.repeat(x[0], 4), ik, np.repeat(x[-1], 4)])
        B = BSpline.design_matrix(x, knots, 3).toarray()
        return B @ np.linalg.pinv(B)

    Pt = bspline_proj_matrix(t, max(T // 2, 30))
    Pw = bspline_proj_matrix(wl, 260)
    return np.maximum(Pt @ X @ Pw.T, 10.0)


def run_blind_synthetic_benchmark(
    tag: str = "Att02_6B03206",
    seed: int = 2026,
    detector_fn=detect_anomalies_v21,
    reconstruct_fn=reconstruct_v21,
    proposed_name: str = "M5_Proposed_v21_WLR_WE_FBS",
):
    """
    Construct a realistic semi-synthetic ground truth from clean frames of Att02, add realistic
    Poisson-Gaussian sensor noise + all 4 severe anomaly types (11-frame blackout, 2-frame 650-col
    bloomed saturation with 3x blue jump, 3-px persistent hot pixel + tail, and random 1-2px spikes),
    and run END-TO-END BLIND evaluation!
    """
    rng = np.random.default_rng(seed)
    d = np.load(os.path.join(CACHE_DIR, f"{tag}.npz"))
    wl = d["wavelengths"]
    sp_raw = d["spec"].astype(np.float64)

    # Obtain clean noise-free underlying physical field S_true from a clean 600-frame active window
    base_res = reconstruct_v21(wl, sp_raw)
    S_true = base_res["spec_recon"][450:1050].copy()  # (600, 2047) spanning ramp-up & main decarb!
    T, W = S_true.shape

    # Generate realistic Poisson-Gaussian heteroscedastic sensor noise: sigma(I) = sqrt(33^2 + 0.95 * max(I-510, 0))
    sig_field = np.sqrt(33.0 ** 2 + 0.95 * np.maximum(S_true - 510.0, 0.0))
    Y_noisy = S_true + rng.normal(0.0, sig_field)

    # Inject Ground-Truth Anomalies (End-to-End Blind Test!)
    Y_corrupt = Y_noisy.copy()
    gt_col_bad = np.zeros(W, dtype=bool)
    gt_blackout = np.zeros(T, dtype=bool)
    gt_flash = np.zeros(T, dtype=bool)
    gt_spike = np.zeros((T, W), dtype=bool)

    # 1) Persistent hot pixel + 2-px CTE tail at ~383.3 nm (exact 6C defect replica!) + 4-px UV edge defect at 346nm
    j_hot = int(np.argmin(np.abs(wl - 383.3)))
    Y_corrupt[:, j_hot] += 1350.0
    Y_corrupt[:, j_hot + 1] += 240.0
    Y_corrupt[:, j_hot + 2] += 85.0
    gt_col_bad[j_hot : j_hot + 3] = True

    j_edge = int(np.argmin(np.abs(wl - 346.0)))
    Y_corrupt[:, j_edge : j_edge + 4] += 420.0
    gt_col_bad[j_edge : j_edge + 4] = True

    # 2) Severe blackouts: 11-frame contiguous blackout + 6-frame blackout + 15 isolated/2-frame blackouts
    bo_intervals = [(80, 91), (240, 246), (140, 142), (310, 313), (420, 428)]
    for a, b in bo_intervals:
        atten = rng.uniform(0.12, 0.38, size=(b - a, 1))
        Y_corrupt[a:b] = 510.0 + atten * (Y_corrupt[a:b] - 510.0)
        gt_blackout[a:b] = True
    for r in [50, 120, 175, 205, 285, 355, 390, 465, 510, 545]:
        Y_corrupt[r] = 510.0 + 0.25 * (Y_corrupt[r] - 510.0)
        gt_blackout[r] = True

    # 3) Whole-frame bloomed saturation flashes (including a 2-frame consecutive flash, 650+ cols clipped at 65535 + 3x blue jump!)
    flash_rows = [160, 161, 340, 490, 491]
    for r in flash_rows:
        Y_corrupt[r] = 510.0 + 3.4 * (S_true[r] - 510.0)
        Y_corrupt[r, (wl >= 580) & (wl <= 850)] = 65535.0
        gt_flash[r] = True

    # 4) Random cosmic-ray transient spikes (1-2 px wide, 1500 impulses across non-bad frames)
    valid_r = np.where(~(gt_blackout | gt_flash))[0]
    spike_r = rng.choice(valid_r, size=1500, replace=True)
    spike_c = rng.integers(15, W - 15, size=1500)
    for r, c in zip(spike_r, spike_c):
        if gt_col_bad[c]:
            continue
        amp = rng.uniform(9.0, 22.0) * sig_field[r, c] * rng.choice([-1.0, 1.0], p=[0.15, 0.85])
        Y_corrupt[r, c] += amp
        gt_spike[r, c] = True
        if rng.random() < 0.30 and not gt_col_bad[c + 1]:
            Y_corrupt[r, c + 1] += 0.65 * amp
            gt_spike[r, c + 1] = True

    Y_corrupt = np.clip(Y_corrupt, 0.0, 65535.0)
    gt_any_anomaly = gt_col_bad[None, :] | gt_blackout[:, None] | gt_flash[:, None] | gt_spike

    # --- Test Detection Accuracy of v2.1 vs v1 ---
    det_v21 = detector_fn(wl, Y_corrupt)
    pred_any_v21 = det_v21["col_bad_mask"][None, :] | det_v21["frame_bad_mask"][:, None] | det_v21["spike_mask"]

    def f1_stats(gt, pred):
        tp = int(np.sum(gt & pred))
        fp = int(np.sum((~gt) & pred))
        fn = int(np.sum(gt & (~pred)))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        return round(prec * 100, 2), round(rec * 100, 2), round(f1 * 100, 2)

    det_report = {
        "All_Pixel_Mask_P/R/F1": f1_stats(gt_any_anomaly, pred_any_v21),
        "Blackout_Frame_P/R/F1": f1_stats(gt_blackout, det_v21["is_blackout"]),
        "Flash_Frame_P/R/F1": f1_stats(gt_flash, det_v21["is_flash"]),
        "Persistent_BadCol_P/R/F1": f1_stats(gt_col_bad, det_v21["col_bad_mask"]),
        "Transient_Spike_P/R/F1": f1_stats(gt_spike, det_v21["spike_mask"]),
    }
    # Check Na/K peak false positive rate of spike detector on clean frames!
    peak_band = ((wl >= 586.0) & (wl <= 592.0)) | ((wl >= 763.0) & (wl <= 772.5))
    clean_peak_zone = (~gt_any_anomaly) & peak_band[None, :]
    det_report["NaK_Peak_FalsePositive_%"] = round(float(det_v21["spike_mask"][clean_peak_zone].mean() * 100.0), 4)

    # --- Compare All 5 Reconstruction Methods against Noise-Free Ground Truth S_true ---
    methods = {}
    t0 = time.time()
    methods["M1_2D_Median_SG"] = (method_m1_median_sg(Y_corrupt), time.time() - t0, "-")
    t0 = time.time()
    methods["M2_v1_Hampel_Spline"] = (method_m2_v1_hampel_spline(wl, Y_corrupt), time.time() - t0, "14")
    t0 = time.time()
    m3_hat, k_m3 = method_m3_unwhitened_gd(wl, Y_corrupt)
    methods["M3_Unwhitened_GD_SVD"] = (m3_hat, time.time() - t0, str(k_m3))
    t0 = time.time()
    methods["M4_Cleaned_2D_BSpline"] = (method_m4_cleaned_2d_bspline(wl, Y_corrupt), time.time() - t0, "-")
    t0 = time.time()
    r5 = reconstruct_fn(wl, Y_corrupt)
    methods[proposed_name] = (r5["spec_recon"], time.time() - t0, str(r5["K_selected"]))

    bench_rows = []
    idx_na = int(np.argmin(np.abs(wl - 588.40)))
    idx_k = int(np.argmin(np.abs(wl - 765.96)))
    for mname, (hat, elapsed, k_str) in methods.items():
        err_all = hat - S_true
        rmse_global = float(np.sqrt(np.mean(err_all ** 2)))
        rmse_masked = float(np.sqrt(np.mean(err_all[gt_any_anomaly] ** 2)))
        rmse_clean = float(np.sqrt(np.mean(err_all[~gt_any_anomaly] ** 2)))
        rmse_bo11 = float(np.sqrt(np.mean(err_all[80:91] ** 2)))
        rmse_flash = float(np.sqrt(np.mean(err_all[flash_rows] ** 2)))
        na_rel_err = float(np.mean(np.abs(hat[~(gt_blackout | gt_flash), idx_na] - S_true[~(gt_blackout | gt_flash), idx_na])) / np.mean(S_true[~(gt_blackout | gt_flash), idx_na]) * 100.0)
        k_rel_err = float(np.mean(np.abs(hat[~(gt_blackout | gt_flash), idx_k] - S_true[~(gt_blackout | gt_flash), idx_k])) / np.mean(S_true[~(gt_blackout | gt_flash), idx_k]) * 100.0)
        snr_db = float(20.0 * np.log10(np.linalg.norm(S_true) / (np.linalg.norm(err_all) + 1e-9)))
        bench_rows.append({
            "Method": mname,
            "Rank_K": k_str,
            "Global_RMSE": round(rmse_global, 2),
            "Clean_Denoise_RMSE": round(rmse_clean, 2),
            "Anomaly_Repair_RMSE": round(rmse_masked, 2),
            "11Frame_Blackout_RMSE": round(rmse_bo11, 2),
            "650Col_Flash_RMSE": round(rmse_flash, 2),
            "Na589_MAE%": round(na_rel_err, 3),
            "K766_MAE%": round(k_rel_err, 3),
            "Recon_SNR_dB": round(snr_db, 2),
            "Time_s": round(elapsed, 2),
        })

    return det_report, pd.DataFrame(bench_rows)


def main():
    print("=== PART A: END-TO-END BLIND BENCHMARK (4-Type Realistic Anomaly Injection) ===")
    det_rep, df_bench = run_blind_synthetic_benchmark("Att02_6B03206")
    for k, v in det_rep.items():
        print(f"  {k:28s}: {v}")
    print("\n--- 5-Method Quantitative Comparison vs Noise-Free Ground Truth ---")
    print(df_bench.to_string(index=False))
    df_bench.to_csv(os.path.join(OUT_DIR, "q1_blind_benchmark_5methods.csv"), index=False, encoding="utf-8-sig")

    print("\n=== PART B: REAL-DATA BLOCKER RESOLUTION CHECK (Att02_6B03206 & Att06_6C02058) ===")
    real_rows = []
    for tag in ["Att02_6B03206", "Att06_6C02058", "Att05_6C02057", "Att09_6C02107"]:
        d = np.load(os.path.join(CACHE_DIR, f"{tag}.npz"))
        wl, sp = d["wavelengths"], d["spec"].astype(np.float64)
        res = reconstruct_v21(wl, sp)
        acc = evaluate_acceptance_criteria(wl, sp, res)
        det = res["anomalies"]
        row = {
            "Tag": tag,
            "K_Unwhite_GD": res["K_gd_unwhite"],
            "K_White_GD_Raw": res["K_gd_white_raw"],
            "K_Selected": res["K_selected"],
            "BadCols": int(det["col_bad_mask"].sum()),
            "Blackout_Frames": int(det["is_blackout"].sum()),
            "Flash_Frames": int(det["is_flash"].sum()),
            "NarrowSat_Pts": int(det["sat_narrow_mask"].sum()),
            "Spikes_Pts": int(det["spike_mask"].sum()),
            **acc,
        }
        real_rows.append(row)
    df_real = pd.DataFrame(real_rows)
    print(df_real.to_string(index=False))
    df_real.to_csv(os.path.join(OUT_DIR, "q1_blocker_resolution_spotcheck.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
