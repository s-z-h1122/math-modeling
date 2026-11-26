import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CACHE_DIR = os.environ.get("Q1_CACHE_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache_npz"))
OUT_DIR = os.environ.get("Q1_ANATOMY_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "results", "q1_anatomy"))
os.makedirs(OUT_DIR, exist_ok=True)


def mad_sigma(x, axis=None):
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    return 1.4826 * mad


def analyze_all_heats():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    data = {os.path.basename(f)[:-4]: np.load(f) for f in files}

    summary_rows = []
    sat_details = []
    peak_details = []
    noise_details = []
    svd_details = []

    for tag, d in data.items():
        wl = d["wavelengths"]
        spec = d["spec"].astype(np.float64)  # (T, W)
        T, W = spec.shape

        # 0. Dark baseline per frame
        dark_mask = wl < 420.0
        dark_base = np.median(spec[:, dark_mask], axis=1)  # (T,)
        spec_net = spec - dark_base[:, None]

        # 1. Type A: Whole-frame occlusion / blackout analysis
        frame_energy = np.mean(spec_net[:, (wl >= 550) & (wl <= 950)], axis=1)
        roll_med = pd.Series(frame_energy).rolling(21, center=True, min_periods=5).median().values
        ratio = frame_energy / (roll_med + 1e-6)
        # Mild dip (<0.7), severe blackout (<0.5), near-total (<0.25) where signal is active (roll_med > 1000)
        active = roll_med > 1000.0
        dip_70 = active & (ratio < 0.70)
        dip_50 = active & (ratio < 0.50)
        dip_25 = active & (ratio < 0.25)

        # Max & distribution of consecutive blackout runs (ratio < 0.5)
        runs = []
        cur = 0
        for flag in dip_50:
            if flag:
                cur += 1
            elif cur > 0:
                runs.append(cur)
                cur = 0
        if cur > 0:
            runs.append(cur)
        max_run = max(runs) if runs else 0
        run_len_1 = sum(1 for r in runs if r == 1)
        run_len_2_3 = sum(1 for r in runs if 2 <= r <= 3)
        run_len_4p = sum(1 for r in runs if r >= 4)

        # Also check positive whole-frame flash spikes (ratio > 1.5)
        flash_150 = active & (ratio > 1.50)

        # 2. Type B: Wavelength-axis spikes vs persistent bad pixels
        # High-pass along wavelength: second difference d2 = s[j] - 0.5*(s[j-1]+s[j+1])
        d2_wl = spec[:, 1:-1] - 0.5 * (spec[:, :-2] + spec[:, 2:])
        # Robust noise sigma per frame from d2_wl (for white noise, std(d2) = sqrt(1.5)*sigma)
        sigma_wl_frame = np.median(np.abs(d2_wl), axis=1) * 1.4826 / np.sqrt(1.5)
        # Local spike indicator: |d2_wl| > 8 * sqrt(1.5) * sigma_wl_frame (avoiding Na/K peaks? Let's check!)
        thresh_wl = 8.0 * np.sqrt(1.5) * sigma_wl_frame[:, None]
        wl_spike_mask = np.abs(d2_wl) > thresh_wl
        # Check which wavelengths trigger most frequently (persistent bad pixels vs real sharp peaks)
        col_trigger_rate = wl_spike_mask.mean(axis=0)
        persistent_cols = np.where(col_trigger_rate > 0.20)[0] + 1  # offset by 1
        persistent_wls = [round(float(wl[c]), 2) for c in persistent_cols[:15]]
        # Random isolated spikes (excluding persistent cols)
        non_pers = np.ones(W - 2, dtype=bool)
        non_pers[col_trigger_rate > 0.20] = False
        rand_wl_spike_rate = float(wl_spike_mask[:, non_pers].mean())

        # Check dark-zone (<420nm) persistent bad pixels specifically!
        dark_spec = spec[:, dark_mask]
        dark_col_med = np.median(dark_spec, axis=0)
        dark_col_dev = np.abs(dark_col_med - np.median(dark_col_med))
        dark_bad_pixels = int(np.sum(dark_col_dev > 5.0 * mad_sigma(dark_col_med).item()))

        # 3. Type C: Saturation at 65535
        sat_mask = spec >= 65535.0
        sat_cnt = int(sat_mask.sum())
        sat_frames = int(np.sum(sat_mask.any(axis=1)))
        sat_wls = wl[sat_mask.any(axis=0)]
        sat_wl_range = (round(float(sat_wls.min()), 2), round(float(sat_wls.max()), 2)) if len(sat_wls) else (None, None)
        # Max contiguous saturated pixels in a single frame
        max_sat_width = 0
        if sat_cnt > 0:
            for r_idx in np.where(sat_mask.any(axis=1))[0]:
                row_s = sat_mask[r_idx]
                c = 0
                for v in row_s:
                    if v:
                        c += 1
                        if c > max_sat_width:
                            max_sat_width = c
                    else:
                        c = 0
        sat_details.append({
            "Tag": tag, "sat_pts": sat_cnt, "sat_frames": sat_frames,
            "sat_wl_min": sat_wl_range[0], "sat_wl_max": sat_wl_range[1],
            "max_contig_sat_px": max_sat_width,
        })

        # 4. Real atomic peak FWHM (Na ~589nm, K ~766.5nm) on late-blow mean spectrum
        late_spec = np.median(spec[int(0.6 * T):], axis=0)
        peak_info = {"Tag": tag}
        for pname, lo, hi in [("Na_589", 580, 598), ("K_766", 755, 780), ("K_770", 768.5, 773.5)]:
            m = (wl >= lo) & (wl <= hi)
            sub_wl = wl[m]
            sub_y = late_spec[m]
            # Local continuum baseline from endpoints
            base = np.linspace(sub_y[0], sub_y[-1], len(sub_y))
            prom = sub_y - base
            p_idx = int(np.argmax(prom))
            p_val = float(prom[p_idx])
            half_mask = prom >= 0.5 * p_val
            fwhm_px = int(half_mask.sum())
            fwhm_nm = float(sub_wl[half_mask][-1] - sub_wl[half_mask][0]) if fwhm_px > 1 else 0.34
            peak_info[f"{pname}_wl"] = round(float(sub_wl[p_idx]), 2)
            peak_info[f"{pname}_prom"] = round(p_val, 1)
            peak_info[f"{pname}_fwhm_px"] = fwhm_px
            peak_info[f"{pname}_fwhm_nm"] = round(fwhm_nm, 2)
        peak_details.append(peak_info)

        # 5. Noise heteroscedasticity: dark band (<420nm) vs mid band (650-700nm, smooth continuum) vs bright band (800-850nm)
        noise_info = {"Tag": tag}
        for bname, lo, hi in [("dark_350_410", 350, 410), ("mid_650_700", 650, 700), ("high_800_850", 800, 850)]:
            m = (wl >= lo) & (wl <= hi)
            sub = spec[:, m]
            d2 = sub[:, 1:-1] - 0.5 * (sub[:, :-2] + sub[:, 2:])
            sig = float(np.median(np.abs(d2)) * 1.4826 / np.sqrt(1.5))
            mean_i = float(np.mean(sub))
            noise_info[f"{bname}_mean"] = round(mean_i, 1)
            noise_info[f"{bname}_sigma"] = round(sig, 2)
            noise_info[f"{bname}_snr"] = round(mean_i / (sig + 1e-6), 1)
        # Temporal lag-1 autocorrelation on non-blackout continuum (700nm)
        idx_700 = int(np.argmin(np.abs(wl - 700.0)))
        s700 = spec[:, idx_700]
        ac1 = float(np.corrcoef(s700[:-1], s700[1:])[0, 1])
        ac2 = float(np.corrcoef(s700[:-2], s700[2:])[0, 1])
        ac5 = float(np.corrcoef(s700[:-5], s700[5:])[0, 1])
        noise_info["temp_ac1"] = round(ac1, 4)
        noise_info["temp_ac2"] = round(ac2, 4)
        noise_info["temp_ac5"] = round(ac5, 4)
        noise_details.append(noise_info)

        # 6. SVD low-rank energy & peak fidelity (global RMSE vs Na/K peak RMSE)
        # Center along time
        mu_wl = spec.mean(axis=0, keepdims=True)
        Xc = spec - mu_wl
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        energy_tot = np.sum(S ** 2)
        m_peaks = ((wl >= 585) & (wl <= 593)) | ((wl >= 762) & (wl <= 773))
        svd_row = {"Tag": tag}
        for K in [3, 5, 10, 15, 20, 30]:
            Xhat = (U[:, :K] * S[:K]) @ Vt[:K, :] + mu_wl
            err = spec - Xhat
            rmse_all = float(np.sqrt(np.mean(err ** 2)))
            rmse_peak = float(np.sqrt(np.mean(err[:, m_peaks] ** 2)))
            var_exp = float(np.sum(S[:K] ** 2) / energy_tot * 100.0)
            svd_row[f"K{K}_var%"] = round(var_exp, 3)
            svd_row[f"K{K}_rmse"] = round(rmse_all, 1)
            svd_row[f"K{K}_peak_rmse"] = round(rmse_peak, 1)
        svd_details.append(svd_row)

        summary_rows.append({
            "Tag": tag, "T": T, "W": W,
            "dip70_cnt": int(dip_70.sum()), "dip50_cnt": int(dip_50.sum()),
            "dip25_cnt": int(dip_25.sum()), "flash150_cnt": int(flash_150.sum()),
            "max_blackout_run": max_run,
            "runs_1": run_len_1, "runs_2_3": run_len_2_3, "runs_4p": run_len_4p,
            "pers_wl_cols": len(persistent_cols), "dark_bad_px": dark_bad_pixels,
            "rand_wl_spike%": round(rand_wl_spike_rate * 100, 4),
            "sat_cnt": sat_cnt,
            "pers_wls_sample": str(persistent_wls[:6]),
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sat = pd.DataFrame(sat_details)
    df_peak = pd.DataFrame(peak_details)
    df_noise = pd.DataFrame(noise_details)
    df_svd = pd.DataFrame(svd_details)

    print("=== 1. OUTLIER TAXONOMY (Blackouts, Flashes, Wavelength Spikes, Persistent Bad Pixels) ===")
    print(df_sum.to_string(index=False))
    print("\n=== 2. SATURATION (65535) ANATOMY ===")
    print(df_sat.to_string(index=False))
    print("\n=== 3. REAL ATOMIC PEAK WIDTHS (FWHM) — CRITICAL FOR FILTER DESIGN ===")
    print(df_peak.to_string(index=False))
    print("\n=== 4. NOISE HETEROSCEDASTICITY & TEMPORAL AUTOCORRELATION ===")
    print(df_noise.to_string(index=False))
    print("\n=== 5. SVD LOW-RANK ENERGY & ATOMIC PEAK FIDELITY ===")
    print(df_svd[["Tag", "K3_var%", "K5_var%", "K10_var%", "K20_var%", "K5_rmse", "K10_rmse", "K20_rmse", "K5_peak_rmse", "K10_peak_rmse", "K20_peak_rmse"]].to_string(index=False))

    # Save tables
    df_sum.to_csv(os.path.join(OUT_DIR, "q1_outlier_taxonomy.csv"), index=False, encoding="utf-8-sig")
    df_sat.to_csv(os.path.join(OUT_DIR, "q1_saturation_anatomy.csv"), index=False, encoding="utf-8-sig")
    df_peak.to_csv(os.path.join(OUT_DIR, "q1_peak_widths.csv"), index=False, encoding="utf-8-sig")
    df_noise.to_csv(os.path.join(OUT_DIR, "q1_noise_profile.csv"), index=False, encoding="utf-8-sig")
    df_svd.to_csv(os.path.join(OUT_DIR, "q1_svd_benchmark.csv"), index=False, encoding="utf-8-sig")

    # 7. Generate a 4-panel diagnostic figure on Att06_6C02058 (the most challenging heat: most blackouts + most saturation + highest glitches)
    d6 = data["Att06_6C02058"]
    wl6, sp6 = d6["wavelengths"], d6["spec"].astype(np.float64)
    T6 = len(sp6)
    t6 = np.arange(T6) * 0.5
    fe6 = np.mean(sp6[:, (wl6 >= 550) & (wl6 <= 950)], axis=1)
    rm6 = pd.Series(fe6).rolling(21, center=True, min_periods=5).median().values

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=150)

    # Panel A: Blackout frame zoom (t=250..350s)
    ax = axes[0, 0]
    m_t = (t6 >= 250) & (t6 <= 350)
    ax.plot(t6[m_t], fe6[m_t], lw=1.1, label="原始帧均值(550~950nm)")
    ax.plot(t6[m_t], rm6[m_t], lw=1.6, ls="--", color="tab:red", label="21点滑动中位数")
    b_idx = m_t & (fe6 < 0.5 * rm6)
    ax.scatter(t6[b_idx], fe6[b_idx], color="black", s=24, zorder=5, label=f"整帧遮挡黑帧(<50%)")
    ax.set_title("A. 整帧遮挡/黑帧局部放大 (Att06, 250~350s)")
    ax.set_xlabel("时间 t (s)")
    ax.set_ylabel("波段平均光强")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel B: Saturation & K doublet zoom (755..780nm on a saturated frame)
    ax = axes[0, 1]
    sat_rows = np.where((sp6 >= 65535).any(axis=1))[0]
    r_sat = sat_rows[len(sat_rows) // 2]
    m_k = (wl6 >= 755) & (wl6 <= 780)
    ax.plot(wl6[m_k], sp6[r_sat, m_k], "o-", ms=2.5, lw=1.2, label=f"饱和帧 t={r_sat*0.5:.1f}s")
    ax.axhline(65535, color="tab:red", ls="--", lw=1, label="16-bit 饱和上限 65535")
    # Also plot a nearby non-saturated high frame
    non_sat = [r for r in range(max(0, r_sat-10), min(T6, r_sat+10)) if sp6[r, m_k].max() < 65000 and fe6[r] > 0.8 * rm6[r]]
    if non_sat:
        ax.plot(wl6[m_k], sp6[non_sat[0], m_k], lw=1.2, alpha=0.8, label=f"邻近未饱和帧 t={non_sat[0]*0.5:.1f}s")
    ax.set_title("B. K 双峰(766.5/769.9nm) 16-bit 饱和平顶截断")
    ax.set_xlabel("波长 λ (nm)")
    ax.set_ylabel("光强 (W/cm²)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel C: Wavelength-axis noise & persistent bad pixels in dark zone (340..420nm)
    ax = axes[1, 0]
    m_dk = wl6 < 420
    ax.plot(wl6[m_dk], np.median(sp6[:, m_dk], axis=0), lw=1.0, label="Att06 暗区时间中位数谱")
    d2 = data["Att02_6B03206"]
    wl2, sp2 = d2["wavelengths"], d2["spec"].astype(np.float64)
    m_dk2 = wl2 < 420
    ax.plot(wl2[m_dk2], np.median(sp2[:, m_dk2], axis=0), lw=1.0, label="Att02 暗区时间中位数谱")
    ax.set_title("C. 暗区(340~420nm) 固定坏像素与 6B/6C 暗基线对比")
    ax.set_xlabel("波长 λ (nm)")
    ax.set_ylabel("中位数光强")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel D: SVD singular value decay across all 10 heats
    ax = axes[1, 1]
    for tag, d in data.items():
        sp = d["spec"].astype(np.float64)
        Xc = sp - sp.mean(axis=0, keepdims=True)
        _, S, _ = np.linalg.svd(Xc, full_matrices=False)
        ax.semilogy(np.arange(1, 51), S[:50] / S[0], lw=1.1, alpha=0.85, label=tag[:5])
    ax.axvline(10, color="gray", ls="--", lw=0.9)
    ax.axvline(20, color="gray", ls=":", lw=0.9)
    ax.set_title("D. 全部 10 炉光谱 SVD 归一化奇异值衰减谱 (前 50 阶)")
    ax.set_xlabel("奇异值阶数 k")
    ax.set_ylabel("σ_k / σ_1 (对数坐标)")
    ax.legend(ncol=2, fontsize=7.5)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "q1_spectral_anatomy_diagnostic.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nSaved diagnostic figure: {fig_path}")


if __name__ == "__main__":
    analyze_all_heats()
