"""
Deep Physical Reality & Feasibility Audit for Problem 1 v2.1 Reconstructed Outputs
Independently scrutinizes the actual saved Phase 2 .npz files across 7 real-world feasibility tests:
  1) Off-grid C^2 continuity & Runge oscillation check (t+0.25s, wl+0.15nm)
  2) Real longest contiguous blackout gap inspection (Att06 & Att09)
  3) Real 650-col bloomed saturation flash repair inspection (Att06)
  4) Root-cause audit of NarrowSat_Pts=0 on Att02/Att04/Att07 (true flash vs over-flagging?)
  5) Persistent 383.29nm hot pixel + CTE tail & 340-355nm UV edge check
  6) Real atomic peak profile fidelity (Na 589nm, K 766.5/769.9nm: height, FWHM, area, shift)
  7) Downstream Question 3 feasibility check (6B/6C alignment on unified 341-1023nm grid,
     dark baseline zeroing, spectral-to-flue-gas correlation preservation, and Att05 frame-loss behavior)
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CACHE_DIR = os.environ.get("Q1_CACHE_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache_npz"))
NATIVE_DIR = os.environ.get("Q1_NATIVE_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "results", "q1_v21", "recon_native_npz"))
ALIGN_DIR = os.environ.get("Q1_ALIGN_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "results", "q1_v21", "recon_aligned_q3_npz"))
AUDIT_DIR = os.environ.get("Q1_AUDIT_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "results", "q1_v21", "reality_audit"))
os.makedirs(AUDIT_DIR, exist_ok=True)


def eval_from_saved_npz(npz_data, t_q, wl_q):
    wl = npz_data["wavelengths"]
    t_grid = npz_data["t_grid"]
    mu_vec = npz_data["mu_vec"]
    psi_basis = npz_data["psi_basis"]
    c_full = npz_data["c_full"]
    K = int(npz_data["K_selected"])

    mu_q = CubicSpline(wl, mu_vec, bc_type="not-a-knot")(wl_q)
    C_q = np.column_stack([CubicSpline(t_grid, c_full[:, k], bc_type="not-a-knot")(t_q) for k in range(K)])
    Psi_q = np.vstack([CubicSpline(wl, psi_basis[k], bc_type="not-a-knot")(wl_q) for k in range(K)])
    return np.maximum(mu_q[None, :] + C_q @ Psi_q, 10.0)


def main():
    tags = [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))]

    print("=========================================================================")
    print("AUDIT 1: OFF-GRID C^2 CONTINUITY & RUNGE OSCILLATION CHECK (ALL 10 HEATS)")
    print("=========================================================================")
    c2_rows = []
    for tag in tags:
        nat = np.load(os.path.join(NATIVE_DIR, f"{tag}_q1_v21.npz"))
        wl = nat["wavelengths"]
        t_grid = nat["t_grid"]
        sp_on = nat["spec_recon"].astype(np.float64)
        # Half-step off-grid evaluation
        t_mid = t_grid[:-1] + 0.25
        wl_mid = 0.5 * (wl[:-1] + wl[1:])
        sp_off = eval_from_saved_npz(nat, t_mid, wl_mid)
        # Compare off-grid vs bilinear neighbor envelope to detect any Runge overshoot/undershoot
        on_local_max = np.maximum.reduce([sp_on[:-1, :-1], sp_on[:-1, 1:], sp_on[1:, :-1], sp_on[1:, 1:]])
        on_local_min = np.minimum.reduce([sp_on[:-1, :-1], sp_on[:-1, 1:], sp_on[1:, :-1], sp_on[1:, 1:]])
        overshoot = np.maximum(sp_off - on_local_max, 0.0)
        undershoot = np.maximum(on_local_min - sp_off, 0.0)
        c2_rows.append({
            "Tag": tag,
            "OffGrid_NaN": int(np.isnan(sp_off).sum()),
            "OffGrid_Min": round(float(sp_off.min()), 2),
            "OffGrid_Max": round(float(sp_off.max()), 1),
            "OnGrid_Max": round(float(sp_on.max()), 1),
            "Max_Overshoot": round(float(overshoot.max()), 2),
            "P999_Overshoot": round(float(np.percentile(overshoot, 99.9)), 2),
            "Max_Undershoot": round(float(undershoot.max()), 2),
        })
    df_c2 = pd.DataFrame(c2_rows)
    print(df_c2.to_string(index=False))

    print("\n=========================================================================")
    print("AUDIT 2: INVESTIGATING NarrowSat_Pts=0 ON Att02, Att04, Att07, Att08")
    print("=========================================================================")
    for tag in ["Att02_6B03206", "Att04_6B03271", "Att07_6C02059", "Att08_6C02060"]:
        raw = np.load(os.path.join(CACHE_DIR, f"{tag}.npz"))
        nat = np.load(os.path.join(NATIVE_DIR, f"{tag}_q1_v21.npz"))
        wl = raw["wavelengths"]
        sp = raw["spec"].astype(np.float64)
        sp_hat = nat["spec_recon"].astype(np.float64)
        sat_rows = np.where((sp >= 65530).any(axis=1))[0]
        dark = nat["dark_base"]
        blue_m = (wl >= 430) & (wl <= 510)
        main_m = (wl >= 550) & (wl <= 950)
        e_blue = np.mean(sp[:, blue_m] - dark[:, None], axis=1)
        e_main = np.mean(sp[:, main_m] - dark[:, None], axis=1)
        print(f"-- {tag}: {len(sat_rows)} frames with >=65530 --")
        for r in sat_rows[:8]:
            lo, hi = max(0, r - 20), min(len(sp), r + 21)
            eb_med = np.median(e_blue[lo:hi])
            em_med = np.median(e_main[lo:hi])
            ns = int((sp[r] >= 65530).sum())
            sat_cols = np.where(sp[r] >= 65530)[0]
            print(
                f"   frame={r:4d} (t={r*0.5:5.1f}s) | n_sat={ns:3d} ({wl[sat_cols[0]]:.1f}..{wl[sat_cols[-1]]:.1f}nm) | "
                f"r_blue={e_blue[r]/max(eb_med,1):.2f}x | r_main={e_main[r]/max(em_med,1):.2f}x | "
                f"flagged_flash={bool(nat['is_flash'][r])} | recon_peak_at_sat={sp_hat[r, sat_cols].max():.0f}"
            )

    print("\n=========================================================================")
    print("AUDIT 3: ATOMIC PEAK SHAPE FIDELITY (HEIGHT, FWHM, INTEGRATED AREA)")
    print("=========================================================================")
    pk_rows = []
    for tag in tags:
        raw = np.load(os.path.join(CACHE_DIR, f"{tag}.npz"))
        nat = np.load(os.path.join(NATIVE_DIR, f"{tag}_q1_v21.npz"))
        wl = raw["wavelengths"]
        sp = raw["spec"].astype(np.float64)
        sp_hat = nat["spec_recon"].astype(np.float64)
        good = nat["good_frames"]
        late = good.copy()
        late[: int(0.55 * len(sp))] = False
        late &= ~((sp >= 65530).any(axis=1))
        r_med = np.median(sp[late], axis=0)
        h_med = np.median(sp_hat[late], axis=0)
        row = {"Tag": tag}
        for pname, lo, hi in [("Na589", 583, 595), ("K766", 758, 776)]:
            m = (wl >= lo) & (wl <= hi)
            w_sub = wl[m]
            yr, yh = r_med[m], h_med[m]
            base_r = np.linspace(yr[0], yr[-1], len(yr))
            base_h = np.linspace(yh[0], yh[-1], len(yh))
            pr, ph = np.maximum(yr - base_r, 0), np.maximum(yh - base_h, 0)
            area_r = float(np.trapezoid(pr, w_sub))
            area_h = float(np.trapezoid(ph, w_sub))
            area_err = abs(area_h - area_r) / (area_r + 1e-6) * 100.0
            fwhm_r = int(np.sum(pr >= 0.5 * pr.max()))
            fwhm_h = int(np.sum(ph >= 0.5 * ph.max()))
            row[f"{pname}_AreaErr%"] = round(area_err, 2)
            row[f"{pname}_FWHM_px(R/H)"] = f"{fwhm_r}/{fwhm_h}"
        pk_rows.append(row)
    print(pd.DataFrame(pk_rows).to_string(index=False))

    print("\n=========================================================================")
    print("AUDIT 4: DOWNSTREAM Q3 FEASIBILITY (6B vs 6C ALIGNMENT & FLUE-GAS CORR)")
    print("=========================================================================")
    q3_rows = []
    for tag in tags:
        raw = np.load(os.path.join(CACHE_DIR, f"{tag}.npz"))
        al = np.load(os.path.join(ALIGN_DIR, f"{tag}_aligned.npz"))
        cwl = al["common_wl"]
        sp_net = al["spec_aligned_net"].astype(np.float64)
        co2, co = al["co2"], al["co"]
        # Check dark residual (355-415nm)
        dm = (cwl >= 355) & (cwl <= 415)
        dark_med = float(np.median(sp_net[:, dm]))
        # Check Na & K aligned peak wavelengths on late blow
        late_med = np.median(sp_net[int(0.55 * len(sp_net)):], axis=0)
        m_na = (cwl >= 584) & (cwl <= 594)
        m_k = (cwl >= 760) & (cwl <= 771)
        na_wl = float(cwl[m_na][np.argmax(late_med[m_na])])
        k_wl = float(cwl[m_k][np.argmax(late_med[m_k])])
        # Compare Spearman correlation with CO & CO2 BEFORE vs AFTER cleaning!
        raw_mean = raw["spec"].astype(np.float64).mean(axis=1)
        clean_mean = sp_net.mean(axis=1)
        if np.std(co) > 0:
            r_raw_co = float(pd.Series(raw_mean).corr(pd.Series(co), method="spearman"))
            r_cln_co = float(pd.Series(clean_mean).corr(pd.Series(co), method="spearman"))
            r_raw_co2 = float(pd.Series(raw_mean).corr(pd.Series(co2), method="spearman"))
            r_cln_co2 = float(pd.Series(clean_mean).corr(pd.Series(co2), method="spearman"))
        else:
            r_raw_co = r_cln_co = r_raw_co2 = r_cln_co2 = np.nan
        q3_rows.append({
            "Tag": tag,
            "Aligned_DarkMed": round(dark_med, 2),
            "Na_Peak_nm": round(na_wl, 2),
            "K_Peak_nm": round(k_wl, 2),
            "Raw_CO_Spear": round(r_raw_co, 3) if not np.isnan(r_raw_co) else np.nan,
            "Clean_CO_Spear": round(r_cln_co, 3) if not np.isnan(r_cln_co) else np.nan,
            "Raw_CO2_Spear": round(r_raw_co2, 3) if not np.isnan(r_raw_co2) else np.nan,
            "Clean_CO2_Spear": round(r_cln_co2, 3) if not np.isnan(r_cln_co2) else np.nan,
        })
    df_q3 = pd.DataFrame(q3_rows)
    print(df_q3.to_string(index=False))
    df_q3.to_csv(os.path.join(AUDIT_DIR, "q1_downstream_q3_feasibility.csv"), index=False, encoding="utf-8-sig")

    print("\n=========================================================================")
    print("AUDIT 5: GENERATING VISUAL REALITY VERIFICATION PANELS (6-PANEL AUDIT)")
    print("=========================================================================")
    # Visual Inspection Figure: 6 tough real-world cases!
    fig, axes = plt.subplots(3, 2, figsize=(16, 13), dpi=150)

    # Load Att06 & Att09 & Att02 & Att05
    r6 = np.load(os.path.join(CACHE_DIR, "Att06_6C02058.npz"))
    n6 = np.load(os.path.join(NATIVE_DIR, "Att06_6C02058_q1_v21.npz"))
    r9 = np.load(os.path.join(CACHE_DIR, "Att09_6C02107.npz"))
    n9 = np.load(os.path.join(NATIVE_DIR, "Att09_6C02107_q1_v21.npz"))
    r2 = np.load(os.path.join(CACHE_DIR, "Att02_6B03206.npz"))
    n2 = np.load(os.path.join(NATIVE_DIR, "Att02_6B03206_q1_v21.npz"))
    r5 = np.load(os.path.join(CACHE_DIR, "Att05_6C02057.npz"))
    n5 = np.load(os.path.join(NATIVE_DIR, "Att05_6C02057_q1_v21.npz"))

    # Panel 1: Att09 longest contiguous blackout time-series zoom at 700nm & 766.7nm
    bo9 = n9["is_blackout"]
    # Find longest run
    best_s, best_len, cur_s, cur_len = 0, 0, 0, 0
    for i, f in enumerate(bo9):
        if f:
            if cur_len == 0: cur_s = i
            cur_len += 1
            if cur_len > best_len: best_s, best_len = cur_s, cur_len
        else:
            cur_len = 0
    print(f"Att09 longest contiguous blackout: frames {best_s}..{best_s+best_len-1} (len={best_len}, t={best_s*0.5:.1f}..{(best_s+best_len-1)*0.5:.1f}s)")
    ax = axes[0, 0]
    w_lo, w_hi = max(0, best_s - 25), min(len(bo9), best_s + best_len + 25)
    t9 = n9["t_grid"][w_lo:w_hi]
    j700_9 = int(np.argmin(np.abs(r9["wavelengths"] - 700.0)))
    ax.plot(t9, r9["spec"][w_lo:w_hi, j700_9], "o-", ms=3, lw=1.0, alpha=0.75, label="原始 700nm")
    ax.plot(t9, n9["spec_recon"][w_lo:w_hi, j700_9], "-", lw=2.0, color="tab:red", label="v2.1 重构 700nm")
    ax.axvspan(best_s * 0.5, (best_s + best_len - 1) * 0.5, color="gray", alpha=0.22, label=f"连续{best_len}帧黑帧区")
    ax.set_title(f"1. 真实最长连续黑帧修复检验 (Att09, 连黑 {best_len} 帧)")
    ax.set_xlabel("时间 t (s)")
    ax.set_ylabel("光强 (W/cm^2)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: Att06 650-col bloomed flash frame spectrum (raw clipped vs reconstructed vs neighbor clean)
    ax = axes[0, 1]
    wl6 = r6["wavelengths"]
    sat_cnt6 = (r6["spec"] >= 65530).sum(axis=1)
    r_flash = int(np.argmax(sat_cnt6))
    # Find nearest good frame before & after
    g6 = np.where(n6["good_frames"])[0]
    g_prev = g6[g6 < r_flash][-1]
    g_next = g6[g6 > r_flash][0]
    ax.plot(wl6, r6["spec"][r_flash], lw=1.1, alpha=0.8, color="tab:orange", label=f"原始过曝闪帧 t={r_flash*0.5:.1f}s (饱和{sat_cnt6[r_flash]}列)")
    ax.plot(wl6, n6["spec_recon"][r_flash], lw=2.0, color="tab:red", label="v2.1 重构该帧")
    ax.plot(wl6, r6["spec"][g_prev], lw=1.0, ls="--", color="tab:blue", alpha=0.75, label=f"前邻正常帧 t={g_prev*0.5:.1f}s")
    ax.plot(wl6, r6["spec"][g_next], lw=1.0, ls=":", color="tab:green", alpha=0.85, label=f"后邻正常帧 t={g_next*0.5:.1f}s")
    ax.set_title("2. 真实 650+ 列饱和过曝闪帧修复检验 (Att06)")
    ax.set_xlabel("波长 (nm)")
    ax.set_ylabel("光强 (W/cm^2)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: Att06 UV/Dark zone (340-420nm) hot pixel 383.29nm + CTE tail & edge defects
    ax = axes[1, 0]
    m_uv = wl6 < 420
    ax.plot(wl6[m_uv], np.median(r6["spec"][:, m_uv], axis=0), lw=1.3, label="Att06 原始时间中位数")
    ax.plot(wl6[m_uv], np.median(n6["spec_recon"][:, m_uv], axis=0), lw=2.0, color="tab:red", label="Att06 v2.1 重构中位数")
    ax.set_title("3. 暗区固定热像素 (383.29nm 含拖尾) 与短波边缘修复")
    ax.set_xlabel("波长 (nm)")
    ax.set_ylabel("中位数光强")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 4: Atomic Peak Fidelity zoom (Na 589nm & K 766.5/769.9nm on Att02 & Att06)
    ax = axes[1, 1]
    wl2 = r2["wavelengths"]
    g2_late = n2["good_frames"].copy()
    g2_late[: int(0.6 * len(g2_late))] = False
    m_k2 = (wl2 >= 756) & (wl2 <= 776)
    m_k6 = (wl6 >= 756) & (wl6 <= 776)
    g6_late = n6["good_frames"].copy()
    g6_late[: int(0.6 * len(g6_late))] = False
    ax.plot(wl2[m_k2], np.median(r2["spec"][g2_late][:, m_k2], axis=0), "o", ms=3, alpha=0.65, label="Att02 原始 K 双峰")
    ax.plot(wl2[m_k2], np.median(n2["spec_recon"][g2_late][:, m_k2], axis=0), "-", lw=2.0, label="Att02 v2.1 重构")
    ax.plot(wl6[m_k6], np.median(r6["spec"][g6_late][:, m_k6], axis=0), "s", ms=2.5, alpha=0.55, label="Att06 原始 K 双峰")
    ax.plot(wl6[m_k6], np.median(n6["spec_recon"][g6_late][:, m_k6], axis=0), "--", lw=2.0, label="Att06 v2.1 重构")
    ax.set_title("4. K 双峰 (766.5/769.9nm) 零削峰与峰位保真检验")
    ax.set_xlabel("波长 (nm)")
    ax.set_ylabel("光强 (W/cm^2)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 5: Att06 whole-heat energy timeline (200-450s) raw vs v2.1 reconstructed
    ax = axes[2, 0]
    m_main6 = (wl6 >= 550) & (wl6 <= 950)
    e_raw6 = r6["spec"][:, m_main6].mean(axis=1)
    e_hat6 = n6["spec_recon"][:, m_main6].mean(axis=1)
    t6 = n6["t_grid"]
    m_t = (t6 >= 200) & (t6 <= 420)
    ax.plot(t6[m_t], e_raw6[m_t], lw=0.9, alpha=0.6, label="Att06 原始波段均值 (含大量黑帧+闪帧)")
    ax.plot(t6[m_t], e_hat6[m_t], lw=1.6, color="tab:red", label="Att06 v2.1 重构波段均值")
    ax.set_title("5. 最复杂炉次 Att06 中段全时序去异常保真对比 (200~420s)")
    ax.set_xlabel("时间 t (s)")
    ax.set_ylabel("550~950nm 平均光强")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 6: Att05 largest O2 frame-loss jump (dO2 = 54.91 m3) boundary behavior
    ax = axes[2, 1]
    o2d5 = np.diff(r5["o2"])
    jump_idx = int(np.argmax(o2d5))
    lo5, hi5 = max(0, jump_idx - 35), min(len(r5["o2"]), jump_idx + 35)
    t5 = n5["t_grid"][lo5:hi5]
    j700_5 = int(np.argmin(np.abs(r5["wavelengths"] - 700.0)))
    ax.plot(t5, r5["spec"][lo5:hi5, j700_5], "o-", ms=3, lw=1.0, alpha=0.7, label="Att05 原始 700nm")
    ax.plot(t5, n5["spec_recon"][lo5:hi5, j700_5], "-", lw=2.0, color="tab:red", label="Att05 v2.1 重构 700nm")
    ax.axvline(jump_idx * 0.5, color="purple", ls="--", lw=1.5, label=f"最大丢帧跳变点 (dO2={o2d5[jump_idx]:.1f}m3)")
    ax.set_title("6. Att05 最大丢帧跳变点邻域重构稳定性检验")
    ax.set_xlabel("名义采样时间 t (s)")
    ax.set_ylabel("光强 (W/cm^2)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(AUDIT_DIR, "q1_physical_reality_audit_6panel.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved 6-panel visual reality audit figure: {fig_path}")


if __name__ == "__main__":
    main()
