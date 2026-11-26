import os
import re
import time
import json
import numpy as np
import pandas as pd
from scipy import stats

RAW_DIR = os.environ.get("RAW_XLSB_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw_xlsb"))
REF_DIR = os.environ.get("REF_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "ref"))
OUT_DIR = os.environ.get("VERIFY_OUT_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "results", "verification"))
CACHE_DIR = os.environ.get("Q1_CACHE_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache_npz"))
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def load_xlsb_fast(path):
    """Load first sheet (and report all sheet names) using calamine or pyxlsb."""
    try:
        from python_calamine import CalamineWorkbook
        wb = CalamineWorkbook.from_path(path)
        sheets = wb.sheet_names
        rows = wb.get_sheet_by_index(0).to_python()
        return sheets, rows
    except Exception:
        from pyxlsb import open_workbook
        with open_workbook(path) as wb:
            sheets = wb.sheets
            with wb.get_sheet(1) as sheet:
                rows = [[c.v for c in r] for r in sheet.rows()]
        return sheets, rows


def parse_heat_file(path):
    sheets, rows = load_xlsb_fast(path)
    # Inspect header rows to find where data starts
    header_info = []
    data_start = None
    wavelengths = None
    for i in range(min(10, len(rows))):
        r = rows[i]
        preview = [x for x in r[:6]] + ["..."] + [x for x in r[-4:]]
        header_info.append((i, len(r), preview))
        # Detect row where col 0 is numeric 1 (or 1.0)
        if data_start is None and len(r) > 2000:
            v0 = r[0]
            if isinstance(v0, (int, float)) and abs(v0 - 1) < 1e-6:
                data_start = i
                # Previous row should contain wavelengths
                wl_row = rows[i - 1]
                wavelengths = [x for x in wl_row[1:-3]]

    if data_start is None:
        raise RuntimeError(f"Could not detect data start in {path}: {header_info}")

    # Filter out trailing empty rows
    valid_rows = [r for r in rows[data_start:] if r and r[0] is not None]
    ncols = len(valid_rows[0])
    arr = np.empty((len(valid_rows), ncols), dtype=np.float64)
    non_numeric_count = 0
    for i, r in enumerate(valid_rows):
        for j in range(ncols):
            v = r[j] if j < len(r) else None
            if v is None or v == "":
                arr[i, j] = np.nan
            else:
                try:
                    arr[i, j] = float(v)
                except Exception:
                    arr[i, j] = np.nan
                    non_numeric_count += 1

    wl = np.array([float(x) for x in wavelengths], dtype=np.float64)
    seq = arr[:, 0]
    spec = arr[:, 1 : 1 + len(wl)]
    o2 = arr[:, -3]
    co2 = arr[:, -2]
    co = arr[:, -1]
    return {
        "sheets": sheets,
        "header_info": header_info,
        "data_start": data_start,
        "ncols": ncols,
        "non_numeric_count": non_numeric_count,
        "wavelengths": wl,
        "seq": seq,
        "spec": spec,
        "o2": o2,
        "co2": co2,
        "co": co,
    }


def main():
    manifest = pd.read_csv(os.path.join(REF_DIR, "00_dataset_manifest.csv"))
    ref_qc = pd.read_csv(os.path.join(REF_DIR, "01_数据质量体检总表.csv"))
    ref_lag = pd.read_csv(os.path.join(REF_DIR, "04_有效炉次跨模态滞后初估表.csv"))

    results = []
    o2_step_reports = []
    gas_step_reports = []
    spectral_reports = []
    all_wls = []
    heats_data = {}

    t0_all = time.time()
    for _, mrow in manifest.iterrows():
        att = int(mrow["Att"])
        tag = mrow["Tag"]
        grp = mrow["Group"]
        fname = mrow["File"]
        fpath = os.path.join(RAW_DIR, fname)
        t0 = time.time()
        d = parse_heat_file(fpath)
        dt = time.time() - t0
        print(f"[{tag}] loaded in {dt:.2f}s | sheets={d['sheets']} | rows={len(d['seq'])} | cols={d['ncols']} | wl={len(d['wavelengths'])} ({d['wavelengths'][0]:.3f}..{d['wavelengths'][-1]:.3f})")

        # Save cache npz (float32 for spec to keep fast & compact, float64 for time/o2/gas/wl)
        cache_path = os.path.join(CACHE_DIR, f"{tag}.npz")
        np.savez_compressed(
            cache_path,
            wavelengths=d["wavelengths"],
            seq=d["seq"],
            spec=d["spec"].astype(np.float32),
            o2=d["o2"],
            co2=d["co2"],
            co=d["co"],
        )
        heats_data[tag] = d
        all_wls.append(d["wavelengths"])

        seq = d["seq"]
        spec = d["spec"]
        o2 = d["o2"]
        co2 = d["co2"]
        co = d["co"]
        n = len(seq)

        # Basic checks
        seq_diff = np.diff(seq)
        seq_cont = bool(np.all(seq_diff == 1) and seq[0] == 1)
        spec_nan_rate = float(np.isnan(spec).mean())
        spec_min = float(np.nanmin(spec))
        spec_max = float(np.nanmax(spec))
        spec_mean = float(np.nanmean(spec))
        sat_exact_65535 = int(np.sum(spec == 65535.0))
        sat_gt_60000_rate = float(np.mean(spec > 60000.0))

        # Dark baseline (lambda < 420 nm)
        dark_mask = d["wavelengths"] < 420.0
        dark_mean = float(np.nanmean(spec[:, dark_mask]))
        dark_std = float(np.nanstd(spec[:, dark_mask]))
        # Whole-frame occlusion / blackout detection: frame mean drops < 50% of local rolling median (window=11)
        frame_mean = np.nanmean(spec, axis=1)
        frame_med = pd.Series(frame_mean).rolling(15, center=True, min_periods=3).median().values
        blackout_frames = int(np.sum((frame_mean < 0.5 * frame_med) & (frame_med > dark_mean + 500)))

        # O2 step analysis
        o2_diff = np.diff(o2)
        o2_neg = int(np.sum(o2_diff < 0))
        o2_zero_ratio = float(np.mean(o2_diff == 0))
        o2_pos = o2_diff[o2_diff > 0]
        # Quantiles & distribution of positive O2 steps
        q_pos = np.percentile(o2_pos, [5, 25, 50, 75, 90, 95, 99, 100]) if len(o2_pos) else [0]*8
        # Count double jumps (> 1.5 * median positive step)
        med_step = float(np.median(o2_pos)) if len(o2_pos) else 0.0
        double_jumps = int(np.sum(o2_pos > 1.5 * med_step))
        # Consecutive zero run lengths in o2_diff
        zero_runs = []
        cur = 0
        for v in o2_diff:
            if v == 0:
                cur += 1
            else:
                if cur > 0:
                    zero_runs.append(cur)
                cur = 0
        if cur > 0:
            zero_runs.append(cur)
        zero_run_counts = pd.Series(zero_runs).value_counts().head(5).to_dict() if zero_runs else {}

        # CO2 / CO analysis
        co2_diff = np.diff(co2)
        co_diff = np.diff(co)
        co2_zero_ratio = float(np.mean(co2 == 0))
        co_zero_ratio = float(np.mean(co == 0))
        co2_neg = int(np.sum(co2 < 0))
        co_neg = int(np.sum(co < 0))
        co2_flat_ratio = float(np.mean(co2_diff == 0))
        co_flat_ratio = float(np.mean(co_diff == 0))

        # Oxygen mass balance integral check (assuming dt=0.5s)
        dt_s = 0.5
        v_co2 = float(np.trapezoid(co2, dx=dt_s))
        v_co = float(np.trapezoid(co, dx=dt_s))
        v_o2_gas = v_co2 + 0.5 * v_co
        v_o2_blown = float(o2[-1] - o2[0])
        o2_balance_ratio = v_o2_gas / v_o2_blown if v_o2_blown > 0 else np.nan

        results.append({
            "Tag": tag, "Group": grp, "Rows": n, "Duration_0.5s": n * 0.5,
            "Seq_Cont": seq_cont, "NonNumeric": d["non_numeric_count"],
            "Spec_Min": spec_min, "Spec_Max": spec_max, "Spec_Mean": round(spec_mean, 2),
            "Sat_65535_Cnt": sat_exact_65535, "Sat_gt60k_Rate": sat_gt_60000_rate,
            "Dark_Mean_lt420": round(dark_mean, 2), "Dark_Std": round(dark_std, 2),
            "Blackout_Frames": blackout_frames,
            "O2_Start": float(o2[0]), "O2_End": float(o2[-1]), "O2_Neg": o2_neg,
            "O2_Flat_Ratio": round(o2_zero_ratio, 4),
            "O2_AvgFlow_0.5s": round((o2[-1] - o2[0]) / (n * 0.5), 4),
            "O2_PosStep_Med": round(med_step, 4), "O2_PosStep_P95": round(q_pos[5], 4),
            "O2_PosStep_Max": round(q_pos[7], 4), "O2_DoubleJumps": double_jumps,
            "CO2_Max": float(np.nanmax(co2)), "CO2_Mean": round(float(np.nanmean(co2)), 4),
            "CO2_ZeroRatio": round(co2_zero_ratio, 4), "CO2_FlatRatio": round(co2_flat_ratio, 4),
            "CO_Max": float(np.nanmax(co)), "CO_Mean": round(float(np.nanmean(co)), 4),
            "CO_ZeroRatio": round(co_zero_ratio, 4), "CO_FlatRatio": round(co_flat_ratio, 4),
            "V_CO2_0.5s": round(v_co2, 1), "V_CO_0.5s": round(v_co, 1),
            "V_O2_gas_0.5s": round(v_o2_gas, 1), "O2_Gas_vs_Blown": round(o2_balance_ratio, 4),
        })
        o2_step_reports.append({
            "Tag": tag,
            "n_pos_steps": int(len(o2_pos)),
            "pos_step_quantiles_5_25_50_75_90_95_99_100": [round(float(x), 3) for x in q_pos],
            "zero_run_len_top5": zero_run_counts,
        })

    print(f"\nAll 10 files parsed & cached in {time.time()-t0_all:.2f}s")

    # 1. Wavelength grid consistency check
    wl0 = all_wls[0]
    wl_max_diff = max(float(np.max(np.abs(w - wl0))) for w in all_wls)
    print(f"\nWavelength grid identical across all 10 heats? max_abs_diff = {wl_max_diff}")
    print(f"Wavelength range: {wl0[0]} .. {wl0[-1]} (len={len(wl0)}, step range {np.min(np.diff(wl0)):.4f}..{np.max(np.diff(wl0)):.4f})")

    # 2. Compare against reference 01_数据质量体检总表.csv
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, "audit_summary.csv"), index=False, encoding="utf-8-sig")

    print("\n=== DISCREPANCY CHECK vs 01_数据质量体检总表.csv ===")
    comp_cols = [
        ("Rows", "采样点数", 0),
        ("Spec_Min", "光谱最小值", 1e-6),
        ("Spec_Max", "光谱最大值", 1e-6),
        ("Spec_Mean", "光谱均值", 0.05),
        ("O2_Start", "耗氧起点_m3", 1e-6),
        ("O2_End", "耗氧终点_m3", 1e-6),
        ("O2_Neg", "耗氧回退点数", 0),
        ("O2_Flat_Ratio", "耗氧平台比例(diff=0)", 5e-4),
        ("CO2_Max", "CO2最大值", 1e-6),
        ("CO2_Mean", "CO2均值", 1e-3),
        ("CO2_ZeroRatio", "CO2零值比例", 5e-4),
        ("CO_Max", "CO最大值", 1e-6),
        ("CO_Mean", "CO均值", 1e-3),
        ("CO_ZeroRatio", "CO零值比例", 5e-4),
    ]
    mismatches = []
    for i, r in res_df.iterrows():
        rr = ref_qc[ref_qc["炉次"] == r["Tag"]].iloc[0]
        for my_c, ref_c, tol in comp_cols:
            diff = abs(float(r[my_c]) - float(rr[ref_c]))
            if diff > tol:
                mismatches.append((r["Tag"], my_c, r[my_c], rr[ref_c], diff))
    if not mismatches:
        print("ALL 14 core metrics match 01_数据质量体检总表.csv within tolerance!")
    else:
        print(f"Found {len(mismatches)} mismatches:")
        for m in mismatches:
            print(" ", m)

    # 3. Print key audit table (O2 mass balance, dark baseline, blackout frames, O2 step median)
    print("\n=== DEEP AUDIT TABLE (O2 Balance, Dark Baseline, Blackouts, O2 Steps) ===")
    cols_show = ["Tag", "Group", "Rows", "Dark_Mean_lt420", "Blackout_Frames", "Sat_65535_Cnt",
                 "O2_Flat_Ratio", "O2_PosStep_Med", "O2_PosStep_Max", "O2_DoubleJumps",
                 "V_O2_gas_0.5s", "O2_End", "O2_Gas_vs_Blown"]
    print(res_df[cols_show].to_string(index=False))

    # 4. Deep dive on Att05_6C02057 vs Att06_6C02058 (solving the 775-row mystery!)
    print("\n=== DEEP DIVE: Att05_6C02057 vs Att06_6C02058 ===")
    for tag in ["Att05_6C02057", "Att06_6C02058", "Att02_6B03206"]:
        d = heats_data[tag]
        o2d = np.diff(d["o2"])
        print(f"\n-- {tag} (rows={len(d['seq'])}) --")
        print("First 25 rows (seq, o2, d_o2, co2, co):")
        for k in range(25):
            do2 = d["o2"][k] - d["o2"][k-1] if k > 0 else 0.0
            print(f"  {int(d['seq'][k]):4d} | o2={d['o2'][k]:8.2f} (d={do2:6.2f}) | co2={d['co2'][k]:6.2f} | co={d['co'][k]:6.2f}")
        print("Mid-blow rows 300..325:")
        for k in range(300, 325):
            do2 = d["o2"][k] - d["o2"][k-1]
            print(f"  {int(d['seq'][k]):4d} | o2={d['o2'][k]:8.2f} (d={do2:6.2f}) | co2={d['co2'][k]:6.2f} | co={d['co'][k]:6.2f}")
        print("O2 step summary:", [x for x in o2_step_reports if x["Tag"] == tag][0])

    # 5. Verify 6B vs 6C PCA separation BEFORE and AFTER dark-baseline subtraction & SNV
    print("\n=== PCA 6B vs 6C SEPARATION AUDIT ===")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Sample evenly 200 frames per heat
    X_raw_list, X_dark_list, X_snv_list, grp_list = [], [], [], []
    dark_mask = wl0 < 420.0
    for tag, d in heats_data.items():
        spec = d["spec"]
        idx = np.linspace(0, len(spec) - 1, 200, dtype=int)
        sub = spec[idx]
        X_raw_list.append(sub)
        # Subtract dark baseline per frame
        dark_base = np.median(sub[:, dark_mask], axis=1, keepdims=True)
        sub_dark = sub - dark_base
        X_dark_list.append(sub_dark)
        # SNV per frame
        mu = sub.mean(axis=1, keepdims=True)
        sd = sub.std(axis=1, keepdims=True) + 1e-8
        X_snv_list.append((sub - mu) / sd)
        grp_list.extend([tag[:5] + "_" + ("6B" if "6B" in tag else "6C")] * len(idx))

    is_6c = np.array([1 if "6C" in g else 0 for g in grp_list])
    for name, X in [("Raw (StandardScaled)", StandardScaler().fit_transform(np.vstack(X_raw_list))),
                    ("Raw (Unscaled)", np.vstack(X_raw_list)),
                    ("Dark-Subtracted (Unscaled)", np.vstack(X_dark_list)),
                    ("SNV Normalized", np.vstack(X_snv_list))]:
        pca = PCA(n_components=5, random_state=42)
        Z = pca.fit_transform(X)
        evr = pca.explained_variance_ratio_
        # Linear separability score via simple 1D/2D logistic regression AUC
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        clf = LogisticRegression().fit(Z[:, :2], is_6c)
        auc = roc_auc_score(is_6c, clf.predict_proba(Z[:, :2])[:, 1])
        print(f"  {name:28s}: PC1={evr[0]*100:5.2f}%, PC1~3={evr[:3].sum()*100:5.2f}%, 6B-vs-6C AUC(PC1,2)={auc:.4f}")

    # 6. Audit CO2/CO onset & O2 step-up timing (TRUE physical delay probe!)
    print("\n=== TRUE PHYSICAL DELAY PROBE: O2 Flow Step-Up vs Flue-Gas Response ===")
    for tag, d in heats_data.items():
        if tag == "Att01_6B03205":
            continue
        o2 = d["o2"]
        co2 = d["co2"]
        co = d["co"]
        n = len(o2)
        t = np.arange(n) * 0.5
        # Smooth O2 flow over 5s window (10 steps)
        o2_flow_5s = np.gradient(pd.Series(o2).rolling(11, center=True, min_periods=1).mean().values, 0.5)
        # Find early blow flow ramp-up: where o2_flow_5s crosses 3.5 m3/s (for normal heats)
        # First inspect o2_flow_5s at t=20, 40, 60, 80, 100, 120, 200, 300s
        sample_t = [10, 30, 50, 70, 90, 110, 150, 200, 300]
        flows = [round(float(o2_flow_5s[min(int(st*2), n-1)]), 2) for st in sample_t]
        # Find CO2 onset (first time CO2 > 1.0 m3/s) and CO onset (first time CO > 1.0 m3/s)
        t_co2_on = float(t[np.argmax(co2 > 1.0)]) if np.any(co2 > 1.0) else np.nan
        t_co_on = float(t[np.argmax(co > 1.0)]) if np.any(co > 1.0) else np.nan
        # Find O2 start of blowing (first time o2_flow_5s > 1.0) and step-up (> 3.5)
        t_o2_on = float(t[np.argmax(o2_flow_5s > 1.0)]) if np.any(o2_flow_5s > 1.0) else np.nan
        t_o2_hi = float(t[np.argmax(o2_flow_5s > 3.5)]) if np.any(o2_flow_5s > 3.5) else np.nan
        # Check detrended CCF between o2_flow_5s diff and Q_o2_gas diff
        q_gas = co2 + 0.5 * co
        q_gas_s = pd.Series(q_gas).rolling(11, center=True, min_periods=1).mean().values
        dq_o2 = np.diff(o2_flow_5s)
        dq_gas = np.diff(q_gas_s)
        # search lag 0..120 steps (0..60s)
        best_lag, best_r = 0, -2
        for lag in range(0, 121):
            if lag == 0:
                a, b = dq_o2, dq_gas
            else:
                a, b = dq_o2[:-lag], dq_gas[lag:]
            r = np.corrcoef(a, b)[0, 1]
            if r > best_r:
                best_r = r
                best_lag = lag * 0.5
        print(f"  {tag}: t_O2>1={t_o2_on:5.1f}s, t_O2>3.5={t_o2_hi:5.1f}s | t_CO2>1={t_co2_on:5.1f}s (dt={t_co2_on-t_o2_on:4.1f}s), t_CO>1={t_co_on:5.1f}s | dQ_CCF best_lag={best_lag:4.1f}s (r={best_r:.3f}) | flows@t={flows}")

    # Save json report
    with open(os.path.join(OUT_DIR, "o2_step_reports.json"), "w", encoding="utf-8") as f:
        json.dump(o2_step_reports, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
