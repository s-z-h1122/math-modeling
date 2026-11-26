r"""
Phase 2: Full 10-Heat Continuous Spectral Reconstruction & Automated Acceptance Verification (v2.1)
Outputs:
  1) ./q1_v21_outputs\q1_all10_acceptance_table.csv
  2) ./q1_v21_outputs\recon_native_npz\{Tag}_q1_v21.npz  (Native grid + B-spline basis parameters)
  3) ./q1_v21_outputs\recon_aligned_q3_npz\{Tag}_aligned.npz (Unified 341-1023nm dark-subtracted for Q3)
"""
import os
import glob
import time
import json
import numpy as np
import pandas as pd

from q1_core_v21 import reconstruct_v21, evaluate_acceptance_criteria

CACHE_DIR = os.environ.get("Q1_CACHE_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache_npz"))
OUT_DIR = os.environ.get("Q1_OUT_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "results", "q1_v21"))
NATIVE_DIR = os.path.join(OUT_DIR, "recon_native_npz")
ALIGN_DIR = os.path.join(OUT_DIR, "recon_aligned_q3_npz")
os.makedirs(NATIVE_DIR, exist_ok=True)
os.makedirs(ALIGN_DIR, exist_ok=True)

COMMON_WL = np.linspace(341.0, 1023.0, 1365)  # Exact 0.5 nm unified cross-converter grid


def main():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    rows = []
    t_start_all = time.time()

    for fpath in files:
        tag = os.path.basename(fpath)[:-4]
        grp = "6B" if "6B" in tag else "6C"
        t0 = time.time()

        d = np.load(fpath)
        wl = d["wavelengths"]
        spec_raw = d["spec"].astype(np.float64)
        seq = d["seq"]
        o2 = d["o2"]
        co2 = d["co2"]
        co = d["co"]

        res = reconstruct_v21(wl, spec_raw, dt=0.5)
        acc = evaluate_acceptance_criteria(wl, spec_raw, res)
        det = res["anomalies"]
        dt_elapsed = time.time() - t0

        # 1. Save native-grid continuous model & discrete evaluation
        np.savez_compressed(
            os.path.join(NATIVE_DIR, f"{tag}_q1_v21.npz"),
            wavelengths=wl,
            t_grid=res["t_grid"],
            seq=seq,
            spec_recon=res["spec_recon"].astype(np.float32),
            mu_vec=res["mu_vec"],
            psi_basis=res["psi_basis"],
            c_full=res["c_full"],
            sigma_col=res["sigma_col"],
            K_selected=np.int32(res["K_selected"]),
            dark_base=det["dark_base"],
            good_frames=det["good_frames"],
            is_blackout=det["is_blackout"],
            is_flash=det["is_flash"],
            col_bad_mask=det["col_bad_mask"],
            sat_narrow_mask=det["sat_narrow_mask"],
        )

        # 2. Evaluate C^2 continuous surface on unified 341.0..1023.0 nm grid & subtract smooth dark baseline for Q3
        spec_common = res["eval_surface"](res["t_grid"], COMMON_WL)
        dark_common_mask = (COMMON_WL >= 355.0) & (COMMON_WL <= 415.0)
        dark_common = np.median(spec_common[:, dark_common_mask], axis=1)
        spec_common_net = np.maximum(spec_common - dark_common[:, None], 0.0)

        np.savez_compressed(
            os.path.join(ALIGN_DIR, f"{tag}_aligned.npz"),
            common_wl=COMMON_WL,
            t_grid=res["t_grid"],
            seq=seq,
            spec_aligned_raw=spec_common.astype(np.float32),
            spec_aligned_net=spec_common_net.astype(np.float32),
            dark_common=dark_common.astype(np.float32),
            good_frames=det["good_frames"],
            o2=o2,
            co2=co2,
            co=co,
        )

        bad_wls = [round(float(w), 2) for w in wl[det["col_bad_mask"]]]
        row = {
            "Tag": tag,
            "Group": grp,
            "Rows_T": len(seq),
            "Duration_s": round(len(seq) * 0.5, 1),
            "K_Unwhite_GD": res["K_gd_unwhite"],
            "K_White_GD_Raw": res["K_gd_white_raw"],
            "K_Selected": res["K_selected"],
            "BadCols_Cnt": int(det["col_bad_mask"].sum()),
            "Blackout_Frames": int(det["is_blackout"].sum()),
            "Flash_Frames": int(det["is_flash"].sum()),
            "GoodFrame_Ratio%": round(float(det["good_frames"].mean() * 100.0), 2),
            "NarrowSat_Pts": int(det["sat_narrow_mask"].sum()),
            "Spike_Pts": int(det["spike_mask"].sum()),
            **acc,
            "Time_s": round(dt_elapsed, 2),
            "BadCols_Wavelengths_Sample": str(bad_wls[:8]),
        }
        rows.append(row)
        status = "PASS" if acc["Overall_Pass"] else "CHECK"
        print(
            f"[{tag}] {status} in {dt_elapsed:.2f}s | K_rawGD={res['K_gd_unwhite']} -> K_white={res['K_selected']} | "
            f"BadCols={row['BadCols_Cnt']}, BO={row['Blackout_Frames']}, Flash={row['Flash_Frames']}, Spikes={row['Spike_Pts']} | "
            f"PeakErr(Na/K1/K2)={acc['Na589_Err%']:.2f}%/{acc['K766_Err%']:.2f}%/{acc['K770_Err%']:.2f}% | "
            f"NoiseRatio(D/B)={acc['Dark_Ratio']:.2f}/{acc['Bright_Ratio']:.2f}"
        )

    df_all = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "q1_all10_acceptance_table.csv")
    df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\nAll 10 heats reconstructed & verified in {time.time() - t_start_all:.2f}s!")
    print(f"Overall Pass Count: {int(df_all['Overall_Pass'].sum())} / {len(df_all)}")
    print("\n=== FULL 10-HEAT ACCEPTANCE SUMMARY TABLE ===")
    show_cols = [
        "Tag", "K_Unwhite_GD", "K_Selected", "BadCols_Cnt", "Blackout_Frames", "Flash_Frames",
        "Spike_Pts", "Overall_Pass", "Hot383_Dev_Sigma", "Remaining_Blackouts",
        "Na589_Err%", "K766_Err%", "K770_Err%", "NaK_SpikeFlag_Rate%",
        "Dark_Ratio", "Bright_Ratio"
    ]
    print(df_all[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
