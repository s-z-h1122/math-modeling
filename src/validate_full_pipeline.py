"""Independent acceptance checks for the saved full-pipeline outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="run_outputs")
    args = ap.parse_args()
    out = Path(args.output_dir)
    q1 = pd.read_csv(out / "q1_v22" / "q1_v22_all10_acceptance.csv")
    assert len(q1) == 10 and q1["Finite_Output"].all()
    assert (q1["NIST_Max_Error_nm"] <= 0.25).all()
    a5 = q1.set_index("Tag").loc["Att05_6C02057"]
    a6 = q1.set_index("Tag").loc["Att06_6C02058"]
    assert int(a5["Segments"]) == 2 and 5.5 <= a5["Main_Blow_Gas_O2_Ratio"] <= 6.5
    native5 = np.load(out / "q1_v22" / "recon_native_npz" / "Att05_6C02057_q1_v22.npz")
    assert 700 <= native5["t_phys"][751] <= 730
    assert 7 <= a6["Strict_Blackout_Rate_pct"] <= 10
    assert (q1["Observed_Valid_Pixels_Preserved_pct"] >= 92).all()

    q2h = pd.read_csv(out / "q2" / "q2_delay_by_heat.csv")
    q2f = pd.read_csv(out / "q2" / "q2_delay_by_flow.csv")
    assert len(q2h) == 10 and (q2h["O2_Conservation_RelErr"] < 1e-8).all()
    for _, g in q2f.groupby("Tag"):
        g = g.sort_values("Flow_Mean")
        assert np.all(np.diff(g["Physics_Constrained_Delay_s"]) <= 1e-9)

    fold = pd.read_csv(out / "q3" / "q3_loho_fold_metrics.csv")
    assert fold["Held_Out_Heat"].nunique() == 10
    assert fold["Method"].nunique() == 6 and len(fold) == 120
    assert np.isfinite(fold[["MAE", "RMSE", "R2"]]).all().all()
    manifest = json.loads((out / "q3" / "q3_feature_manifest.json").read_text(encoding="utf-8"))
    assert manifest["causal"] is True and manifest["right_censor_s"] == 35.0

    print("FULL PIPELINE VALIDATION: PASS")
    print("Q1=10 heats; Q2=10 conserved oxygen curves; Q3=10x6 LOHO model runs")


if __name__ == "__main__":
    main()
