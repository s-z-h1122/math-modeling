"""Run the APMCM Problem B analysis from the ten original spreadsheets."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from signals import (delay_by_flow, estimate_delay_windows, gas_signals, load_heats,
                     reconstruct_spectrum, spectrum_figure)
from predict import run_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing the ten XLSB files")
    parser.add_argument("--out", type=Path, required=True, help="Destination for results")
    args = parser.parse_args()
    start = perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    q1, q2, q3 = [args.out / f"q{i}" for i in (1, 2, 3)]
    for folder in (q1, q2, q3):
        folder.mkdir(exist_ok=True)

    heats = load_heats(args.data)
    inventory, spectral_audit, gas_audit, delay_rows = [], [], [], []
    reconstructed_gas = {}
    for heat in heats:
        file = args.data / f"Attachment {heat.number}. Heat No. {heat.name}.xlsb"
        with file.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        inventory.append({
            "attachment": heat.number, "heat": heat.name, "file": file.name,
            "sha256": checksum, "samples": len(heat.time), "channels": len(heat.wavelength),
            "wavelength_min_nm": heat.wavelength[0], "wavelength_max_nm": heat.wavelength[-1],
            "gas_both_constant_zero": not heat.valid_gas,
        })
        cleaned, invalid, spikes = reconstruct_spectrum(heat)
        spectrum_figure(heat, cleaned, q1 / f"{heat.name}_spectrum.png")
        spectral_audit.append({
            "heat": heat.name, "saturated": int((heat.spectrum >= 65535).sum()),
            "invalid_or_saturated_repaired": invalid, "spikes_repaired": spikes,
        })
        gas_record, flow, gas = gas_signals(heat, q2)
        gas_audit.append(gas_record)
        reconstructed_gas[heat.name] = gas
        delay_rows.extend(estimate_delay_windows(heat, flow, gas))
        print(f"Reconstructed signals: {heat.name}", flush=True)

    pd.DataFrame(inventory).to_csv(args.out / "input_manifest.csv", index=False)
    pd.DataFrame(spectral_audit).to_csv(q1 / "spectral_audit.csv", index=False)
    pd.DataFrame(gas_audit).to_csv(q2 / "gas_totals_and_associations.csv", index=False, float_format="%.6f")
    windows = pd.DataFrame(delay_rows)
    if windows.empty:
        raise ValueError("No delay windows could be evaluated.")
    windows.to_csv(q2 / "delay_windows.csv", index=False, float_format="%.6f")
    delay_by_flow(windows, q2)
    summary = run_prediction(heats, reconstructed_gas, windows, q3)
    metadata = {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        "scikit_learn": sklearn.__version__, "elapsed_s": round(perf_counter() - start, 2),
        "seed_ensemble": [42, 123, 2026], "optical_windows_s": [20, 60, 150],
        "sampling_interval_s": 0.5, "pca_components": 40,
        "validation": "leave one heat out; the constant-zero heat is predicted but not used in training or aggregate scores",
    }
    (args.out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(summary[["target", "mode", "model", "macro_r2", "macro_rmse"]].to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
