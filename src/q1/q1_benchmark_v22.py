"""Blind five-method benchmark with the v2.2 strict detector/reconstructor."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from q1_benchmark_v21 import run_blind_synthetic_benchmark
from q1_core_v22 import detect_anomalies_v22, reconstruct_segment_v22


OUT_DIR = Path(os.environ.get("Q1_OUT_DIR", Path(__file__).resolve().parents[2] / "results" / "q1_v22_benchmark"))


def _reconstruct(wl, spec):
    return reconstruct_segment_v22(wl, spec, np.arange(len(spec), dtype=float) * 0.5)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    det, bench = run_blind_synthetic_benchmark(
        "Att02_6B03206",
        seed=2026,
        detector_fn=detect_anomalies_v22,
        reconstruct_fn=_reconstruct,
        proposed_name="M5_Proposed_v22_Strict_PCHIP",
    )
    bench.to_csv(OUT_DIR / "q1_blind_benchmark_5methods.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "q1_v22_detection_metrics.txt").write_text("\n".join(f"{k}: {v}" for k, v in det.items()), encoding="utf-8")
    print(det)
    print(bench.to_string(index=False))


if __name__ == "__main__":
    main()
