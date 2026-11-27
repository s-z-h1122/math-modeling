"""Question 1 v2.2: strict blackout mask and segment-safe PCHIP surface."""
from __future__ import annotations

import numpy as np
import pandas as pd

from q1_core_v21 import detect_anomalies_v21, reconstruct_v21


def detect_anomalies_v22(wl: np.ndarray, spec: np.ndarray) -> dict:
    """Use the audited strict blackout definition without shoulder dilation."""
    det = detect_anomalies_v21(wl, spec)
    e = np.asarray(det["E_main"], float)
    med51 = (
        pd.Series(e)
        .rolling(51, center=True, min_periods=5)
        .median()
        .bfill()
        .ffill()
        .to_numpy()
    )
    strict = e < np.maximum(350.0, 0.18 * med51)
    det["E_med51"] = med51
    det["is_blackout"] = strict
    det["frame_bad_mask"] = strict | det["is_flash"]
    det["good_frames"] = ~det["frame_bad_mask"]
    return det


def reconstruct_segment_v22(
    wl: np.ndarray,
    spec: np.ndarray,
    t_grid: np.ndarray,
    rank_bounds=(12, 20),
    n_svd_iter: int = 8,
) -> dict:
    """Reconstruct one physically continuous blow segment."""
    return reconstruct_v21(
        wl,
        spec,
        rank_bounds=rank_bounds,
        n_svd_iter=n_svd_iter,
        detector=detect_anomalies_v22,
        t_grid_override=t_grid,
        stage3_kind="pchip",
    )


def pchip_surface_overshoot(res: dict, n_mid: int = 2) -> float:
    """Return off-grid overshoot beyond adjacent discrete endpoint ranges."""
    t = res["t_grid"]
    if len(t) < 2:
        return 0.0
    tq = np.concatenate([
        t[:-1] + (j + 1) / (n_mid + 1) * np.diff(t) for j in range(n_mid)
    ])
    tq.sort()
    yq = res["eval_surface"](tq, res["wl"])
    pos = np.searchsorted(t, tq, side="right") - 1
    pos = np.clip(pos, 0, len(t) - 2)
    lo = np.minimum(res["spec_recon"][pos], res["spec_recon"][pos + 1])
    hi = np.maximum(res["spec_recon"][pos], res["spec_recon"][pos + 1])
    return float(max(np.max(lo - yq), np.max(yq - hi), 0.0))
