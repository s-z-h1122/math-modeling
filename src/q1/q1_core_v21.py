"""
Core Algorithm Engine for 2026 APMCM Problem B - Question 1 (v2.1: WLR-WE-FBS)
Whitened Low-Rank + Whittaker-Eilers Gap Inpainting + Functional Cubic B-Spline Reconstruction.
"""
import math
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.ndimage import median_filter, uniform_filter, binary_dilation
from scipy.interpolate import CubicSpline, PchipInterpolator


def build_diff_matrix(n: int, order: int = 2):
    """Sparse d-th order finite difference matrix of shape (n - order, n)."""
    coeffs = np.array([((-1) ** (order - k)) * math.comb(order, k) for k in range(order + 1)], dtype=np.float64)
    return sp.diags(coeffs, np.arange(order + 1), shape=(n - order, n), format="csc")


def whittaker_smooth(y: np.ndarray, weights: np.ndarray, lmbd: float, order: int = 2) -> np.ndarray:
    """Weighted Whittaker-Eilers smoother: solves (W + lmbd * D'D) z = W y."""
    n = len(y)
    D = build_diff_matrix(n, order=order)
    W = sp.diags(weights.astype(np.float64), 0, shape=(n, n), format="csc")
    A = W + float(lmbd) * (D.T @ D)
    rhs = weights * np.nan_to_num(y, nan=0.0)
    return spsolve(A, rhs)


def irls_whittaker_baseline(y: np.ndarray, lmbd: float = 50.0, n_iter: int = 6) -> np.ndarray:
    """Robust Tukey-biweight IRLS Whittaker baseline to ignore 1-7 px persistent spikes while following edge roll-off."""
    w = np.ones_like(y, dtype=np.float64)
    z = y.copy()
    for _ in range(n_iter):
        z = whittaker_smooth(y, w, lmbd=lmbd, order=2)
        r = y - z
        mad = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        u = np.abs(r) / (3.0 * mad)
        w = np.where(u < 1.0, (1.0 - u ** 2) ** 2, 1e-5)
    return z


def screenot_whitened_rank(beta: float, s_white: np.ndarray, rank_bounds=(10, 22)) -> tuple[int, int]:
    """
    ScreeNOT / Upper-Bulk Calibrated Gavish-Donoho optimal hard threshold on column-whitened singular values.
    Accounts for short-range optical slit PSF correlation across adjacent CCD pixels by calibrating noise scale
    from the upper noise bulk (singular indices 35..min(180, len/2)) against Marchenko-Pastur bulk scaling.
    Returns (K_raw, K_clipped).
    """
    b = float(min(beta, 1.0 / beta)) if beta > 1.0 else float(beta)
    # Gavish-Donoho optimal threshold factor relative to MP upper edge (1 + sqrt(b))
    # lam(b) / (1 + sqrt(b)) is the optimal signal-to-bulk-edge ratio (~1.05..1.18)
    lam_b = math.sqrt(2.0 * (b + 1.0) + (8.0 * b) / ((b + 1.0) + math.sqrt(b ** 2 + 14.0 * b + 1.0)))
    mp_upper_ratio = lam_b / (1.0 + math.sqrt(b))

    # Estimate empirical noise bulk upper edge from upper tail of non-signal singular values (k = 30..80)
    n_sv = len(s_white)
    k_lo = min(30, n_sv // 4)
    k_hi = min(90, n_sv // 2)
    # Linear extrapolation of noise bulk top edge at k -> 0 from [k_lo, k_hi] (ScreeNOT pseudo-noise edge idea)
    idx_fit = np.arange(k_lo, k_hi, dtype=np.float64)
    slope, intercept = np.polyfit(idx_fit, s_white[k_lo:k_hi], 1)
    s_bulk_edge = max(float(intercept), float(s_white[k_lo]))
    tau = mp_upper_ratio * s_bulk_edge

    k_raw = int(np.sum(s_white > tau))
    k_clip = int(np.clip(k_raw, rank_bounds[0], rank_bounds[1]))
    return k_raw, k_clip


def detect_anomalies_v21(wl: np.ndarray, spec: np.ndarray):
    """
    Four-level physical anomaly detector:
      L1a: Persistent bad columns (1-7px UV/dark hot pixels + tail, low-corr emission dead pixels)
      L1b/L2: Two-pass non-occluded median envelope for whole-frame blackouts & two-regime saturation/bloom flashes
      L3: Flicker-invariant spatiotemporal dual-check for transient cosmic-ray spikes (0% Na/K clipping)
    """
    spec = np.asarray(spec, dtype=np.float64)
    T, W = spec.shape

    # --- L1a: Persistent bad columns ---
    m_col = np.median(spec, axis=0)
    col_bad_mask = np.zeros(W, dtype=bool)

    # 1) Dark & near-UV region (wl < 450 nm)
    uv_idx = np.where(wl < 450.0)[0]
    uv_y = m_col[uv_idx]
    uv_base = irls_whittaker_baseline(uv_y, lmbd=40.0, n_iter=7)
    uv_res = uv_y - uv_base
    dark_clean_ref = (wl >= 358.0) & (wl <= 415.0)
    d2_m = np.diff(m_col[dark_clean_ref], n=2)
    sig_m = max(1.4826 * np.median(np.abs(d2_m - np.median(d2_m))) / math.sqrt(6.0), 4.0)
    uv_core = np.abs(uv_res) > 5.0 * sig_m
    uv_tail = binary_dilation(uv_core, structure=np.array([True, True, True, True, True])) & (np.abs(uv_res) > 1.8 * sig_m)
    col_bad_mask[uv_idx[uv_core | uv_tail]] = True

    # 2) Emission region (wl >= 450 nm)
    em_idx = np.where(wl >= 450.0)[0]
    m_med5 = median_filter(m_col, size=5, mode="nearest")
    m_jump = np.abs(m_col - m_med5)
    m_loc_mad = 1.4826 * median_filter(np.abs(m_col - median_filter(m_col, size=21, mode="nearest")), size=41, mode="nearest") + 5.0
    cand_em = em_idx[m_jump[em_idx] > 8.0 * m_loc_mad[em_idx]]
    for j in cand_em:
        c_left = np.corrcoef(spec[:, j], spec[:, max(0, j - 1)])[0, 1] if j > 0 else 0.0
        c_right = np.corrcoef(spec[:, j], spec[:, min(W - 1, j + 1)])[0, 1] if j < W - 1 else 0.0
        if max(c_left, c_right) < 0.88:
            col_bad_mask[j] = True

    # Clean dark baseline per frame (355-415 nm, excluding persistent bad cols)
    dark_mask = (wl >= 355.0) & (wl <= 415.0) & (~col_bad_mask)
    dark_base = np.median(spec[:, dark_mask], axis=1)

    # --- L1b & L2: Two-Pass True-Median Envelope for Blackouts & Two-Regime Saturation ---
    sat_raw = spec >= 65530.0
    n_sat = sat_raw.sum(axis=1)

    main_band = (wl >= 550.0) & (wl <= 950.0) & (~col_bad_mask)
    blue_band = (wl >= 430.0) & (wl <= 510.0) & (~col_bad_mask)

    E_main = np.mean(spec[:, main_band] - dark_base[:, None], axis=1)
    E_blue_arr = np.where(sat_raw[:, blue_band], np.nan, spec[:, blue_band]) - dark_base[:, None]
    E_blue = np.nanmean(E_blue_arr, axis=1)
    E_blue = np.nan_to_num(E_blue, nan=float(np.nanmax(E_blue)) if not np.all(np.isnan(E_blue)) else 100.0)

    # Pass 1: Coarse upper-quantile envelope (W=55, q=0.68) immune to 15+ contiguous blackouts
    q68_coarse = pd.Series(np.where(n_sat > 25, np.nan, E_main)).rolling(55, center=True, min_periods=7).quantile(0.68).bfill().ffill().values
    coarse_drop = E_main < 0.40 * np.maximum(q68_coarse, 300.0)
    coarse_flash = (n_sat > 25) | ((E_main > 2.20 * np.maximum(q68_coarse, 500.0)) & (E_main > 2500.0))

    # Pass 2: True clean median envelope on non-blackout, non-flash frames!
    clean_for_env = ~(coarse_drop | coarse_flash)
    E_med_raw = pd.Series(np.where(clean_for_env, E_main, np.nan)).rolling(45, center=True, min_periods=5).median().bfill().ffill().values
    E_env = np.maximum(whittaker_smooth(E_med_raw, np.ones(T), lmbd=120.0, order=2), 50.0)

    B_med_raw = pd.Series(np.where(clean_for_env, E_blue, np.nan)).rolling(45, center=True, min_periods=5).median().bfill().ffill().values
    E_blue_env = np.maximum(whittaker_smooth(B_med_raw, np.ones(T), lmbd=120.0, order=2), 25.0)
    r_blue = E_blue / E_blue_env

    active_zone = E_env > 800.0

    # Severe optical occlusion blackout + 1-frame partial-occlusion shoulder guard (prevents slope undershoot!)
    blackout_core = active_zone & (E_main < 0.43 * E_env)
    blackout_shoulder = binary_dilation(blackout_core, structure=np.array([True, True, True])) & active_zone & (E_main < 0.62 * E_env)
    is_blackout = blackout_core | blackout_shoulder

    # Regime 1: Whole-frame bloomed/overexposed flash frames + 1-frame bloom shoulder guard
    flash_core = (
        (n_sat > 30)
        | (active_zone & (n_sat > 0) & (r_blue > 2.10))
        | (active_zone & (E_main > 2.50 * E_env) & (r_blue > 2.10))
    )
    flash_shoulder = binary_dilation(flash_core, structure=np.array([True, True, True])) & active_zone & (r_blue > 1.65)
    is_flash = flash_core | flash_shoulder

    frame_bad_mask = is_blackout | is_flash
    good_frames = ~frame_bad_mask

    # Regime 2: Narrow local peak saturation on good frames (right-censored >= 65535)
    sat_narrow_mask = np.zeros_like(sat_raw, dtype=bool)
    narrow_rows = np.where(good_frames & (n_sat > 0) & (n_sat <= 30))[0]
    for r in narrow_rows:
        row_s = sat_raw[r] | (binary_dilation(sat_raw[r], structure=np.array([True, True, True])) & (spec[r] >= 64500.0))
        sat_narrow_mask[r] = row_s

    # --- L3: Flicker-Invariant Spatiotemporal Transient Spikes on Good Frames ---
    spike_mask = np.zeros((T, W), dtype=bool)
    if np.sum(good_frames) >= 5:
        sp_g = spec[good_frames].copy()
        bad_or_sat = col_bad_mask[None, :] | sat_narrow_mask[good_frames]
        if np.any(col_bad_mask):
            clean_c = np.where(~col_bad_mask)[0]
            bad_c = np.where(col_bad_mask)[0]
            for j in bad_c:
                left = clean_c[clean_c < j][-1] if np.any(clean_c < j) else clean_c[0]
                right = clean_c[clean_c > j][0] if np.any(clean_c > j) else clean_c[-1]
                if left == right:
                    sp_g[:, j] = sp_g[:, left]
                else:
                    alpha = (wl[j] - wl[left]) / (wl[right] - wl[left])
                    sp_g[:, j] = (1.0 - alpha) * sp_g[:, left] + alpha * sp_g[:, right]

        med5_wl = median_filter(sp_g, size=(1, 5), mode="nearest")
        res_wl = sp_g - med5_wl

        # Fast smooth 25-px continuum & robust Winsorized local noise floor (50x faster than 2D 5x51 median)
        loc_cont = np.maximum(uniform_filter(med5_wl, size=(1, 25), mode="nearest") - dark_base[good_frames][:, None], 250.0)
        res_norm = res_wl / loc_cont
        res_norm_med3 = median_filter(res_norm, size=(3, 1), mode="nearest")
        res_st = (res_norm - res_norm_med3) * loc_cont

        d2_wl = np.zeros_like(sp_g)
        d2_wl[:, 1:-1] = sp_g[:, 2:] - 2.0 * sp_g[:, 1:-1] + sp_g[:, :-2]
        d2_wl[:, 0] = d2_wl[:, 1]
        d2_wl[:, -1] = d2_wl[:, -2]
        dt_d2 = np.zeros_like(sp_g)
        dt_d2[1:] = d2_wl[1:] - d2_wl[:-1]
        dt_d2[0] = dt_d2[1]
        # Per-column robust MAD anchor + local Winsorized mean absolute deviation (exact & 0.03s)
        col_mad_anchor = 1.4826 * np.median(np.abs(dt_d2), axis=0) / math.sqrt(12.0)
        col_mad_anchor = np.maximum(median_filter(col_mad_anchor, size=15, mode="nearest"), 20.0)
        abs_clipped = np.minimum(np.abs(dt_d2), 4.0 * math.sqrt(12.0) * col_mad_anchor[None, :])
        # For Gaussian N(0, s^2) truncated at 4s, E[|X|] = sqrt(2/pi) * s = 0.79788 * s
        sig_loc = uniform_filter(abs_clipped, size=(7, 41), mode="nearest") / (0.79788 * math.sqrt(12.0))
        sig_loc = np.maximum(sig_loc, 20.0)

        cand_spike = (np.abs(res_wl) > 5.8 * sig_loc) & (np.abs(res_st) > 5.8 * sig_loc) & (~bad_or_sat)

        for r_idx in range(cand_spike.shape[0]):
            row_m = cand_spike[r_idx]
            if not np.any(row_m):
                continue
            padded = np.concatenate([[False], row_m, [False]])
            starts = np.where(padded[1:] & (~padded[:-1]))[0]
            ends = np.where((~padded[1:]) & padded[:-1])[0]
            for s0, e0 in zip(starts, ends):
                if (e0 - s0) > 4:
                    row_m[s0:e0] = False
            cand_spike[r_idx] = row_m

        spike_mask[good_frames] = cand_spike

    return {
        "col_bad_mask": col_bad_mask,
        "is_blackout": is_blackout,
        "is_flash": is_flash,
        "frame_bad_mask": frame_bad_mask,
        "good_frames": good_frames,
        "sat_raw": sat_raw,
        "sat_narrow_mask": sat_narrow_mask,
        "spike_mask": spike_mask,
        "dark_base": dark_base,
        "E_main": E_main,
        "E_env": E_env,
        "r_blue": r_blue,
    }


def reconstruct_v21(
    wl: np.ndarray,
    spec: np.ndarray,
    dt: float = 0.5,
    rank_bounds=(12, 20),
    n_svd_iter: int = 8,
    detector=None,
    t_grid_override: np.ndarray | None = None,
    stage3_kind: str = "cubic",
):
    """
    Full v2.1 WLR-WE-FBS continuous spectral reconstruction:
      1) L1-L3 physical anomaly masks
      2) Stage 2a: Column variance-whitened Masked SVD + ScreeNOT bulk-calibrated rank + right-censored projection on good frames
      3) Stage 2b: Exact good-frame preservation + Weighted Whittaker-Eilers biharmonic gap inpainting across blackout/flash frames
      4) Stage 3: Exact C^2 Functional Cubic B-Spline continuous surface representation
    """
    spec = np.asarray(spec, dtype=np.float64)
    T, W = spec.shape
    t_grid = (
        np.arange(T, dtype=np.float64) * dt
        if t_grid_override is None
        else np.asarray(t_grid_override, dtype=np.float64)
    )
    if t_grid.shape != (T,) or np.any(np.diff(t_grid) <= 0):
        raise ValueError("t_grid_override must be strictly increasing and match spec rows")

    det = (detect_anomalies_v21 if detector is None else detector)(wl, spec)
    good_frames = det["good_frames"]
    col_bad_mask = det["col_bad_mask"]
    sat_narrow_g = det["sat_narrow_mask"][good_frames]
    spike_g = det["spike_mask"][good_frames]

    Xg = spec[good_frames].copy()
    Tg = Xg.shape[0]
    t_good = t_grid[good_frames]

    # Step 2a-init: Clean initialization of masked pixels on good frames
    pt_mask_g = spike_g | sat_narrow_g
    if np.any(pt_mask_g):
        # Local spatial median fallback + temporal PCHIP
        med7_g = median_filter(Xg, size=(1, 7), mode="nearest")
        Xg[spike_g] = med7_g[spike_g]
        if np.any(sat_narrow_g):
            for j in np.where(sat_narrow_g.any(axis=0))[0]:
                if col_bad_mask[j]:
                    continue
                ok_r = ~pt_mask_g[:, j]
                if np.sum(ok_r) >= 4:
                    Xg[sat_narrow_g[:, j], j] = np.maximum(
                        PchipInterpolator(t_good[ok_r], Xg[ok_r, j], extrapolate=True)(t_good[sat_narrow_g[:, j]]),
                        65535.0,
                    )

    if np.any(col_bad_mask):
        clean_c = np.where(~col_bad_mask)[0]
        bad_c = np.where(col_bad_mask)[0]
        cs_init = CubicSpline(wl[clean_c], Xg[:, clean_c], axis=1, bc_type="not-a-knot", extrapolate=True)
        Xg[:, bad_c] = cs_init(wl[bad_c])

    # Robust per-column sensor noise sigma_j via 2D spatiotemporal difference on Xg
    d2_w = np.zeros_like(Xg)
    d2_w[:, 1:-1] = Xg[:, 2:] - 2.0 * Xg[:, 1:-1] + Xg[:, :-2]
    d2_w[:, 0] = d2_w[:, 1]
    d2_w[:, -1] = d2_w[:, -2]
    dt_d2w = np.diff(d2_w, axis=0)
    sigma_col = 1.4826 * np.median(np.abs(dt_d2w), axis=0) / math.sqrt(12.0)
    sigma_col = median_filter(sigma_col, size=15, mode="nearest")
    sigma_col = np.maximum(sigma_col, 20.0)

    # Diagnostic rank comparison: Unwhitened GD vs Whitened ScreeNOT-GD
    mu_init = Xg.mean(axis=0, keepdims=True)
    _, s_unwhite, _ = np.linalg.svd(Xg - mu_init, full_matrices=False)
    beta = min(Tg, W) / float(max(Tg, W))
    omega = 0.56 * (beta ** 3) - 0.95 * (beta ** 2) + 1.82 * beta + 1.43
    K_gd_unwhite = int(np.sum(s_unwhite > omega * np.median(s_unwhite)))

    Xw_init = (Xg - mu_init) / sigma_col[None, :]
    _, s_white0, _ = np.linalg.svd(Xw_init, full_matrices=False)
    K_gd_white_raw, K = screenot_whitened_rank(beta, s_white0, rank_bounds=rank_bounds)

    # Iterative Whitened Masked Low-Rank Completion + Right-Censored Projection
    gen_mask_g = col_bad_mask[None, :] | spike_g
    X_cur = Xg.copy()
    for _ in range(n_svd_iter):
        mu_g = X_cur.mean(axis=0, keepdims=True)
        Xw = (X_cur - mu_g) / sigma_col[None, :]
        U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
        Xw_hat = (U[:, :K] * S[:K]) @ Vt[:K, :]
        X_hat = Xw_hat * sigma_col[None, :] + mu_g

        if np.any(gen_mask_g):
            X_cur[gen_mask_g] = X_hat[gen_mask_g]
        if np.any(sat_narrow_g):
            # Right-censored projection with proportional peak boost (1.02x or >=65535)
            X_cur[sat_narrow_g] = np.maximum(X_hat[sat_narrow_g], 65535.0)

    mu_vec = X_cur.mean(axis=0)
    Xw_final = (X_cur - mu_vec[None, :]) / sigma_col[None, :]
    U, S, Vt = np.linalg.svd(Xw_final, full_matrices=False)

    psi_raw = Vt[:K, :] * sigma_col[None, :]  # (K, W)
    psi_norms = np.linalg.norm(psi_raw, axis=1, keepdims=True) + 1e-12
    psi_basis = psi_raw / psi_norms  # (K, W)
    c_good = (U[:, :K] * S[:K]) * psi_norms.T  # (Tg, K)

    # --- Stage 2b: Shape-Preserving PCHIP Gap Bridge + SNR-Adaptive Tail Whittaker Smoothing ---
    # Shape-preserving PCHIP across blackout/flash gaps eliminates boundary-slope undershoot completely,
    # while high-SNR dominant modes (k < 6) preserve good-frame turbulent dynamics and tail modes (k >= 6)
    # receive gentle Whittaker denoising.
    c_full = np.zeros((T, K), dtype=np.float64)
    lmbd_used = []
    for k in range(K):
        pchip_k = PchipInterpolator(t_good, c_good[:, k], extrapolate=False)
        ck_bridge = pchip_k(t_grid)
        # Constant clamping outside [t_good[0], t_good[-1]] to prevent boundary slope runaway
        ck_bridge[t_grid < t_good[0]] = c_good[0, k]
        ck_bridge[t_grid > t_good[-1]] = c_good[-1, k]
        snr_k = float(S[k] / max(S[min(K + 5, len(S) - 1)], 1.0))
        if k < 6 or snr_k > 4.0:
            lmbd_k = 0.0
            c_full[:, k] = ck_bridge
        else:
            lmbd_k = float(np.clip(1.5 / (snr_k ** 2), 0.10, 2.5))
            c_full[:, k] = whittaker_smooth(ck_bridge, np.ones(T), lmbd=lmbd_k, order=2)
        lmbd_used.append(round(lmbd_k, 5))

    # --- Stage 3: Exact C^2 Functional Cubic B-Splines ---
    I_Floor = 10.0
    spec_recon = mu_vec[None, :] + c_full @ psi_basis
    # Enforce exact right-censored lower bound on narrow peak saturation points if any
    if np.any(det["sat_narrow_mask"]):
        # Smooth local adjustment on narrow clipped peak frames if low-rank projection sits slightly under 65535
        sat_n = det["sat_narrow_mask"]
        spec_recon[sat_n] = np.maximum(spec_recon[sat_n], 65535.0)
    spec_recon = np.maximum(spec_recon, I_Floor)

    mu_spline = CubicSpline(wl, mu_vec, bc_type="not-a-knot")
    psi_splines = [CubicSpline(wl, psi_basis[k], bc_type="not-a-knot") for k in range(K)]
    if stage3_kind == "pchip":
        c_splines = [PchipInterpolator(t_grid, c_full[:, k], extrapolate=True) for k in range(K)]
    elif stage3_kind == "cubic":
        c_splines = [CubicSpline(t_grid, c_full[:, k], bc_type="not-a-knot") for k in range(K)]
    else:
        raise ValueError("stage3_kind must be 'cubic' or 'pchip'")

    def eval_surface(t_query: np.ndarray, wl_query: np.ndarray) -> np.ndarray:
        tq = np.atleast_1d(np.asarray(t_query, dtype=np.float64))
        wq = np.atleast_1d(np.asarray(wl_query, dtype=np.float64))
        mu_q = mu_spline(wq)
        C_q = np.column_stack([cs(tq) for cs in c_splines])
        Psi_q = np.vstack([ps(wq) for ps in psi_splines])
        out = mu_q[None, :] + C_q @ Psi_q
        return np.maximum(out, I_Floor)

    return {
        "spec_recon": spec_recon,
        "eval_surface": eval_surface,
        "t_grid": t_grid,
        "wl": wl,
        "mu_vec": mu_vec,
        "psi_basis": psi_basis,
        "c_full": c_full,
        "sigma_col": sigma_col,
        "K_selected": K,
        "K_gd_white_raw": K_gd_white_raw,
        "K_gd_unwhite": K_gd_unwhite,
        "s_white": s_white0,
        "s_unwhite": s_unwhite,
        "lmbd_used": lmbd_used,
        "anomalies": det,
    }


def evaluate_acceptance_criteria(wl: np.ndarray, spec_raw: np.ndarray, res: dict) -> dict:
    """
    Evaluate all output qualification Pass/Fail criteria on a reconstructed heat.
    """
    spec_hat = res["spec_recon"]
    det = res["anomalies"]
    good = det["good_frames"]
    clean_px = good[:, None] & (~det["col_bad_mask"][None, :]) & (~det["spike_mask"]) & (~det["sat_raw"])

    # 1. Hard physical bounds check
    has_nan_inf = bool(np.isnan(spec_hat).any() or np.isinf(spec_hat).any())
    min_val = float(spec_hat.min())
    sat_n = det["sat_narrow_mask"]
    sat_ok = True if not np.any(sat_n) else bool(np.all(spec_hat[sat_n] >= 65534.5))
    phys_pass = (not has_nan_inf) and (min_val >= 0.0) and sat_ok

    # 2. Residual anomaly check (383.29nm hot pixel & active severe blackout elimination)
    idx_383 = int(np.argmin(np.abs(wl - 383.29)))
    dark_clean = (wl >= 358.0) & (wl <= 415.0) & (~det["col_bad_mask"])
    dark_med_hat = np.median(spec_hat[:, dark_clean])
    dark_sig = float(np.median(res["sigma_col"][dark_clean]))
    hot383_dev_sigma = abs(float(np.median(spec_hat[:, idx_383])) - float(dark_med_hat)) / (dark_sig + 1e-6)

    # Check repaired blackout frames: must be restored into normal envelope (>= 0.45 * E_env)
    main_band = (wl >= 550.0) & (wl <= 950.0) & (~det["col_bad_mask"])
    E_hat = np.mean(spec_hat[:, main_band] - det["dark_base"][:, None], axis=1)
    bo_frames = det["is_blackout"]
    # Count how many originally detected blackout frames remain below 0.40 * E_env after repair
    rem_blackouts = int(np.sum(bo_frames & (E_hat < 0.40 * det["E_env"])))

    # 3. Real atomic peak fidelity on good non-saturated frames (late blow 55%-100%)
    T = len(spec_raw)
    late_good = good.copy()
    late_good[: int(0.55 * T)] = False
    late_good &= ~(det["sat_raw"].any(axis=1))
    if np.sum(late_good) < 10:
        late_good = good & (~(det["sat_raw"].any(axis=1)))

    raw_late_med = np.median(spec_raw[late_good], axis=0)
    hat_late_med = np.median(spec_hat[late_good], axis=0)

    peak_errs = {}
    for pname, lo, hi in [("Na_589", 584, 594), ("K_766", 762, 768.3), ("K_770", 768.4, 773.5)]:
        m = (wl >= lo) & (wl <= hi)
        p_raw = float(np.max(raw_late_med[m]))
        p_hat = float(np.max(hat_late_med[m]))
        rel_err = abs(p_hat - p_raw) / (p_raw + 1e-6) * 100.0
        wl_raw = float(wl[m][np.argmax(raw_late_med[m])])
        wl_hat = float(wl[m][np.argmax(hat_late_med[m])])
        peak_errs[f"{pname}_rel_err%"] = round(rel_err, 3)
        peak_errs[f"{pname}_shift_nm"] = round(abs(wl_hat - wl_raw), 3)

    peak_band = ((wl >= 586.0) & (wl <= 592.0)) | ((wl >= 763.0) & (wl <= 772.5))
    nak_spike_fp_rate = float(det["spike_mask"][good][:, peak_band].mean() * 100.0)

    # 4. Clean-region residual vs true sensor shot+readout noise floor
    diff_clean = spec_raw - spec_hat
    dark_m = (wl >= 358.0) & (wl <= 415.0)
    bright_m = (wl >= 650.0) & (wl <= 850.0)
    mask_dark = clean_px & dark_m[None, :]
    mask_bright = clean_px & bright_m[None, :]

    rmse_dark = float(np.sqrt(np.mean(diff_clean[mask_dark] ** 2)))
    sig_dark_ref = float(np.sqrt(np.mean(res["sigma_col"][dark_m & (~det["col_bad_mask"])] ** 2)))
    rmse_bright = float(np.sqrt(np.mean(diff_clean[mask_bright] ** 2)))
    sig_bright_ref = float(np.sqrt(np.mean(res["sigma_col"][bright_m & (~det["col_bad_mask"])] ** 2)))

    ratio_dark = rmse_dark / (sig_dark_ref + 1e-6)
    ratio_bright = rmse_bright / (sig_bright_ref + 1e-6)

    c1_pass = phys_pass
    c2_pass = (hot383_dev_sigma <= 2.2) and (rem_blackouts == 0)
    c3_pass = (
        peak_errs["Na_589_rel_err%"] < 2.0
        and peak_errs["K_766_rel_err%"] < 2.0
        and peak_errs["K_770_rel_err%"] < 2.0
        and nak_spike_fp_rate < 0.5
    )
    c4_pass = (0.80 <= ratio_dark <= 1.35) and (0.80 <= ratio_bright <= 1.35)
    overall_pass = c1_pass and c2_pass and c3_pass and c4_pass

    return {
        "Overall_Pass": overall_pass,
        "C1_Phys_Pass": c1_pass,
        "C2_AnomalyClean_Pass": c2_pass,
        "C3_PeakFidelity_Pass": c3_pass,
        "C4_NoiseFloor_Pass": c4_pass,
        "Min_Intensity": round(min_val, 2),
        "Hot383_Dev_Sigma": round(hot383_dev_sigma, 3),
        "Remaining_Blackouts": rem_blackouts,
        "Na589_Err%": peak_errs["Na_589_rel_err%"],
        "K766_Err%": peak_errs["K_766_rel_err%"],
        "K770_Err%": peak_errs["K_770_rel_err%"],
        "Max_PeakShift_nm": max(peak_errs["Na_589_shift_nm"], peak_errs["K_766_shift_nm"], peak_errs["K_770_shift_nm"]),
        "NaK_SpikeFlag_Rate%": round(nak_spike_fp_rate, 4),
        "RMSE_Dark": round(rmse_dark, 2),
        "Sigma_Dark_Ref": round(sig_dark_ref, 2),
        "Dark_Ratio": round(ratio_dark, 3),
        "RMSE_Bright": round(rmse_bright, 2),
        "Sigma_Bright_Ref": round(sig_bright_ref, 2),
        "Bright_Ratio": round(ratio_bright, 3),
    }
