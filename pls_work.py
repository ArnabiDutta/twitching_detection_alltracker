# path_by_disp.py
# Robust repetition score computed relative to global body motion.
# Author: you + ChatGPT
# Usage:
#   from path_by_disp import compute_repetition_ratio, plot_repetition_ratios
#   ratios = compute_repetition_ratio(trajs_np, visibs_np)

from __future__ import annotations
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

def _gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    """1D Gaussian smoothing along time for each coordinate independently."""
    if sigma <= 0:
        return arr
    # Build kernel
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    # Convolve along time for each column independently
    out = np.empty_like(arr)
    for d in range(arr.shape[1]):
        out[:, d] = np.convolve(arr[:, d], k, mode="same")
    return out

def _per_frame_center(
    trajs_np: np.ndarray,
    visibs_np: np.ndarray,
    method: str = "median",
) -> np.ndarray:
    """
    Compute a robust center-of-mass / global motion per frame using *visible* points only.
    trajs_np: [N, T, 2]  (float)
    visibs_np: [N, T]    (bool)
    returns com: [T, 2]
    """
    N, T, _ = trajs_np.shape
    com = np.zeros((T, 2), dtype=np.float32)
    for t in range(T):
        mask = visibs_np[:, t]
        if not np.any(mask):
            # If nothing visible, copy previous (or keep zeros for t=0)
            com[t] = com[t - 1] if t > 0 else 0.0
            continue
        pts = trajs_np[mask, t]  # [M,2]
        if method == "mean":
            com[t] = np.nanmean(pts, axis=0)
        else:
            # default robust median
            com[t] = np.nanmedian(pts, axis=0)
    return com

def _path_len_and_net_disp(traj_xy: np.ndarray) -> Tuple[float, float]:
    """
    traj_xy: [T_vis, 2] (after visibility filtering & COM subtraction)
    returns (path_len, net_disp)
    """
    if len(traj_xy) < 2:
        return 0.0, 0.0
    diffs = np.diff(traj_xy, axis=0)                     # [T_vis-1, 2]
    step = np.linalg.norm(diffs, axis=1)                 # [T_vis-1]
    path_len = float(np.nansum(step))
    net_disp = float(np.linalg.norm(traj_xy[-1] - traj_xy[0]))
    return path_len, net_disp

def compute_repetition_ratio(
    trajs_np: np.ndarray,
    visibs_np: np.ndarray,
    *,
    max_cap: float = 5.0,
    min_visible: int = 5,
    com_method: str = "median",
    com_smooth_sigma: float = 0.0,
    per_point_smooth_sigma: float = 0.0,
    eps: float = 1e-6,
    return_debug: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute normalized repetition ratios *relative to global motion*.

    Inputs
    ------
    trajs_np : [N, T, 2] float
        Keypoint trajectories in image coords.
    visibs_np : [N, T] bool
        Visibility per keypoint per frame.
    max_cap : float
        Upper cap for raw ratio before normalization (stabilizes the scale).
    min_visible : int
        Require at least this many visible frames for a keypoint; otherwise NaN.
    com_method : {"median","mean"}
        How to compute per-frame body center.
    com_smooth_sigma : float
        Optional Gaussian smoothing sigma for the COM curve (in frames).
    per_point_smooth_sigma : float
        Optional Gaussian smoothing sigma for per-point trajectories before metrics.
    eps : float
        Numerical stability.

    Returns
    -------
    ratios_norm : [N] float in [0,1] (NaN if not enough visibility)
        0 ~ smooth, 1 ~ highly repetitive relative motion.
    (optional) debug dict with raw_ratio, net_disp_rel, path_len_rel, com

    Notes
    -----
    - We first compute a robust center-of-mass per frame from all visible points.
    - We subtract this COM from every keypoint before computing path_len/net_disp.
    - This makes locomotion (walking) look smooth (ratio→~0) while localized oscillations
      (waving, twitching) look repetitive (ratio→~1).
    """
    assert trajs_np.ndim == 3 and trajs_np.shape[2] == 2, "trajs_np must be [N,T,2]"
    assert visibs_np.shape[:2] == trajs_np.shape[:2], "visibs_np shape mismatch"

    N, T, _ = trajs_np.shape
    # 1) Center-of-mass over time (robust to occlusions via per-frame visibility)
    com = _per_frame_center(trajs_np, visibs_np, method=com_method)  # [T,2]
    if com_smooth_sigma > 0:
        com = _gaussian_smooth(com, sigma=com_smooth_sigma)

    ratios_norm = np.full((N,), np.nan, dtype=np.float32)
    raw_ratio = np.full((N,), np.nan, dtype=np.float32)
    net_disp_rel = np.full((N,), np.nan, dtype=np.float32)
    path_len_rel = np.full((N,), np.nan, dtype=np.float32)

    # 2) For each keypoint, subtract COM then measure path vs displacement
    for n in range(N):
        vis = visibs_np[n]                 # [T] bool
        if vis.sum() < min_visible:
            continue

        traj = trajs_np[n].astype(np.float32)  # [T,2]
        traj_rel = traj - com                  # relative to body motion

        # keep only visible time steps (preserve order)
        traj_rel_vis = traj_rel[vis]
        if per_point_smooth_sigma > 0 and len(traj_rel_vis) >= 2:
            traj_rel_vis = _gaussian_smooth(traj_rel_vis, sigma=per_point_smooth_sigma)

        p_len, n_disp = _path_len_and_net_disp(traj_rel_vis)
        path_len_rel[n] = p_len
        net_disp_rel[n] = n_disp
        # Handle static or background cases
        if p_len < 1e-3 and n_disp < 1e-3:
            ratios_norm[n] = 0.0   # treat as smooth
            continue
        if n_disp < 1.0:  # very low displacement = probably static background
            continue
        
        raw = (p_len + eps) / (n_disp + eps)
        raw_ratio[n] = raw
        
        # Map so that raw≈1 → 0 (smooth), raw≫1 → 1 (repetitive)
        raw_clipped = min(max_cap, raw)
        ratios_norm[n] = (raw_clipped - 1.0) / (max_cap - 1.0)
        ratios_norm[n] = max(0.0, ratios_norm[n])  # keep in [0,1]

    if return_debug:
        return ratios_norm, {
            "raw_ratio": raw_ratio,
            "path_len_rel": path_len_rel,
            "net_disp_rel": net_disp_rel,
            "com": com,
        }
    return ratios_norm

def plot_repetition_ratios(
    ratios: np.ndarray,
    save_path: str = "repetition_ratio.png",
    title: str = "Normalized Repetition Ratio per Keypoint (relative to body)",
) -> None:
    """
    Simple scatter plot with semantic colors:
      green < 0.2, orange < 0.6, red otherwise; gray = NaN
    """
    valid = ~np.isnan(ratios)
    x = np.arange(len(ratios))

    # color coding
    colors = []
    for r in ratios:
        if not np.isfinite(r):
            colors.append("gray")
        elif r < 0.2:
            colors.append("green")
        elif r < 0.6:
            colors.append("orange")
        else:
            colors.append("red")

    plt.figure(figsize=(12, 4))
    plt.scatter(x[valid], ratios[valid], c=np.array(colors)[valid], s=16, edgecolors="black", linewidths=0.2)
    if valid.any():
        plt.plot(x[valid], ratios[valid], linestyle="--", alpha=0.25)

    plt.title(title)
    plt.xlabel("Keypoint ID")
    plt.ylabel("Normalized Ratio (0–1)")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    # hints
    n = len(ratios)
    plt.text(n * 0.02, 0.05, "smooth", color="green", fontsize=9, ha="left")
    plt.text(n * 0.35, 0.35, "medium", color="orange", fontsize=9, ha="left")
    plt.text(n * 0.70, 0.80, "highly repetitive", color="red", fontsize=9, ha="left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Saved] Plot at {save_path}")
