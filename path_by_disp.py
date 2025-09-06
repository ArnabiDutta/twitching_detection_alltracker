import numpy as np
import matplotlib.pyplot as plt

def compute_repetition_ratio(trajs_np, visibs_np, max_cap=5.0):
    """
    trajs_np: [N, T, 2] keypoint trajectories
    visibs_np: [N, T] visibility mask (bool)
    Returns: normalized repetition_ratios: [N] in [0, 1]
    """
    N, T, _ = trajs_np.shape
    repetition_ratios = []

    for n in range(N):
        traj = trajs_np[n]          # [T, 2]
        vis = visibs_np[n]          # [T]
        traj_vis = traj[vis]        # Only visible points

        if len(traj_vis) < 2:
            repetition_ratios.append(np.nan)
            continue

        net_disp = np.linalg.norm(traj_vis[-1] - traj_vis[0])
        diffs = np.diff(traj_vis, axis=0)
        path_len = np.sum(np.linalg.norm(diffs, axis=1))

        if path_len == 0 or net_disp == 0:
            ratio = 0
        else:
            ratio = path_len / net_disp

        # --- Normalize between 0–1 ---
        ratio = min(ratio, max_cap)   # clip at max_cap
        norm_ratio = (ratio - 1) / (max_cap - 1)  # maps [1, max_cap] → [0, 1]

        repetition_ratios.append(norm_ratio)

    return np.array(repetition_ratios)

def plot_repetition_ratios(ratios, save_path="repetition_ratio.png"):
    plt.figure(figsize=(10, 4))

    # color coding
    colors = []
    for r in ratios:
        if np.isnan(r):
            colors.append("gray")
        elif r < 0.2:
            colors.append("green")   # smooth
        elif r < 0.6:
            colors.append("orange")  # medium
        else:
            colors.append("red")     # highly repetitive

    plt.scatter(range(len(ratios)), ratios, c=colors, s=60, edgecolors="black")
    plt.plot(ratios, linestyle="--", alpha=0.5)

    plt.title("Normalized Repetition Ratio per Keypoint")
    plt.xlabel("Keypoint ID")
    plt.ylabel("Normalized Ratio (0–1)")
    plt.grid(True)
    plt.ylim(0, 1.05)

    # text annotations
    plt.text(len(ratios) * 0.02, 0.02, "0 move", color="blue", fontsize=10, ha="left")
    plt.text(len(ratios) * 0.3, 0.2, "smooth", color="green", fontsize=10, ha="left")
    plt.text(len(ratios) * 0.6, 0.6, "highly repetitive", color="red", fontsize=10, ha="left")

    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] Plot at {save_path}")
