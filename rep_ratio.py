import numpy as np
import matplotlib.pyplot as plt

def compute_repetition_ratio(trajs_np, visibs_np):
    """
    trajs_np: [N, T, 2] keypoint trajectories
    visibs_np: [N, T] visibility mask (bool)
    Returns: repetition_ratios: [N] array of ratios per keypoint
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

        # Net displacement: from first to last visible point
        net_disp = np.linalg.norm(traj_vis[-1] - traj_vis[0])

        # Path length: sum of distances between consecutive visible points
        diffs = np.diff(traj_vis, axis=0)
        path_len = np.sum(np.linalg.norm(diffs, axis=1))

        if path_len == 0:
            ratio = 0
        else:
            ratio = net_disp / path_len

        repetition_ratios.append(ratio)

    return np.array(repetition_ratios)

def plot_repetition_ratios(ratios, save_path="repetition_ratio.png"):
    plt.figure(figsize=(10, 4))
    plt.plot(ratios, marker='o')
    plt.title("Repetition Ratio per Keypoint")
    plt.xlabel("Keypoint ID")
    plt.ylabel("Net Displacement / Path Length")
    plt.grid(True)
    plt.ylim(0, 1.1)
    plt.savefig(save_path)
    plt.close()          # <— frees the figure
    print(f"[Saved] Plot at {save_path}")

# Example usage
# Assuming `trajs_np` is [N, T, 2] and `visibs_np` is [N, T]
# Load or pass them from your demo.py result
# trajs_np = ...
# visibs_np = ...

# ratios = compute_repetition_ratio(trajs_np, visibs_np)
# plot_repetition_ratios(ratios)
