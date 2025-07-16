import numpy as np
import matplotlib.pyplot as plt
import os

def plot_time_series(trajs, visibs, fps, selected_ids=None, out_dir="."):
    """
    Plots and saves velocity and acceleration time-series for selected tracks.

    trajs:       np.array [N, T, 2]
    visibs:      bool np.array [N, T] or torch.Tensor
    fps:         frames per second (scalar)
    selected_ids: list of trajectory indices to plot (default: random 5)
    out_dir:     directory to save the plots (default: current directory)
    """
    os.makedirs(out_dir, exist_ok=True)

    if hasattr(visibs, 'numpy'):
        visibs = visibs.cpu().numpy()

    N, T, _ = trajs.shape
    if selected_ids is None:
        rng = np.random.default_rng(0)
        selected_ids = rng.choice(N, size=min(5, N), replace=False)

    velocity     = np.zeros((N, T-1))
    acceleration = np.zeros((N, T-2))

    for n in selected_ids:
        valid = visibs[n]
        pts   = trajs[n]

        diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        vel   = diffs * fps
        mask_v = valid[1:] & valid[:-1]
        velocity[n, :] = vel * mask_v

        acc = np.diff(vel)
        mask_a = mask_v[1:] & mask_v[:-1]
        acceleration[n, :] = acc * mask_a

    # Plot and save velocity
    plt.figure(figsize=(10, 4))
    for n in selected_ids:
        plt.plot(np.arange(1, T), velocity[n], label=f"traj {n}")
    plt.xlabel("Frame")
    plt.ylabel("Velocity (px/sec)")
    plt.title("Velocity over Time")
    plt.legend()
    plt.tight_layout()
    vel_path = os.path.join(out_dir, "velocity_plot.png")
    plt.savefig(vel_path)
    plt.show()

    # Plot and save acceleration
    plt.figure(figsize=(10, 4))
    for n in selected_ids:
        plt.plot(np.arange(2, T), acceleration[n], label=f"traj {n}")
    plt.xlabel("Frame")
    plt.ylabel("Acceleration (px/sec²)")
    plt.title("Acceleration over Time")
    plt.legend()
    plt.tight_layout()
    acc_path = os.path.join(out_dir, "acceleration_plot.png")
    plt.savefig(acc_path)
    plt.show()

    print(f"Saved velocity plot to: {vel_path}")
    print(f"Saved acceleration plot to: {acc_path}")
