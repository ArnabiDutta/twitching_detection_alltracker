import numpy as np
import matplotlib.pyplot as plt

def plot_time_series(trajs, visibs, fps, selected_ids=None):
    """
    Plots velocity and acceleration time-series for a handful of tracks.
    
    trajs:   np.array [N, T, 2]
    visibs:  bool np.array [N, T] or torch.Tensor
    fps:     frames per second (scalar)
    selected_ids: list of trajectory indices to plot (default: random 5)
    """
    # Convert visibs to numpy if it's a tensor
    if hasattr(visibs, 'numpy'):
        visibs = visibs.cpu().numpy()

    N, T, _ = trajs.shape
    if selected_ids is None:
        rng = np.random.default_rng(0)
        selected_ids = rng.choice(N, size=min(5, N), replace=False)
    
    # Pre-allocate
    velocity     = np.zeros((N, T-1))
    acceleration = np.zeros((N, T-2))
    
    for n in selected_ids:
        valid = visibs[n]
        pts   = trajs[n]
        
        # velocity (px/sec)
        diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        vel   = diffs * fps
        mask_v = valid[1:] & valid[:-1]
        velocity[n, :] = vel * mask_v  # already numpy array

        # acceleration (px/sec²)
        acc = np.diff(vel)
        mask_a = mask_v[1:] & mask_v[:-1]
        acceleration[n, :] = acc * mask_a  # mask_a is also numpy array now

    # Plot velocity
    plt.figure(figsize=(10,4))
    for n in selected_ids:
        plt.plot(np.arange(1, T), velocity[n], label=f"traj {n}")
    plt.xlabel("Frame")
    plt.ylabel("Velocity (px/sec)")
    plt.title("Velocity over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot acceleration
    plt.figure(figsize=(10,4))
    for n in selected_ids:
        plt.plot(np.arange(2, T), acceleration[n], label=f"traj {n}")
    plt.xlabel("Frame")
    plt.ylabel("Acceleration (px/sec²)")
    plt.title("Acceleration over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()
