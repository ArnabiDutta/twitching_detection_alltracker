import numpy as np
import matplotlib.pyplot as plt

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# 1. THRESHOLDS & CONFIGURATION
# These values are your main knobs for tuning the classification.
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
THRESH_PATH_NOMOVE = 10.0     # (pixels) If total path is less than this, it's "No Movement".
THRESH_DR_TWITCH = 8.0       # (unitless) If Directness Ratio is > this, it's "Twitching".
THRESH_ACCEL_TWITCH = 25.0    # (pixels/frame^2) If max accel is > this, it's also "Twitching".
EPSILON = 1e-6               # A small number to prevent division by zero.


def compute_motion_metrics(trajs_np, visibs_np, framerate=30):
    """
    Analyzes trajectories to classify them into 'No Movement', 'Walking', or 'Twitching'.

    Args:
        trajs_np (np.array): [N, T, 2] keypoint trajectories.
        visibs_np (np.array): [N, T] visibility mask (bool).
        framerate (int): The framerate of the video for calculating real-world acceleration.

    Returns:
        list[dict]: A list of dictionaries, one for each keypoint, containing all
                    computed metrics and the final classification label.
    """
    N, T, _ = trajs_np.shape
    analysis_results = []

    for n in range(N):
        traj = trajs_np[n]
        vis = visibs_np[n]
        traj_vis = traj[vis]

        if len(traj_vis) < 3:  # Need at least 3 points for acceleration
            analysis_results.append({'id': n, 'label': 'Invalid', 'path_len': np.nan})
            continue

        # --- --- --- --- --- --- --- --- --- ---
        # 2. COMPUTE CORE METRICS
        # --- --- --- --- --- --- --- --- --- ---
        # Path Length
        path_len = np.sum(np.linalg.norm(np.diff(traj_vis, axis=0), axis=1))

        # Net Displacement
        net_disp = np.linalg.norm(traj_vis[-1] - traj_vis[0])

        # Directness Ratio (The fixed version of your metric)
        directness_ratio = path_len / (net_disp + EPSILON)

        # Acceleration
        velocities = np.diff(traj_vis, axis=0) * framerate
        accelerations = np.diff(velocities, axis=0) * framerate
        accel_magnitudes = np.linalg.norm(accelerations, axis=1)
        max_accel_mag = np.max(accel_magnitudes) if len(accel_magnitudes) > 0 else 0

        # --- --- --- --- --- --- --- --- --- ---
        # 3. CLASSIFICATION LOGIC
        # --- --- --- --- --- --- --- --- --- ---
        label = ''
        if path_len < THRESH_PATH_NOMOVE:
            label = 'No Movement'
        elif directness_ratio > THRESH_DR_TWITCH or max_accel_mag > THRESH_ACCEL_TWITCH:
            label = 'Twitching'
        else:
            label = 'Walking'

        analysis_results.append({
            'id': n,
            'label': label,
            'path_len': path_len,
            'net_disp': net_disp,
            'directness_ratio': directness_ratio,
            'max_accel_mag': max_accel_mag,
        })

    return analysis_results


def plot_motion_analysis(results, save_path="motion_analysis.png"):
    """
    Generates a multi-panel plot to visualize the motion analysis results.
    """
    if not results:
        print("No results to plot.")
        return

    # Unpack results for easier plotting
    ids = [r['id'] for r in results if 'path_len' in r and not np.isnan(r['path_len'])]
    labels = np.array([r['label'] for r in results if r['id'] in ids])
    ratios = np.array([r['directness_ratio'] for r in results if r['id'] in ids])
    accels = np.array([r['max_accel_mag'] for r in results if r['id'] in ids])
    paths = np.array([r['path_len'] for r in results if r['id'] in ids])

    # Define colors for each category
    color_map = {
        'No Movement': 'blue',
        'Walking': 'green',
        'Twitching': 'red',
        'Invalid': 'gray'
    }
    colors = [color_map.get(label, 'gray') for label in labels]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    fig.suptitle("Motion Analysis of Tracked Points", fontsize=16)

    # --- --- --- --- --- --- --- --- --- ---
    # PLOT 1: Classification by Directness Ratio
    # --- --- --- --- --- --- --- --- --- ---
    ax1 = axes[0]
    for label, color in color_map.items():
        mask = labels == label
        if np.any(mask):
            ax1.scatter(np.array(ids)[mask], ratios[mask], c=color, label=label, s=50, alpha=0.7, edgecolors='black')

    ax1.set_yscale('log')  # Log scale is crucial for viewing the wide range of ratios
    ax1.set_title("Per-Point Classification based on Directness Ratio")
    ax1.set_xlabel("Keypoint ID")
    ax1.set_ylabel("Directness Ratio (Path / Disp) - Log Scale")
    ax1.grid(True, which="both", ls="--")
    ax1.legend()
    ax1.axhline(y=THRESH_DR_TWITCH, color='red', linestyle='--', label=f'Twitch Threshold ({THRESH_DR_TWITCH})')
    ax1.legend()


    # --- --- --- --- --- --- --- --- --- ---
    # PLOT 2: Diagnostic Plot (Acceleration vs. Path Length)
    # --- --- --- --- --- --- --- --- --- ---
    ax2 = axes[1]
    for label, color in color_map.items():
        mask = labels == label
        if np.any(mask):
            ax2.scatter(paths[mask], accels[mask], c=color, label=label, s=50, alpha=0.7, edgecolors='black')

    ax2.set_title("Diagnostic View: Max Acceleration vs. Total Path Length")
    ax2.set_xlabel("Total Path Length (pixels)")
    ax2.set_ylabel("Max Acceleration (pixels/frame²)")
    ax2.grid(True, ls="--")
    ax2.legend()
    ax2.axhline(y=THRESH_ACCEL_TWITCH, color='red', linestyle='--', label=f'Twitch Accel Threshold')
    ax2.axvline(x=THRESH_PATH_NOMOVE, color='blue', linestyle='--', label=f'No-Move Path Threshold')
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] Motion analysis plot at {save_path}")
