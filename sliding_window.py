import numpy as np
import matplotlib.pyplot as plt

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# 1. THRESHOLDS & CONFIGURATION
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Global thresholds (for initial filtering)
THRESH_PATH_NOMOVE = 10.0      # (pixels) Global path length for 'No Movement'

# Sliding Window Configuration
WINDOW_SIZE = 15               # (frames) How long each local analysis window is. ~0.5s at 30fps
STEP_SIZE = 5                  # (frames) How far to slide the window each time.

# Classification thresholds based on window statistics
THRESH_MAX_DR_TWITCH = 10.0    # (unitless) If the MAX DR in any window is > this, it's a potential twitch.
THRESH_MEAN_DR_WALK = 2.5      # (unitless) If the MEAN DR is > this, it's not smooth walking.

EPSILON = 1e-6


def compute_motion_metrics(trajs_np, visibs_np, framerate=30):
    """
    Analyzes trajectories using a sliding window to better distinguish walking
    from twitching and other non-linear movements.
    """
    N, T, _ = trajs_np.shape
    analysis_results = []

    for n in range(N):
        traj = trajs_np[n]
        vis = visibs_np[n]
        traj_vis = traj[vis]

        if len(traj_vis) < WINDOW_SIZE:
            analysis_results.append({'id': n, 'label': 'Invalid', 'global_path_len': np.nan})
            continue

        # --- --- --- --- --- --- --- --- --- ---
        # 1. GLOBAL METRIC FOR 'NO MOVEMENT' CHECK
        # --- --- --- --- --- --- --- --- --- ---
        global_path_len = np.sum(np.linalg.norm(np.diff(traj_vis, axis=0), axis=1))

        if global_path_len < THRESH_PATH_NOMOVE:
            analysis_results.append({
                'id': n, 'label': 'No Movement', 'global_path_len': global_path_len,
                'mean_dr': 1.0, 'max_dr': 1.0, 'std_dr': 0.0
            })
            continue

        # --- --- --- --- --- --- --- --- --- ---
        # 2. SLIDING WINDOW ANALYSIS FOR MOVING POINTS
        # --- --- --- --- --- --- --- --- --- ---
        local_directness_ratios = []
        num_windows = (len(traj_vis) - WINDOW_SIZE) // STEP_SIZE + 1

        for i in range(num_windows):
            start = i * STEP_SIZE
            end = start + WINDOW_SIZE
            window_traj = traj_vis[start:end]

            if len(window_traj) < 2:
                continue

            window_path_len = np.sum(np.linalg.norm(np.diff(window_traj, axis=0), axis=1))
            window_net_disp = np.linalg.norm(window_traj[-1] - window_traj[0])
            
            # Avoid division by zero for stationary windows
            if window_path_len < EPSILON:
                local_dr = 1.0
            else:
                local_dr = window_path_len / (window_net_disp + EPSILON)
            
            local_directness_ratios.append(local_dr)

        if not local_directness_ratios:
             analysis_results.append({'id': n, 'label': 'Invalid', 'global_path_len': global_path_len})
             continue

        # --- --- --- --- --- --- --- --- --- ---
        # 3. STATISTICS FROM SLIDING WINDOW
        # --- --- --- --- --- --- --- --- --- ---
        mean_dr = np.mean(local_directness_ratios)
        max_dr = np.max(local_directness_ratios)
        std_dr = np.std(local_directness_ratios)

        # --- --- --- --- --- --- --- --- --- ---
        # 4. NEW CLASSIFICATION LOGIC
        # --- --- --- --- --- --- --- --- --- ---
        label = ''
        # A twitch is a brief, extreme event: high max DR, but not necessarily a high mean.
        if max_dr > THRESH_MAX_DR_TWITCH and mean_dr < (max_dr / 2.0):
             label = 'Twitching'
        # Walking is consistently smooth: low mean DR.
        elif mean_dr < THRESH_MEAN_DR_WALK:
             label = 'Walking'
        # If it's consistently non-direct (like spinning), it's not a twitch.
        else:
             label = 'General Movement' # New category for things like spinning, etc.

        analysis_results.append({
            'id': n, 'label': label, 'global_path_len': global_path_len,
            'mean_dr': mean_dr, 'max_dr': max_dr, 'std_dr': std_dr,
        })

    return analysis_results


def plot_motion_analysis(results, save_path="motion_analysis.png"):
    """
    Generates a multi-panel plot to visualize the new sliding window analysis.
    """
    if not results:
        print("No results to plot.")
        return

    # Filter out invalid results before unpacking
    valid_results = [r for r in results if r['label'] != 'Invalid']
    if not valid_results:
        print("No valid points to plot.")
        return

    ids = [r['id'] for r in valid_results]
    labels = np.array([r['label'] for r in valid_results])
    mean_drs = np.array([r['mean_dr'] for r in valid_results])
    max_drs = np.array([r['max_dr'] for r in valid_results])
    global_paths = np.array([r['global_path_len'] for r in valid_results])

    color_map = {
        'No Movement': 'blue',
        'Walking': 'green',
        'Twitching': 'red',
        'General Movement': 'purple',
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    fig.suptitle("Sliding Window Motion Analysis", fontsize=16)

    # --- --- --- --- --- --- --- --- --- ---
    # PLOT 1: Mean DR vs Max DR
    # --- --- --- --- --- --- --- --- --- ---
    ax1 = axes[0]
    for label, color in color_map.items():
        mask = labels == label
        if np.any(mask):
            ax1.scatter(mean_drs[mask], max_drs[mask], c=color, label=label, s=50, alpha=0.7, edgecolors='black')
    
    ax1.set_title("Per-Point Classification (Mean vs. Max Directness Ratio)")
    ax1.set_xlabel("Mean Directness Ratio (across all windows)")
    ax1.set_ylabel("Max Directness Ratio (in any single window)")
    ax1.grid(True, which="both", ls="--")
    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.axhline(y=THRESH_MAX_DR_TWITCH, color='red', linestyle='--', label=f'Twitch Max DR Threshold')
    ax1.axvline(x=THRESH_MEAN_DR_WALK, color='green', linestyle='--', label=f'Walking Mean DR Threshold')
    ax1.legend()

    # --- --- --- --- --- --- --- --- --- ---
    # PLOT 2: Label Distribution by Keypoint ID
    # --- --- --- --- --- --- --- --- --- ---
    ax2 = axes[1]
    bottoms = {} # Used for stacking bars
    for label, color in color_map.items():
        mask = labels == label
        if np.any(mask):
            # Create a simple bar for each point for visualization
            ax2.bar(np.array(ids)[mask], 1, bottom=bottoms.get(label, 0), color=color, label=label, width=1.0)
            # This is a visual trick; we just want to see the color for each ID.
    
    ax2.set_title("Final Classification per Keypoint ID")
    ax2.set_xlabel("Keypoint ID")
    ax2.set_yticks([]) # No y-axis needed
    ax2.set_ylabel("Classification")
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] Motion analysis plot at {save_path}")
