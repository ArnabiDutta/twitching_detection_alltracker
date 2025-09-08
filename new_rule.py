# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# In motion_analyzer.py

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# 1. THRESHOLDS & CONFIGURATION
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Global thresholds
THRESH_PATH_NOMOVE = 10.0

# Sliding Window Configuration
WINDOW_SIZE = 15
STEP_SIZE = 5

# Classification thresholds
THRESH_MAX_DR_TWITCH = 10.0
THRESH_MEAN_DR_WALK = 2.5
# THRESHOLD FOR THE "CONTEXT" RULE
THRESH_WINDOW_PATH_CALM = 5.0
# *** NEW THRESHOLD TO PREVENT FALSE POSITIVES FROM NOISE ***
THRESH_WINDOW_PATH_SIGNIFICANT = 8.0 # (pixels) The spike itself must have at least this much movement.

EPSILON = 1e-6


def compute_motion_metrics(trajs_np, visibs_np, framerate=30):
    """
    Analyzes trajectories using a sliding window and contextual analysis,
    now with a magnitude check to reject low-amplitude noise.
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

        global_path_len = np.sum(np.linalg.norm(np.diff(traj_vis, axis=0), axis=1))

        if global_path_len < THRESH_PATH_NOMOVE:
            analysis_results.append({
                'id': n, 'label': 'No Movement', 'global_path_len': global_path_len,
                'mean_dr': 1.0, 'max_dr': 1.0, 'std_dr': 0.0
            })
            continue

        local_directness_ratios = []
        window_path_lengths = []
        num_windows = (len(traj_vis) - WINDOW_SIZE) // STEP_SIZE + 1

        for i in range(num_windows):
            start = i * STEP_SIZE
            end = start + WINDOW_SIZE
            window_traj = traj_vis[start:end]

            if len(window_traj) < 2: continue

            window_path_len = np.sum(np.linalg.norm(np.diff(window_traj, axis=0), axis=1))
            window_net_disp = np.linalg.norm(window_traj[-1] - window_traj[0])
            
            window_path_lengths.append(window_path_len)
            
            if window_path_len < EPSILON:
                local_directness_ratios.append(1.0)
            else:
                local_directness_ratios.append(window_path_len / (window_net_disp + EPSILON))

        if not local_directness_ratios:
             analysis_results.append({'id': n, 'label': 'Invalid', 'global_path_len': global_path_len})
             continue

        mean_dr = np.mean(local_directness_ratios)
        max_dr = np.max(local_directness_ratios)
        std_dr = np.std(local_directness_ratios)

        # --- --- --- --- --- --- --- --- --- ---
        # FINAL, MOST ROBUST CLASSIFICATION LOGIC
        # --- --- --- --- --- --- --- --- --- ---
        label = ''
        is_a_twitch = False
        if max_dr > THRESH_MAX_DR_TWITCH:
            max_dr_idx = np.argmax(local_directness_ratios)
            
            # Condition 1: Is the spike's MAGNITUDE significant?
            is_significant = window_path_lengths[max_dr_idx] > THRESH_WINDOW_PATH_SIGNIFICANT

            # Condition 2: Is the spike ISOLATED? (calm before and after)
            is_calm_before = True
            if max_dr_idx > 0:
                is_calm_before = window_path_lengths[max_dr_idx - 1] < THRESH_WINDOW_PATH_CALM
            
            is_calm_after = True
            if max_dr_idx < len(window_path_lengths) - 1:
                is_calm_after = window_path_lengths[max_dr_idx + 1] < THRESH_WINDOW_PATH_CALM

            if is_significant and is_calm_before and is_calm_after:
                is_a_twitch = True

        if is_a_twitch:
             label = 'Twitching'
        elif mean_dr < THRESH_MEAN_DR_WALK:
             label = 'Walking'
        else:
             label = 'General Movement'

        analysis_results.append({
            'id': n, 'label': label, 'global_path_len': global_path_len,
            'mean_dr': mean_dr, 'max_dr': max_dr, 'std_dr': std_dr,
        })

    return analysis_results

# NOTE: The plot_motion_analysis function does not need to be changed.
