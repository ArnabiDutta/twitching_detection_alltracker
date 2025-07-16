import numpy as np
import torch
import cv2

def plot_global_trajectories(trajs, visibs, colors, H, W, selected_ids=None, save_path="trajectory_video.mp4", fps=10):
    """
    trajs: [N, T, 2] tensor
    visibs: [N, T] bool tensor
    colors: [N, 3] np.array in [0,1]
    """
    N, T, _ = trajs.shape
    if selected_ids is None:
        selected_ids = np.random.choice(N, size=min(50, N), replace=False)

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    video_writer = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    prev_pts = {n: None for n in selected_ids}

    for t in range(T):
        for n in selected_ids:
            if not visibs[n, t]:
                continue
            x, y = trajs[n, t].numpy().astype(int)
            color = (colors[n] * 255).astype(np.uint8).tolist()
            if prev_pts[n] is not None:
                cv2.line(canvas, prev_pts[n], (x, y), color, 1)
            prev_pts[n] = (x, y)
        video_writer.write(canvas.copy())

    video_writer.release()
    print(f"[✓] Saved trajectory-only video: {save_path}")