# demo_batch.py
import torch
import cv2
import argparse
import utils.saveload
import utils.basic
import utils.improc
import PIL.Image
import numpy as np
import os
from prettytable import PrettyTable
import time
import pandas as pd

# import your existing modules
from plot_trajectories import plot_global_trajectories
from rep_ratio import compute_repetition_ratio, plot_repetition_ratios

def read_mp4(name_path):
    vidcap = cv2.VideoCapture(name_path)
    framerate = int(round(vidcap.get(cv2.CAP_PROP_FPS)))
    print(f"[INFO] '{name_path}' → framerate {framerate}")
    frames = []
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    vidcap.release()
    return frames, framerate

# --- your draw_pts_gpu() unchanged ---
def draw_pts_gpu(rgbs, trajs, visibs, colormap, rate=1, bkg_opacity=0.0):
    # … copy exactly your existing implementation here …
    # (omitted for brevity)
    return rgbs

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        if param > 100_000:
            table.add_row([name, param])
        total_params += param
    print(table)
    print(f"total params: {total_params/1e6:.2f} M")
    return total_params

def forward_video(rgbs, framerate, model, args, basename):
    B,T,C,H,W = rgbs.shape
    device = rgbs.device
    # … same forward logic up through computing trajs_np & visibs_np …
    # For brevity, assume trajs_np, visibs_np are now:
    #   trajs_np: [N, T, 2] np.ndarray
    #   visibs_np: [N, T] bool np.ndarray

    # 1) save trajectory-only video
    traj_vid = os.path.join("results", f"{basename}_trajectories.mp4")
    plot_global_trajectories(trajs_np, visibs_np, 
                             utils.improc.get_2d_colors(trajs_np[:,0], H, W),
                             H, W,
                             save_path=traj_vid,
                             fps=framerate)

    # 2) compute & save repetition ratio CSV + plot PNG
    ratios = compute_repetition_ratio(trajs_np, visibs_np)
    # raw CSV
    df = pd.DataFrame({
        "track_id": np.arange(len(ratios)),
        "repetition_ratio": ratios
    })
    csv_path = os.path.join("results", f"{basename}_repetition_ratio.csv")
    df.to_csv(csv_path, index=False)
    # plot PNG
    png_path = os.path.join("results", f"{basename}_repetition_ratio.png")
    plot_repetition_ratios(ratios, save_path=png_path)

    # 3) now your original overlay video (optional—comment out if you only want the tra j video)
    # … your existing code that calls draw_pts_gpu() and writes the mp4 …

def run_on_folder(model, args):
    os.makedirs("results", exist_ok=True)
    for fname in sorted(os.listdir(args.cropped_folder)):
        if not fname.lower().endswith(".mp4"):
            continue
        path = os.path.join(args.cropped_folder, fname)
        basename = os.path.splitext(fname)[0]
        print(f"\n=== Processing {fname} ===")
        frames, fps = read_mp4(path)
        # shrink & stack exactly as before
        if args.max_frames:
            frames = frames[: args.max_frames]
        H0,W0 = frames[0].shape[:2]
        scale = min(1024/H0, 1024/W0)
        H,W = int(H0*scale)//8*8, int(W0*scale)//8*8
        frames = [cv2.resize(f, (W,H)) for f in frames]
        # to tensor
        rgbs = torch.stack(
            [torch.from_numpy(f).permute(2,0,1) for f in frames],
            dim=0
        ).unsqueeze(0).float().cuda()
        forward_video(rgbs, fps, model, args, basename)

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_init", type=str, default="")
    parser.add_argument("--cropped_folder", type=str, default="cropped_videos")
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--inference_iters", type=int, default=4)
    parser.add_argument("--window_len", type=int, default=16)
    parser.add_argument("--rate", type=int, default=16)
    parser.add_argument("--conf_thr", type=float, default=0.1)
    parser.add_argument("--bkg_opacity", type=float, default=0.0)
    parser.add_argument("--vstack", action="store_true")
    parser.add_argument("--hstack", action="store_true")
    args = parser.parse_args()

    # load model once
    from nets.alltracker import Net
    model = Net(args.window_len).cuda().eval()
    if args.ckpt_init:
        utils.saveload.load(None, args.ckpt_init, model, None, None, None, True, False)
    else:
        url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
        sd = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(sd["model"], strict=True)
    for p in model.parameters():
        p.requires_grad = False
    count_parameters(model)

    run_on_folder(model, args)
