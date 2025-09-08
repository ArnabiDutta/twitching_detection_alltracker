import torch
import cv2
import argparse
import utils.saveload
import utils.basic
import utils.improc
import PIL.Image
import numpy as np
from ultralytics import YOLO
import os
from prettytable import PrettyTable
import time
from sliding_window import compute_motion_metrics, plot_motion_analysis


def read_mp4(name_path):
    vidcap = cv2.VideoCapture(name_path)
    framerate = int(round(vidcap.get(cv2.CAP_PROP_FPS)))
    print('framerate', framerate)
    frames = []
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    vidcap.release()
    return frames, framerate
def draw_twitching_visualization(rgbs, trajs, visibs, rate=1):
    """
    Draws only the provided trajectories as bright red points and lines
    on a black background to highlight twitching motion.
    
    Args:
        rgbs (torch.Tensor): A tensor of video frames (used for shape info).
        trajs (torch.Tensor): The trajectories of twitching points [T, N, 2].
        visibs (torch.Tensor): Visibility data for the twitching points [T, N].
        rate (int): The subsampling rate used.
    """
    device = trajs.device
    # Force a black background by creating a new tensor on the correct device
    T, C, H, W = rgbs.shape
    
    # --- THIS IS THE CORRECTED LINE ---
    output_rgbs = torch.zeros((T, C, H, W), device=device, dtype=torch.float32)

    # Prepare trajectory data
    trajs = trajs.permute(1,0,2) # N,T,2
    visibs = visibs.permute(1,0) # N,T
    N = trajs.shape[0]

    # Use a single, bright red color for all twitching points
    twitch_color = torch.tensor([255.0, 50.0, 50.0], dtype=torch.float32, device=device)
    colors = twitch_color.unsqueeze(0).repeat(N, 1) # [N, 3]

    # --- This point-drawing logic is from the original draw_pts_gpu ---
    opacity = 1.0
    radius = 2 if rate > 4 else 1 # Make points a bit bigger to be visible
    sharpness = 0.2
    
    D = radius * 2 + 1
    y = torch.arange(D, device=device).float()[:, None] - radius
    x = torch.arange(D, device=device).float()[None, :] - radius
    dist2 = x**2 + y**2
    icon = torch.clamp(1 - (dist2 - (radius**2) / 2.0) / (radius * 2 * sharpness), 0, 1)
    icon = icon.view(1, D, D)
    dx = torch.arange(-radius, radius + 1, device=device)
    dy = torch.arange(-radius, radius + 1, device=device)
    disp_y, disp_x = torch.meshgrid(dy, dx, indexing="ij")

    for t in range(T):
        mask = visibs[:, t]
        if mask.sum() == 0: continue
            
        xy = trajs[mask, t] + 0.5
        xy[:, 0] = xy[:, 0].clamp(0, W - 1)
        xy[:, 1] = xy[:, 1].clamp(0, H - 1)
        colors_now = colors[mask]
        N_now = xy.shape[0]
        cx, cy = xy[:, 0].long(), xy[:, 1].long()
        x_grid, y_grid = cx[:, None, None] + disp_x, cy[:, None, None] + disp_y
        
        valid = (x_grid >= 0) & (x_grid < W) & (y_grid >= 0) & (y_grid < H)
        x_valid, y_valid = x_grid[valid], y_grid[valid]
        icon_weights = icon.expand(N_now, D, D)[valid]
        colors_valid = colors_now[:, :, None, None].expand(N_now, 3, D, D).permute(1, 0, 2, 3)[:, valid]
        idx_flat = (y_valid * W + x_valid).long()

        accum = torch.zeros_like(output_rgbs[t])
        weight = torch.zeros(1, H * W, device=device)
        img_flat = accum.view(C, -1)
        
        weighted_colors = colors_valid * icon_weights
        img_flat.scatter_add_(1, idx_flat.unsqueeze(0).expand(C, -1), weighted_colors)
        weight.scatter_add_(1, idx_flat.unsqueeze(0), icon_weights.unsqueeze(0))
        weight = weight.view(1, H, W)

        alpha = weight.clamp(0, 1) * opacity
        accum = accum / (weight + 1e-6)
        output_rgbs[t] = output_rgbs[t] * (1 - alpha) + accum * alpha

    # --- Now, draw lines connecting the trajectory points ---
    # Convert to NumPy for cv2 line drawing
    output_rgbs_np = np.ascontiguousarray(output_rgbs.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy())    
    line_color_bgr = (50, 50, 255) # BGR format for OpenCV
    
    for n in range(N): # For each trajectory
        prev_point = None
        for t in range(T): # For each frame
            if visibs[n, t]:
                x, y = trajs[n, t].cpu().numpy().astype(int)
                if prev_point is not None:
                    cv2.line(output_rgbs_np[t], tuple(prev_point), (x, y), line_color_bgr, 1)
                prev_point = (x, y)
            else:
                prev_point = None # Reset if point becomes invisible

    return output_rgbs_np
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        if param > 100000:
            table.add_row([name, param])
        total_params+=param
    print(table)
    print('total params: %.2f M' % (total_params/1000000.0))
    return total_params
def get_person_masks(frames):
    """
    Runs YOLO segmentation (global model) on frames.
    Returns: np.array of shape [T, H, W] with binary person masks.
    """
    masks = []
    kernel = np.ones((5, 5), np.uint8)
    for frame in frames:
        results = yolo_model(frame, verbose=False)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        if results[0].masks is not None:
            for cls_id, seg in zip(results[0].boxes.cls.cpu().numpy(),
                                   results[0].masks.xy):
                if int(cls_id) == 0:  # class 0 = person
                    poly = np.array([seg], dtype=np.int32)
                    cv2.fillPoly(mask, poly, 1)
        mask = cv2.erode(mask, kernel, iterations=1)  # Removes noisy edges
        mask = cv2.dilate(mask, kernel, iterations=1) # Fills holes and restores size

        masks.append(mask)
    return np.stack(masks, axis=0)  # [T, H, W]

def forward_video(rgbs, framerate, model, args, basename):
    B,T,C,H,W = rgbs.shape
    assert C == 3
    device = rgbs.device
    assert(B==1)

    grid_xy = utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float()
    grid_xy = grid_xy.permute(0,2,1).reshape(1,1,2,H,W)

    torch.cuda.empty_cache()
    print('starting forward...')
    f_start_time = time.time()

    flows_e, visconf_maps_e, _, _ = \
        model(rgbs[:, args.query_frame:], iters=args.inference_iters, sw=None, is_training=False)
    traj_maps_e = flows_e + grid_xy
    if args.query_frame > 0:
        backward_flows_e, backward_visconf_maps_e, _, _ = \
            model(rgbs[:, :args.query_frame+1].flip([1]), iters=args.inference_iters, sw=None, is_training=False)
        backward_traj_maps_e = backward_flows_e + grid_xy
        backward_traj_maps_e = backward_traj_maps_e.flip([1])[:, :-1]
        backward_visconf_maps_e = backward_visconf_maps_e.flip([1])[:, :-1]
        traj_maps_e = torch.cat([backward_traj_maps_e, traj_maps_e], dim=1)
        visconf_maps_e = torch.cat([backward_visconf_maps_e, visconf_maps_e], dim=1)
    ftime = time.time()-f_start_time
    print('finished forward; %.2f seconds / %d frames; %d fps' % (ftime, T, round(T/ftime)))
    utils.basic.print_stats('traj_maps_e', traj_maps_e)
    utils.basic.print_stats('visconf_maps_e', visconf_maps_e)

    rate = args.rate
    trajs_e = traj_maps_e[:,:,:,::rate,::rate].reshape(B,T,2,-1).permute(0,1,3,2)
    visconfs_e = visconf_maps_e[:,:,:,::rate,::rate].reshape(B,T,2,-1).permute(0,1,3,2)

    trajs_np = trajs_e[0].cpu().permute(1, 0, 2)  # [N, T, 2]
    visibs_np = visconfs_e[0,:,:,1].cpu().permute(1, 0).bool()  # [N, T]

      # === Run YOLO segmentation inline (global model) ===
    human_masks = get_person_masks([f.permute(1,2,0).cpu().numpy().astype(np.uint8) 
                                    for f in rgbs[0]])  # [T,H,W]
    mask_small = human_masks[:, ::rate, ::rate]
    mask_flat = mask_small.reshape(T, -1).T.astype(bool)  # [N,T]
    print(f"[forward_video] Applied YOLO person mask → {mask_flat.shape}")
    visibs_np = visibs_np & mask_flat

    # === NEW Motion Analysis Plot ===
    print("[forward_video] Starting motion analysis...")
    motion_results = compute_motion_metrics(trajs_np, visibs_np, framerate=framerate)
    out_png = os.path.join("results_yolo", f"{basename}_motion_analysis.png")
    plot_motion_analysis(motion_results, save_path=out_png)
    print(f"[forward_video] Saved full motion analysis plot → {out_png}")


    # === NEW: GENERATE TWITCHING VISUALIZATION VIDEO ===
    print("[forward_video] Generating twitching visualization...")
    
    # 1. Get the IDs of all points classified as 'Twitching'
    twitching_ids = [r['id'] for r in motion_results if r['label'] == 'Twitching']
    
    if len(twitching_ids) > 0:
        print(f"Found {len(twitching_ids)} twitching points to visualize.")
        
        # 2. Filter the original trajectory and visibility tensors to keep only twitching points
        # The 'N' dimension in these tensors corresponds to the point ID
        trajs_twitch = trajs_e[:, :, twitching_ids, :]         # Shape [B, T, N_twitch, 2]
        visconfs_twitch = visconfs_e[:, :, twitching_ids, :]   # Shape [B, T, N_twitch, 2]
        
        # Extract the relevant visibility component (visconfs has fwd/bwd vis)
        visibs_twitch = visconfs_twitch[0, :, :, 1] > args.conf_thr # Shape [T, N_twitch]

        # 3. Call our new drawing function with the filtered data
        # We pass rgbs[0] just to get the shape for the black background
        twitch_frames = draw_twitching_visualization(
            rgbs[0], 
            trajs_twitch[0], 
            visibs_twitch, 
            rate=rate
        )

        # 4. Save the frames as an MP4 video
        vid_out_f = os.path.join("results_yolo", f"{basename}_twitching_vis.mp4")
        temp_dir = f"temp_twitch_vis_{basename}"
        utils.basic.mkdir(temp_dir)

        print(f"Writing twitching visualization to {vid_out_f}...")
        for ti, frame in enumerate(twitch_frames):
            temp_out_f = os.path.join(temp_dir, f"{ti:04d}.jpg")
            PIL.Image.fromarray(frame).save(temp_out_f)
        
        os.system(f'/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate {framerate} -pattern_type glob -i "./{temp_dir}/*.jpg" -c:v libx264 -crf 20 -pix_fmt yuv420p "{vid_out_f}"')
        print("Done.")

    else:
        print("No twitching detected, skipping visualization video.")

    return None


    return None

def run(model, args):
    if args.ckpt_init:
        _ = utils.saveload.load(
            None,
            args.ckpt_init,
            model,
            optimizer=None,
            scheduler=None,
            ignore_load=None,
            strict=True,
            verbose=False,
            weights_only=False,
        )
        print('loaded weights from', args.ckpt_init)
    else:
        url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=True)
        print('loaded weights from', url)

    model.cuda()
    for n, p in model.named_parameters():
        p.requires_grad = False
    model.eval()

    os.makedirs("results_yolo", exist_ok=True)

    for fname in sorted(os.listdir(args.videos_folder)):
        if not fname.lower().endswith(".mp4"):
            continue
        path = os.path.join(args.videos_folder, fname)
        basename = os.path.splitext(fname)[0]
        print(f"\n=== Processing {fname} ===")

        rgbs, framerate = read_mp4(path)
        H,W = rgbs[0].shape[:2]

        if args.max_frames:
            rgbs = rgbs[:args.max_frames]
        HH = 512
        scale = min(HH/H, HH/W)
        H, W = int(H*scale), int(W*scale)
        H, W = H//8 * 8, W//8 * 8
        rgbs = [cv2.resize(rgb, dsize=(W, H), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]

        rgbs = [torch.from_numpy(rgb).permute(2,0,1) for rgb in rgbs]
        rgbs = torch.stack(rgbs, dim=0).unsqueeze(0).float()

        with torch.no_grad():
            forward_video(rgbs, framerate, model, args, basename)

    return None

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_init", type=str, default='')
    parser.add_argument("--videos_folder", type=str, default='./demo_video')
    parser.add_argument("--query_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=1000)
    parser.add_argument("--inference_iters", type=int, default=4)
    parser.add_argument("--window_len", type=int, default=16)
    parser.add_argument("--rate", type=int, default=16)
    parser.add_argument("--conf_thr", type=float, default=0.1)
    parser.add_argument("--bkg_opacity", type=float, default=0.7)
    parser.add_argument("--vstack", action='store_true', default=False)
    parser.add_argument("--hstack", action='store_true', default=False)
    args = parser.parse_args()

    from nets.alltracker import Net; model = Net(args.window_len)
    count_parameters(model)
    global yolo_model
    yolo_model = YOLO("yolov8n-seg.pt")

    run(model, args)
