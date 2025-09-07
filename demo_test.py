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

def draw_pts_gpu(rgbs, trajs, visibs, colormap, rate=1, bkg_opacity=0.0):
    device = rgbs.device
    T, C, H, W = rgbs.shape
    trajs = trajs.permute(1,0,2) # N,T,2
    visibs = visibs.permute(1,0) # N,T
    N = trajs.shape[0]
    colors = torch.tensor(colormap, dtype=torch.float32, device=device)  # [N,3]

    rgbs = rgbs * bkg_opacity # darken, to see the point tracks better
    
    opacity = 1.0
    if rate==1:
        radius = 1
        opacity = 0.9
    elif rate==2:
        radius = 1
    elif rate== 4:
        radius = 2
    elif rate== 8:
        radius = 4
    else:
        radius = 6
    sharpness = 0.15 + 0.05 * np.log2(rate)
    
    D = radius * 2 + 1
    y = torch.arange(D, device=device).float()[:, None] - radius
    x = torch.arange(D, device=device).float()[None, :] - radius
    dist2 = x**2 + y**2
    icon = torch.clamp(1 - (dist2 - (radius**2) / 2.0) / (radius * 2 * sharpness), 0, 1)  # [D,D]
    icon = icon.view(1, D, D)
    dx = torch.arange(-radius, radius + 1, device=device)
    dy = torch.arange(-radius, radius + 1, device=device)
    disp_y, disp_x = torch.meshgrid(dy, dx, indexing="ij")  # [D,D]
    
    for t in range(T):
        mask = visibs[:, t]  # [N]
        if mask.sum() == 0:
            continue
        xy = trajs[mask, t] + 0.5  # [N,2]
        xy[:, 0] = xy[:, 0].clamp(0, W - 1)
        xy[:, 1] = xy[:, 1].clamp(0, H - 1)
        colors_now = colors[mask]  # [N,3]
        N = xy.shape[0]
        cx = xy[:, 0].long()  # [N]
        cy = xy[:, 1].long()
        x_grid = cx[:, None, None] + disp_x  # [N,D,D]
        y_grid = cy[:, None, None] + disp_y  # [N,D,D]
        valid = (x_grid >= 0) & (x_grid < W) & (y_grid >= 0) & (y_grid < H)
        x_valid = x_grid[valid]  # [K]
        y_valid = y_grid[valid]
        icon_weights = icon.expand(N, D, D)[valid]  # [K]
        colors_valid = colors_now[:, :, None, None].expand(N, 3, D, D).permute(1, 0, 2, 3)[
            :, valid
        ]  # [3, K]
        idx_flat = (y_valid * W + x_valid).long()  # [K]

        accum = torch.zeros_like(rgbs[t])  # [3, H, W]
        weight = torch.zeros(1, H * W, device=device)  # [1, H*W]
        img_flat = accum.view(C, -1)  # [3, H*W]
        weighted_colors = colors_valid * icon_weights  # [3, K]
        img_flat.scatter_add_(1, idx_flat.unsqueeze(0).expand(C, -1), weighted_colors)
        weight.scatter_add_(1, idx_flat.unsqueeze(0), icon_weights.unsqueeze(0))
        weight = weight.view(1, H, W)

        alpha = weight.clamp(0, 1) * opacity
        accum = accum / (weight + 1e-6)  # [3, H, W]
        rgbs[t] = rgbs[t] * (1 - alpha) + accum * alpha

    # Convert back to T,H,W,3 format before return
    rgbs = rgbs.clamp(0, 255).byte().cpu().numpy()  # [T, 3, H, W]
    rgbs = np.transpose(rgbs, (0, 2, 3, 1))  # [T,H,W,3]

    # Optional: boost saturation if background was black
    if bkg_opacity == 0.0:
        for t in range(T):
            hsv_frame = cv2.cvtColor(rgbs[t], cv2.COLOR_RGB2HSV)
            saturation_factor = 1.5
            hsv_frame[..., 1] = np.clip(hsv_frame[..., 1] * saturation_factor, 0, 255)
            rgbs[t] = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)

    print("[draw_pts_gpu] Final rgbs shape:", rgbs.shape)
    print("[draw_pts_gpu] Sample pixel value (frame 0):", rgbs[0, 0, 0])
    print("[draw_pts_gpu] Visibility stats per frame:")
    for t in range(T):
        print(f"  Frame {t}: {visibs[:, t].sum().item()} points visible")
    return rgbs


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
    #human_masks = get_person_masks([f.permute(1,2,0).cpu().numpy().astype(np.uint8) 
                                    #for f in rgbs[0]])  # [T,H,W]
    #mask_small = human_masks[:, ::rate, ::rate]
    #mask_flat = mask_small.reshape(T, -1).T.astype(bool)  # [N,T]
    #print(f"[forward_video] Applied YOLO person mask → {mask_flat.shape}")
    #visibs_np = visibs_np & mask_flat

    # === Repetition Ratio Plot ===
    from path_by_disp import compute_repetition_ratio, plot_repetition_ratios
    ratios = compute_repetition_ratio(trajs_np, visibs_np)
    out_png = os.path.join("results_yolo", f"{basename}_repetition_ratio.png")
    plot_repetition_ratios(ratios, save_path=out_png)
    print(f"[forward_video] Saved repetition ratio plot → {out_png}")

    # === Map ratios to colors (R=red, Y=yellow, G=green) ===
    colors = []
    mask_keep = []
    for r in ratios:
        if not np.isnan(r) and r >= 0.6:   # high repetition threshold
            colors.append([255, 0, 0])     # red
            mask_keep.append(True)
        else:
            colors.append([0, 0, 0])       # invisible / black
            mask_keep.append(False)
    colors = np.array(colors, dtype=np.float32) / 255.0
    mask_keep = np.array(mask_keep)
    trajs_e = trajs_e[:, :, mask_keep, :]          # [B, T, N_red, 2]
    visconfs_e = visconfs_e[:, :, mask_keep, :]    # [B, T, N_red, 2]
    colors = colors[mask_keep]                     # [N_red, 3]
    # === Draw points & trajectories on video frames ===
    frames = draw_pts_gpu(rgbs[0].to('cuda:0'), trajs_e[0], visconfs_e[0,:,:,1] > args.conf_thr,
                          colors, rate=rate, bkg_opacity=args.bkg_opacity)
    print('[forward_video] frames.shape=', frames.shape)

    # Optional stacking of input and overlay
    if args.vstack:
        frames_top = rgbs[0].clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy() # T,H,W,3
        frames = np.concatenate([frames_top, frames], axis=1)
    elif args.hstack:
        frames_left = rgbs[0].clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy() # T,H,W,3
        frames = np.concatenate([frames_left, frames], axis=2)

    # Save frames to temporary folder
    temp_dir = f'temp_pt_vis_{basename}_rate{rate}_q{args.query_frame}'
    utils.basic.mkdir(temp_dir)
    T = frames.shape[0]
    for ti in range(T):
        temp_out_f = f'{temp_dir}/{ti:03d}.jpg'
        im = PIL.Image.fromarray(frames[ti])
        im.save(temp_out_f)

    # Convert to final mp4
    rgb_out_f = f'./pt_vis_{basename}_rate{rate}_q{args.query_frame}.mp4'
    os.system(f'/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate {framerate} -pattern_type glob -i "./{temp_dir}/*.jpg" -c:v libx264 -crf 20 -pix_fmt yuv420p {rgb_out_f}')
    print(f"[forward_video] Saved final overlay video → {rgb_out_f}")

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
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--inference_iters", type=int, default=4)
    parser.add_argument("--window_len", type=int, default=16)
    parser.add_argument("--rate", type=int, default=16)
    parser.add_argument("--conf_thr", type=float, default=0.1)
    parser.add_argument("--bkg_opacity", type=float, default=0.0)
    parser.add_argument("--vstack", action='store_true', default=False)
    parser.add_argument("--hstack", action='store_true', default=False)
    args = parser.parse_args()

    from nets.alltracker import Net; model = Net(args.window_len)
    count_parameters(model)

    run(model, args)
