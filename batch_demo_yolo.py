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
def get_person_masks(frames):
    """
    Runs YOLO segmentation (global model) on frames.
    Returns: np.array of shape [T, H, W] with binary person masks.
    """
    masks = []
    for frame in frames:
        results = yolo_model(frame, verbose=False)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        if results[0].masks is not None:
            for cls_id, seg in zip(results[0].boxes.cls.cpu().numpy(),
                                   results[0].masks.xy):
                if int(cls_id) == 0:  # class 0 = person
                    poly = np.array([seg], dtype=np.int32)
                    cv2.fillPoly(mask, poly, 1)

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

    # === Repetition Ratio Plot ===
    from rep_ratio import compute_repetition_ratio, plot_repetition_ratios
    ratios = compute_repetition_ratio(trajs_np, visibs_np)
    out_png = os.path.join("results_yolo", f"{basename}_repetition_ratio.png")
    plot_repetition_ratios(ratios, save_path=out_png)
    print(f"[forward_video] Saved repetition ratio plot → {out_png}")

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
    global yolo_model
    yolo_model = YOLO("yolov8n-seg.pt")

    run(model, args)
