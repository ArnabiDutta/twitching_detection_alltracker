import cv2
import os
import numpy as np
import subprocess
from tqdm import tqdm
from ultralytics import YOLO

def segment_people_yolo(input_video, output_dir='masked_frames', output_video='outputs/segmented_video.mp4', model_size='n', conf=0.3):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    # Automatically download YOLOv8 segmentation weights
    model_path = f'yolov8{model_size}-seg.pt'
    print(f"[INFO] Loading YOLOv8-segmentation model: {model_path}")
    model = YOLO(model_path)

    print(f"[INFO] Opening video: {input_video}")
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    print(f"[INFO] Segmenting people in video (total {total_frames} frames)...")
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=conf)
            masks = results[0].masks
            cls = results[0].boxes.cls if results[0].boxes is not None else None

            # Compose person mask
            if masks is not None and cls is not None and (cls == 0).any():
                person_mask = (cls == 0)
                combined_mask = masks.data[person_mask].sum(dim=0).clamp(0, 1).cpu().numpy()
                combined_mask = (combined_mask > 0.5).astype(np.uint8)
                combined_mask_resized = cv2.resize(combined_mask, (W, H), interpolation=cv2.INTER_NEAREST)

                fg = frame * combined_mask_resized[..., None]
            else:
                fg = np.zeros_like(frame)

            out_path = os.path.join(output_dir, f"{frame_idx:04d}.jpg")
            cv2.imwrite(out_path, fg)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"[INFO] Saved {frame_idx} masked frames to '{output_dir}'")

    # Create segmented video using ffmpeg
    print(f"[INFO] Creating video using ffmpeg...")
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', os.path.join(output_dir, '%04d.jpg'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        output_video
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"[DONE] Segmented video saved to: {output_video}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, default='demo_video/monkey.mp4', help='Path to input .mp4')
    parser.add_argument('--output_dir', type=str, default='masked_frames', help='Directory to store masked frames')
    parser.add_argument('--output_video', type=str, default='outputs/segmented_video.mp4', help='Path to save output video')
    parser.add_argument('--model_size', type=str, default='n', help='YOLOv8 model size: n, s, m, l, x')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    args = parser.parse_args()

    segment_people_yolo(
        input_video=args.input_video,
        output_dir=args.output_dir,
        output_video=args.output_video,
        model_size=args.model_size,
        conf=args.conf
    )
