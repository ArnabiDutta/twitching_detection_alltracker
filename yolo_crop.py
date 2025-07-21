import cv2
import os
import numpy as np
import subprocess
from tqdm import tqdm
from ultralytics import YOLO
import torch


def segment_people_yolo_bbox(
    input_video,
    output_video,
    temp_dir='temp_frames',
    model_size='n',
    conf=0.3,
    crop_size=(256, 256)
):
    os.makedirs(temp_dir, exist_ok=True)

    # Select GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    # Ensure YOLO weights are downloaded
    model_path = f'yolov8{model_size}.pt'
    if not os.path.exists(model_path):
        print(f"[INFO] Downloading YOLOv8-{model_size} weights...")
        YOLO(model_path)  # This will auto-download if not found

    model = YOLO(model_path)
    model.to(device)

    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    with tqdm(total=total_frames, desc=f"[{os.path.basename(input_video)}]") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection on GPU
            results = model(frame, conf=conf, device=device)
            boxes = results[0].boxes
            cls = boxes.cls.cpu().numpy() if boxes is not None else []

            person_crop = None
            if len(cls) > 0 and (cls == 0).any():
                for i, c in enumerate(cls):
                    if c == 0:  # Person class
                        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        person_crop = frame[y1:y2, x1:x2]
                        break

            if person_crop is not None and person_crop.size > 0:
                crop_resized = cv2.resize(person_crop, crop_size, interpolation=cv2.INTER_LINEAR)
            else:
                crop_resized = np.zeros((crop_size[1], crop_size[0], 3), dtype=np.uint8)

            out_path = os.path.join(temp_dir, f"{frame_idx:04d}.jpg")
            cv2.imwrite(out_path, crop_resized)
            frame_idx += 1
            pbar.update(1)

    cap.release()

    # Create cropped video using ffmpeg
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', os.path.join(temp_dir, '%04d.jpg'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        output_video
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Clean up temporary frames
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))

    print(f"[DONE] Cropped video saved to: {output_video}")


def process_video_folder(input_folder, output_folder, model_size='n', conf=0.3, crop_size=(256, 256)):
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    print(f"[INFO] Found {len(video_files)} videos in {input_folder}")
    for video_name in video_files:
        input_video = os.path.join(input_folder, video_name)
        output_video = os.path.join(output_folder, os.path.splitext(video_name)[0] + '_cropped.mp4')

        segment_people_yolo_bbox(
            input_video=input_video,
            output_video=output_video,
            temp_dir='temp_frames',
            model_size=model_size,
            conf=conf,
            crop_size=crop_size
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='videos', help='Folder containing input videos')
    parser.add_argument('--output_folder', type=str, default='cropped_videos', help='Folder to save cropped videos')
    parser.add_argument('--model_size', type=str, default='n', help='YOLOv8 model size: n, s, m, l, x')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[256, 256], help='Crop size (width height)')
    args = parser.parse_args()

    process_video_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        model_size=args.model_size,
        conf=args.conf,
        crop_size=tuple(args.crop_size)
    )
