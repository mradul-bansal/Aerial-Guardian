import os
import cv2
import time
import glob
import numpy as np
import torch

from detector import TiledDetector
from gmc import GlobalMotionCompensation
from reid import ReIDExtractor
from tracker import DroneTracker
from smoothing import smooth_bbox, smooth_trajectory

def get_color(idx):
    """
    Generate a unique color based on ID.
    Uses a vibrant palette suited for dark/light themes.
    """
    # Palette of 10 modern colors (RGB)
    palette = [
        (255, 99, 132),   # Pinkish Red
        (54, 162, 235),   # Blue
        (255, 206, 86),   # Yellow
        (75, 192, 192),   # Teal
        (153, 102, 255),  # Purple
        (255, 159, 64),   # Orange
        (233, 30, 99),    # Deep Pink
        (0, 150, 136),    # Green-Teal
        (139, 195, 74),   # Light Green
        (63, 81, 181)     # Indigo
    ]
    return palette[idx % len(palette)]

def run_pipeline(sequence_path, output_path, model_path="yolo11s.pt", 
                 tile_size=640, overlap=160, use_sahi=True, use_gmc=True, use_reid=True,
                 conf_threshold=0.3, tracker_max_lost=30, limit_frames=None):
    """
    sequence_path: Path to sequence folder containing JPEG files OR path to video file
    output_path: Path to write the output MP4 video
    """
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running pipeline on device: {device}")
    
    # 1. Initialize Pipeline Components
    print("Initializing detector...")
    detector = TiledDetector(model_path=model_path, tile_size=tile_size, overlap=overlap, device=device)
    
    print("Initializing GMC...")
    gmc = GlobalMotionCompensation() if use_gmc else None
    
    print("Initializing ReID extractor...")
    reid = ReIDExtractor(device=device) if use_reid else None
    
    print("Initializing Tracker...")
    tracker = DroneTracker(max_time_lost=tracker_max_lost, high_conf_threshold=conf_threshold)

    # 2. Setup Input Source (frames sequence or video file)
    is_video = os.path.isfile(sequence_path) and sequence_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    if is_video:
        cap = cv2.VideoCapture(sequence_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Source: Video file, {w}x{h}, {total_frames} frames, {fps} FPS")
    else:
        # Check folders under sequence_path (sometimes VisDrone has a subfolder or sequence directly)
        image_files = sorted(glob.glob(os.path.join(sequence_path, '*.jpg')))
        if not image_files:
            image_files = sorted(glob.glob(os.path.join(sequence_path, 'left', '*.png'))) # Kitti/other style fallback
        if not image_files:
            raise FileNotFoundError(f"No image files (*.jpg) found in {sequence_path}")
            
        first_img = cv2.imread(image_files[0])
        h, w = first_img.shape[:2]
        fps = 20 # Standard drone FPS
        total_frames = len(image_files)
        print(f"Source: Image sequence folder, {w}x{h}, {total_frames} frames")

    if limit_frames is not None:
        total_frames = min(total_frames, limit_frames)
        print(f"Limiting execution to first {limit_frames} frames.")

    # 3. Setup Video Writer
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Metrics
    frame_times = []
    frame_idx = 0

    try:
        while True:
            # Read frame
            if is_video:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                if frame_idx >= total_frames:
                    break
                frame = cv2.imread(image_files[frame_idx])
                if frame is None:
                    print(f"Error reading image {image_files[frame_idx]}")
                    break
            
            frame_idx += 1
            t_start = time.time()
            
            # --- Pipeline execution ---
            # 1. Global Motion Compensation (GMC)
            H = None
            if use_gmc and gmc is not None:
                # To optimize, we can pass bounding boxes of active tracks from previous frame
                active_bboxes = [track.get_bbox() for track in tracker.tracked_tracks] if hasattr(tracker, 'tracked_tracks') else None
                # Apply GMC
                H = gmc.apply(frame, active_bboxes)
                
            # 2. Object Detection
            if use_sahi:
                # Custom Tiled Inference
                boxes, scores, class_ids = detector.detect(
                    frame, conf_threshold=conf_threshold, iou_threshold=0.4, classes=[0]
                )
            else:
                # Standard full-image inference
                results = detector.model(frame, imgsz=max(w, h), conf=conf_threshold, verbose=False, device=device)[0]
                boxes = []
                scores = []
                class_ids = []
                for box in results.boxes:
                    c_id = int(box.cls[0].item())
                    if c_id == 0:  # COCO Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        score = box.conf[0].item()
                        boxes.append([x1, y1, x2, y2])
                        scores.append(score)
                        class_ids.append(c_id)
                boxes = np.array(boxes) if boxes else np.empty((0, 4))
                scores = np.array(scores) if scores else np.empty((0,))
                class_ids = np.array(class_ids, dtype=int) if class_ids else np.empty((0,), dtype=int)

            # Map COCO person class (0) to VisDrone pedestrian class (1) for tracking consistency
            if len(class_ids) > 0:
                class_ids = np.where(class_ids == 0, 1, class_ids)

            # 3. Appearance Feature Extraction
            embeddings = None
            if use_reid and reid is not None and len(boxes) > 0:
                embeddings = reid.extract(frame, boxes)

            # 4. Update Tracker
            tracks = tracker.update(boxes, scores, class_ids, embeddings, H)
            
            # Record processing time
            t_end = time.time()
            frame_times.append(t_end - t_start)
            current_fps = 1.0 / (t_end - t_start)

            # --- Visualisation ---
            annotated_frame = frame.copy()
            
            for track in tracks:
                track_id = track['id']
                bbox = track['bbox']
                class_id = track['class_id']
                score = track['score']
                history = track['history']
                
                # Apply EMA smoothing on coordinates to eliminate box jitter
                smoothed_box = smooth_bbox(history, alpha=0.3)
                x1, y1, x2, y2 = map(int, smoothed_box)
                
                # Draw bounding box
                color = get_color(track_id)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw text label: ID + Score
                label = f"ID:{track_id} P:{score:.2f}"
                (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - txt_h - 4), (x1 + txt_w, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Draw Trajectory Tail (Last 30 frames)
                if len(history) > 1:
                    # Calculate center coordinates for history
                    centers = []
                    for h_box in history[-30:]: # Show last 30 frames
                        hx1, hy1, hx2, hy2 = h_box
                        centers.append([int(hx1 + (hx2-hx1)/2), int(hy1 + (hy2-hy1)/2)])
                    
                    # Smooth trajectory line
                    smoothed_centers = smooth_trajectory(centers, window_size=5)
                    
                    # Draw polyline representing path
                    for i in range(len(smoothed_centers) - 1):
                        p1 = tuple(map(int, smoothed_centers[i]))
                        p2 = tuple(map(int, smoothed_centers[i+1]))
                        # Fading line width or intensity could be added, but solid line is highly visible
                        cv2.line(annotated_frame, p1, p2, color, 2, cv2.LINE_AA)

            # Draw overlay dashboard (opaque backdrop + text)
            avg_fps = 1.0 / np.mean(frame_times) if frame_times else 0.0
            info_overlay = [
                f"Frame: {frame_idx}/{total_frames}",
                f"Active Targets: {len(tracks)}",
                f"FPS: {current_fps:.1f} (Avg: {avg_fps:.1f})",
                f"SAHI: {'ON' if use_sahi else 'OFF'} | GMC: {'ON' if use_gmc else 'OFF'} | ReID: {'ON' if use_reid else 'OFF'}",
                f"Device: {device.upper()}"
            ]
            
            # Backdrop box
            bh_offset = 20
            db_w, db_h = 320, len(info_overlay) * bh_offset + 15
            cv2.rectangle(annotated_frame, (10, 10), (10 + db_w, 10 + db_h), (0, 0, 0), -1)
            # Add semitransparent alpha blending
            cv2.addWeighted(annotated_frame, 0.7, frame, 0.3, 0, frame) # Mix overlays
            # Re-draw on top for crispness
            cv2.rectangle(annotated_frame, (10, 10), (10 + db_w, 10 + db_h), (30, 30, 30), -1)
            for i, text in enumerate(info_overlay):
                cv2.putText(annotated_frame, text, (20, 28 + i*bh_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Write frame to video
            writer.write(annotated_frame)
            
            if frame_idx % 20 == 0 or frame_idx == total_frames:
                print(f"Processed frame {frame_idx}/{total_frames} (Current FPS: {current_fps:.1f}, Avg FPS: {avg_fps:.1f})")

    finally:
        # Cleanup
        if is_video:
            cap.release()
        writer.release()
        print("Video writing completed.")

    overall_fps = 1.0 / np.mean(frame_times) if frame_times else 0.0
    print(f"Pipeline finished! Processing speed: {overall_fps:.2f} FPS")
    return overall_fps

if __name__ == '__main__':
    # For testing, can be run directly from terminal
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to sequence folder or video file')
    parser.add_argument('--output', type=str, required=True, help='Path to output MP4 video file')
    parser.add_argument('--model', type=str, default='yolo11s.pt', help='YOLO model path')
    parser.add_argument('--no-sahi', action='store_true', help='Disable SAHI tiled inference')
    parser.add_argument('--no-gmc', action='store_true', help='Disable GMC')
    parser.add_argument('--no-reid', action='store_true', help='Disable ReID appearance embeddings')
    parser.add_argument('--conf', type=float, default=0.3, help='Detection confidence threshold')
    parser.add_argument('--limit', type=int, default=None, help='Limit frame count for fast tests')
    
    args = parser.parse_args()
    
    run_pipeline(
        sequence_path=args.input,
        output_path=args.output,
        model_path=args.model,
        use_sahi=not args.no_sahi,
        use_gmc=not args.no_gmc,
        use_reid=not args.no_reid,
        conf_threshold=args.conf,
        limit_frames=args.limit
    )
