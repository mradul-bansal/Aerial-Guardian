import os
import glob
import time
import numpy as np
import pandas as pd
import cv2
import torch
import motmetrics as mm

from detector import TiledDetector
from gmc import GlobalMotionCompensation
from reid import ReIDExtractor
from tracker import DroneTracker
from analyze_dataset import find_dataset_paths

def load_gt(ann_path):
    """
    Loads VisDrone MOT ground truth annotations.
    Format: <frame_idx>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    We filter for Pedestrian (1) and People (2).
    Returns:
        gt_dict: dict mapping frame_idx -> list of {'id': target_id, 'bbox': [x, y, w, h]}
    """
    gt_dict = {}
    if not os.path.exists(ann_path):
        return gt_dict
        
    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            try:
                frame_idx = int(parts[0])
                target_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                score = int(parts[6])
                category = int(parts[7])
                
                # Keep only Pedestrian (1) and People (2), and non-ignored targets (score == 1)
                if category in [1, 2] and score == 1:
                    if frame_idx not in gt_dict:
                        gt_dict[frame_idx] = []
                    gt_dict[frame_idx].append({
                        'id': target_id,
                        'bbox': [x, y, w, h] # [x_left, y_top, width, height]
                    })
            except ValueError:
                continue
    return gt_dict

def evaluate_sequence(seq_dir, ann_file, detector, use_sahi=True, gmc_enabled=True, reid_extractor=None, 
                      conf_threshold=0.3, limit_frames=None):
    """
    Evaluates tracking on a single sequence and returns a motmetrics accumulator.
    """
    seq_name = os.path.basename(seq_dir)
    print(f"\nEvaluating sequence: {seq_name}")
    
    # Load GT
    gt_data = load_gt(ann_file)
    if not gt_data:
        print(f"No ground truth found for sequence {seq_name}")
        return None, 0.0
        
    # Get image frames
    image_files = sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(seq_dir, 'left', '*.png')))
        
    if not image_files:
        print(f"No frames found for sequence {seq_name}")
        return None, 0.0
        
    if limit_frames is not None:
        image_files = image_files[:limit_frames]
        
    total_frames = len(image_files)
    
    # Initialize GMC and Tracker
    gmc = GlobalMotionCompensation() if gmc_enabled else None
    tracker = DroneTracker(high_conf_threshold=conf_threshold)
    
    # Create accumulator
    acc = mm.MOTAccumulator(auto_id=True)
    
    frame_times = []
    
    for frame_idx, img_path in enumerate(image_files, start=1):
        t0 = time.time()
        
        # Read frame
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        h, w = frame.shape[:2]
        
        # 1. GMC
        H = None
        if gmc is not None:
            active_bboxes = [track.get_bbox() for track in tracker.tracked_tracks] if hasattr(tracker, 'tracked_tracks') else None
            H = gmc.apply(frame, active_bboxes)
            
        # 2. Detect
        if use_sahi:
            boxes, scores, class_ids = detector.detect(
                frame, conf_threshold=conf_threshold, iou_threshold=0.4, classes=[0]
            )
        else:
            # Standard full-frame inference
            results = detector.model(frame, imgsz=max(w, h), conf=conf_threshold, verbose=False, device=detector.device)[0]
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
        
        # Map COCO class 0 (person) to class 1
        if len(class_ids) > 0:
            class_ids = np.where(class_ids == 0, 1, class_ids)
            
        # 3. ReID
        embeddings = None
        if reid_extractor is not None and len(boxes) > 0:
            embeddings = reid_extractor.extract(frame, boxes)
            
        # 4. Tracker Update
        tracks = tracker.update(boxes, scores, class_ids, embeddings, H)
        
        t1 = time.time()
        frame_times.append(t1 - t0)
        
        # --- Evaluate Frame ---
        # Get ground truth targets for this frame
        gt_frame = gt_data.get(frame_idx, [])
        gt_ids = [obj['id'] for obj in gt_frame]
        gt_boxes = [obj['bbox'] for obj in gt_frame]
        
        # Get tracker targets
        pred_ids = [t['id'] for t in tracks]
        # Tracker returns bounding box in [x1, y1, x2, y2], but motmetrics expects [x_left, y_top, width, height]
        pred_boxes = []
        for t in tracks:
            bx1, by1, bx2, by2 = t['bbox']
            pred_boxes.append([bx1, by1, bx2 - bx1, by2 - by1])
            
        # Compute distances (1 - IoU). Distance gate is 0.5 (meaning IoU must be >= 0.5)
        # In motmetrics, 'iou_matrix' returns distance where value > max_iou is set to NaN
        # Wait, the parameter 'max_iou' in mm.distances.iou_matrix is actually the max allowed DISTANCE!
        # Thus, max_iou=0.5 means max allowed distance is 0.5 (so min IoU is 0.5).
        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        
        # Update accumulator
        acc.update(gt_ids, pred_ids, distances)
        
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            print(f"Processed frame {frame_idx}/{total_frames} (Avg FPS: {1.0 / np.mean(frame_times):.1f})")

    avg_fps = 1.0 / np.mean(frame_times) if frame_times else 0.0
    return acc, avg_fps

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluation device: {device}")
    
    try:
        ann_dir, seq_dir = find_dataset_paths()
    except Exception as e:
        print(f"Error finding dataset: {e}")
        return

    # Initialize shared detector and ReID extractor
    print("Loading model and initializing detector...")
    detector = TiledDetector(model_path="yolo11s.pt", tile_size=640, overlap=160, device=device)
    reid_extractor = ReIDExtractor(device=device)

    # We evaluate on 3 representative sequences from validation set for benchmarking
    # VisDrone Val sequences: e.g. uav0000086_00000_v, uav0000117_00000_v, uav0000268_00000_v
    val_sequences = sorted(os.listdir(seq_dir))
    eval_sequences = [s for s in val_sequences if os.path.isdir(os.path.join(seq_dir, s))]
    
    # Let's take the first 3 sequences for a robust but reasonably fast evaluation
    eval_sequences = eval_sequences[:3]
    print(f"Evaluating on sequences: {eval_sequences}")

    accumulators = []
    names = []
    fps_list = []

    for seq in eval_sequences:
        seq_path = os.path.join(seq_dir, seq)
        ann_file = os.path.join(ann_dir, seq + '.txt')
        
        acc, avg_fps = evaluate_sequence(
            seq_path, ann_file, detector, 
            use_sahi=True, gmc_enabled=True, reid_extractor=reid_extractor, 
            conf_threshold=0.3, limit_frames=None
        )
        
        if acc is not None:
            accumulators.append(acc)
            names.append(seq)
            fps_list.append(avg_fps)

    if not accumulators:
        print("No sequences evaluated successfully.")
        return

    # Compute metrics
    print("\nComputing overall MOT metrics...")
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accumulators, 
        names=names, 
        metrics=['num_frames', 'mota', 'motp', 'idf1', 'num_switches', 'precision', 'recall'], 
        generate_overall=True
    )
    
    # Print results
    print("\n=== Tracking Evaluation Summary ===")
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    
    # Save results to a CSV file
    os.makedirs('results', exist_ok=True)
    summary.to_csv('results/evaluation_metrics.csv')
    print("Saved evaluation results to results/evaluation_metrics.csv")

if __name__ == '__main__':
    main()
