import os
import time
import pandas as pd
import numpy as np
import torch
import motmetrics as mm

from detector import TiledDetector
from reid import ReIDExtractor
from evaluate import evaluate_sequence, load_gt, find_dataset_paths

def run_ablation():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Ablation Study on device: {device}")
    
    try:
        ann_dir, seq_dir = find_dataset_paths()
    except Exception as e:
        print(f"Error finding dataset: {e}")
        return

    # Choose a representative sequence for evaluation
    # uav0000086_00000_v is a standard, challenging moving camera sequence
    seq_name = "uav0000086_00000_v"
    seq_path = os.path.join(seq_dir, seq_name)
    ann_file = os.path.join(ann_dir, seq_name + ".txt")
    
    if not os.path.exists(seq_path):
        # Fallback to the first available sequence in seq_dir
        seq_names = [s for s in sorted(os.listdir(seq_dir)) if os.path.isdir(os.path.join(seq_dir, s))]
        if not seq_names:
            print("No sequences found.")
            return
        seq_name = seq_names[0]
        seq_path = os.path.join(seq_dir, seq_name)
        ann_file = os.path.join(ann_dir, seq_name + ".txt")
        
    print(f"Selected sequence for ablation: {seq_name}")
    
    # We limit to 150 frames to make the ablation study fast (approx. 1-2 minutes total)
    limit_frames = 150
    
    # Pre-load models to avoid compilation/loading time in metrics
    print("Loading models...")
    detector = TiledDetector(model_path="yolo11s.pt", tile_size=640, overlap=160, device=device)
    reid_extractor = ReIDExtractor(device=device)

    # Define the 4 ablation configurations
    configs = [
        {
            "name": "1. Baseline (YOLOv11s Only)",
            "use_sahi": False,
            "use_gmc": False,
            "use_reid": False
        },
        {
            "name": "2. YOLOv11s + SAHI Tiles",
            "use_sahi": True,
            "use_gmc": False,
            "use_reid": False
        },
        {
            "name": "3. YOLOv11s + SAHI + GMC",
            "use_sahi": True,
            "use_gmc": True,
            "use_reid": False
        },
        {
            "name": "4. YOLOv11s + SAHI + GMC + ReID (Full)",
            "use_sahi": True,
            "use_gmc": True,
            "use_reid": True
        }
    ]

    results = []

    for cfg in configs:
        print(f"\n--- Running Configuration: {cfg['name']} ---")
        
        # Configure ReID extractor based on config
        curr_reid = reid_extractor if cfg['use_reid'] else None
        
        # Run tracking and gather accumulator
        acc, avg_fps = evaluate_sequence(
            seq_path, ann_file, detector,
            use_sahi=cfg['use_sahi'],
            gmc_enabled=cfg['use_gmc'],
            reid_extractor=curr_reid,
            conf_threshold=0.3,
            limit_frames=limit_frames
        )
        
        if acc is not None:
            # Compute metrics
            mh = mm.metrics.create()
            summary = mh.compute(
                acc,
                metrics=['mota', 'idf1', 'num_switches', 'precision', 'recall'],
                name=cfg['name']
            )
            
            # Extract scores
            mota = summary.loc[cfg['name'], 'mota'] * 100.0
            idf1 = summary.loc[cfg['name'], 'idf1'] * 100.0
            ids = int(summary.loc[cfg['name'], 'num_switches'])
            prec = summary.loc[cfg['name'], 'precision'] * 100.0
            rec = summary.loc[cfg['name'], 'recall'] * 100.0
            
            results.append({
                "Configuration": cfg['name'],
                "MOTA (%)": f"{mota:.2f}%",
                "IDF1 (%)": f"{idf1:.2f}%",
                "ID Switches": ids,
                "Precision (%)": f"{prec:.2f}%",
                "Recall (%)": f"{rec:.2f}%",
                "FPS": f"{avg_fps:.1f}"
            })
            
    # Format and display results
    df = pd.DataFrame(results)
    print("\n\n=================== ABLATION STUDY RESULTS ===================")
    print(df.to_markdown(index=False))
    print("==============================================================")
    
    # Save table to CSV
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/ablation_results.csv', index=False)
    print("Ablation study saved to results/ablation_results.csv")

if __name__ == '__main__':
    run_ablation()
