import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def find_dataset_paths(base_path='data'):
    # Search recursively for annotations directory
    ann_paths = glob.glob(os.path.join(base_path, '**/annotations'), recursive=True)
    seq_paths = glob.glob(os.path.join(base_path, '**/sequences'), recursive=True)
    
    if not ann_paths or not seq_paths:
        # Check if they are directly in data/
        if os.path.exists(os.path.join(base_path, 'annotations')) and os.path.exists(os.path.join(base_path, 'sequences')):
            return os.path.join(base_path, 'annotations'), os.path.join(base_path, 'sequences')
        raise FileNotFoundError(f"Could not find annotations or sequences folders in {base_path}")
        
    return ann_paths[0], seq_paths[0]

def main():
    try:
        ann_dir, seq_dir = find_dataset_paths()
        print(f"Found annotations in: {ann_dir}")
        print(f"Found sequences in: {seq_dir}")
    except Exception as e:
        print(f"Dataset not ready yet: {e}")
        return

    widths = []
    heights = []
    areas = []
    aspect_ratios = []
    categories = []

    ann_files = glob.glob(os.path.join(ann_dir, '*.txt'))
    print(f"Analyzing {len(ann_files)} annotation files...")

    for ann_file in ann_files:
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                
                # Format: <frame_idx>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                try:
                    category = int(parts[7])
                    # Keep only Pedestrian (1) and People (2)
                    if category in [1, 2]:
                        w = float(parts[4])
                        h = float(parts[5])
                        
                        widths.append(w)
                        heights.append(h)
                        areas.append(w * h)
                        aspect_ratios.append(w / (h + 1e-6))
                        categories.append(category)
                except ValueError:
                    continue

    if not widths:
        print("No person annotations found!")
        return

    # Convert to numpy arrays
    widths = np.array(widths)
    heights = np.array(heights)
    areas = np.array(areas)
    aspect_ratios = np.array(aspect_ratios)

    print("\n--- Person Size Distribution Statistics ---")
    print(f"Total person annotations: {len(widths)}")
    print(f"Width (px)  - Mean: {widths.mean():.2f}, Median: {np.median(widths):.2f}, Min: {widths.min():.2f}, Max: {widths.max():.2f}")
    print(f"Height (px) - Mean: {heights.mean():.2f}, Median: {np.median(heights):.2f}, Min: {heights.min():.2f}, Max: {heights.max():.2f}")
    print(f"Area (px^2) - Mean: {areas.mean():.2f}, Median: {np.median(areas):.2f}, Min: {areas.min():.2f}, Max: {areas.max():.2f}")
    print(f"Aspect Ratio (W/H) - Mean: {aspect_ratios.mean():.2f}, Median: {np.median(aspect_ratios):.2f}")

    # Create plots directory
    os.makedirs('plots', exist_ok=True)

    # Plot width & height distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(widths.mean(), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean ({widths.mean():.1f}px)')
    plt.title('Person Width Distribution')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    plt.axvline(heights.mean(), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean ({heights.mean():.1f}px)')
    plt.title('Person Height Distribution')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = 'plots/person_size_distribution.png'
    plt.savefig(plot_path)
    print(f"Saved size distribution plot to {plot_path}")

    # Area distribution plot
    plt.figure(figsize=(8, 5))
    plt.hist(areas, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.axvline(areas.mean(), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean ({areas.mean():.1f}px^2)')
    plt.title('Person Area (W * H) Distribution')
    plt.xlabel('Area (pixels^2)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_area_path = 'plots/person_area_distribution.png'
    plt.savefig(plot_area_path)
    print(f"Saved area distribution plot to {plot_area_path}")

if __name__ == '__main__':
    main()
