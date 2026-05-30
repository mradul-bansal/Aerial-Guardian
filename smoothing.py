import numpy as np

def smooth_bbox(history, alpha=0.3):
    """
    Applies Exponential Moving Average (EMA) to the bounding box history.
    history: list of np.ndarray bboxes [x1, y1, x2, y2]
    alpha: smoothing factor (lower means more smoothing, higher means more responsive)
    Returns:
        smoothed_bbox: np.ndarray of shape (4,) [x1, y1, x2, y2]
    """
    if not history:
        return None
    if len(history) == 1:
        return history[0]
        
    smoothed = history[0].copy()
    for bbox in history[1:]:
        smoothed = alpha * bbox + (1 - alpha) * smoothed
        
    return smoothed

def smooth_trajectory(points, window_size=5):
    """
    Applies a simple moving average filter to a list of 2D points (x, y)
    to smooth out the trajectory tails.
    points: list of tuple or list [x, y]
    window_size: odd integer representing size of sliding window
    Returns:
        smoothed_points: list of [x, y]
    """
    if len(points) < 3:
        return points

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    smoothed_x = []
    smoothed_y = []
    
    n = len(points)
    for i in range(n):
        # Determine dynamic window size near boundaries
        w = min(window_size, i + 1, n - i)
        # Force window to be odd
        if w % 2 == 0:
            w = max(1, w - 1)
            
        half_w = w // 2
        start_idx = i - half_w
        end_idx = i + half_w + 1
        
        smoothed_x.append(np.mean(x_coords[start_idx:end_idx]))
        smoothed_y.append(np.mean(y_coords[start_idx:end_idx]))
        
    return [[sx, sy] for sx, sy in zip(smoothed_x, smoothed_y)]
