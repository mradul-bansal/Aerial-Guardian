import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def bbox_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0.0:
        return 0.0
    return inter_area / union_area

def iou_distance(tracks, detections):
    """
    Computes IoU distance matrix (1 - IoU).
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, track in enumerate(tracks):
        track_box = track.get_bbox()
        for j, det in enumerate(detections):
            cost_matrix[i, j] = 1.0 - bbox_iou(track_box, det)
    return cost_matrix

def embedding_distance(tracks, embeddings):
    """
    Computes cosine distance matrix (1 - cosine_similarity) using a multi-view gallery.
    Compares the detection embedding against all historical embeddings in the track's
    gallery, taking the minimum distance.
    """
    cost_matrix = np.zeros((len(tracks), len(embeddings)), dtype=np.float32)
    for i, track in enumerate(tracks):
        if not track.features:
            cost_matrix[i, :] = 1.0
            continue
        for j, emb in enumerate(embeddings):
            # Compute distance to all gallery features and take minimum
            min_dist = 1.0
            for feat in track.features:
                dot_prod = np.dot(feat, emb)
                dist = max(0.0, min(1.0, 1.0 - dot_prod))
                if dist < min_dist:
                    min_dist = dist
            cost_matrix[i, j] = min_dist
    return cost_matrix


class Track:
    count = 0
    def __init__(self, bbox, score, class_id, embedding=None):
        self.track_id = Track.count
        Track.count += 1
        
        # Convert bbox [x1, y1, x2, y2] to center state [cx, cy, w, h]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2
        cy = y1 + h/2
        
        # Initialize Kalman Filter (8 states: cx, cy, w, h, dx, dy, dw, dh)
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        # State transition matrix F (constant velocity model)
        self.kf.F = np.eye(8, dtype=np.float32)
        dt = 1.0
        self.kf.F[0, 4] = dt
        self.kf.F[1, 5] = dt
        self.kf.F[2, 6] = dt
        self.kf.F[3, 7] = dt
        
        # Measurement matrix H
        self.kf.H = np.zeros((4, 8), dtype=np.float32)
        self.kf.H[0, 0] = 1.0
        self.kf.H[1, 1] = 1.0
        self.kf.H[2, 2] = 1.0
        self.kf.H[3, 3] = 1.0
        
        # Measurement noise R
        std_p = 2.0
        std_s = 2.0
        self.kf.R = np.diag([std_p**2, std_p**2, std_s**2, std_s**2]).astype(np.float32)
        
        # State covariance matrix P
        self.kf.P = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0]).astype(np.float32)
        
        # Process noise matrix Q
        self.kf.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1]).astype(np.float32)
        
        self.state = 0  # 0: Tentative, 1: Confirmed, 2: Lost
        self.age = 0
        self.time_since_update = 0
        self.history = [bbox]
        self.class_id = class_id
        self.score = score
        
        # Appearance ReID
        self.embedding = embedding
        self.features = []
        if embedding is not None:
            self.features.append(embedding)

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, score, embedding=None):
        """
        Updates tracker state with a new detection.
        bbox: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2
        cy = y1 + h/2
        
        z = np.array([cx, cy, w, h], dtype=np.float32)
        self.kf.update(z)
        self.time_since_update = 0
        self.score = score
        
        # Confirm track if it updates
        self.state = 1
        
        # Update embedding history
        if embedding is not None:
            self.features.append(embedding)
            if len(self.features) > 10:
                self.features.pop(0)
            
            # EMA for appearance representation
            alpha = 0.90
            if self.embedding is None:
                self.embedding = embedding
            else:
                self.embedding = alpha * self.embedding + (1.0 - alpha) * embedding
                self.embedding /= (np.linalg.norm(self.embedding) + 1e-6)

        self.history.append(self.get_bbox())
        if len(self.history) > 60:
            self.history.pop(0)

    def gmc_compensate(self, H):
        """
        Compensate Kalman Filter state mean and covariance for camera movement.
        H: 3x3 homography matrix
        """
        # Extract 2x3 affine matrix
        M = H[:2, :3]
        A = M[:, :2] # rotation and scaling
        t = M[:, 2]  # translation
        
        # Compute zoom scale along x and y
        s_x = np.linalg.norm(A[0])
        s_y = np.linalg.norm(A[1])
        
        # Compensate state vector: x = [cx, cy, w, h, dx, dy, dw, dh]
        self.kf.x[:2] = A @ self.kf.x[:2] + t
        self.kf.x[2] = s_x * self.kf.x[2]
        self.kf.x[3] = s_y * self.kf.x[3]
        self.kf.x[4:6] = A @ self.kf.x[4:6]
        self.kf.x[6] = s_x * self.kf.x[6]
        self.kf.x[7] = s_y * self.kf.x[7]
        
        # Compensate covariance P: P = R @ P @ R^T
        R = np.zeros((8, 8), dtype=np.float32)
        R[0:2, 0:2] = A
        R[2, 2] = s_x
        R[3, 3] = s_y
        R[4:6, 4:6] = A
        R[6, 6] = s_x
        R[7, 7] = s_y
        
        self.kf.P = R @ self.kf.P @ R.T

    def get_bbox(self):
        cx, cy, w, h = self.kf.x[:4]
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return np.array([x1, y1, x2, y2], dtype=np.float32)


class DroneTracker:
    def __init__(self, max_time_lost=30, high_conf_threshold=0.5, low_conf_threshold=0.1):
        self.max_time_lost = max_time_lost
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
        
        self.tracked_tracks = []   # Active/Confirmed tracks
        self.lost_tracks = []      # Lost tracks
        self.tentative_tracks = [] # New unconfirmed tracks
        
        self.frame_id = 0

    def update(self, bboxes, scores, class_ids, embeddings, H=None):
        """
        bboxes: np.ndarray of shape (N, 4) in [x1, y1, x2, y2]
        scores: np.ndarray of shape (N,)
        class_ids: np.ndarray of shape (N,)
        embeddings: np.ndarray of shape (N, 576)
        H: 3x3 homography matrix representing camera motion
        """
        self.frame_id += 1
        
        # 1. Apply Global Motion Compensation (GMC) to all tracks
        if H is not None:
            for track in self.tracked_tracks:
                track.gmc_compensate(H)
            for track in self.lost_tracks:
                track.gmc_compensate(H)
            for track in self.tentative_tracks:
                track.gmc_compensate(H)

        # 2. Predict next state using Kalman Filter
        for track in self.tracked_tracks:
            track.predict()
        for track in self.lost_tracks:
            track.predict()
        for track in self.tentative_tracks:
            track.predict()

        # 3. Categorize Detections (ByteTrack logic)
        high_indices = scores >= self.high_conf_threshold
        low_indices = (scores >= self.low_conf_threshold) & (scores < self.high_conf_threshold)
        
        dets_high = bboxes[high_indices]
        scores_high = scores[high_indices]
        classes_high = class_ids[high_indices]
        embs_high = embeddings[high_indices] if embeddings is not None else [None] * len(dets_high)
        
        dets_low = bboxes[low_indices]
        scores_low = scores[low_indices]
        classes_low = class_ids[low_indices]

        # 4. Association Stage 1: Active Tracks + High-confidence Detections (IoU + ReID)
        # Combine tracked and lost tracks for matching
        pool_tracks = self.tracked_tracks + self.lost_tracks
        
        unmatched_tracks = pool_tracks.copy()
        unmatched_dets_high = list(range(len(dets_high)))
        
        matched_stage1 = []
        
        if len(pool_tracks) > 0 and len(dets_high) > 0:
            # Compute costs
            d_iou = iou_distance(pool_tracks, dets_high)
            
            if embeddings is not None:
                d_emb = embedding_distance(pool_tracks, embs_high)
                # Combine: 0.5 * IoU_dist + 0.5 * ReID_dist
                cost_matrix = 0.5 * d_iou + 0.5 * d_emb
                
                # Gate matches: if IoU dist is very large or ReID dist is very large, prevent matching
                for i in range(len(pool_tracks)):
                    for j in range(len(dets_high)):
                        if d_iou[i, j] > 0.8 or d_emb[i, j] > 0.4:
                            cost_matrix[i, j] = 1e6
            else:
                cost_matrix = d_iou
                for i in range(len(pool_tracks)):
                    for j in range(len(dets_high)):
                        if d_iou[i, j] > 0.8:
                            cost_matrix[i, j] = 1e6
            
            row_inds, col_inds = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_inds, col_inds):
                if cost_matrix[r, c] < 1e5:
                    track = pool_tracks[r]
                    track.update(dets_high[c], scores_high[c], embs_high[c])
                    
                    matched_stage1.append(track)
                    unmatched_tracks.remove(track)
                    unmatched_dets_high.remove(c)

        # Update lists: what was matched is confirmed active
        self.tracked_tracks = [t for t in matched_stage1 if t.state == 1]
        self.lost_tracks = [t for t in unmatched_tracks if t.state == 2]

        # 5. Association Stage 2: Active Tracks + Low-confidence Detections (IoU Only)
        # Match remaining active tracks (from Stage 1 unmatched) with low confidence detections
        active_unmatched = [t for t in unmatched_tracks if t in self.tracked_tracks or t.state == 1]
        matched_stage2 = []
        
        if len(active_unmatched) > 0 and len(dets_low) > 0:
            d_iou = iou_distance(active_unmatched, dets_low)
            cost_matrix = d_iou.copy()
            for i in range(len(active_unmatched)):
                for j in range(len(dets_low)):
                    if d_iou[i, j] > 0.7:  # Stricter IoU gate for low-conf detections
                        cost_matrix[i, j] = 1e6
                        
            row_inds, col_inds = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_inds, col_inds):
                if cost_matrix[r, c] < 1e5:
                    track = active_unmatched[r]
                    # Note: no embedding update for low confidence crop (likely poor quality)
                    track.update(dets_low[c], scores_low[c], None)
                    matched_stage2.append(track)
                    active_unmatched.remove(track)
                    
        # Update lists again
        for t in matched_stage2:
            if t not in self.tracked_tracks:
                self.tracked_tracks.append(t)
            if t in self.lost_tracks:
                self.lost_tracks.remove(t)

        # 6. Association Stage 3: Tentative Tracks + Remaining High-confidence Detections
        # Tentative tracks (new tracks from previous frame) matched with remaining high-conf dets
        matched_tentative = []
        unmatched_dets_remaining = unmatched_dets_high.copy()
        
        if len(self.tentative_tracks) > 0 and len(unmatched_dets_remaining) > 0:
            dets_rem = dets_high[unmatched_dets_remaining]
            scores_rem = scores_high[unmatched_dets_remaining]
            embs_rem = [embs_high[c] for c in unmatched_dets_remaining]
            
            d_iou = iou_distance(self.tentative_tracks, dets_rem)
            cost_matrix = d_iou.copy()
            for i in range(len(self.tentative_tracks)):
                for j in range(len(dets_rem)):
                    if d_iou[i, j] > 0.7:
                        cost_matrix[i, j] = 1e6
                        
            row_inds, col_inds = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_inds, col_inds):
                if cost_matrix[r, c] < 1e5:
                    track = self.tentative_tracks[r]
                    det_idx = unmatched_dets_remaining[c]
                    track.update(dets_rem[c], scores_rem[c], embs_rem[c])
                    track.state = 1 # Promoted to active
                    self.tracked_tracks.append(track)
                    matched_tentative.append(track)
                    unmatched_dets_high.remove(det_idx)
                    
        # Remove matched tentative tracks
        self.tentative_tracks = [t for t in self.tentative_tracks if t not in matched_tentative]
        # Any remaining unmatched tentative tracks are discarded
        self.tentative_tracks = []

        # 7. Post-Association Management
        # Mark remaining active unmatched tracks as Lost
        for t in active_unmatched:
            if t not in matched_stage2:
                t.state = 2 # Lost
                self.lost_tracks.append(t)
                if t in self.tracked_tracks:
                    self.tracked_tracks.remove(t)

        # Manage lost tracks (remove if lost for too long)
        self.lost_tracks = [t for t in self.lost_tracks if t.time_since_update <= self.max_time_lost]

        # Init new tentative tracks from remaining high-confidence detections
        for idx in unmatched_dets_high:
            track = Track(dets_high[idx], scores_high[idx], classes_high[idx], embs_high[idx])
            self.tentative_tracks.append(track)

        # Reset Track Counter on sequence change (just to keep IDs clean, optional)
        # In a generic multi-sequence tracker, we keep them unique or reset them.
        
        # Compile active tracks to return
        active_results = []
        for track in self.tracked_tracks:
            bbox = track.get_bbox()
            active_results.append({
                'id': track.track_id,
                'bbox': bbox,
                'class_id': track.class_id,
                'score': track.score,
                'history': track.history
            })
            
        return active_results
