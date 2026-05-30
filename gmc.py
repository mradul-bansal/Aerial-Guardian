import cv2
import numpy as np

class GlobalMotionCompensation:
    def __init__(self, method='lk', max_features=1000):
        """
        method: 'lk' (Lucas-Kanade optical flow)
        max_features: maximum number of features to track
        """
        self.method = method
        self.max_features = max_features
        self.prev_frame = None
        self.prev_pts = None

    def apply(self, frame, active_bboxes=None):
        """
        Estimates camera motion between previous frame and current frame.
        If active_bboxes is provided, features inside these bounding boxes are ignored.
        active_bboxes: list/array of [x1, y1, x2, y2]
        Returns:
            H: 3x3 homography matrix (identity if first frame or motion estimation fails)
        """
        H = np.eye(3, dtype=np.float32)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initial frame setup
        if self.prev_frame is None:
            self.prev_frame = gray
            # Detect initial good features to track
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10
            )
            return H

        # If previous points are empty, re-detect and return identity
        if self.prev_pts is None or len(self.prev_pts) < 4:
            self.prev_frame = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10
            )
            return H

        # 1. Track points from previous frame to current frame using Lucas-Kanade Optical Flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, gray, self.prev_pts, None
        )
        
        if curr_pts is None or status is None:
            # Tracking failed completely
            self.prev_frame = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10
            )
            return H

        # Filter successful matches
        valid = (status == 1).reshape(-1)
        pts_prev = self.prev_pts[valid]
        pts_curr = curr_pts[valid]
        
        if len(pts_prev) < 4:
            self.prev_frame = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10
            )
            return H

        # 2. Filter out points inside moving target bounding boxes to focus on static background
        if active_bboxes is not None and len(active_bboxes) > 0:
            mask = np.ones(len(pts_prev), dtype=bool)
            for i in range(len(pts_prev)):
                px, py = pts_prev[i][0]
                for bbox in active_bboxes:
                    x1, y1, x2, y2 = bbox
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        mask[i] = False
                        break
            pts_prev = pts_prev[mask]
            pts_curr = pts_curr[mask]

        if len(pts_prev) < 4:
            # Not enough background points left
            self.prev_frame = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10
            )
            return H

        # 3. Estimate partial affine transformation (translation, rotation, scale)
        M, inliers = cv2.estimateAffinePartial2D(
            pts_prev, pts_curr, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        
        if M is not None:
            H[:2, :3] = M
        else:
            # Fallback to homography if affine fails
            H_hom, inliers = cv2.findHomography(pts_prev, pts_curr, cv2.RANSAC, 3.0)
            if H_hom is not None:
                H = H_hom

        # 4. Prepare for next frame: re-detect features to avoid point depletion and drift
        self.prev_frame = gray
        self.prev_pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10
        )
        
        return H
