import cv2
import numpy as np
from ultralytics import YOLO

class TiledDetector:
    def __init__(self, model_path="yolo11s.pt", tile_size=640, overlap=160, device="cpu"):
        """
        model_path: path to YOLO model (e.g. yolo11s.pt or community fine-tuned model)
        tile_size: width and height of each square tile
        overlap: pixel overlap between adjacent tiles
        device: device to run model on ('cpu', 'cuda', 'mps')
        """
        self.model = YOLO(model_path)
        self.tile_size = tile_size
        self.overlap = overlap
        self.device = device
        self.model.to(device)

    def detect(self, img, conf_threshold=0.25, iou_threshold=0.45, classes=[0]):
        """
        Runs tiled inference on the image and merges detections.
        classes: class indices to keep (default [0] for COCO person)
        Returns:
            boxes: np.ndarray of shape (N, 4) in [x1, y1, x2, y2] format
            scores: np.ndarray of shape (N,)
            class_ids: np.ndarray of shape (N,)
        """
        h, w = img.shape[:2]
        
        # Lists to collect detections across all tiles and full image
        all_boxes_xywh = []
        all_scores = []
        all_classes = []

        # 1. Run inference on full image (resized) to capture larger targets
        # Standardize full-image size (e.g. resize to 1280 or 1536 depending on original resolution)
        full_img_sz = max(w, h)
        # Cap full-image size to avoid memory overflow but keep resolution high for small targets
        full_img_sz = min(1280, full_img_sz)
        
        full_res = self.model(img, imgsz=full_img_sz, conf=conf_threshold, verbose=False, device=self.device)[0]
        for box in full_res.boxes:
            c_id = int(box.cls[0].item())
            if c_id in classes:
                # box.xyxy format: [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = box.conf[0].item()
                
                # Convert to [x, y, w, h] for OpenCV NMS
                all_boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
                all_scores.append(score)
                all_classes.append(c_id)

        # 2. Run tiled inference for small objects
        stride = self.tile_size - self.overlap
        
        # Calculate grid coordinates
        y_coords = list(range(0, h - self.tile_size + 1, stride))
        if len(y_coords) == 0 or y_coords[-1] + self.tile_size < h:
            y_coords.append(max(0, h - self.tile_size))
            
        x_coords = list(range(0, w - self.tile_size + 1, stride))
        if len(x_coords) == 0 or x_coords[-1] + self.tile_size < w:
            x_coords.append(max(0, w - self.tile_size))

        # Collect all tiles and their offsets for parallel batched inference
        tiles = []
        offsets = []
        for y_offset in y_coords:
            for x_offset in x_coords:
                # Crop tile
                tile = img[y_offset:y_offset + self.tile_size, x_offset:x_offset + self.tile_size]
                
                # Check tile shape is valid (pad if near borders)
                if tile.shape[0] != self.tile_size or tile.shape[1] != self.tile_size:
                    tile = cv2.copyMakeBorder(
                        tile, 0, self.tile_size - tile.shape[0], 0, self.tile_size - tile.shape[1],
                        cv2.BORDER_CONSTANT, value=[0, 0, 0]
                    )
                tiles.append(tile)
                offsets.append((x_offset, y_offset))

        # Perform parallel batched inference on all tiles in a single call
        if tiles:
            batch_results = self.model(tiles, conf=conf_threshold, verbose=False, device=self.device)
            
            for idx, results in enumerate(batch_results):
                x_offset, y_offset = offsets[idx]
                for box in results.boxes:
                    c_id = int(box.cls[0].item())
                    if c_id in classes:
                        tx1, ty1, tx2, ty2 = box.xyxy[0].cpu().numpy()
                        score = box.conf[0].item()
                        
                        # Project coordinates back to full image coordinate system
                        gx1 = tx1 + x_offset
                        gy1 = ty1 + y_offset
                        gx2 = tx2 + x_offset
                        gy2 = ty2 + y_offset
                        
                        # Filter out boxes that are padded outside the image boundaries
                        gx1 = max(0, min(gx1, w - 1))
                        gy1 = max(0, min(gy1, h - 1))
                        gx2 = max(0, min(gx2, w - 1))
                        gy2 = max(0, min(gy2, h - 1))
                        
                        bw = gx2 - gx1
                        bh = gy2 - gy1
                        
                        # Filter out zero-area boxes
                        if bw > 2 and bh > 2:
                            all_boxes_xywh.append([gx1, gy1, bw, bh])
                            all_scores.append(score)
                            all_classes.append(c_id)

        if not all_boxes_xywh:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

        # 3. Apply Non-Maximum Suppression (NMS) to merge overlapping predictions
        indices = cv2.dnn.NMSBoxes(all_boxes_xywh, all_scores, conf_threshold, iou_threshold)
        
        if len(indices) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
            
        indices = np.array(indices).flatten()
        
        final_boxes_xyxy = []
        final_scores = []
        final_classes = []
        
        for idx in indices:
            x, y, bw, bh = all_boxes_xywh[idx]
            final_boxes_xyxy.append([x, y, x + bw, y + bh])
            final_scores.append(all_scores[idx])
            final_classes.append(all_classes[idx])

        return np.array(final_boxes_xyxy), np.array(final_scores), np.array(final_classes, dtype=int)
