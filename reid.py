import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2
import numpy as np
import ssl

# Bypass macOS SSL certificate verification issues for model downloads
ssl._create_default_https_context = ssl._create_unverified_context

class ReIDExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        # Load lightweight MobileNetV3 Small
        try:
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        except AttributeError:
            # Fallback for older torchvision versions
            try:
                model = models.mobilenet_v3_small(pretrained=True)
            except Exception:
                # Offline/no internet fallback
                model = models.mobilenet_v3_small(weights=None)
                
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Freeze parameters
        for p in self.features.parameters():
            p.requires_grad = False
            
        self.features.eval()
        self.features.to(device)
        
        # Standard ReID preprocessing transforms:
        # Resize to 128x64, convert to float tensor, normalize with ImageNet statistics
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract(self, img, bboxes):
        """
        Extract ReID features for cropped bounding boxes.
        img: full BGR frame
        bboxes: np.ndarray or list of shape (N, 4) in [x1, y1, x2, y2] format
        Returns:
            embeddings: np.ndarray of shape (N, 576)
        """
        if len(bboxes) == 0:
            return np.empty((0, 576))
            
        h, w = img.shape[:2]
        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Clip coordinates to frame dimensions
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            
            # Ensure crop has positive width and height
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
                # Convert BGR (OpenCV default) to RGB
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor_crop = self.transform(crop)
                crops.append(tensor_crop)
            else:
                # Handle tiny edge case: black crop if coordinates are invalid
                dummy = np.zeros((128, 64, 3), dtype=np.uint8)
                crops.append(self.transform(dummy))
                
        if not crops:
            return np.empty((0, 576))
            
        # Implement Batch-Size Padding for Apple MPS / GPU execution stability
        # Find the next power of 2 to lock batch size shapes (prevents compilation lag on MPS)
        n = len(crops)
        padded_size = 1
        while padded_size < n:
            padded_size *= 2
            
        # Pad by repeating the first crop
        padded_crops = list(crops)
        while len(padded_crops) < padded_size:
            padded_crops.append(crops[0].clone())
            
        # Form batch and run inference
        batch = torch.stack(padded_crops).to(self.device)
        features = self.features(batch)
        pooled = self.pool(features).view(features.size(0), -1)
        
        # Slice output back to original size and compute L2 Norm
        original_pooled = pooled[:n]
        norms = torch.norm(original_pooled, p=2, dim=1, keepdim=True)
        normalized = original_pooled / (norms + 1e-6)
        
        return normalized.cpu().numpy()
