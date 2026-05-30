import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO

class ReIDWrapper(nn.Module):
    def __init__(self, features, pool):
        super().__init__()
        self.features = features
        self.pool = pool
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # L2 normalization
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        return x / (norm + 1e-6)

def export_detector(model_path="yolo11s.pt"):
    print(f"Exporting YOLO detector {model_path} to ONNX...")
    model = YOLO(model_path)
    # Export with dynamic axes for variable input resolutions if needed
    onnx_path = model.export(format="onnx", imgsz=640, dynamic=True)
    print(f"Detector successfully exported to: {onnx_path}")

def export_reid():
    print("Exporting ReID MobileNetV3-Small to ONNX...")
    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except AttributeError:
        model = models.mobilenet_v3_small(pretrained=True)
        
    features = model.features
    pool = nn.AdaptiveAvgPool2d(1)
    
    wrapper = ReIDWrapper(features, pool)
    wrapper.eval()
    
    # Dummy input [batch_size, channels, height, width]
    # Standard ReID input crop size is 128x64
    dummy_input = torch.randn(1, 3, 128, 64)
    
    output_path = "reid_mobilenet_v3.onnx"
    
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        # Define the batch axis as dynamic because the number of person crops changes per frame
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=12,
        do_constant_folding=True
    )
    print(f"ReID extractor successfully exported to: {output_path}")

def main():
    # Export detector
    try:
        export_detector("yolo11s.pt")
    except Exception as e:
        print(f"Failed to export detector: {e}")
        
    # Export ReID
    try:
        export_reid()
    except Exception as e:
        print(f"Failed to export ReID: {e}")

if __name__ == '__main__':
    main()
