"""
Generate Eigen-CAM heatmaps for a trained YOLOv8 model.
Requires: pip install grad-cam ultralytics opencv-python
Reference approach: pytorch-grad-cam's EigenCAM applied to the last
backbone/neck layer before detection heads (works model-agnostically,
no need to unpack YOLO's multi-scale outputs).
"""
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = PROJECT_ROOT / "runs" / "yolov8n_baseline" / "weights" / "best.pt"
OUT_DIR = PROJECT_ROOT / "outputs" / "eigencam"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class YoloEigenCamWrapper(torch.nn.Module):
    """Wraps YOLO's underlying nn.Module so EigenCAM can call it like a plain model."""
    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model.model

    def forward(self, x):
        return self.model(x)


def load_image(path, size=640):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    rgb_float = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb_float).permute(2, 0, 1).unsqueeze(0)
    return rgb_float, tensor


def run_eigencam(image_path, yolo_model, target_layer_idx=-4):
    wrapped = YoloEigenCamWrapper(yolo_model)
    wrapped.eval()

    # last conv block before the detect head tends to give the cleanest CAMs
    target_layers = [wrapped.model.model[target_layer_idx]]

    rgb_float, tensor = load_image(image_path)

    cam = EigenCAM(wrapped, target_layers, task="od")
    grayscale_cam = cam(tensor)[0, :, :]
    visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return visualization, grayscale_cam


def main():
    yolo = YOLO(WEIGHTS.as_posix())

    # point this at a handful of images spanning majority + minority classes
    sample_images = list((PROJECT_ROOT / "data" / "yolo" / "images" / "test").glob("*.jpg"))[:20]

    for img_path in sample_images:
        vis, cam_map = run_eigencam(img_path, yolo)
        out_path = OUT_DIR / f"{img_path.stem}_eigencam.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        np.save(OUT_DIR / f"{img_path.stem}_cam.npy", cam_map)   # raw map for later comparison
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
