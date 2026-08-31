"""
Simulate airborne image degradation (blur, noise, low-res) and measure how
much the Eigen-CAM explanation shifts, for majority vs minority classes.
Stability metric: cosine similarity + SSIM between clean-image CAM and
degraded-image CAM (higher = more stable explanation).
Requires: pip install scikit-image
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
import torch
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- inlined from 04_eigencam.py (module names can't start with a digit) ---
class YoloEigenCamWrapper(torch.nn.Module):
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
    target_layers = [wrapped.model.model[target_layer_idx]]
    rgb_float, tensor = load_image(image_path)
    cam = EigenCAM(wrapped, target_layers, task="od")
    grayscale_cam = cam(tensor)[0, :, :]
    visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return visualization, grayscale_cam
# --- end inlined ---

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = PROJECT_ROOT / "runs" / "yolov8n_baseline" / "weights" / "best.pt"
TEST_IMAGES = PROJECT_ROOT / "data" / "yolo" / "images" / "test"
CLASS_COUNTS_CSV = None  # optionally point to counts if you want auto majority/minority split
RESULTS_CSV = PROJECT_ROOT / "outputs" / "degradation_stability.csv"


def degrade(img_bgr, kind):
    if kind == "blur":
        return cv2.GaussianBlur(img_bgr, (9, 9), sigmaX=3)
    if kind == "noise":
        noise = np.random.normal(0, 15, img_bgr.shape).astype(np.float32)
        noisy = img_bgr.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if kind == "lowres":
        h, w = img_bgr.shape[:2]
        small = cv2.resize(img_bgr, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(kind)


def cam_similarity(cam_a, cam_b):
    a, b = cam_a.flatten(), cam_b.flatten()
    cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    ssim_score = float(ssim(cam_a, cam_b, data_range=1.0))
    return cos_sim, ssim_score


def main():
    yolo = YOLO(WEIGHTS.as_posix())
    tmp_dir = PROJECT_ROOT / "outputs" / "_tmp_degraded"
    tmp_dir.mkdir(exist_ok=True)

    rows = []
    images = sorted(TEST_IMAGES.glob("*.jpg"))[:40]  # sample; expand as needed

    for img_path in images:
        clean = cv2.imread(str(img_path))
        _, clean_cam = run_eigencam(img_path, yolo)

        for kind in ["blur", "noise", "lowres"]:
            degraded = degrade(clean, kind)
            deg_path = tmp_dir / f"{img_path.stem}_{kind}.jpg"
            cv2.imwrite(str(deg_path), degraded)

            _, deg_cam = run_eigencam(deg_path, yolo)
            cos_sim, ssim_score = cam_similarity(clean_cam, deg_cam)

            rows.append({
                "image": img_path.name,
                "degradation": kind,
                "cosine_similarity": cos_sim,
                "ssim": ssim_score,
            })
            print(f"{img_path.name} | {kind:6s} | cos={cos_sim:.3f} ssim={ssim_score:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved -> {RESULTS_CSV}")

    print("\nMean stability by degradation type:")
    print(df.groupby("degradation")[["cosine_similarity", "ssim"]].mean())


if __name__ == "__main__":
    main()
