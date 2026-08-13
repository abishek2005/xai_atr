import argparse
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image

class YOLOTarget:
    def __init__(self, class_idx=None):
        self.class_idx = class_idx
    def __call__(self, model_output):
        if isinstance(model_output, (list, tuple)):
            model_output = model_output[0]
        nc = model_output.shape[1] - 4
        cls_scores = model_output[0, 4:4+nc, :]
        scores = cls_scores[self.class_idx] if self.class_idx is not None else cls_scores.max(dim=0).values
        return scores.max()

def torch_device(d):
    if d in ("cpu", "cuda"):
        return d
    return f"cuda:{d}"

def letterbox_load(img_source, imgsz):
    if isinstance(img_source, (str, Path)):
        img = cv2.imread(str(img_source))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.array(img_source.convert("RGB"))
    img = cv2.resize(img, (imgsz, imgsz))
    rgb_float = np.float32(img) / 255.0
    tensor = torch.from_numpy(rgb_float).permute(2, 0, 1).unsqueeze(0)
    return rgb_float, tensor

def build_cam(weights, device="cpu"):
    torch_dev = torch_device(device)
    yolo = YOLO(weights)
    yolo.model = yolo.model.to(torch_dev)
    model = yolo.model
    model.eval()
    # prevent Ultralytics auto-fusing Conv+BN on first predict() call,
    # which replaces target-layer weights with non-grad-tracked tensors
    model.fuse = lambda *a, **k: model
    for p in model.parameters():
        p.requires_grad_(True)
    cam = GradCAM(model=model, target_layers=[model.model[-2]])
    return yolo, cam

def run_gradcam(yolo, cam, img_source, imgsz=640, device="cpu", class_idx=None):
    torch_dev = torch_device(device)
    for p in yolo.model.parameters():
        p.requires_grad_(True)  # defensive: undo any grad-disabling from a prior predict() call

    rgb_float, tensor = letterbox_load(img_source, imgsz)
    tensor = tensor.to(torch_dev)
    raw_cam = cam(tensor, targets=[YOLOTarget(class_idx)])
    grayscale_cam = scale_cam_image(raw_cam, (imgsz, imgsz))[0]
    cam_overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    src = str(img_source) if isinstance(img_source, (str, Path)) else img_source
    results = yolo.predict(src, imgsz=imgsz, device=device, verbose=False)
    return rgb_float, cam_overlay, results[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="outputs/cam")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    yolo, cam = build_cam(args.weights, args.device)
    src = Path(args.source)
    images = [src] if src.is_file() else sorted(list(src.glob("*.jpg")) + list(src.glob("*.png")))
    for img_path in images:
        _, cam_overlay, result = run_gradcam(yolo, cam, img_path, args.imgsz, args.device)
        det = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
        det = cv2.resize(det, (args.imgsz, args.imgsz))
        combined = np.hstack([det, cam_overlay])
        cv2.imwrite(str(Path(args.out) / f"{img_path.stem}_cam.jpg"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"wrote {img_path.stem}_cam.jpg")

if __name__ == "__main__":
    main()
