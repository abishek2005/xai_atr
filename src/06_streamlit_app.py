"""
Streamlit UI: upload an image -> YOLO detection + Eigen-CAM overlay,
with optional degradation applied before inference.
Run: streamlit run 06_streamlit_app.py
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = (PROJECT_ROOT / "runs" / "yolov8n_baseline" / "weights" / "best.pt").as_posix()


@st.cache_resource
def load_model():
    return YOLO(WEIGHTS)


class YoloEigenCamWrapper(torch.nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model.model

    def forward(self, x):
        return self.model(x)


def degrade(img_bgr, kind):
    if kind == "None":
        return img_bgr
    if kind == "Blur":
        return cv2.GaussianBlur(img_bgr, (9, 9), sigmaX=3)
    if kind == "Noise":
        noise = np.random.normal(0, 15, img_bgr.shape).astype(np.float32)
        return np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if kind == "Low-res":
        h, w = img_bgr.shape[:2]
        small = cv2.resize(img_bgr, (w // 4, h // 4))
        return cv2.resize(small, (w, h))


def run(model, img_bgr):
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (640, 640)), cv2.COLOR_BGR2RGB)
    rgb_float = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb_float).permute(2, 0, 1).unsqueeze(0)

    results = model.predict(img_rgb, verbose=False)[0]
    annotated = results.plot()

    wrapped = YoloEigenCamWrapper(model)
    wrapped.eval()
    target_layers = [wrapped.model.model[-4]]
    cam = EigenCAM(wrapped, target_layers, task="od")
    grayscale_cam = cam(tensor)[0, :, :]
    cam_overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    return annotated, cam_overlay, results


st.set_page_config(page_title="XAI for ATR", layout="wide")
st.title("Explainable AI for ATR in Airborne Surveillance")

uploaded = st.file_uploader("Upload aircraft image", type=["jpg", "jpeg", "png"])
degradation = st.selectbox("Simulated degradation", ["None", "Blur", "Noise", "Low-res"])

if uploaded:
    model = load_model()
    pil_img = Image.open(uploaded).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_bgr = degrade(img_bgr, degradation)

    annotated, cam_overlay, results = run(model, img_bgr)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detection")
        st.image(annotated, channels="BGR")
    with col2:
        st.subheader("Eigen-CAM explanation")
        st.image(cam_overlay)

    st.subheader("Predicted classes")
    names = results.names
    for box in results.boxes:
        cid = int(box.cls[0])
        conf = float(box.conf[0])
        st.write(f"- {names[cid]}: {conf:.2f}")
