import os
import uuid
import cv2
from flask import Flask, request, render_template, jsonify
from src.explain import build_cam, run_gradcam

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]:
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)

WEIGHTS = os.environ.get("ATR_WEIGHTS", "outputs/models/best.pt")
DEVICE = os.environ.get("ATR_DEVICE", "0")
IMGSZ = 640

print("Loading model...")
yolo, cam = build_cam(WEIGHTS, device=DEVICE)
class_names = yolo.names
print(f"Model loaded. {len(class_names)} classes.")


@app.route('/')
def index():
    return render_template('index.html', class_names=list(class_names.values()))


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    class_idx = None
    class_filter = request.form.get('class_filter', 'auto')
    if class_filter != 'auto':
        matches = [k for k, v in class_names.items() if v == class_filter]
        if matches:
            class_idx = matches[0]

    uid = str(uuid.uuid4())[:8]
    filename = f"{uid}_{file.filename}"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    _, cam_overlay, result = run_gradcam(yolo, cam, upload_path, imgsz=IMGSZ, device=DEVICE, class_idx=class_idx)

    det_img = result.plot()  # BGR
    det_filename = f"det_{uid}.jpg"
    det_path = os.path.join(app.config['RESULT_FOLDER'], det_filename)
    cv2.imwrite(det_path, det_img)

    cam_filename = f"cam_{uid}.jpg"
    cam_path = os.path.join(app.config['RESULT_FOLDER'], cam_filename)
    cv2.imwrite(cam_path, cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))

    boxes = [
        {"class": class_names[int(b.cls[0])], "confidence": round(float(b.conf[0]), 3)}
        for b in result.boxes
    ]

    return jsonify({
        'detections': boxes,
        'count': len(boxes),
        'detection_image': f'/{det_path.replace(os.sep, "/")}',
        'gradcam_image': f'/{cam_path.replace(os.sep, "/")}',
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
