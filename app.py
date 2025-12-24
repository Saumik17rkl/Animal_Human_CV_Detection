import os
import time
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory,send_file, abort, render_template, Response
from werkzeug.utils import secure_filename
import mimetypes
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import json
from concurrent.futures import ThreadPoolExecutor

# Initialize Flask app
app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 * 1024  # 1GB max file size

# Ensure upload and processed directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Global variables for models and processing state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_models = []
classifier = None
processing_status = {}

# Initialize models
def init_models():
    global yolo_models, classifier
    
    # Initialize YOLO models
    models_dir = Path("models")
    for path in (models_dir / "yolov8_detection").rglob("best.pt"):
        print(f"[INFO] Loading YOLO: {path}")
        yolo_models.append(YOLO(str(path)))
    
    assert yolo_models, "❌ No YOLO models found"
    
    # Initialize EfficientNet classifier
    classifier = models.efficientnet_b0(weights=None)
    classifier.classifier[1] = nn.Linear(
        classifier.classifier[1].in_features, 2
    )
    classifier.load_state_dict(
        torch.load(
            models_dir / "efficientnet_classifier" / "best_efficientnet.pt",
            map_location=DEVICE,
        )
    )
    classifier.to(DEVICE).eval()

# Image transformation for classifier
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Class names
CLASS_NAMES = ["Human", "Animal"]

# Utility functions
def classify_crop(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(crop_rgb)
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(classifier(tensor), dim=1)
        conf, pred = torch.max(probs, 1)

    return CLASS_NAMES[pred.item()], float(conf.item())

def collect_boxes(frame):
    boxes = []
    meta = []

    for model_idx, model in enumerate(yolo_models):
        results = model(frame, conf=0.25, iou=0.5, verbose=False)[0]

        if results.boxes is None:
            continue

        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            det_conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            yolo_label = model.names[cls_id].lower()

            boxes.append([x1, y1, x2, y2, det_conf])
            meta.append({"label": yolo_label, "model": model_idx})

    return np.array(boxes, dtype=np.float32), meta

def nms(boxes, iou_thresh=0.5):
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep

# Global executor (IMPORTANT)
executor = ThreadPoolExecutor(max_workers=2)

def process_video_task(video_path, task_id):
    output_path = os.path.join(
        app.config['PROCESSED_FOLDER'],
        f"annotated_{os.path.basename(video_path)}"
    )

    processing_status[task_id] = {
        "status": "processing",
        "progress": 0,
        "result_path": output_path
    }

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"H264")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (w, h)
    )


    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes_np, meta = collect_boxes(frame)
            keep_ids = nms(boxes_np)

            for idx in keep_ids:
                x1, y1, x2, y2, det_conf = boxes_np[idx]
                yolo_label = meta[idx]["label"]

                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                # 🔥 Junk removal (CRITICAL)
                if ((x2 - x1) * (y2 - y1)) / (w * h) < 0.005:
                    continue

                # 🔥 Decision fusion (MATCHES REFERENCE)
                if yolo_label == "person":
                    final_label = "Human"
                    final_conf = det_conf
                else:
                    crop = frame[y1:y2, x1:x2]
                    clf_label, clf_conf = classify_crop(crop)

                    if clf_label == "Animal" and clf_conf >= 0.7:
                        final_label = "Animal"
                        final_conf = clf_conf
                    else:
                        final_label = "Animal"
                        final_conf = det_conf

                color = (0, 255, 0) if final_label == "Human" else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{final_label} ({final_conf:.2f})",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            out.write(frame)
            frame_idx += 1
            processing_status[task_id]["progress"] = int((frame_idx / total) * 100)

        processing_status[task_id]["status"] = "completed"

    except Exception as e:
        processing_status[task_id]["status"] = "error"
        processing_status[task_id]["error"] = str(e)

    finally:
        cap.release()
        out.release()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file or file.filename == "":
        return jsonify({"error": "No file"}), 400

    task_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    executor.submit(process_video_task, path, task_id)

    return jsonify({"task_id": task_id}), 202

@app.route('/status/<task_id>')
def status(task_id):
    if task_id not in processing_status:
        return jsonify({'error': 'Invalid task ID'}), 404
        
    status_info = processing_status[task_id].copy()
    
    # If processing is complete, add download link
    if status_info['status'] == 'completed':
        status_info['download_url'] = f'/processed/{os.path.basename(status_info["result_path"])}'
    
    return jsonify(status_info)



@app.route("/processed/<path:filename>")
def processed_file(filename):
    file_path = os.path.join(app.config["PROCESSED_FOLDER"], filename)

    if not os.path.isfile(file_path):
        abort(404)

    return send_file(
        file_path,
        mimetype="video/mp4",
        as_attachment=False,   # 🔴 CRITICAL
        conditional=True       # 🔴 CRITICAL (range requests)
    )



if __name__ == '__main__':
    # Initialize models
    print("Initializing models...")
    init_models()
    print("Models loaded successfully!")
    
    # Run the app
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
