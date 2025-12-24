
"""
============================================================
Multi-Model Animal–Human Detection & Classification
============================================================

- Uses ALL YOLOv8 models in models/yolov8_detection/
- YOLO = localization ONLY
- EfficientNet = final Human / Animal decision
- Cross-model NMS to remove duplicates
- Robust annotated video output
============================================================
"""

import os
import time
from pathlib import Path
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ------------------------------------------------------------
# Environment safety
# ------------------------------------------------------------
os.environ["ULTRALYTICS_MLFLOW"] = "false"
os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlruns"

# ------------------------------------------------------------
# Paths (notebook + script safe)
# ------------------------------------------------------------
PROJECT_ROOT = Path.cwd()
TEST_VIDEOS_DIR = PROJECT_ROOT / "test_videos"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

TEST_VIDEOS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ------------------------------------------------------------
# Load ALL YOLO models
# ------------------------------------------------------------
yolo_models = []
for path in (MODELS_DIR / "yolov8_detection").rglob("best.pt"):
    print(f"[INFO] Loading YOLO: {path}")
    yolo_models.append(YOLO(str(path)))

assert yolo_models, "❌ No YOLO models found"

# ------------------------------------------------------------
# Load EfficientNet classifier
# ------------------------------------------------------------
classifier = models.efficientnet_b0(weights=None)
classifier.classifier[1] = nn.Linear(
    classifier.classifier[1].in_features, 2
)
classifier.load_state_dict(
    torch.load(
        MODELS_DIR / "efficientnet_classifier" / "best_efficientnet.pt",
        map_location=DEVICE,
    )
)
classifier.to(DEVICE).eval()

CLASS_NAMES = ["Human", "Animal"]

clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ------------------------------------------------------------
# Classify ANY crop (final authority)
# ------------------------------------------------------------
def classify_crop(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(crop_rgb)
    tensor = clf_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(classifier(tensor), dim=1)
        conf, pred = torch.max(probs, 1)

    return CLASS_NAMES[pred.item()], float(conf.item())

# ------------------------------------------------------------
# Collect boxes from all YOLO models
# ------------------------------------------------------------
def collect_boxes(frame):
    """
    Returns:
        boxes_np: np.ndarray [N, 5] -> x1, y1, x2, y2, det_conf
        meta:     list of dicts -> label, model_idx
    """
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
            meta.append({
                "label": yolo_label,
                "model": model_idx
            })

    return np.array(boxes, dtype=np.float32), meta


# ------------------------------------------------------------
# Cross-model NMS
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Video processing
# ------------------------------------------------------------
def process_video(video_path):
    print(f"[INFO] Processing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = OUTPUTS_DIR / f"annotated_{video_path.name}"
    out = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- DETECTION ----
        boxes_np, meta = collect_boxes(frame)
        keep_ids = nms(boxes_np)

        for idx in keep_ids:
            x1, y1, x2, y2, det_conf = boxes_np[idx]
            yolo_label = meta[idx]["label"]

            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue

            # Remove tiny junk
            if ((x2 - x1) * (y2 - y1)) / (w * h) < 0.005:
                continue

            # ---- DECISION FUSION ----
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

            # ---- DRAW ----
            color = (0, 255, 0) if final_label == "Human" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{final_label} ({final_conf:.2f})",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        out.write(frame)

    cap.release()
    out.release()
    print(f"[INFO] Saved → {out_path.name}")




# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
def main():
    print("[INFO] Multi-model inference started")
    processed = set()

    while True:
        videos = list(TEST_VIDEOS_DIR.glob("*.mp4")) + list(TEST_VIDEOS_DIR.glob("*.avi"))

        new_videos_found = False

        for video in videos:
            if video.name not in processed:
                new_videos_found = True
                process_video(video)
                processed.add(video.name)

        # 👉 Sleep ONLY if nothing new was found
        if not new_videos_found:
            time.sleep(5)


if __name__ == "__main__":
    main()
