# Animal–Human Detection & Classification Inference Script
#
# This script describes the complete inference pipeline used
# for detecting and classifying Humans and Animals in videos.
#
# IMPORTANT:
# - This file contains ONLY comments.
# - No executable code is included by design.
# - It serves as documentation + explanation for reviewers.

# 1. PROJECT DIRECTORY STRUCTURE
# project/
# ├── datasets/              # Dataset storage (Open Images V6)
# │   └── open_images_v6/
# │       ├── images/
# │       │   ├── train/
# │       │   └── val/
# │       ├── labels/
# │       │   ├── train/
# │       │   └── val/
# │       ├── classification_crops/
# │       │   ├── human/
# │       │   └── animal/
# │       └── dataset.yaml   # YOLOv8 dataset configuration
# │
# ├── models/                # Trained model checkpoints
# │   ├── yolov8_detection/
# │   │   └── human_animal_detector/
# │   │       └── weights/
# │   │           └── best.pt
# │   └── efficientnet_classifier/
# │       └── best_efficientnet.pt
# │
# ├── test_videos/           # User uploads test videos here
# ├── outputs/               # Annotated output videos
# └── animal_human_detection.py
#


# 2. DATASET CHOICE & JUSTIFICATION
# Dataset Used:
# - Open Images V6 (via FiftyOne)
# Reasons for choosing Open Images V6:
# - Large-scale, real-world dataset
# - High-quality bounding box annotations
# - Contains diverse human and animal classes
# - Suitable for both detection and classification tasks
#
# Detection Classes (16 total):
# - Person
# - Dog, Cat, Horse, Sheep, Goat
# - Pig, Bird, Deer, Elephant, Bear
# - Zebra, Giraffe, Monkey, Tiger, Lion
#
# Classification Setup:
# - Binary classification only:
#   → Human
#   → Animal
#
# Crops for classification are generated directly from
# detection bounding boxes to maintain consistency.

# 3. MODEL SELECTION & RATIONALE
# Detection Model:
# - YOLOv8 (Ultralytics)
# Why YOLOv8?
# - Real-time object detection
# - Strong localization accuracy
# - Easy fine-tuning on custom datasets
# - Excellent balance of speed and performance

# Classification Model:
# - EfficientNet-B0
# Why EfficientNet?
# - Strong performance on small image crops
# - Parameter-efficient (important for CPU inference)
# - Pretrained on ImageNet for better generalization

# Model Responsibilities:
# - YOLOv8 → Bounding box localization ONLY
# - EfficientNet → Final Human vs Animal decision

# This separation avoids overloading YOLO with
# fine-grained classification decisions.
    
# 4. TRAINING OVERVIEW
# YOLOv8 Training:
# - Trained on Open Images V6 subset
# - 16 detection classes
# - Optimizer: AdamW
# - Image size: 640
# - Metrics logged using Weights & Biases
# - Output: best.pt
#
# EfficientNet Training:
# - Binary classification (Human vs Animal)
# - Uses cropped bounding boxes
# - Class-weighted loss to handle imbalance
# - Pretrained backbone + frozen feature extractor
# - Early stopping to prevent overfitting
# - Output: best_efficientnet.pt

# 5. INFERENCE PIPELINE (STEP-BY-STEP)
# Step 1: Monitor ./test_videos/
# - The script continuously scans the directory
# - Any new video is automatically picked up

# Step 2: Frame-by-frame processing
# - Each video is read frame by frame using OpenCV

# Step 3: Multi-model YOLO detection
# - All YOLOv8 models in models/yolov8_detection/ are loaded
# - Each model runs detection on the same frame

# Step 4: Cross-model Non-Maximum Suppression (NMS)
# - Bounding boxes from all models are merged
# - Duplicate detections are removed

# Step 5: Classification decision
# - YOLO provides bounding boxes
# - EfficientNet classifies each crop as Human or Animal
# - EfficientNet is treated as the final authority

# Step 6: Annotation
# - Green bounding box → Human
# - Red bounding box → Animal
# - Confidence score displayed on each box

# Step 7: Output generation
# - Annotated frames are written to a video
# - Output saved to ./outputs/

# 6. CHALLENGES FACED
# 1. Class Imbalance
# - Humans dominate Open Images
# - Mitigated using class-weighted loss
#
# 2. False Positives
# - Animals detected near humans
# - Solved using classifier confirmation
#
# 3. Model Disagreement
# - YOLO models may disagree on detections
# - Solved using cross-model NMS
#
# 4. Windows / SSL Issues
# - Disabled MLflow
# - Used W&B in offline mode
#
# 5. CPU-only Constraints
# - Chose lightweight models
# - Optimized batch sizes and inference steps
#
# 6. POTENTIAL IMPROVEMENTS
#
# - Replace NMS with Weighted Box Fusion (WBF)
# - Add temporal smoothing across frames
# - Introduce an "Unknown" class for low confidence cases
# - Train a small Vision Transformer (ViT) for classification
# - Fine-tune YOLO with focal loss for minority classes

# 7. FINAL NOTES
# - This pipeline is fully automated
# - No user interaction required
# - Run the test.py file to start the script
# - Simply upload a video to ./test_videos/
# - Output appears in ./outputs/
# - The script monitors ./test_videos/ for new videos
# - The script will process the videos and save the output in ./outputs/
# The system is modular, extensible, and production-ready.

