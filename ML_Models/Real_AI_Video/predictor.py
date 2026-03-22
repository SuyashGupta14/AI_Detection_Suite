"""
ML_Models/Real_AI_Video/predictor.py
Model 1 — Real vs AI Video Detection
Uses: xgb_cnn.pkl + scaler_cnn.pkl + EfficientNet-B0
"""

import os
import pickle
import numpy as np
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from flask import Blueprint, render_template, request, jsonify, current_app

video_bp  = Blueprint("video", __name__)
MODEL_DIR = Path(__file__).resolve().parent

# ── Load once at startup ──────────────────────────────────────────
print("  Loading Model 1 — Real vs AI Video...")
_ready = False
_error = None

try:
    with open(MODEL_DIR / "xgb_cnn.pkl",    "rb") as f:
        _xgb = pickle.load(f)
    with open(MODEL_DIR / "scaler_cnn.pkl", "rb") as f:
        _scaler = pickle.load(f)

    _device = torch.device("cpu")
    _cnn    = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    _cnn.classifier = nn.Identity()
    _cnn    = _cnn.to(_device).eval()

    _transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    _ready = True
    print("  ✅ Model 1 ready — XGBoost + EfficientNet-B0")

except Exception as e:
    _error = str(e)
    print(f"  ❌ Model 1 failed: {e}")


# ── Helpers ───────────────────────────────────────────────────────
def _extract_frames(path, max_frames=10, size=(224, 224)):
    cap   = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames


def _predict_video(path):
    frames = _extract_frames(path)
    if not frames:
        return None

    tensors = torch.stack([_transform(f) for f in frames]).to(_device)
    with torch.no_grad():
        emb = _cnn(tensors).mean(dim=0).cpu().numpy().reshape(1, -1)

    proba      = _xgb.predict_proba(_scaler.transform(emb))[0]
    pred       = 1 if proba[1] >= 0.45 else 0
    confidence = round(float(proba[pred]) * 100, 1)

    return {
        "label":      "AI Generated" if pred == 1 else "Real Video",
        "confidence": confidence,
        "real_prob":  round(float(proba[0]) * 100, 1),
        "ai_prob":    round(float(proba[1]) * 100, 1),
        "frames":     len(frames),
        "conf_level": "High" if confidence >= 80 else "Medium" if confidence >= 60 else "Low",
    }


# ── Routes ────────────────────────────────────────────────────────
@video_bp.route("/model1")
def page():
    return render_template("model1.html")


@video_bp.route("/predict", methods=["POST"])
def predict():
    if not _ready:
        return jsonify({"error": f"Model not loaded: {_error}"}), 500
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files["video"]
    if video.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    temp = os.path.join(current_app.config["UPLOAD_FOLDER"], "temp_video.mp4")
    video.save(temp)
    result = _predict_video(temp)
    if os.path.exists(temp):
        os.remove(temp)
    if result is None:
        return jsonify({"error": "Could not process video"}), 422

    return jsonify(result)