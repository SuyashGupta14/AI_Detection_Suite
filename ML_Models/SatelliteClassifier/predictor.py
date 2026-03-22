"""
ML_Models/SatelliteClassifier/predictor.py
Model 2 — Satellite Image Classification
Uses: ResNet50_best.pth OR CustomCNN_best.pth + satellite_meta.json
Auto-picks correct .pth from best_model field in satellite_meta.json
"""

import os
import json
import uuid
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models as tv_models
from PIL import Image
from flask import Blueprint, render_template, request, jsonify, current_app

satellite_bp = Blueprint("satellite", __name__)
MODEL_DIR    = Path(__file__).resolve().parent

# ── Class display info ────────────────────────────────────────────
CLASS_META = {
    "cloudy":     {"emoji": "☁️",  "color": "#89DCEB", "label": "Cloudy",     "desc": "Cloud formations detected"},
    "desert":     {"emoji": "🏜️",  "color": "#F9E2AF", "label": "Desert",     "desc": "Arid / sandy terrain"},
    "green_area": {"emoji": "🌿",  "color": "#A6E3A1", "label": "Green Area", "desc": "Vegetation / forest detected"},
    "water":      {"emoji": "💧",  "color": "#89B4FA", "label": "Water",      "desc": "Water body detected"},
}

# ── Model architectures (must match training exactly) ─────────────
class _CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        def blk(i, o, d):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout2d(d),
            )
        self.block1 = blk(3, 32, .10)
        self.block2 = blk(32, 64, .15)
        self.block3 = blk(64, 128, .20)
        self.block4 = blk(128, 256, .25)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(.5),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        return self.classifier(x)


def _build_resnet50(num_classes):
    m    = tv_models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 512), nn.ReLU(), nn.Dropout(.4),
        nn.Linear(512, 128),              nn.ReLU(), nn.Dropout(.3),
        nn.Linear(128, num_classes),
    )
    return m


# ── Load model once at startup ────────────────────────────────────
print("  Loading Model 2 — Satellite Classifier...")
_ready     = False
_error     = None
_model     = None
_transform = None
_classes   = []
_meta      = {}
_arch      = "—"

try:
    # Step 1 — Read metadata JSON
    meta_path = MODEL_DIR / "satellite_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("satellite_meta.json not found in SatelliteClassifier/")

    with open(meta_path) as f:
        _meta = json.load(f)

    _classes  = _meta.get("classes", ["cloudy", "desert", "green_area", "water"])
    _img_size = _meta.get("img_size", 224)
    _arch     = _meta.get("best_model", "ResNet50")

    # Step 2 — Pick correct .pth file
    _pth = MODEL_DIR / f"{_arch}_best.pth"
    if not _pth.exists():
        # Fallback: try both known names
        for candidate in ["ResNet50_best.pth", "CustomCNN_best.pth"]:
            if (MODEL_DIR / candidate).exists():
                _pth  = MODEL_DIR / candidate
                _arch = candidate.replace("_best.pth", "")
                print(f"  ⚠️  Using fallback: {candidate}")
                break
        else:
            raise FileNotFoundError(
                "No .pth file found. Expected ResNet50_best.pth or CustomCNN_best.pth"
            )

    # Step 3 — Build model and load weights
    _model = _build_resnet50(len(_classes)) if _arch == "ResNet50" \
             else _CustomCNN(len(_classes))

    _model.load_state_dict(torch.load(_pth, map_location="cpu"), strict=False)
    _model.eval()

    # Step 4 — Preprocessing pipeline
    _transform = transforms.Compose([
        transforms.Resize((_img_size, _img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            _meta.get("mean", [0.485, 0.456, 0.406]),
            _meta.get("std",  [0.229, 0.224, 0.225]),
        ),
    ])

    _ready = True
    print(f"  ✅ Model 2 ready — {_arch} | classes: {_classes}")
    print(f"     Test accuracy : {_meta.get('test_accuracy', 'N/A')}")

except Exception as e:
    _error = str(e)
    print(f"  ❌ Model 2 failed: {e}")


# ── Prediction function ───────────────────────────────────────────
@torch.no_grad()
def _predict_image(image_path: str) -> dict:
    img    = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0)
    logits = _model(tensor)
    probs  = torch.softmax(logits, dim=1).squeeze().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_class = _classes[pred_idx]
    meta       = CLASS_META.get(pred_class, {
        "emoji": "🛰️", "color": "#CDD6F4",
        "label": pred_class.title(), "desc": "Classified",
    })

    conf       = round(float(probs[pred_idx]) * 100, 1)
    conf_level = "High" if conf >= 80 else "Medium" if conf >= 60 else "Low"

    return {
        "class":      pred_class,
        "label":      meta["label"],
        "emoji":      meta["emoji"],
        "color":      meta["color"],
        "desc":       meta["desc"],
        "confidence": conf,
        "conf_level": conf_level,
        "probabilities": [
            {
                "class": cls,
                "label": CLASS_META.get(cls, {}).get("label", cls.title()),
                "emoji": CLASS_META.get(cls, {}).get("emoji", "🛰️"),
                "color": CLASS_META.get(cls, {}).get("color", "#ccc"),
                "pct":   round(float(p) * 100, 1),
            }
            for cls, p in zip(_classes, probs)
        ],
    }


# ── Routes ────────────────────────────────────────────────────────
@satellite_bp.route("/model2")
def page():
    val_accs = _meta.get("val_accuracies", {})
    return render_template(
        "model2.html",
        model_ready = _ready,
        model_error = _error,
        arch        = _arch,
        test_acc    = _meta.get("test_accuracy", "N/A") if _ready else "—",
        cnn_acc     = val_accs.get("CustomCNN", "—"),
        resnet_acc  = val_accs.get("ResNet50",  "—"),
    )


@satellite_bp.route("/predict_satellite", methods=["POST"])
def predict():
    if not _ready:
        return jsonify({"error": f"Model not loaded: {_error}"}), 500
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        return jsonify({"error": f"Unsupported format {ext} — use JPG or PNG"}), 400

    temp_path = os.path.join(
        current_app.config["UPLOAD_FOLDER"],
        f"sat_{uuid.uuid4().hex}{ext}"
    )
    file.save(temp_path)

    try:
        result = _predict_image(temp_path)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify(result)