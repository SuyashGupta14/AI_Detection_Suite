"""
ML_Models/Deepfake_audio_detection/predictor.py
Model 4 - Deepfake Audio Detection
Expected files in this folder:
  - best_model.pkl
  - scaler.pkl
"""

from pathlib import Path
import os
import uuid

import joblib
import librosa
import numpy as np
from flask import Blueprint, jsonify, render_template, request, current_app

deepfake_bp = Blueprint("deepfake_audio", __name__)
MODEL_DIR = Path(__file__).resolve().parent

SAMPLE_RATE = 22050
DURATION = 2.0
MAX_SAMPLES = int(SAMPLE_RATE * DURATION)
N_MFCC = 40

print("  Loading Model 4 - Deepfake Audio Detection...")
_ready = False
_error = None
_model = None
_scaler = None
_model_path = None
_scaler_path = None
_feature_count = None


def _normalize_label(v):
    return str(v).strip().lower()


def _resolve_class_indices(model, n_classes: int):
    """
    Resolve fake/real probability indices robustly from model.classes_.
    Falls back to binary convention (0=real, 1=fake) when uncertain.
    """
    fake_aliases = {
        "1", "fake", "ai", "ai_generated", "deepfake", "synthetic", "yes", "true", "positive"
    }
    real_aliases = {"0", "real", "human", "genuine", "authentic", "no", "false", "negative"}

    classes = getattr(model, "classes_", None)
    if classes is not None:
        norm_classes = [_normalize_label(c) for c in classes]

        # Prefer explicit fake/real labels if present
        for i, c in enumerate(norm_classes):
            if c in fake_aliases:
                fake_idx = i
                real_idx = 1 - i if n_classes == 2 else 0
                return fake_idx, real_idx

        for i, c in enumerate(norm_classes):
            if c in real_aliases:
                real_idx = i
                fake_idx = 1 - i if n_classes == 2 else min(1, n_classes - 1)
                return fake_idx, real_idx

        # Numeric convention where class 1 means fake
        for i, c in enumerate(classes):
            if c == 1:
                fake_idx = i
                real_idx = 1 - i if n_classes == 2 else 0
                return fake_idx, real_idx

        for i, c in enumerate(classes):
            if c == 0:
                real_idx = i
                fake_idx = 1 - i if n_classes == 2 else min(1, n_classes - 1)
                return fake_idx, real_idx

    # Safe fallback
    if n_classes >= 2:
        return 1, 0
    return 0, 0


def _first_existing(candidates):
    for name in candidates:
        path = MODEL_DIR / name
        if path.exists():
            return path
    return None


try:
    # Prefer explicit RandomForest artifact names first.
    _model_path = _first_existing([
        "random_forest.pkl",
        "rf_model.pkl",
        "best_model.pkl",
        "deepfake_audio_model.pkl",
    ])
    _scaler_path = _first_existing(["scaler.pkl"])

    if _model_path is None:
        raise FileNotFoundError("best_model.pkl not found in Deepfake_audio_detection/")
    if _scaler_path is None:
        raise FileNotFoundError("scaler.pkl not found in Deepfake_audio_detection/")

    _model = joblib.load(_model_path)
    _scaler = joblib.load(_scaler_path)

    # Fix for scikit-learn 1.5+ which removed 'multi_class' constructor param
    # but predict_proba still references self.multi_class internally
    if not hasattr(_model, 'multi_class'):
        _model.multi_class = 'auto'

    

    if hasattr(_scaler, "mean_"):
        _feature_count = int(_scaler.mean_.shape[0])

    _ready = True
    print(
        "  ✅ Model 4 ready - "
        f"{type(_model).__name__} | features: {_feature_count or 'unknown'}"
    )

except Exception as e:
    _error = str(e)
    print(f"  ❌ Model 4 failed: {e}")


def _load_audio(file_path: str) -> np.ndarray:
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
    if len(y) < MAX_SAMPLES:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)), mode="constant")
    else:
        y = y[:MAX_SAMPLES]
    return y


def _extract_features(file_path: str) -> np.ndarray:
    y = _load_audio(file_path)

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta_mfcc = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=20)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=SAMPLE_RATE,
        )
        f0_clean = f0[~np.isnan(f0)]
        pitch_mean = float(np.mean(f0_clean)) if len(f0_clean) else 0.0
        pitch_std = float(np.std(f0_clean)) if len(f0_clean) else 0.0
    except Exception:
        pitch_mean = 0.0
        pitch_std = 0.0

    features = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            delta_mfcc.mean(axis=1),
            delta_mfcc.std(axis=1),
            chroma.mean(axis=1),
            chroma.std(axis=1),
            contrast.mean(axis=1),
            contrast.std(axis=1),
            [float(zcr.mean()), float(zcr.std())],
            [float(rms.mean()), float(rms.std())],
            mel_db.mean(axis=1),
            mel_db.std(axis=1),
            [pitch_mean, pitch_std],
        ]
    ).astype(np.float32)

    if _feature_count and features.shape[0] != _feature_count:
        raise ValueError(
            f"Feature mismatch: extracted {features.shape[0]}, expected {_feature_count}"
        )

    return features.reshape(1, -1)


def _predict_audio(file_path: str) -> dict:
    feat = _extract_features(file_path)
    if feat is None:
        raise ValueError("Could not extract features from audio")

    feat_scaled = _scaler.transform(feat)

    if hasattr(_model, "predict_proba"):
        proba = np.asarray(_model.predict_proba(feat_scaled)[0], dtype=float)
        # Match Gradio behavior: index 0 = Real, index 1 = Fake
        real_prob = float(proba[0]) * 100
        fake_prob = float(proba[1]) * 100
    else:
        pred = int(_model.predict(feat_scaled)[0])
        # fallback mapping
        fake_prob = 100.0 if pred == 1 else 0.0
        real_prob = 100.0 - fake_prob

    is_fake = fake_prob > real_prob
    confidence = fake_prob if is_fake else real_prob
    conf_level = "High" if confidence >= 80 else "Medium" if confidence >= 60 else "Low"

    return {
        "label": "AI Generated (Fake)" if is_fake else "Real Human Audio",
        "prediction": "FAKE" if is_fake else "REAL",
        "confidence": round(confidence, 2),
        "real_prob": round(real_prob, 2),
        "fake_prob": round(fake_prob, 2),
        "conf_level": conf_level,
        "emoji": "??" if is_fake else "??",
        "model_used": type(_model).__name__,
        "features": int(_feature_count or feat.shape[1]),
        "is_fake": is_fake
    }

@deepfake_bp.route("/model4")
def page():
    return render_template(
        "model4.html",
        model_ready=_ready,
        model_error=_error,
        model_name=type(_model).__name__ if _ready else "-",
        feature_count=_feature_count or "unknown",
        model_file=_model_path.name if _model_path else "best_model.pkl",
        scaler_file=_scaler_path.name if _scaler_path else "scaler.pkl",
    )


@deepfake_bp.route("/predict_audio_deepfake", methods=["POST"])
def predict_audio_deepfake():
    if not _ready:
        return jsonify({"error": f"Model not loaded: {_error}"}), 500

    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio = request.files["audio"]
    if audio.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(audio.filename).suffix.lower()
    if ext not in {".wav", ".mp3", ".m4a", ".ogg", ".flac"}:
        return jsonify({"error": f"Unsupported format {ext}"}), 400

    temp_path = os.path.join(current_app.config["UPLOAD_FOLDER"], f"audio_{uuid.uuid4().hex}{ext}")
    audio.save(temp_path)

    try:
        result = _predict_audio(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
