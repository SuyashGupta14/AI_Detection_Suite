"""
ML_Models/Fake_Job_Postings/predictor.py
Model 5 - Fake Job Posting Detection
Expected files in this folder:
  - fake_job_model.pkl
  - tfidf_vectorizer.pkl
"""

from pathlib import Path

import joblib
import pandas as pd
from flask import Blueprint, jsonify, render_template, request

fake_job_bp = Blueprint("fake_job", __name__)
MODEL_DIR = Path(__file__).resolve().parent

print("  Loading Model 5 - Fake Job Posting Detection...")
_ready = False
_error = None
_model = None
_vectorizer = None
_model_path = None
_vectorizer_path = None


def _normalize_label(v):
    return str(v).strip().lower()


def _resolve_class_indices(model, n_classes: int):
    """
    Resolve fake/real indices from model.classes_.
    Handles numeric or string labels and unexpected class ordering.
    """
    fake_aliases = {"1", "fake", "fraud", "fraudulent", "spam", "yes", "true", "positive"}
    real_aliases = {"0", "real", "legit", "legitimate", "genuine", "no", "false", "negative"}

    classes = getattr(model, "classes_", None)
    if classes is not None:
        norm_classes = [_normalize_label(c) for c in classes]

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
    _model_path = _first_existing(["fake_job_model.pkl", "model.pkl"])
    _vectorizer_path = _first_existing(["tfidf_vectorizer.pkl", "vectorizer.pkl"])

    if _model_path is None:
        raise FileNotFoundError("fake_job_model.pkl not found in Fake_Job_Postings/")
    if _vectorizer_path is None:
        raise FileNotFoundError("tfidf_vectorizer.pkl not found in Fake_Job_Postings/")

    _model = joblib.load(_model_path)
    _vectorizer = joblib.load(_vectorizer_path)

    # Fix for scikit-learn 1.5+ which removed 'multi_class' constructor param
    # but predict_proba still references self.multi_class internally
    if not hasattr(_model, 'multi_class'):
        _model.multi_class = 'auto'

    _ready = True
    print(
        "  ✅ Model 5 ready - "
        f"{type(_model).__name__} + {type(_vectorizer).__name__}"
    )

except Exception as e:
    _error = str(e)
    print(f"  ❌ Model 5 failed: {e}")


def _confidence_level(confidence: float) -> str:
    if confidence >= 80:
        return "High"
    if confidence >= 60:
        return "Medium"
    return "Low"


def _predict_text(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text input")

    vec = _vectorizer.transform([text])
    pred = int(_model.predict(vec)[0])

    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(vec)[0]
        fake_idx, real_idx = _resolve_class_indices(_model, len(proba))
        fake_prob = float(proba[fake_idx]) * 100
        real_prob = float(proba[real_idx]) * 100
    else:
        fake_prob = 100.0 if pred == 1 else 0.0
        real_prob = 100.0 - fake_prob

    is_fake = bool(pred == 1)
    confidence = fake_prob if is_fake else real_prob
    conf_level = _confidence_level(confidence)

    return {
        "prediction": "Fake Job Posting" if is_fake else "Real Job Posting",
        "is_fake": is_fake,
        "confidence": round(confidence, 2),
        "fake_prob": round(fake_prob, 2),
        "real_prob": round(real_prob, 2),
        "conf_level": conf_level,
        "emoji": "⚠️" if is_fake else "✅",
        "color": "#f06060" if is_fake else "#4caf82",
        "model_used": type(_model).__name__,
        "vectorizer": type(_vectorizer).__name__,
    }


def _row_to_text(row: pd.Series) -> str:
    preferred = [
        "text",
        "job_posting",
        "description",
        "job_description",
        "description_text",
    ]
    for key in preferred:
        if key in row and str(row.get(key, "")).strip():
            return str(row.get(key, ""))

    parts = []
    for key in [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits",
        "industry",
        "function",
        "department",
    ]:
        if key in row and str(row.get(key, "")).strip():
            parts.append(str(row.get(key, "")))

    if parts:
        return "\n".join(parts)

    non_empty = [str(v) for v in row.values if str(v).strip() and str(v) != "nan"]
    return "\n".join(non_empty[:4])


@fake_job_bp.route("/model5")
def page():
    return render_template(
        "model5.html",
        model_ready=_ready,
        model_error=_error,
        model_name=type(_model).__name__ if _ready else "-",
        vectorizer_name=type(_vectorizer).__name__ if _ready else "-",
        model_file=_model_path.name if _model_path else "fake_job_model.pkl",
        vectorizer_file=_vectorizer_path.name if _vectorizer_path else "tfidf_vectorizer.pkl",
    )


@fake_job_bp.route("/predict_fake_job", methods=["POST"])
def predict_fake_job():
    if not _ready:
        return jsonify({"error": f"Model not loaded: {_error}"}), 500

    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    try:
        result = _predict_text(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@fake_job_bp.route("/predict_fake_job_csv", methods=["POST"])
def predict_fake_job_csv():
    if not _ready:
        return jsonify({"error": f"Model not loaded: {_error}"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"CSV read failed: {str(e)}"}), 400

    rows = []
    fake_count = 0

    for i, row in df.iterrows():
        text = _row_to_text(row)
        if not text.strip():
            continue

        try:
            pred = _predict_text(text)
        except Exception:
            continue

        if pred["is_fake"]:
            fake_count += 1

        rows.append(
            {
                "row": int(i + 1),
                "preview": text.replace("\n", " ")[:90],
                "prediction": pred["prediction"],
                "is_fake": pred["is_fake"],
                "confidence": pred["confidence"],
                "emoji": pred["emoji"],
            }
        )

        if len(rows) >= 200:
            break

    total = len(rows)
    fake_pct = round((fake_count / total) * 100, 2) if total else 0.0

    return jsonify(
        {
            "total": total,
            "fake_count": fake_count,
            "real_count": total - fake_count,
            "fake_pct": fake_pct,
            "model_used": type(_model).__name__,
            "vectorizer": type(_vectorizer).__name__,
            "rows": rows,
        }
    )
