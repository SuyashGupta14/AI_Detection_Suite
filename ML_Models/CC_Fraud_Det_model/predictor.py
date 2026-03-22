"""
ML_Models/CC_Fraud_Det_model/predictor.py
Model 3 — Credit Card Fraud Detection
Uses feature_engineer.pkl for exact same encoding as training
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, current_app

fraud_bp  = Blueprint("fraud", __name__)
MODEL_DIR = Path(__file__).resolve().parent

# ── Model metrics ─────────────────────────────────────────────────
MODEL_METRICS = {
    "xgboost": {
        "label": "XGBoost",       "color": "#F9E2AF",
        "accuracy": "99.20%",     "roc_auc": "0.9955",
        "f1": "0.4653",           "file": "xgboost.pkl",
    },
}

# ── Exact 28 features in correct order ───────────────────────────
FEATURE_NAMES = [
    'merchant', 'category', 'amt', 'gender', 'city', 'state',
    'lat', 'long', 'city_pop', 'job', 'dob', 'merch_lat', 'merch_long',
    'trans_date_trans_time_year', 'trans_date_trans_time_month',
    'trans_date_trans_time_day', 'trans_date_trans_time_hour',
    'trans_date_trans_time_dayofweek', 'trans_date_trans_time_is_weekend',
    'unix_time_year', 'unix_time_month', 'unix_time_day',
    'unix_time_hour', 'unix_time_dayofweek', 'unix_time_is_weekend',
    'merchant_x_category', 'category_x_amt', 'amt_x_gender'
]

CATEGORIES = [
    'food_dining', 'gas_transport', 'grocery_net', 'grocery_pos',
    'health_fitness', 'home', 'kids_pets', 'misc_net', 'misc_pos',
    'personal_care', 'shopping_net', 'shopping_pos', 'travel', 'entertainment',
]

# ── Load feature engineer + all models at startup ─────────────────
print("  Loading Model 3 — Credit Card Fraud Detection...")
_models   = {}
_engineer = None
_ready    = False
_error    = None

try:
    # Step 1 — Load feature engineer (exact encoders + scaler from training)
    fe_path = MODEL_DIR / "feature_engineer.pkl"
    if not fe_path.exists():
        raise FileNotFoundError("feature_engineer.pkl not found in CC_Fraud_Det_model/")

    with open(fe_path, "rb") as f:
        _engineer = pickle.load(f)
    print("    ✅ feature_engineer.pkl loaded")

    # Step 2 — Load all model pkl files
    for key, info in MODEL_METRICS.items():
        pth = MODEL_DIR / info["file"]
        if pth.exists():
            with open(pth, "rb") as f:
                _models[key] = pickle.load(f)
            print(f"    ✅ {info['label']} loaded")
        else:
            print(f"    ⚠️  {info['file']} not found — skipping")

    if not _models:
        raise FileNotFoundError("No model .pkl files found")

    _ready = True
    print(f"  ✅ Model 3 ready — {len(_models)} models loaded")

except Exception as e:
    _error = str(e)
    print(f"  ❌ Model 3 failed: {e}")


# ── Feature engineering using exact same pipeline as training ─────
def engineer_features(row: dict) -> np.ndarray:
    """
    Uses the SAME FeatureEngineer fitted during training.
    Guarantees identical encoding for correct predictions.
    """
    # Parse datetime
    dt_str = str(row.get("trans_date_trans_time", "2020-06-21 12:14:25"))
    dt_str = dt_str.replace("T", " ")
    if len(dt_str) == 16:
        dt_str += ":00"
    try:
        dt = datetime.strptime(dt_str[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        dt = datetime(2020, 6, 21, 12, 14, 25)

    unix_time = int(dt.timestamp())

    # Build raw row exactly as original training data
    raw = {
        "trans_date_trans_time": pd.to_datetime(dt_str[:19]),
        "merchant":   str(row.get("merchant",  "fraud_Kirlin and Sons")),
        "category":   str(row.get("category",  "personal_care")),
        "amt":        float(row.get("amt",      0)),
        "gender":     str(row.get("gender",     "M")),
        "city":       str(row.get("city",       "Columbia")),
        "state":      str(row.get("state",      "SC")),
        "lat":        float(row.get("lat",      33.9659)),
        "long":       float(row.get("long",     -80.9355)),
        "city_pop":   float(row.get("city_pop", 333497)),
        "job":        str(row.get("job",        "Mechanical engineer")),
        "dob":        str(row.get("dob",        "1968-03-19")),
        "merch_lat":  float(row.get("merch_lat",  33.986)),
        "merch_long": float(row.get("merch_long", -81.200)),
        "unix_time":  unix_time,
    }

    df = pd.DataFrame([raw])

    # Convert to datetime so extract_datetime_features works

    df["unix_time"] = pd.to_datetime(df["unix_time"], unit="s")

    # Run exact same pipeline as training
    df = _engineer.extract_datetime_features(
        df, datetime_cols=["trans_date_trans_time", "unix_time"]
    )
    df = _engineer.encode_categorical_features(df, method='label')
    df = _engineer.create_interaction_features(df, is_train=False)
    df = _engineer.scale_features(df, method='robust')

    # Fill any missing features with 0
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

    return df[FEATURE_NAMES].values.reshape(1, -1)


# ── Prediction ────────────────────────────────────────────────────
def _run_predict(X: np.ndarray, model_key: str) -> dict:
    model  = _models[model_key]
    pred   = model.predict(X)[0]
    proba  = model.predict_proba(X)[0] if hasattr(model, "predict_proba") \
             else np.array([1 - float(pred), float(pred)])

    is_fraud   = bool(pred == 1)
    fraud_prob = round(float(proba[1]) * 100, 1)
    legit_prob = round(float(proba[0]) * 100, 1)
    confidence = round(float(max(proba)) * 100, 1)
    conf_level = "High" if confidence >= 80 else "Medium" if confidence >= 60 else "Low"
    info       = MODEL_METRICS[model_key]

    return {
        "prediction":  "Fraudulent" if is_fraud else "Legitimate",
        "is_fraud":    is_fraud,
        "confidence":  confidence,
        "fraud_prob":  fraud_prob,
        "legit_prob":  legit_prob,
        "conf_level":  conf_level,
        "model_used":  info["label"],
        "model_acc":   info["accuracy"],
        "model_roc":   info["roc_auc"],
        "color":       "#f06060" if is_fraud else "#4caf82",
        "emoji":       "🚨" if is_fraud else "✅",
    }


# ── Routes ────────────────────────────────────────────────────────
@fraud_bp.route("/model3")
def page():
    return render_template(
        "model3.html",
        model_ready   = _ready,
        model_error   = _error,
        model_metrics = MODEL_METRICS,
        available     = list(_models.keys()),
        categories    = CATEGORIES,
        best_model    = "xgboost",
    )


@fraud_bp.route("/predict_fraud_single", methods=["POST"])
def predict_single():
    if not _ready:
        return jsonify({"error": f"Model not loaded: {_error}"}), 500
    data      = request.get_json()
    model_key = data.get("model", "xgboost")
    if model_key not in _models:
        model_key = next(iter(_models))
    try:
        X      = engineer_features(data)
        result = _run_predict(X, model_key)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@fraud_bp.route("/predict_fraud_csv", methods=["POST"])
def predict_csv():
    if not _ready:
        return jsonify({"error": f"Model not loaded: {_error}"}), 500
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file      = request.files["file"]
    model_key = request.form.get("model", "xgboost")
    if model_key not in _models:
        model_key = next(iter(_models))

    try:
        df    = pd.read_csv(file)
        rows  = []
        fraud_count = 0

        for _, row in df.iterrows():
            try:
                X      = engineer_features(row.to_dict())
                result = _run_predict(X, model_key)
                if result["is_fraud"]:
                    fraud_count += 1
                rows.append({
                    "row":        len(rows) + 1,
                    "amount":     round(float(row.get("amt", 0)), 2),
                    "merchant":   str(row.get("merchant", "—"))[:30],
                    "category":   str(row.get("category", "—")),
                    "prediction": result["prediction"],
                    "is_fraud":   result["is_fraud"],
                    "confidence": result["confidence"],
                    "emoji":      result["emoji"],
                })
            except Exception:
                continue
            if len(rows) >= 200:
                break

        total = len(rows)
        return jsonify({
            "total":       total,
            "fraud_count": fraud_count,
            "legit_count": total - fraud_count,
            "fraud_pct":   round(fraud_count / total * 100, 1) if total else 0,
            "model_used":  MODEL_METRICS[model_key]["label"],
            "model_acc":   MODEL_METRICS[model_key]["accuracy"],
            "rows":        rows,
        })
    except Exception as e:
        return jsonify({"error": f"CSV failed: {str(e)}"}), 500