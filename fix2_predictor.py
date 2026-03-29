import re

with open('ML_Models/Deepfake_audio_detection/predictor.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace _predict_audio logic to be identical to notebook
new_predict_audio = '''def _predict_audio(file_path: str) -> dict:
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
    }'''

text = re.sub(r'def _predict_audio\(file_path: str\) -> dict:.*?(?=\n\n(?:@deepfake_bp\.route|#|$))', new_predict_audio, text, flags=re.DOTALL)

with open('ML_Models/Deepfake_audio_detection/predictor.py', 'w', encoding='utf-8') as f:
    f.write(text)
