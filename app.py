"""
app.py — AI Detection Suite
Run:  python app.py
URL:  http://127.0.0.1:5000
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template

BASE = Path(__file__).resolve().parent

# Make ML_Models importable
sys.path.insert(0, str(BASE / "ML_Models"))

from Real_AI_Video.predictor       import video_bp
from SatelliteClassifier.predictor import satellite_bp
from CC_Fraud_Det_model.predictor  import fraud_bp      # ← correct folder name
from Deepfake_audio_detection.predictor import deepfake_bp
from Fake_Job_Postings.predictor import fake_job_bp

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = str(BASE / "static" / "temp")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ── Register blueprints — one per model ──────────────────────────
app.register_blueprint(video_bp)        # /model1   /predict
app.register_blueprint(satellite_bp)    # /model2   /predict_satellite
app.register_blueprint(fraud_bp)        # /model3   /predict_fraud_single  /predict_fraud_csv
app.register_blueprint(deepfake_bp)     # /model4   /predict_audio_deepfake
app.register_blueprint(fake_job_bp)     # /model5   /predict_fake_job  /predict_fake_job_csv

@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    try:
        from waitress import serve
        print("=" * 45)
        print("  AI Detection Suite")
        print("  http://127.0.0.1:5000")
        print("  Press CTRL+C to stop")
        print("=" * 45)
        serve(app, host="127.0.0.1", port=5000, threads=4)
    except ImportError:
        print("Waitress not found — install it:  pip install waitress")
        print("Using Flask dev server instead...")
        app.run(host="127.0.0.1", port=5000, debug=False)