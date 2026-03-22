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

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = str(BASE / "static" / "temp")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ── Register blueprints — one per model ──────────────────────────
app.register_blueprint(video_bp)        # /model1   /predict
app.register_blueprint(satellite_bp)    # /model2   /predict_satellite
app.register_blueprint(fraud_bp)        # /model3   /predict_fraud_single  /predict_fraud_csv
# app.register_blueprint(model4_bp)
# app.register_blueprint(model5_bp)

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