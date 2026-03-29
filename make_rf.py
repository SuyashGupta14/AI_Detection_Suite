import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load existing scaler to get the right feature dimension
scaler = joblib.load('ML_Models/Deepfake_audio_detection/scaler.pkl')
n_features = scaler.mean_.shape[0]

# Generate some dummy data
X_dummy = np.random.randn(100, n_features)
y_dummy = np.random.randint(0, 2, 100)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_dummy, y_dummy)

# Save it to best_model.pkl
joblib.dump(rf, 'ML_Models/Deepfake_audio_detection/best_model.pkl')
print("Dummy RF created successfully.")
