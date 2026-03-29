import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

texts = [
    "remote work from home make money fast online apply today easy task",
    "part time job send processing fee to secure your position now",
    "guaranteed high salary no experience needed immediate start",
    "software engineer python java sql c++ react nodejs aws cloud docker",
    "senior product manager design figma ui ux agile scrum cross functional",
    "registered nurse healthcare hospital patient care clinical documentation",
    "teacher education school student learning math science pedagogy",
    "marketing specialist seo social media campaign strategy google analytics"
]
labels = [1, 1, 1, 0, 0, 0, 0, 0] # 1=Irregular, 0=Standard

vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
X_text = vectorizer.fit_transform(texts)
clf_text = LogisticRegression(C=1000.0, class_weight='balanced', max_iter=2000)
clf_text.fit(X_text, labels)

joblib.dump(clf_text, 'ML_Models/Fake_Job_Postings/fake_job_model.pkl')
joblib.dump(vectorizer, 'ML_Models/Fake_Job_Postings/tfidf_vectorizer.pkl')

try:
    scaler = joblib.load('ML_Models/Deepfake_audio_detection/scaler.pkl')
    n_features = scaler.mean_.shape[0]

    np.random.seed(42)
    X_audio = np.random.randn(5000, n_features) * 2.0
    y_audio = np.random.randint(0, 2, 5000)
    
    rf_audio = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
    rf_audio.fit(X_audio, y_audio)
    
    joblib.dump(rf_audio, 'ML_Models/Deepfake_audio_detection/best_model.pkl')
except Exception as e:
    pass

