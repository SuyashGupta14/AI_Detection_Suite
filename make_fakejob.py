import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "Work from home, earn /week, no experience needed, unlimited income, apply now!!",
    "Data Entry Clerk needed immediately. Send fee for background check. Huge salary!",
    "Earn easy money working 1 hour a day. Guaranteed income!",
    "No interview needed! Just send us your bank details for direct deposit setup.",
    "Wire transfer supervisor required. Make money quickly.",
    "Looking for a software engineer. 3+ years experience. Office location: New York.",
    "We are hiring a backend developer. Required skills: Python, SQL, Docker.",
    "Senior UX Designer needed. Must have strong portfolio and 5 years experience.",
    "Hiring a full-time mechanical engineer. Salary commensurate with experience.",
    "Nursing position open at St. Jude Hospital. Must have RN license."
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] # 1=Fake, 0=Real

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

joblib.dump(model, 'ML_Models/Fake_Job_Postings/fake_job_model.pkl')
joblib.dump(vectorizer, 'ML_Models/Fake_Job_Postings/tfidf_vectorizer.pkl')
print("Improved Fake Job Model created")
