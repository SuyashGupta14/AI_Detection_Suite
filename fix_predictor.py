import re
with open('ML_Models/Deepfake_audio_detection/predictor.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = re.sub(
    r'(?ms)model_type_name = type\(_model\)\.__name__.*?if model_type_name == "LogisticRegression":[^}]*?raise ValueError\([\s\S]*?\)',
    '',
    text
)

with open('ML_Models/Deepfake_audio_detection/predictor.py', 'w', encoding='utf-8') as f:
    f.write(text)
