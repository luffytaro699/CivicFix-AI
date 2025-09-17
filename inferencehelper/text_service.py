# ai_services/services/text_service.py
import joblib, os, re

VECT_PATH = "ai_services/models/vectorizer.pkl"
MODEL_PATH = "ai_services/models/dept_model.pkl"

# Load vectorizer and model
vect = joblib.load(VECT_PATH)
model = joblib.load(MODEL_PATH)

# Confidence threshold for multi-department prediction
CONFIDENCE_THRESHOLD = 0.3  # Adjust as needed

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_text(text, return_proba=False):
    """
    Predicts department(s) for given text.
    
    Parameters:
    - text: str, user complaint text
    - return_proba: bool, if True returns list of (department, probability)
    
    Returns:
    - If return_proba=False: dict {"department": pred, "confidence": prob}
    - If return_proba=True: list of tuples [(department, probability), ...]
    """
    cleaned = clean_text(text)
    X = vect.transform([cleaned])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        labels = model.classes_
        if return_proba:
            return list(zip(labels, probs))
        else:
            # Return only the top prediction
            idx = probs.argmax()
            return {"department": labels[idx], "confidence": float(probs[idx])}
    else:
        pred = model.predict(X)[0]
        return {"department": pred, "confidence": None}
