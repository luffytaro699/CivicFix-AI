from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib

# Load the trained model and vectorizer
try:
    model = joblib.load("ai_services/models/dept_model.pkl")
    vectorizer = joblib.load("ai_services/models/vectorizer.pkl")
except FileNotFoundError:
    raise RuntimeError("Model or vectorizer file not found. Please train the model first.")

# Create router
router = APIRouter(prefix="/predict/text", tags=["Text Prediction"])

# Define the confidence threshold
CONFIDENCE_THRESHOLD = 0.6  # You can fine-tune this value based on your model's performance

# Define request body format
class ComplaintRequest(BaseModel):
    complaint: str

@router.post("/")
def predict_department(data: ComplaintRequest):
    """
    Predicts the department for a given complaint and provides a confidence score.
    Returns an error if the prediction confidence is below a set threshold.
    """
    try:
        # Convert complaint into a vector
        X = vectorizer.transform([data.complaint])

        # Get prediction probabilities and the highest probability
        probabilities = model.predict_proba(X)
        max_prob = probabilities.max()

        # Get the predicted department
        prediction = model.predict(X)[0]

        # Check if the confidence is below the threshold
        if max_prob < CONFIDENCE_THRESHOLD:
            return {
                "error": "The complaint could not be categorized with high confidence.",
                "confidence": float(max_prob)
            }

        # Return the prediction and confidence score
        return {
            "complaint": data.complaint,
            "predicted_department": prediction,
            "confidence": float(max_prob)
        }

    except Exception as e:
        # Handle potential errors during the prediction process
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")