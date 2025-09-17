# ai_services/services/image_service.py
import json, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as kimage
from PIL import Image
import io

MODEL_PATH = "ai_services/models/image_model.h5"
LABELS_PATH = "ai_services/models/image_labels.json"

# Load model and label map
_image_model = load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    inv_map = json.load(f)  # index -> label

# Confidence threshold for multi-department prediction
CONFIDENCE_THRESHOLD = 0.3  # Adjust as needed

def predict_image_from_pil(pil_image, return_proba=False):
    img = pil_image.resize((224, 224))
    arr = np.array(img).astype(np.float32)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    x = np.expand_dims(arr, 0)
    x = preprocess_input(x)
    
    preds = _image_model.predict(x)[0]  # probabilities for all classes

    if return_proba:
        # Return all labels with their probabilities
        results = [(inv_map.get(str(i), str(i)), float(prob)) for i, prob in enumerate(preds)]
        return results
    else:
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        label = inv_map.get(str(idx), str(idx))
        return {"department": label, "confidence": conf}

# Helper to accept raw bytes
def predict_image_from_bytes(b, return_proba=False):
    pil = Image.open(io.BytesIO(b)).convert("RGB")
    return predict_image_from_pil(pil, return_proba=return_proba)
