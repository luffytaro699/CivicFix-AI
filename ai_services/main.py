# ai_services/main.py
from fastapi import FastAPI, Form, HTTPException
import requests
from collections import Counter

# Correct imports from your inference helper
from ai_services.inferencehelper.text_service import predict_text
from ai_services.inferencehelper.image_service import predict_image_from_bytes
from ai_services.inferencehelper.audio_service import predict_audio
from ai_services.inferencehelper.video_service import predict_video

MIN_AGREEMENT = 2  # Minimum modalities agreeing

app = FastAPI()

def get_confident_departments(predictions_proba, threshold=0.3):
    """Filter predictions above threshold"""
    return [label for label, prob in predictions_proba if prob >= threshold]

def download_file(url):
    """Download a file from a URL and return raw bytes"""
    if not url:
        return None
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.content  # return bytes directly
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

@app.post("/predict_all")
async def predict_all(
    description: str = Form(...),
    image_url: str = Form(None),
    audio_url: str = Form(None),
    video_url: str = Form(None)
):
    # 1️⃣ Text prediction
    text_proba = predict_text(description, return_proba=True)
    text_pred = get_confident_departments(text_proba)

    # 2️⃣ Image prediction
    image_bytes = download_file(image_url)
    image_pred = []
    if image_bytes:
        image_proba = predict_image_from_bytes(image_bytes, return_proba=True)
        image_pred = get_confident_departments(image_proba)

    # 3️⃣ Audio prediction
    audio_bytes = download_file(audio_url)
    audio_pred = []
    if audio_bytes:
        audio_proba = predict_audio(audio_bytes, return_proba=True)
        audio_pred = get_confident_departments(audio_proba)

    # 4️⃣ Video prediction
    video_bytes = download_file(video_url)
    video_pred = []
    if video_bytes:
        video_proba = predict_video(video_bytes, return_proba=True)
        video_pred = get_confident_departments(video_proba)

    # 5️⃣ Aggregate predictions
    all_preds = text_pred + image_pred + audio_pred + video_pred
    counts = Counter(all_preds)
    final_departments = [dept for dept, cnt in counts.items() if cnt >= MIN_AGREEMENT]

    if final_departments:
        return {
            "status": "success",
            "description_pred": text_pred,
            "image_pred": image_pred,
            "audio_pred": audio_pred,
            "video_pred": video_pred,
            "final_departments": final_departments
        }

    # If no consensus
    raise HTTPException(
        status_code=400,
        detail={
            "message": "Mismatch between predictions. Please check inputs.",
            "description_pred": text_pred,
            "image_pred": image_pred,
            "audio_pred": audio_pred,
            "video_pred": video_pred
        }
    )
