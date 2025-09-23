# ai_services/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Optional
from collections import Counter
import httpx

# Correct imports
from ai_services.inferencehelper.text_service import predict_text
from ai_services.inferencehelper.image_service import predict_image_from_bytes
from ai_services.inferencehelper.audio_service import predict_audio
from ai_services.inferencehelper.video_service import predict_video

MIN_AGREEMENT = 1  # Minimum modalities agreeing

app = FastAPI()

def get_confident_departments(predictions_proba, threshold=0.3):
    return [label for label, prob in predictions_proba if prob >= threshold]

async def fetch_bytes(url: str):
    """Fetch content from a URL as bytes."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content

@app.post("/predict_all")
async def predict_all(
    description: str = Form(...),
    image_file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
    video_file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    audio_url: Optional[str] = Form(None),
    video_url: Optional[str] = Form(None)
):
    preds = {}

    # Text (always required)
    text_proba = predict_text(description, return_proba=True)
    preds["description_pred"] = get_confident_departments(text_proba)

    # Image (file or URL)
    image_bytes = None
    if image_file:
        image_bytes = await image_file.read()
    elif image_url:
        image_bytes = await fetch_bytes(image_url)
    
    if image_bytes:
        image_proba = predict_image_from_bytes(image_bytes, return_proba=True)
        preds["image_pred"] = get_confident_departments(image_proba)
    else:
        preds["image_pred"] = []

    # Audio (file or URL)
    audio_bytes = None
    if audio_file:
        audio_bytes = await audio_file.read()
    elif audio_url:
        audio_bytes = await fetch_bytes(audio_url)
    
    if audio_bytes:
        audio_proba = predict_audio(audio_bytes, return_proba=True)
        preds["audio_pred"] = get_confident_departments(audio_proba)
    else:
        preds["audio_pred"] = []

    # Video (file or URL)
    video_bytes = None
    if video_file:
        video_bytes = await video_file.read()
    elif video_url:
        video_bytes = await fetch_bytes(video_url)
    
    if video_bytes:
        video_proba = predict_video(video_bytes, return_proba=True)
        preds["video_pred"] = get_confident_departments(video_proba)
    else:
        preds["video_pred"] = []

    # Aggregate predictions across available modalities
    all_preds = preds["description_pred"] + preds["image_pred"] + preds["audio_pred"] + preds["video_pred"]
    counts = Counter(all_preds)
    final_departments = [dept for dept, cnt in counts.items() if cnt >= MIN_AGREEMENT]

    if final_departments:
        return {
            "status": "success",
            **preds,
            "final_departments": final_departments
        }
    else:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Mismatch between predictions. Please check inputs.",
                **preds
            }
        )
