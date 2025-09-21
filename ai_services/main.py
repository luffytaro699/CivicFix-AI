# ai_services/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from collections import Counter

# Correct imports
from ai_services.inferencehelper.text_service import predict_text
from ai_services.inferencehelper.image_service import predict_image_from_bytes
from ai_services.inferencehelper.audio_service import predict_audio
from ai_services.inferencehelper.video_service import predict_video

MIN_AGREEMENT = 2  # Minimum modalities agreeing

app = FastAPI()

def get_confident_departments(predictions_proba, threshold=0.3):
    return [label for label, prob in predictions_proba if prob >= threshold]

@app.post("/predict_all")
async def predict_all(
    description: str = Form(...),
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    video: UploadFile = File(...)
):
    # Text
    text_proba = predict_text(description, return_proba=True)
    text_pred = get_confident_departments(text_proba)

    # Image
    image_bytes = await image.read()
    image_proba = predict_image_from_bytes(image_bytes, return_proba=True)
    image_pred = get_confident_departments(image_proba)

    # Audio
    audio_bytes = await audio.read()
    audio_proba = predict_audio(audio_bytes, return_proba=True)
    audio_pred = get_confident_departments(audio_proba)

    # Video
    video_bytes = await video.read()
    video_proba = predict_video(video_bytes, return_proba=True)
    video_pred = get_confident_departments(video_proba)

    # Aggregate
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
    else:
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
