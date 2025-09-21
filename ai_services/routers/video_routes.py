import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from fastapi import APIRouter, UploadFile, File
from ai_services.inferencehelper.video_service import predict_video

router = APIRouter(prefix="/video", tags=["Video"])

@router.post("/predict")
async def predict_video_route(file: UploadFile = File(...)):
    video_bytes = await file.read()
    prediction = predict_video(video_bytes)
    return {"predicted_department": prediction}
