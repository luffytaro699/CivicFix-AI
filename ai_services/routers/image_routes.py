# ai_services/routers/image_routes.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from fastapi import APIRouter, UploadFile, File, HTTPException
from ai_services.inferencehelper.image_service import predict_image_from_bytes

router = APIRouter()

@router.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    data = await file.read()
    try:
        out = predict_image_from_bytes(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return out
