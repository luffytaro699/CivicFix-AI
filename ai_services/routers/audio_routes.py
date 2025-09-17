import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from fastapi import APIRouter, UploadFile, File, HTTPException
from ai_services.services.audio_model import transcribe_audio
from ai_services.inferencehelper.audio_service import classify_text

audio_router = APIRouter()

@audio_router.post("/classify_audio")
async def classify_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded audio temporarily
        os.makedirs("temp", exist_ok=True)
        file_location = f"temp/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Step 1: Transcribe audio to text
        result = transcribe_audio(file_location)
        detected_language = result["language"]
        transcribed_text = result["text"]

        # Step 2: Classify transcribed text using text model
        predicted_department = classify_text(transcribed_text)

        # Cleanup temp file
        os.remove(file_location)

        return {
            "detected_language": detected_language,
            "transcribed_text": transcribed_text,
            "predicted_department": predicted_department
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
