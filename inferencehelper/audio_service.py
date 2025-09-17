# ai_services/services/audio_service.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import io
import speech_recognition as sr
from ai_services.inferencehelper.text_service import predict_text as classify_text

# Confidence threshold (optional, for filtering low-prob departments)
CONFIDENCE_THRESHOLD = 0.3

import io
import speech_recognition as sr
from pydub import AudioSegment  # <-- install pydub

def transcribe_audio(audio_bytes: bytes) -> str:
    # Convert MP3/any to WAV (PCM) using pydub
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)

    return text


def predict_audio(audio_bytes: bytes, return_proba=False):
    """
    Transcribes audio and predicts department(s) using text model.
    
    Parameters:
    - audio_bytes: raw audio bytes
    - return_proba: if True, returns list of (department, probability)
    
    Returns:
    - If return_proba=False: dict {"department": top_pred, "confidence": prob}
    - If return_proba=True: list of (department, probability)
    """
    transcript = transcribe_audio(audio_bytes)
    
    if not transcript:
        if return_proba:
            return []
        else:
            return {"department": "Unknown", "confidence": 0.0}
    
    # Use the text classifier for predictions
    return classify_text(transcript, return_proba=return_proba)
