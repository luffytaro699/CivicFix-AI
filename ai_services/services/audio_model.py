import whisper
import os

# Load the Whisper model (multilingual + accent-robust)
model = whisper.load_model("small")  # or "base" for faster, "medium" for better

def transcribe_audio(file_path: str) -> dict:
    """
    Transcribes audio from any language/accent into text.
    Returns both detected language and text.
    """
    result = model.transcribe(file_path)
    return {
        "language": result.get("language", "unknown"),
        "text": result["text"]
    }
