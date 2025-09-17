# ai_services/services/video_service.py
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os

FRAME_SIZE = (128, 128)
FRAMES_PER_VIDEO = 10
CONFIDENCE_THRESHOLD = 0.3  # for multi-department predictions

# Load model and labels
model = tf.keras.models.load_model("ai_services/models/video_model.h5")
with open("ai_services/models/video_labels.txt") as f:
    LABELS = f.read().splitlines()

def predict_video(video_bytes: bytes, return_proba=False):
    """
    Predict department(s) from a video
    - video_bytes: raw video bytes
    - return_proba: if True, returns list of (department, probability)
    """
    # Save video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, FRAMES_PER_VIDEO, dtype=int)

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, FRAME_SIZE)
            frames.append(frame)
    cap.release()
    os.remove(tmp_path)

    if len(frames) == 0:
        return "Invalid Video" if not return_proba else []

    # Average frames for prediction
    avg_frame = np.mean(frames, axis=0)
    avg_frame = np.expand_dims(avg_frame / 255.0, axis=0)

    preds = model.predict(avg_frame)[0]  # probabilities for all classes

    if return_proba:
        return [(LABELS[i], float(prob)) for i, prob in enumerate(preds)]
    else:
        idx = int(np.argmax(preds))
        return {"department": LABELS[idx], "confidence": float(preds[idx])}
