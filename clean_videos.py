import os
import shutil
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

VIDEO_DIR = "ai_services/dataset/videos_dataset"
REVIEW_DIR = "ai_services/dataset/review/videos"

# Make sure review folder exists
os.makedirs(REVIEW_DIR, exist_ok=True)

# Number of frames to check from each video
FRAMES_TO_CHECK = 5

def extract_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()
        return []

    step = max(frame_count // num_frames, 1)
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        if len(frames) >= num_frames:
            break

    cap.release()
    return frames

def is_relevant_frame(frame, keyword, model):
    # Convert frame to PIL image format
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    labels = decode_predictions(preds, top=3)[0]

    # Check if any predicted label loosely matches keyword
    for _, label, _ in labels:
        if keyword.lower() in label.lower():
            return True
    return False

def clean_videos():
    model = mobilenet_v2.MobileNetV2(weights='imagenet')

    for department in os.listdir(VIDEO_DIR):
        dept_path = os.path.join(VIDEO_DIR, department)
        if not os.path.isdir(dept_path):
            continue

        for file in os.listdir(dept_path):
            if not file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue

            video_path = os.path.join(dept_path, file)
            frames = extract_frames(video_path, FRAMES_TO_CHECK)

            if not frames:
                print(f"⚠️ Corrupt video: {file}")
                shutil.move(video_path, os.path.join(REVIEW_DIR, file))
                continue

            relevant = any(is_relevant_frame(frame, department, model) for frame in frames)

            if not relevant:
                print(f"🚩 Moved to review: {file} (from {department})")
                shutil.move(video_path, os.path.join(REVIEW_DIR, file))
            else:
                print(f"✅ Relevant: {file} ({department})")

    print("\n🎯 Cleaning done. Irrelevant videos moved to review folder.")

if __name__ == "__main__":
    clean_videos()
