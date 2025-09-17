import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ----- Config -----
DATASET_DIR = "ai_services/dataset/videos_dataset"
FRAME_SIZE = (128, 128)
FRAMES_PER_VIDEO = 10
EPOCHS = 10
BATCH_SIZE = 16

# ----- Extract frames from each video -----
def load_videos():
    X, y, labels = [], [], []
    class_names = sorted(os.listdir(DATASET_DIR))

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        labels.append(class_name)

        for video_name in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_name)
            cap = cv2.VideoCapture(video_path)
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

            if len(frames) == FRAMES_PER_VIDEO:
                avg_frame = np.mean(frames, axis=0)
                X.append(avg_frame)
                y.append(idx)

    return np.array(X), np.array(y), labels

print("[INFO] Loading videos...")
X, y, LABELS = load_videos()
X = X / 255.0
y = to_categorical(y, num_classes=len(LABELS))

# ----- Split -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- Model -----
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(len(LABELS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----- Train -----
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

# ----- Save -----
model.save("ai_services/models/video_model.h5")
with open("ai_services/models/video_labels.txt", "w") as f:
    f.write("\n".join(LABELS))

print("[INFO] Training complete and model saved.")
