# ai_services/train/train_image.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import json

TRAIN_DIR = "ai_services/dataset/images_dataset"
MODEL_OUT = "ai_services/models/image_model.h5"
LABELS_OUT = "ai_services/models/image_labels.json"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 6

def build_and_train():
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                 validation_split=0.15,
                                 horizontal_flip=True,
                                 rotation_range=10,
                                 zoom_range=0.1)

    train_gen = datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", subset="training"
    )
    val_gen = datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", subset="validation"
    )

    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(train_gen.num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=outputs)

    model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="val_accuracy", mode="max"),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

    # save labels mapping
    mapping = train_gen.class_indices  # name -> index
    inv_map = {v: k for k, v in mapping.items()}
    with open(LABELS_OUT, "w") as f:
        json.dump(inv_map, f, indent=2)
    print("Saved image model to", MODEL_OUT)
    print("Saved label map to", LABELS_OUT)

if __name__ == "__main__":
    build_and_train()
