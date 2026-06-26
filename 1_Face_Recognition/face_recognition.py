"""
Face Recognition using CNN
Dataset: Celebrity Face Image Dataset (Kaggle)
Author: CNN Projects Portfolio
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = "data/Celebrity Faces Dataset"
IMG_SIZE    = 128
BATCH_SIZE  = 32
EPOCHS      = 30
LEARNING_RATE = 1e-4
SEED        = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(data_dir):
    images, labels = [], []
    class_names = sorted(os.listdir(data_dir))
    
    print(f"Found {len(class_names)} classes: {class_names}\n")
    
    for label in tqdm(class_names, desc="Loading images"):
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels), class_names


# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────
def preprocess(images, labels, class_names):
    X = images.astype("float32") / 255.0
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    y = to_categorical(y_encoded, num_classes=len(class_names))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=SEED
    )
    
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test, le


# ─────────────────────────────────────────────
# 3. DATA AUGMENTATION
# ─────────────────────────────────────────────
def get_augmentation():
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )


# ─────────────────────────────────────────────
# 4. BUILD MODEL (Transfer Learning — VGG16)
# ─────────────────────────────────────────────
def build_model(num_classes):
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    # Freeze base layers
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


# ─────────────────────────────────────────────
# 5. TRAIN
# ─────────────────────────────────────────────
def train_model(model, X_train, y_train, X_val, y_val, datagen):
    cb_list = [
        callbacks.EarlyStopping(patience=7, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
        callbacks.ModelCheckpoint("face_recognition_best.h5", save_best_only=True, verbose=1)
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=cb_list,
        verbose=1
    )
    return history


# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, class_names, le):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Test Accuracy: {acc*100:.2f}%  |  Test Loss: {loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix — Face Recognition", fontsize=14)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# 7. PLOT TRAINING CURVES
# ─────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.suptitle("Face Recognition — Training History", fontsize=14)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  FACE RECOGNITION USING CNN (Transfer Learning — VGG16)")
    print("=" * 60)

    images, labels, class_names = load_data(DATA_DIR)
    X_train, X_val, X_test, y_train, y_val, y_test, le = preprocess(images, labels, class_names)

    datagen = get_augmentation()
    model   = build_model(num_classes=len(class_names))

    history = train_model(model, X_train, y_train, X_val, y_val, datagen)

    plot_history(history)
    evaluate_model(model, X_test, y_test, class_names, le)

    model.save("face_recognition_final.h5")
    print("\n✅ Model saved to face_recognition_final.h5")
