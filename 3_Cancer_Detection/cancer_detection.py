"""
Cancer Detection using CNN
Dataset: Histopathologic Cancer Detection (Kaggle)
Task: Binary Classification — Cancerous vs Non-Cancerous Tissue
Author: CNN Projects Portfolio
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR       = "data"
TRAIN_DIR      = os.path.join(DATA_DIR, "train")
LABELS_CSV     = os.path.join(DATA_DIR, "train_labels.csv")
IMG_SIZE       = 96
BATCH_SIZE     = 64
EPOCHS         = 20
LEARNING_RATE  = 1e-4
USE_SUBSET     = True      # Set to False to train on full dataset
SUBSET_FRAC    = 0.20      # Use 20% of data for quick experiments
SEED           = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
# 1. LOAD & PREPARE CSV
# ─────────────────────────────────────────────
def prepare_dataframes():
    df = pd.read_csv(LABELS_CSV)
    df["filename"] = df["id"] + ".tif"
    df["label"]    = df["label"].astype(str)

    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")

    if USE_SUBSET:
        df, _ = train_test_split(
            df, test_size=1 - SUBSET_FRAC, stratify=df["label"], random_state=SEED
        )
        print(f"Using subset: {len(df)} samples ({SUBSET_FRAC*100:.0f}%)")

    train_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=SEED
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, stratify=train_df["label"], random_state=SEED
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


# ─────────────────────────────────────────────
# 2. DATA GENERATORS
# ─────────────────────────────────────────────
def get_generators(train_df, val_df, test_df):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=180,   # Histology images are rotation-invariant
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode="reflect"
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, directory=TRAIN_DIR,
        x_col="filename", y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="binary", seed=SEED
    )
    val_gen = val_test_datagen.flow_from_dataframe(
        val_df, directory=TRAIN_DIR,
        x_col="filename", y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="binary", shuffle=False
    )
    test_gen = val_test_datagen.flow_from_dataframe(
        test_df, directory=TRAIN_DIR,
        x_col="filename", y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="binary", shuffle=False
    )

    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────
# 3. BUILD MODEL (Transfer Learning — EfficientNetB0)
# ─────────────────────────────────────────────
def build_model():
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    # Fine-tune top 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.5)(x)
    x       = layers.Dense(64, activation="relu")(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    model.summary()
    return model


# ─────────────────────────────────────────────
# 4. COMPUTE CLASS WEIGHTS
# ─────────────────────────────────────────────
def get_class_weights(train_df):
    labels = train_df["label"].astype(int).values
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    return {0: weights[0], 1: weights[1]}


# ─────────────────────────────────────────────
# 5. TRAIN
# ─────────────────────────────────────────────
def train_model(model, train_gen, val_gen, class_weights):
    cb_list = [
        callbacks.EarlyStopping(monitor="val_auc", patience=5,
                                 restore_best_weights=True, mode="max", verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_auc", factor=0.5,
                                     patience=3, mode="max", verbose=1),
        callbacks.ModelCheckpoint("cancer_detection_best.h5",
                                   monitor="val_auc", save_best_only=True,
                                   mode="max", verbose=1)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=1
    )
    return history


# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────
def evaluate_model(model, test_gen):
    results = model.evaluate(test_gen, verbose=0)
    print(f"\n✅ Test Accuracy: {results[1]*100:.2f}%  |  Test AUC: {results[2]:.4f}")

    test_gen.reset()
    y_prob = model.predict(test_gen).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = test_gen.labels.astype(int)

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Non-Cancerous", "Cancerous"]))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Non-Cancerous", "Cancerous"],
                yticklabels=["Non-Cancerous", "Cancerous"])
    plt.title("Confusion Matrix — Cancer Detection", fontsize=14)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="crimson", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Cancer Detection")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# 7. PLOT TRAINING CURVES
# ─────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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

    axes[2].plot(history.history["auc"], label="Train")
    axes[2].plot(history.history["val_auc"], label="Validation")
    axes[2].set_title("AUC Score")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.suptitle("Cancer Detection — Training History", fontsize=14)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CANCER DETECTION USING CNN (Transfer Learning — EfficientNetB0)")
    print("=" * 60)

    train_df, val_df, test_df = prepare_dataframes()
    train_gen, val_gen, test_gen = get_generators(train_df, val_df, test_df)

    class_weights = get_class_weights(train_df)
    print(f"\nClass weights: {class_weights}")

    model   = build_model()
    history = train_model(model, train_gen, val_gen, class_weights)

    plot_history(history)
    evaluate_model(model, test_gen)

    model.save("cancer_detection_final.h5")
    print("\n✅ Model saved to cancer_detection_final.h5")
