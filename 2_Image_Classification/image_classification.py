"""
Image Classification using CNN
Dataset: Intel Image Classification (Kaggle)
Classes: buildings, forest, glacier, mountain, sea, street
Author: CNN Projects Portfolio
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_DIR   = "data/seg_train/seg_train"
TEST_DIR    = "data/seg_test/seg_test"
IMG_SIZE    = 150
BATCH_SIZE  = 32
EPOCHS      = 25
LEARNING_RATE = 1e-4
SEED        = 42
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
# 1. DATA GENERATORS
# ─────────────────────────────────────────────
def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.1
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=SEED
    )
    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        seed=SEED
    )
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────
# 2. BUILD MODEL (Transfer Learning — ResNet50)
# ─────────────────────────────────────────────
def build_model(num_classes):
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    # Unfreeze last 20 layers for fine-tuning
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
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
# 3. TRAIN
# ─────────────────────────────────────────────
def train_model(model, train_gen, val_gen):
    cb_list = [
        callbacks.EarlyStopping(patience=6, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        callbacks.ModelCheckpoint("image_classification_best.h5", save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=cb_list,
        verbose=1
    )
    return history


# ─────────────────────────────────────────────
# 4. EVALUATE
# ─────────────────────────────────────────────
def evaluate_model(model, test_gen):
    loss, acc = model.evaluate(test_gen, verbose=0)
    print(f"\n✅ Test Accuracy: {acc*100:.2f}%  |  Test Loss: {loss:.4f}")

    test_gen.reset()
    y_pred = np.argmax(model.predict(test_gen), axis=1)
    y_true = test_gen.classes
    labels = list(test_gen.class_indices.keys())

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix — Image Classification", fontsize=14)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# 5. VISUALIZE PREDICTIONS
# ─────────────────────────────────────────────
def visualize_predictions(model, test_gen):
    test_gen.reset()
    X_batch, y_batch = next(test_gen)
    preds = np.argmax(model.predict(X_batch), axis=1)
    labels = list(test_gen.class_indices.keys())

    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    for i, ax in enumerate(axes.flat):
        if i >= len(X_batch):
            break
        ax.imshow(X_batch[i])
        true_lbl = labels[np.argmax(y_batch[i])]
        pred_lbl = labels[preds[i]]
        color = "green" if true_lbl == pred_lbl else "red"
        ax.set_title(f"T: {true_lbl}\nP: {pred_lbl}", color=color, fontsize=9)
        ax.axis("off")

    plt.suptitle("Sample Predictions — Image Classification", fontsize=14)
    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# 6. PLOT TRAINING CURVES
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

    plt.suptitle("Image Classification — Training History", fontsize=14)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  IMAGE CLASSIFICATION USING CNN (Transfer Learning — ResNet50)")
    print("=" * 60)

    train_gen, val_gen, test_gen = get_generators()
    model = build_model(num_classes=len(CLASS_NAMES))

    history = train_model(model, train_gen, val_gen)

    plot_history(history)
    evaluate_model(model, test_gen)
    visualize_predictions(model, test_gen)

    model.save("image_classification_final.h5")
    print("\n✅ Model saved to image_classification_final.h5")
