# ******************************************************
# ===== 1. IMPORT LIBRARIES =====
# ******************************************************
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, 
    GlobalAveragePooling2D, Input, SpatialDropout2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from global_variables import (
    DATASET_DIR, CATEGORIES, NUM_CLASSES, SPLITED_SAVED_PATH,
    TEST_SPLIT, VAL_SPLIT, IMAGE_SIZE, BATCH_SIZE, EPOCHS, 
    LEARNING_RATE, L2_REG, EARLY_STOPPING_PATIENCE, REDUCE_LR_FACTOR, 
    REDUCE_LR_PATIENCE, MIN_LR, GRAPH_DIR, CLASS_INDEX_FILE, MODEL_DIR,
    BEST_MODEL_PATH, TRAINED_MODEL_PATH, 
    ROTATION_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE, ZOOM_RANGE, HORIZONTAL_FLIP, FILL_MODE, 
    LOSS_FUNCTION,
    start_partition, end_partition,
    download_or_read_image, process_single_image
)


# -----------------------------------------------------------
def check_GPU():
    import tensorflow as tf1
    from tensorflow.keras import mixed_precision
    print("Num GPUs Available: ", len(tf1.config.list_physical_devices('GPU')))
    # Enable FP16 execution
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

check_GPU()
# -----------------------------------------------------------



# ******************************************************
# ===== 2. LOAD & PREPROCESS DATASET =====
# ******************************************************
start_partition("LOADING DATASET")
data, labels = [], []
corrupt_count = 0

for label_idx, category in enumerate(CATEGORIES):
    folder_path = os.path.join(DATASET_DIR, category)
    if not os.path.exists(folder_path):
        print(f"Directory '{folder_path}' missing. Skipping.")
        continue
        
    print(f"Loading category: '{category}'...")
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img_bgr = download_or_read_image(img_path)
            _, model_input = process_single_image(img_bgr)
            data.append(model_input)
            labels.append(label_idx)
        except Exception:
            corrupt_count += 1

X = np.array(data, dtype="float32")
y = np.array(labels, dtype="int32")

print(f"Total valid samples loaded: {len(X)}")
print(f"Corrupt/skipped files: {corrupt_count}")
end_partition()



# ******************************************************
# ===== 3. STRATIFIED SPLITS & CLASS WEIGHTS =====
# ******************************************************
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=(TEST_SPLIT + VAL_SPLIT), random_state=42, stratify=y
)

val_ratio_adjusted = VAL_SPLIT / (TEST_SPLIT + VAL_SPLIT)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=(1.0 - val_ratio_adjusted), random_state=42, stratify=y_temp
)

class_weights_arr = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights_arr)}

start_partition("SPLIT & BIAS BALANCING SUMMARY")
print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
for idx, cat in enumerate(CATEGORIES):
    print(f"Class [{cat}] weight: {class_weight_dict.get(idx, 1.0):.4f}")
end_partition()

y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_val_cat = to_categorical(y_val, NUM_CLASSES)

start_partition("DATASET SIZE SUMMARY")
print(f"Train Dataset Size: {len(X_train)} Images")
print(f"Validation Dataset Size: {len(X_val)} Images")
print(f"Test Dataset Size: {len(X_test)} Images")
end_partition()



# ******************************************************
# ===== 4. REAL-TIME DATA AUGMENTATION =====
# ******************************************************
train_datagen = ImageDataGenerator(
    rotation_range = ROTATION_RANGE,
    width_shift_range = WIDTH_SHIFT_RANGE,
    height_shift_range = HEIGHT_SHIFT_RANGE,
    zoom_range = ZOOM_RANGE,
    horizontal_flip = HORIZONTAL_FLIP,
    fill_mode = FILL_MODE
)



# ******************************************************
# ===== 5. CUSTOM 5-BLOCK CNN ARCHITECTURE =====
# ******************************************************
def build_cnn_model():
    model = Sequential([
        Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),

        # Block 1
        Conv2D(32, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.1),

        # Block 2
        Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.15),

        # Block 3
        Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.2),

        # Block 4
        Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.25),

        # Block 5
        Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.3),

        # Head Layer (Global Average Pooling to eliminate overfitting)
        GlobalAveragePooling2D(),
        Dense(128, activation="relu", kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    return model

model = build_cnn_model()
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=LOSS_FUNCTION,
    metrics=["accuracy", Precision(name="precision"), Recall(name="recall")]
)

start_partition("MODEL SUMMARY")
model.summary()
end_partition()



# ******************************************************
# ===== 6. TRAINING WITH CALLBACKS =====
# ******************************************************
callbacks = [
    EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
    ModelCheckpoint(BEST_MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR, verbose=1)
]

start_partition("TRAINING EXECUTION")
history = model.fit(
    train_datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_val, y_val_cat),
    class_weight=class_weight_dict,
    callbacks=callbacks
)
end_partition()



# ******************************************************
# ===== 7. EXPORT ARTIFACTS & TEST DATA =====
# ******************************************************
best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
best_model.save(TRAINED_MODEL_PATH)

index_to_class = {i: name for i, name in enumerate(CATEGORIES)}
with open(os.path.join(MODEL_DIR, CLASS_INDEX_FILE), "w") as f:
    json.dump(index_to_class, f, indent=2)

np.save(os.path.join(SPLITED_SAVED_PATH, "X_test.npy"), X_test)
np.save(os.path.join(SPLITED_SAVED_PATH, "y_test.npy"), y_test)

np.save(os.path.join(SPLITED_SAVED_PATH, "X_val.npy"), X_val)
np.save(os.path.join(SPLITED_SAVED_PATH, "y_val.npy"), y_val)

np.save(os.path.join(SPLITED_SAVED_PATH, "X_train.npy"), X_train)
np.save(os.path.join(SPLITED_SAVED_PATH, "y_train.npy"), y_train)

def save_metric_plot(train_k, val_k, title, fname):
    plt.figure(figsize=(7, 4))
    plt.plot(history.history[train_k], label=f"Train {train_k.capitalize()}")
    plt.plot(history.history[val_k], label=f"Val {val_k.capitalize()}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, fname))
    plt.close()

save_metric_plot("accuracy", "val_accuracy", "Accuracy Curve", "accuracy_graph.png")
save_metric_plot("loss", "val_loss", "Loss Curve", "loss_graph.png")
save_metric_plot("precision", "val_precision", "Precision Curve", "precision_graph.png")
save_metric_plot("recall", "val_recall", "Recall Curve", "recall_graph.png")

print("Training completed! Best weights saved and test data exported successfully.")
