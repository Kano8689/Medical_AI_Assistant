# train.py
# ******************************************************
# ===== STEP 1: IMPORT LIBRARIES =====
# ******************************************************
import os
import json
import shutil
import cv2
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, 
    Dropout, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import (
    TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

from global_variables import (
    DATASET_DIR, CATEGORIES, NUM_CLASSES, TEST_SAVED_PATH,
    TEST_SPLIT, VAL_FROM_TEMP_SPLIT,
    IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, L2_REG,
    LABEL_SMOOTHING, DROPOUT_HEAD,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_PATIENCE, MIN_LR,
    LOG_DIR, GRAPH_DIR, CLASS_INDEX_FILE, MODEL_DIR,
    BEST_MODEL_PATH, TRAINED_MODEL_PATH,
    ACC_GRAPH, PRECISION_GRAPH, RECALL_GRAPH, F1_GRAPH, LOSS_GRAPH,
    AUG_GRAPH, CLASS_DIST_GRAPH, ORIGINAL_IMAGES_GRAPH,
    AUGMENTATION_ROTATION, AUGMENTATION_ZOOM, AUGMENTATION_SHIFT, AUGMENTATION_FLIP,
    start_partition, end_partition
)

# ******************************************************
# ===== STEP 2: LOAD AND PREPROCESS DATA =====
# ******************************************************
data = []
labels = []

sample_originals = []
sample_processed = []
sample_labels_display = []
MAX_DISPLAY_SAMPLES = 10

corrupt_count = 0

for label, cat_name in enumerate(CATEGORIES):
    folder_path = os.path.join(DATASET_DIR, cat_name)
    print(f"Working in the '{folder_path}' directory")
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("cv2.imread returned None")

            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if len(sample_originals) < MAX_DISPLAY_SAMPLES:
                sample_originals.append(img.copy())
                sample_labels_display.append(cat_name)

            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            img = img.astype("float32") / 255.0

            if len(sample_processed) < MAX_DISPLAY_SAMPLES:
                sample_processed.append(img.copy())

            data.append(img)
            labels.append(label)

        except Exception as e:
            corrupt_count += 1
            print(f"ERROR LOADING IMAGE: {e} || PATH: {img_path}")

X = np.array(data)
y = np.array(labels)

start_partition("CORRUPT FILES")
print(f"Total corrupt/unreadable files skipped: {corrupt_count}")
end_partition()

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
)

X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=VAL_FROM_TEMP_SPLIT, random_state=42, stratify=y_temp
)

start_partition("DATASET SHAPE")
print(f"Original Dataset Shape: {X.shape}")
print(f"X_train Dataset Shape: {X_train.shape}")
print(f"X_val Dataset Shape: {X_val.shape}")
print(f"X_test Dataset Shape: {X_test.shape}")
end_partition()

# ******************************************************
# ===== STEP 3: EDA - CLASS DISTRIBUTION =====
# ******************************************************
label_counts = Counter(labels)
count_by_name = {CATEGORIES[idx]: count for idx, count in label_counts.items()}

start_partition("CLASS DISTRIBUTION")
for name, count in count_by_name.items():
    print(f"{name}: {count}")
end_partition()

plt.figure(figsize=(8, 5))
plt.bar(count_by_name.keys(), count_by_name.values(), color="teal")
plt.title("Class Distribution")
plt.ylabel("Number of Images")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, CLASS_DIST_GRAPH))
plt.show()

# ******************************************************
# ===== STEP 4: SHOW ORIGINAL vs PREPROCESSED =====
# ******************************************************
fig = plt.figure(figsize=(15, 6))
for i in range(len(sample_originals)):
    plt.subplot(2, MAX_DISPLAY_SAMPLES, i + 1)
    plt.imshow(sample_originals[i])
    plt.title(sample_labels_display[i], fontsize=8)
    plt.axis("off")

    plt.subplot(2, MAX_DISPLAY_SAMPLES, MAX_DISPLAY_SAMPLES + i + 1)
    plt.imshow(sample_processed[i])
    plt.axis("off")

fig.text(0.5, 0.95, "Top row: Original | Bottom row: Preprocessed (CLAHE)", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, ORIGINAL_IMAGES_GRAPH))
plt.show()

# ******************************************************
# ===== STEP 5: CLASS WEIGHTS =====
# ******************************************************
class_weights_arr = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}

start_partition("CLASS WEIGHTS")
for class_idx, weight in class_weight_dict.items():
    class_name = CATEGORIES[class_idx]
    print(f"Class: {class_name} (Index: {class_idx}) | Weight: {weight:.4f}")
end_partition()

# ******************************************************
# ===== STEP 6: PREPARE LABELS =====
# ******************************************************
y_train_fit = to_categorical(y_train, NUM_CLASSES)
y_val_fit = to_categorical(y_val, NUM_CLASSES)

# ******************************************************
# ===== STEP 7: DATA AUGMENTATION =====
# ******************************************************
data_gen = ImageDataGenerator(
    rotation_range=AUGMENTATION_ROTATION,
    zoom_range=AUGMENTATION_ZOOM,
    width_shift_range=AUGMENTATION_SHIFT,
    height_shift_range=AUGMENTATION_SHIFT,
    horizontal_flip=AUGMENTATION_FLIP,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,
    fill_mode="nearest"
)

sample = X_train[:1]
fig = plt.figure(figsize=(12, 6))
for i in range(6):
    aug = next(data_gen.flow(sample, batch_size=1))[0]
    plt.subplot(2, 3, i + 1)
    plt.imshow(aug)
    plt.axis("off")

plt.suptitle("Augmentation Examples")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, f"train_{AUG_GRAPH}"))
plt.show()

# ******************************************************
# ===== STEP 8: BUILD CUSTOM CNN =====
# ******************************************************
model = Sequential([
    Conv2D(32, (3, 3), padding="same", activation="relu",
           kernel_regularizer=l2(L2_REG), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding="same", activation="relu",
           kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), padding="same", activation="relu",
           kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding="same", activation="relu",
           kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.30),

    Conv2D(128, (3, 3), padding="same", activation="relu",
           kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding="same", activation="relu",
           kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.40),

    Conv2D(256, (3, 3), padding="same", activation="relu",
           kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.50),

    GlobalAveragePooling2D(),
    Dense(128, activation="relu", kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    Dropout(DROPOUT_HEAD),
    Dense(NUM_CLASSES, activation="softmax")
])

start_partition("MODEL SUMMARY")
model.summary()
end_partition()

# ******************************************************
# ===== STEP 9: COMPILE MODEL =====
# ******************************************************
optimizer = Adam(learning_rate=LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy", Precision(name="precision"), Recall(name="recall"), AUC(name="auc")]
)

# ******************************************************
# ===== STEP 10: CALLBACKS =====
# ******************************************************
log_dir = os.path.join(LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True)
early_stopping_cb = EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                                   restore_best_weights=True, verbose=1, mode="min")
model_checkpoint_cb = ModelCheckpoint(BEST_MODEL_PATH, monitor="val_loss",
                                       mode="min", save_best_only=True, verbose=1)
reduce_lr_cb = ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR,
                                  patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR, verbose=1, mode="min")

# ******************************************************
# ===== STEP 11: TRAIN MODEL =====
# ******************************************************
train_model = model.fit(
    data_gen.flow(X_train, y_train_fit, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_val, y_val_fit),
    class_weight=class_weight_dict,
    callbacks=[early_stopping_cb, model_checkpoint_cb, reduce_lr_cb, tensorboard_cb]
)

print("\nTraining History:")
print(f"Final Training Accuracy: {train_model.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {train_model.history['val_accuracy'][-1]:.4f}")
print(f"Best Validation Loss: {min(train_model.history['val_loss']):.4f}")

# ******************************************************
# ===== STEP 12: PLOT METRICS =====
# ******************************************************
def plot_metric(history, train_key, val_key, title, save_name, isDirect=False):
    plt.figure(figsize=(8, 5))

    if isDirect:
        plt.plot(train_key)
        plt.plot(val_key)
    else:
        plt.plot(history.history[train_key])
        plt.plot(history.history[val_key])

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(train_key if not isDirect else history)
    plt.legend(["Train", "Validation"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, save_name))
    plt.show()

plot_metric(train_model, "accuracy", "val_accuracy", 
            "Training vs Validation Accuracy", f"train_{ACC_GRAPH}")
plot_metric(train_model, "loss", "val_loss", 
            "Training vs Validation Loss", f"train_{LOSS_GRAPH}")
plot_metric(train_model, "precision", "val_precision", 
            "Training vs Validation Precision", f"train_{PRECISION_GRAPH}")
plot_metric(train_model, "recall", "val_recall", 
            "Training vs Validation Recall", f"train_{RECALL_GRAPH}")

train_f1 = [2 * p * r / (p + r + 1e-7) for p, r in zip(train_model.history["precision"], train_model.history["recall"])]
val_f1 = [2 * p * r / (p + r + 1e-7) for p, r in zip(train_model.history["val_precision"], train_model.history["val_recall"])]

plot_metric("F1 Score", train_f1, val_f1, 
            "Training vs Validation F1 Score", f"train_{F1_GRAPH}", True)

# ******************************************************
# ===== STEP 13: SAVE BEST MODEL =====
# ******************************************************
best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
best_model.save(TRAINED_MODEL_PATH)

start_partition("BEST MODEL SAVED")
print(f"Best checkpoint: {BEST_MODEL_PATH}")
print(f"Production model: {TRAINED_MODEL_PATH}")
end_partition()

# ******************************************************
# ===== STEP 14: SAVE CLASS INDEX =====
# ******************************************************
class_index_path = os.path.join(MODEL_DIR, CLASS_INDEX_FILE)
index_to_class = {i: name for i, name in enumerate(CATEGORIES)}
with open(class_index_path, "w") as f:
    json.dump(index_to_class, f, indent=2)

start_partition("CLASS INDEX SAVED")
print(index_to_class)
end_partition()

# ******************************************************
# ===== STEP 15: SAVE TEST DATASET =====
# ******************************************************
if os.path.exists(TEST_SAVED_PATH):
    shutil.rmtree(TEST_SAVED_PATH)
os.makedirs(TEST_SAVED_PATH)

np.save(os.path.join(TEST_SAVED_PATH, "X_test.npy"), X_test)
np.save(os.path.join(TEST_SAVED_PATH, "y_test.npy"), y_test)
np.save(os.path.join(TEST_SAVED_PATH, "X_val.npy"), X_val)
np.save(os.path.join(TEST_SAVED_PATH, "y_val.npy"), y_val)

start_partition("DATASETS SAVED")
print("Test and validation datasets saved successfully!")
end_partition()