# ******************************************************
# ===== STEP 1: IMPORT LIBRARIES =====
# ******************************************************
import global_variables as gv

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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import (
      TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
# NOTE: no pretrained/ResNet imports here - project rules require a
# custom-built CNN, not a prebuilt architecture.


# ******************************************************
# ===== STEP 2: SET DATASET PATHS =====
# ******************************************************
categories = gv.CATEGORIES
Img_Size = gv.IMG_SIZE
Batch_Size = gv.BATCH_SIZE
NUM_CLASSES = gv.NUM_CLASSES

os.makedirs(gv.TRAINED_MODEL_PATH, exist_ok=True)
os.makedirs(gv.GRAPH_DIR, exist_ok=True)


# ******************************************************
# ===== STEP 3: LOAD TRAINING/VALIDATION/TEST DATA =====
# ******************************************************
data = []
labels = []

# Keep a small sample of images BEFORE preprocessing and AFTER
# preprocessing so we can show both grids later (rule #7).
sample_originals = []
sample_processed = []
sample_labels_display = []
MAX_DISPLAY_SAMPLES = 10

corrupt_count = 0

for label, can_name in enumerate(categories):
    folder_path = os.path.join(gv.DATASET_DIR, gv.DATASET_NAME, can_name)
    print(f"Working in the '{folder_path}' directory")
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        try:
            img = cv2.imread(img_path)
            if img is None:
                # cv2.imread silently returns None for unreadable/corrupt files
                raise ValueError("cv2.imread returned None (corrupt or unreadable file)")

            img = cv2.resize(img, (Img_Size, Img_Size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Save a copy of the ORIGINAL (resized, RGB, but not yet
            # contrast-enhanced/normalized) image for display purposes.
            if len(sample_originals) < MAX_DISPLAY_SAMPLES:
                sample_originals.append(img.copy())
                sample_labels_display.append(can_name)

            # L = Lightness (brightness) | A = Green <-> Red | B = Blue <-> Yellow
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0)
            l = clahe.apply(l)

            lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            img = img.astype("float32") / 255.0  # convert uint8 -> float32

            # Save a copy of the PREPROCESSED image (matches the same
            # index/order as sample_originals above).
            if len(sample_processed) < MAX_DISPLAY_SAMPLES:
                sample_processed.append(img.copy())

            data.append(img)
            labels.append(label)

        except Exception as e:
            corrupt_count += 1
            print(f"ERROR LOADING IMAGE: {e} || IMAGE PATH: {img_path}")

X = np.array(data)
y = np.array(labels)

gv.start_partition("CORRUPT/UNREADABLE FILES")
print(f"Total corrupt/unreadable files skipped: {corrupt_count}")
gv.end_partition()

# stratify=y keeps class ratios consistent across train/val/test,
# which matters a lot once you have 5-6 (likely imbalanced) classes.
X_train, X_temp, y_train, y_temp = train_test_split(
     X, y,
     test_size=gv.TEST_SPLIT,
     random_state=42,
     stratify=y
)

X_test, X_val, y_test, y_val = train_test_split(
     X_temp, y_temp,
     test_size=gv.VAL_FROM_TEMP_SPLIT,
     random_state=42,
     stratify=y_temp
)

gv.start_partition("DATASET SHAPE")
print(f"Original Dataset Shape: {X.shape}")
print(f"X_train Dataset Shape: {X_train.shape}")
print(f"X_test Dataset Shape: {X_test.shape}")
print(f"X_val Dataset Shape: {X_val.shape}")
gv.end_partition()


# ******************************************************
# ===== STEP 4: EDA - CLASS DISTRIBUTION =====
# ******************************************************
label_counts = Counter(labels)
count_by_name = {categories[idx]: count for idx, count in label_counts.items()}

gv.start_partition("CLASS DISTRIBUTION")
for name, count in count_by_name.items():
    print(f"{name}: {count}")
gv.end_partition()

plt.figure(figsize=(8, 5))
plt.bar(count_by_name.keys(), count_by_name.values(), color="teal")
plt.title("Class Distribution")
plt.ylabel("Number of Images")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(gv.GRAPH_DIR, gv.CLASS_DIST_GRAPH))
plt.show()


# ******************************************************
# ===== STEP 5: SHOW ORIGINAL vs PREPROCESSED IMAGES =====
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
plt.savefig(os.path.join(gv.GRAPH_DIR, gv.ORIGINAL_IMAGES_GRAPH))
plt.show()


# ******************************************************
# ===== STEP 6: CLASS WEIGHTS (handle imbalance) =====
# ******************************************************
class_weights_arr = compute_class_weight(
      class_weight="balanced",
      classes=np.unique(y_train),
      y=y_train
)
class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}

gv.start_partition("CLASS WEIGHTS")
print(class_weight_dict)
gv.end_partition()


# ******************************************************
# ===== STEP 7: PREPARE LABELS FOR TRAINING =====
# ******************************************************
# 2 classes -> sigmoid + binary_crossentropy (labels stay as-is)
# 3+ classes -> softmax + categorical_crossentropy (labels one-hot encoded)
if NUM_CLASSES == 2:
    OUTPUT_UNITS = 1
    OUTPUT_ACTIVATION = "sigmoid"
    LOSS_FN = "binary_crossentropy"
    y_train_fit = y_train
    y_val_fit = y_val
else:
    OUTPUT_UNITS = NUM_CLASSES
    OUTPUT_ACTIVATION = "softmax"
    LOSS_FN = "categorical_crossentropy"
    y_train_fit = to_categorical(y_train, NUM_CLASSES)
    y_val_fit = to_categorical(y_val, NUM_CLASSES)


# ******************************************************
# ===== STEP 8: IMAGE AUGMENTATION =====
# ******************************************************
data_gen = ImageDataGenerator(
     rotation_range=25,
     zoom_range=0.2,
     width_shift_range=0.2,
     height_shift_range=0.2,
     horizontal_flip=True,
     shear_range=0.15,
     fill_mode="nearest"
)
data_gen.fit(X_train)

# DATA AUGMENTATION GRAPH (rule #7: visualize augmentation of one image)
sample = X_train[:1]
fig = plt.figure(figsize=(12, 6))
for i in range(6):
    aug = next(data_gen.flow(sample, batch_size=1))[0]

    plt.subplot(2, 3, i + 1)
    plt.imshow(aug)
    plt.axis("off")

plt.suptitle("Augmentation Examples (1 sample image)")
plt.tight_layout()
plt.savefig(os.path.join(gv.GRAPH_DIR, gv.AUG_GRAPH))
plt.show()


# ******************************************************
# ===== STEP 9: BUILD MODEL (custom CNN, 4 conv blocks) =====
# ******************************************************
model = Sequential()

# BLOCK 1
model.add(
      Conv2D(
            32, (3, 3), activation='relu',
            kernel_regularizer=l2(0.001),
            input_shape=(Img_Size, Img_Size, 3)
      )
)
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# BLOCK 2
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# BLOCK 3
model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# BLOCK 4
model.add(Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

# ----- FLATTEN -----
model.add(Flatten())

# ----- DENSE LAYER -----
model.add(Dense(256, activation="relu", kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# ----- OUTPUT LAYER (dynamic: binary or multi-class) -----
model.add(Dense(OUTPUT_UNITS, activation=OUTPUT_ACTIVATION))


# ******************************************************
# ===== STEP 10: COMPILE MODEL =====
# ******************************************************
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

model.compile(
     optimizer=optimizer,
     loss=LOSS_FN,
     metrics=[
          "accuracy",
          Precision(name="precision"),
          Recall(name="recall"),
          AUC(name="auc"),
     ]
)

gv.start_partition("MODEL SUMMARY")
model.summary()
gv.end_partition()

# Also save the summary to a text file
summary_path = os.path.join(gv.GRAPH_DIR, "model_summary.txt")
with open(summary_path, "w") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))


# ******************************************************
# ===== STEP 11: CALLBACKS =====
# ******************************************************
log_dir = gv.LOG_DIR + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_cb = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

early_stopping_cb = EarlyStopping(
    monitor="val_loss",
    patience=gv.EARLY_STOPPING_PATIENCE,
    restore_best_weights=True,
    verbose=1
)

best_model_path = os.path.join(gv.TRAINED_MODEL_PATH, gv.BEST_MODEL_NAME)
model_checkpoint_cb = ModelCheckpoint(
    best_model_path,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss",
    factor=gv.REDUCE_LR_FACTOR,
    patience=gv.REDUCE_LR_PATIENCE,
    min_lr=gv.MIN_LR,
    verbose=1
)


# ******************************************************
# ===== STEP 12: TRAIN MODEL =====
# ******************************************************
Epocs = gv.EPOCHS
train_model = model.fit(
     data_gen.flow(
            X_train,
            y_train_fit,
            batch_size=Batch_Size
     ),
     epochs=Epocs,
     validation_data=(X_val, y_val_fit),
     class_weight=class_weight_dict,
     callbacks=[early_stopping_cb, model_checkpoint_cb, reduce_lr_cb, tensorboard_cb]
)


# ******************************************************
# ===== STEP 13: PLOT ACCURACY / LOSS / PRECISION / RECALL / F1 =====
# ******************************************************
def plot_metric(history, train_key, val_key, title, save_name):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history[train_key])
    plt.plot(history.history[val_key])
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(train_key)
    plt.legend(["Train", "Validation"])
    plt.tight_layout()
    plt.savefig(os.path.join(gv.GRAPH_DIR, save_name))
    plt.show()

plot_metric(train_model, "accuracy", "val_accuracy",
            "Training vs Validation Accuracy", gv.ACC_GRAPH)
plot_metric(train_model, "loss", "val_loss",
            "Training vs Validation Loss", gv.LOSS_GRAPH)
plot_metric(train_model, "precision", "val_precision",
            "Training vs Validation Precision", gv.PRECISION_GRAPH)
plot_metric(train_model, "recall", "val_recall",
            "Training vs Validation Recall", gv.RECALL_GRAPH)

# F1 is not a direct Keras metric - compute it per epoch from precision & recall
train_f1 = [
    2 * p * r / (p + r + 1e-7)
    for p, r in zip(train_model.history["precision"], train_model.history["recall"])
]
val_f1 = [
    2 * p * r / (p + r + 1e-7)
    for p, r in zip(train_model.history["val_precision"], train_model.history["val_recall"])
]
plt.figure(figsize=(8, 5))
plt.plot(train_f1)
plt.plot(val_f1)
plt.title("Training vs Validation F1 Score")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend(["Train", "Validation"])
plt.tight_layout()
plt.savefig(os.path.join(gv.GRAPH_DIR, gv.F1_GRAPH))
plt.show()


# ******************************************************
# ===== STEP 14: SAVE FINAL MODEL =====
# ******************************************************
final_model_path = os.path.join(gv.TRAINED_MODEL_PATH, gv.TRAINED_MODEL_NAME)
model.save(final_model_path)

gv.start_partition("FINAL MODEL SAVED")
print(f"Final model saved to: {final_model_path}")
print(f"Best model was saved (during training) to: {best_model_path}")
gv.end_partition()


# ******************************************************
# ===== STEP 15: SAVE CLASS INDEX MAPPING =====
# ******************************************************
# Needed by predict.py / evaluate.py to convert a predicted index
# back into a human-readable class name.
class_index_path = os.path.join(gv.TRAINED_MODEL_PATH, gv.CLASS_INDEX_FILE)
index_to_class = {i: name for i, name in enumerate(categories)}
with open(class_index_path, "w") as f:
    json.dump(index_to_class, f, indent=2)

gv.start_partition("CLASS INDEX MAPPING SAVED")
print(index_to_class)
gv.end_partition()


# ******************************************************
# ===== STEP 16: SAVE TEST DATASET (for evaluate.py) =====
# ******************************************************
test_saved_path = os.path.join(gv.DATASET_DIR, "test_saved_data")

# shutil.rmtree (not os.removedirs) so it works even if the folder
# already has files in it from a previous run.
if os.path.exists(test_saved_path):
    shutil.rmtree(test_saved_path)
os.makedirs(test_saved_path)

np.save(os.path.join(test_saved_path, "X_test.npy"), X_test)
np.save(os.path.join(test_saved_path, "y_test.npy"), y_test)

gv.start_partition("TESTING DATASET SAVED")
print("Test dataset saved successfully..!")
gv.end_partition()