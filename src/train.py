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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from global_variables import DATASET_DIR, CATEGORIES, NUM_CLASSES, TEST_SAVED_PATH
from global_variables import TEST_SPLIT, VAL_FROM_TEMP_SPLIT
from global_variables import IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE
from global_variables import LOG_DIR, GRAPH_DIR, CLASS_INDEX_FILE
from global_variables import BEST_MODEL_PATH, FINAL_BEST_MODEL_PATH, TRAINED_MODEL_PATH
from global_variables import ACC_GRAPH, PRECISION_GRAPH, RECALL_GRAPH, F1_GRAPH, LOSS_GRAPH, AUG_GRAPH, CLASS_DIST_GRAPH,ORIGINAL_IMAGES_GRAPH
from global_variables import REDUCE_LR_FACTOR, REDUCE_LR_PATIENCE, MIN_LR
from global_variables import start_partition, end_partition



# ******************************************************
# ===== STEP 2: LOAD TRAINING/VALIDATION/TEST DATA =====
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
                raise ValueError("cv2.imread returned None (corrupt or unreadable file)")

            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Save a copy of the ORIGINAL (resized, RGB, but not yet contrast-enhanced/normalized) 
            # image for display purposes.
            if len(sample_originals) < MAX_DISPLAY_SAMPLES:
                sample_originals.append(img.copy())
                sample_labels_display.append(cat_name)

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

start_partition("CORRUPT/UNREADABLE FILES")
print(f"Total corrupt/unreadable files skipped: {corrupt_count}")
end_partition()

# split train and temp from original
X_train, X_temp, y_train, y_temp = train_test_split(
     X, y,
     test_size=TEST_SPLIT,
     random_state=42,
     stratify=y
)

# split val and test from temp
X_test, X_val, y_test, y_val = train_test_split(
     X_temp, y_temp,
     test_size=VAL_FROM_TEMP_SPLIT,
     random_state=42,
     stratify=y_temp
)

start_partition("DATASET SHAPE")
print(f"Original Dataset Shape: {X.shape}")
print(f"X_train Dataset Shape: {X_train.shape}")
print(f"X_test Dataset Shape: {X_test.shape}")
print(f"X_val Dataset Shape: {X_val.shape}")
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
# ===== STEP 4: SHOW ORIGINAL vs PREPROCESSED IMAGES =====
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
# ===== STEP 5: CLASS WEIGHTS (handle imbalance) =====
# ******************************************************
class_weights_arr = compute_class_weight(
      class_weight="balanced",
      classes=np.unique(y_train),
      y=y_train
)
class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}

start_partition("CLASS WEIGHTS")
print(class_weight_dict)
end_partition()



# ******************************************************
# ===== STEP 6: PREPARE LABELS FOR TRAINING =====
# ******************************************************
# > 2 classes -> softmax + categorical_crossentropy (labels one-hot encoded)
OUTPUT_UNITS = NUM_CLASSES
OUTPUT_ACTIVATION = "softmax"
LOSS_FN = "categorical_crossentropy"
y_train_fit = to_categorical(y_train, NUM_CLASSES)
y_val_fit = to_categorical(y_val, NUM_CLASSES)



# ******************************************************
# ===== STEP 7: IMAGE AUGMENTATION =====
# ******************************************************
data_gen = ImageDataGenerator(
     rotation_range=25,
     zoom_range=0.2,
     width_shift_range=0.2,
     height_shift_range=0.2,
     horizontal_flip=True,
     fill_mode="nearest"
)
data_gen.fit(X_train)

# DATA AUGMENTATION GRAPH
sample = X_train[:1]
fig = plt.figure(figsize=(12, 6))
for i in range(6):
    aug = next(data_gen.flow(sample, batch_size=1))[0]

    plt.subplot(2, 3, i + 1)
    plt.imshow(aug)
    plt.axis("off")

plt.suptitle("Augmentation Examples (1 sample image)")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, f"train_{AUG_GRAPH}"))
plt.show()



# ******************************************************
# ===== STEP 8: BUILD MODEL (custom CNN, 4 conv blocks) =====
# ******************************************************
model = Sequential([
    # BLOCK 1
    Conv2D(
            32,
            (3, 3),
            activation='relu',
            kernel_regularizer=l2(0.0001),
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.20),

    # BLOCK 2
    Conv2D(
        64,
        (3, 3),
        activation='relu',
        kernel_regularizer=l2(0.0001)
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.20),

    # BLOCK 3
    Conv2D(
        128,
        (3, 3),
        activation='relu',
        kernel_regularizer=l2(0.0001)
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # BLOCK 4
    Conv2D(
        256,
        (3, 3),
        activation='relu',
        kernel_regularizer=l2(0.0001)
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.30),

    # ----- FLATTEN -----
    # Flatten(),
    GlobalAveragePooling2D(),

    # ----- DENSE LAYER -----
    Dense(
        256,
        activation="relu",
        kernel_regularizer=l2(0.0001)
    ),
    # BatchNormalization(),
    Dropout(0.40),

    # ----- OUTPUT LAYER (dynamic: binary or multi-class) -----
    Dense(OUTPUT_UNITS, activation=OUTPUT_ACTIVATION)
])



# ******************************************************
# ===== STEP 9: COMPILE MODEL =====
# ******************************************************
optimizer = Adam(learning_rate=LEARNING_RATE)

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

start_partition("MODEL SUMMARY")
model.summary()
end_partition()



# ******************************************************
# ===== STEP 10: CALLBACKS =====
# ******************************************************
log_dir = LOG_DIR + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_cb = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

early_stopping_cb = EarlyStopping(
    monitor="val_loss",
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True,
    verbose=1
)

best_model_path = os.path.join(BEST_MODEL_PATH)
model_checkpoint_cb = ModelCheckpoint(
    best_model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss",
    factor=REDUCE_LR_FACTOR,
    patience=REDUCE_LR_PATIENCE,
    min_lr=MIN_LR,
    verbose=1
)



# ******************************************************
# ===== STEP 11: TRAIN MODEL =====
# ******************************************************
train_model = model.fit(
     data_gen.flow(
            X_train,
            y_train_fit,
            batch_size=BATCH_SIZE
     ),
     epochs=EPOCHS,
     validation_data=(X_val, y_val_fit),
     class_weight=class_weight_dict,
     callbacks=[early_stopping_cb, model_checkpoint_cb, reduce_lr_cb, tensorboard_cb]
)

print(train_model.history)



# ******************************************************
# ===== STEP 12: PLOT ACCURACY / LOSS / PRECISION / RECALL / F1 =====
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

    if isDirect:
        plt.ylabel(history)
    else:
        plt.ylabel(train_key)

    plt.legend(["Train", "Validation"])
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, save_name))
    plt.show()

plot_metric(train_model, "accuracy", "val_accuracy", "Training vs Validation Accuracy", f"train_{ACC_GRAPH}")
plot_metric(train_model, "loss", "val_loss", "Training vs Validation Loss", f"train_{LOSS_GRAPH}")
plot_metric(train_model, "precision", "val_precision", "Training vs Validation Precision", f"train_{PRECISION_GRAPH}")
plot_metric(train_model, "recall", "val_recall", "Training vs Validation Recall", f"train_{RECALL_GRAPH}")

# F1 is not a direct Keras metric - compute it per epoch from precision & recall
train_f1 = [
    2 * p * r / (p + r + 1e-7)
    for p, r in zip(train_model.history["precision"], train_model.history["recall"])
]
val_f1 = [
    2 * p * r / (p + r + 1e-7)
    for p, r in zip(train_model.history["val_precision"], train_model.history["val_recall"])
]

plot_metric("F1 Score", train_f1, val_f1, "Training vs Validation F1 Score", f"train_{ACC_GRAPH}", True)



# ******************************************************
# ===== STEP 13: SAVE FINAL MODEL =====
# ******************************************************
print(TRAINED_MODEL_PATH)
model.save(TRAINED_MODEL_PATH)

start_partition("FINAL MODEL SAVED")
print(f"Final model saved to: {TRAINED_MODEL_PATH}")
print(f"Best model was saved (during training) to: {best_model_path}")
end_partition()



# ******************************************************
# ===== STEP 14: SAVE CLASS INDEX MAPPING =====
# ******************************************************
# Needed by predict.py / evaluate.py to convert a predicted index
# back into a human-readable class name.
# class_index_path = os.path.join(gv.TRAINED_MODEL_PATH, gv.CLASS_INDEX_FILE)
class_index_path = os.path.join(TRAINED_MODEL_PATH, CLASS_INDEX_FILE)
index_to_class = {i: name for i, name in enumerate(CATEGORIES)}
with open(CLASS_INDEX_FILE, "w") as f:
    json.dump(index_to_class, f, indent=2)

start_partition("CLASS INDEX MAPPING SAVED")
print(index_to_class)
end_partition()



# ******************************************************
# ===== STEP 15: SAVE TEST DATASET (for evaluate.py) =====
# ******************************************************
if os.path.exists(TEST_SAVED_PATH):
    shutil.rmtree(TEST_SAVED_PATH)
os.makedirs(TEST_SAVED_PATH)

np.save(os.path.join(TEST_SAVED_PATH, "X_test.npy"), X_test)
np.save(os.path.join(TEST_SAVED_PATH, "y_test.npy"), y_test)

start_partition("TESTING DATASET SAVED")
print("Test dataset saved successfully..!")
end_partition()
