# ******************************************************
# ===== STEP 1: IMPORT LIBRARIES =====
# ******************************************************
import os
import cv2
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Input

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC
from sklearn.utils.class_weight import compute_class_weight

from global_variables import DATASET_DIR, CATEGORIES
from global_variables import SPLITED_TRAIN_DATASET, SPLITED_VAL_DATASET, SPLITED_TEST_DATASET
from global_variables import IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE
from global_variables import LOG_DIR, GRAPH_DIR
from global_variables import SEED
from global_variables import BEST_MODEL_PATH, FINAL_BEST_MODEL_PATH, TRAINED_MODEL_PATH

from global_variables import set_seeds
from global_variables import preprocessing_for_augmentation, preprocessing_images
from global_variables import scan_dataset, split_train_validation_test
from global_variables import plot_image_grid, plot_original_vs_processed, draw_data_plot, draw_histogram_plot
from global_variables import save_class_indices_map
from global_variables import start_partition, end_partition



# ******************************************************
# ===== STEP 2: REPRODUCIBILITY =====
# ******************************************************
set_seeds()
os.makedirs(BEST_MODEL_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ******************************************************
# ===== STEP 3: SCAN DATASET =====
# ******************************************************
start_partition("SCANNING DATASET")
X, y, categories, currupted_files = scan_dataset(DATASET_DIR)
NUM_OF_CATEGORIES = len(categories)
print(f"Found NUM_OF_CATEGORIES: {NUM_OF_CATEGORIES}")
print(f"Count of Currepted Files: {len(currupted_files)}")
end_partition()



# ******************************************************
# ===== STEP 4: CLASS DISTRIBUTION =====
# ******************************************************
df = pd.DataFrame({"File":X, "Label":y})
start_partition("CLASS DISTRIBUTION")
print(df["Label"].value_counts())

draw_data_plot((8, 5), df["Label"].value_counts(), ["Number of images", "Frequency", "Class Distribution (Raw Merged Dataset)"], "class_distribution.png", "bar")
end_partition()



# ******************************************************
# ===== STEP 5: SAMPLE IMAGES DISTRIBUTION =====
# ******************************************************
# sample_paths = df["File"].sample(min(25, len(df)), random_state=SEED)
# intensities = []
# for p in sample_paths:
#       img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
#       if img is not None:
#             intensities.append(img.mean())

# draw_histogram_plot((8, 5), ["Pixel Intensity Distribution (sampled raw images)", "Mean pixel intensity", "Frequency"], "pixel_intensity_hist.png", 20 )



# ******************************************************
# ===== STEP 5: SHOW ORIGINAL IMAGES =====
# ******************************************************
sample_df = df.sample(16, random_state=SEED)
original_images = [cv2.imread(p) for p in sample_df["File"]]
plot_image_grid(original_images, titels=list(sample_df["Label"]), cols=4,
                 main_title="Original Images (Before Preprocessing)")



# ******************************************************
# ===== STEP 5: SPLIT DATASET =====
# ******************************************************
train_df, val_df, test_df = split_train_validation_test(X, y)



# ******************************************************
# ===== STEP 5: HANDEL CLASS WEIGHT (IMBALANCED) =====
# ******************************************************
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.array(CATEGORIES),
    y=train_df["Label"].values,
)
label_to_index = {label: idx for idx, label in enumerate(CATEGORIES)}
class_weight_dict = {label_to_index[label]: w for label, w in zip(CATEGORIES, class_weights_arr)}
print("Class weights:", class_weight_dict)



# ******************************************************
# ===== STEP 5: DATA AUGMENTATION/GENERATION =====
# ******************************************************
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocessing_for_augmentation,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocessing_for_augmentation,
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="File",
    y_col="Label",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=categories,
    shuffle=True,
    seed=SEED,
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="File",
    y_col="Label",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CATEGORIES,
    shuffle=False,
)



# ******************************************************
# ===== STEP 5: SAVE CLASS INDEX MAPPING =====
# ******************************************************
save_class_indices_map(train_generator.class_indices)
print(f"Class indices: {train_generator.class_indices}")



# ******************************************************
# ===== STEP 5: AUGEMENTATION VISUALIZATION =====
# ******************************************************
sample_img_path = train_df["File"].iloc[0]
sample_img = cv2.imread(sample_img_path)
sample_img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
sample_img_rgb = cv2.resize(sample_img_rgb, (IMAGE_SIZE, IMAGE_SIZE))

sample_batch = np.expand_dims(sample_img_rgb, axis=0)
aug_iter = train_datagen.flow(sample_batch, batch_size=1)

images = [sample_img]
titles = ["Original"]

for i in range(5):
      aug_img_rgb = next(aug_iter)[0]
    
      # Scale float images back to [0, 255] range if datagen rescales (e.g. 1./255)
      if aug_img_rgb.max() <= 1.0:
            aug_img_rgb = aug_img_rgb * 255.0

      # Convert RGB back to BGR so plot_image_grid's cv2.cvtColor works correctly
      aug_img_bgr = cv2.cvtColor(aug_img_rgb.astype("uint8"), cv2.COLOR_RGB2BGR)
    
      images.append(aug_img_bgr)
      titles.append(f"Aug {i+1}")

plot_image_grid(imgs=images, titels=titles, cols=6, main_title="Augmentation Visualization (1 sample image)")



# =========================================================
# STEP 13: Show PREPROCESSED training images (same grid as Step 6)
# =========================================================
print("\n=== Showing preprocessed versions of the same images ===")
processed_images = [preprocessing_images(img) for img in original_images]
plot_image_grid(processed_images, titles=list(sample_df["label"]), cols=5,
                 main_title="Preprocessed Images (After OpenCV Pipeline)")



# =========================================================
# STEP 14: Build CUSTOM CNN (4 conv blocks, no pretrained models)
# =========================================================
def build_cnn(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),

        # Block 1
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        # Block 4
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    return model


print("\n=== Building CNN ===")
model = build_cnn(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=NUM_OF_CATEGORIES)