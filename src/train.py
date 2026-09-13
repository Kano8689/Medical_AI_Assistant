# ******************************************************
# ===== STEP 1: IMPORT LIBRARIES =====
# ******************************************************
import global_variables as gv

import os
import cv2

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.callbacks import TensorBoard
import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D



# ******************************************************
# ===== STEP 2: SET DATASET PATHS =====
# ******************************************************
categories = gv.CATEGORIES
Img_Size = gv.IMG_SIZE
Batch_Size = gv.BATCH_SIZE



# ******************************************************
# ===== STEP 3: LOAD TRAINING/VALIDATION/TEST DATA =====
# ******************************************************
data = []
labels = []

for label, can_name in enumerate(categories):
    folder_path = os.path.join(f"{gv.DATASET_DIR}\{gv.DATASET_NAME}", can_name)
    print(f"Working in the '{folder_path}' directory")
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (Img_Size, Img_Size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # L = Lightness (brightness) | A = Green ↔ Red | B = Blue ↔ Yellow
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l,a,b = cv2.split(lab)

            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0)
            l = clahe.apply(l)

            lab = cv2.merge((l,a,b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            img = img.astype("float32")/255.0 # convert uint8 -> float32


            data.append(img)
            labels.append(label)

        except Exception as e:
             print(f"ERROR LOADING IMAGE: {e} || IMAGE PATH: {img_path}")

X = np.array(data)
y = np.array(labels)

X_train, X_temp, y_train, y_temp = train_test_split(
     X, y,
     test_size = 0.30,
     random_state = 42
)

X_test, X_val, y_test, y_val = train_test_split(
     X_temp, y_temp,
     test_size = 0.50,
     random_state = 42
)

gv.start_partition("DATASET SHAPE")
print(f"Original Dataset Shape: {X.shape}")
print(f"X_train Dataset Shape: {X_train.shape}")
print(f"X_test Dataset Shape: {X_test.shape}")
print(f"X_val Dataset Shape: {X_val.shape}")
gv.end_partition()



# ******************************************************
# ===== STEP 4:  IMAGE PREPROCESSING =====
# ******************************************************

# DATA AUGMENTATION
data_gen = ImageDataGenerator(
     rotation_range = 25,
     zoom_range = 0.2,
     width_shift_range = 0.2,
     height_shift_range = 0.2,
     horizontal_flip = True,
     shear_range=0.15,
     fill_mode="nearest"
)
data_gen.fit(X_train)

# DATA AUGMENTATION GRAPH
sample = X_train[:1]
fig = plt.figure(figsize=(12,6))
for i in range(6):
    aug = next(data_gen.flow(sample, batch_size=1))[0]

    plt.subplot(2,3,i+1)
    plt.imshow(aug)
    plt.axis("off")

plt.title("Augmentation Graph")
plt.tight_layout()
plt.show()



# ******************************************************
# ===== STEP 5: BUILD MODEL =====
# ******************************************************
# ----- CNN MODEL -----
model = Sequential()

# BLOCK 1
model.add(
      Conv2D(
            32,
            (3, 3),
            activation='relu',
            kernel_regularizer=l2(0.001),
            input_shape=(Img_Size, Img_Size, 3)
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

# BLOCK 2
model.add(
      Conv2D(
            64,
            (3, 3),
            activation='relu',
            kernel_regularizer=l2(0.001)
      )
)
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

# BLOCK 3
model.add(
      Conv2D(
            128,
            (3, 3),
            activation='relu',
            kernel_regularizer=l2(0.001)
      )
)
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))


# ----- FLATTEN -----
model.add(Flatten())


# ----- DENSE LAYER -----
model.add(
      Dense(
            256,
            activation="relu",
            kernel_regularizer = l2(0.001)
      )
)
model.add(BatchNormalization())
model.add(Dropout(0.5))


# ----- OUTPUT LAYER -----
model.add(
      Dense(
            1,
            activation="sigmoid"
      )
)



# ******************************************************
# ===== STEP 7: COMPILE MODEL =====
# ******************************************************

# ----- LEARNING RATE -----
learning_rate = 0.001
optimizer = Adam(
     learning_rate=learning_rate
)


# ----- COMPILE MODL -----
model.compile(
     optimizer = optimizer,
     loss="binary_crossentropy",
     metrics=["accuracy"]
)


# ----- SUMMARY -----
gv.start_partition("MODEL SUMMARY")
print(f"Model Summary:\n{model.summary()}")
gv.end_partition()



# ******************************************************
# ===== STEP 8: TRAIN MODEL =====
# ******************************************************
log_dir = gv.LOG_DIR +"/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)



# ******************************************************
# ===== STEP 8: TRAIN MODEL =====
# ******************************************************
Epocs = gv.EPOCHS
train_model = model.fit(
     data_gen.flow(
            X_train,
            y_train,
            batch_size = Batch_Size
     ),
     epochs = Epocs,
     validation_data=(X_val, y_val),
     callbacks=[tensorboard]
)



# ******************************************************
# ===== STEP 9: PLOT ACCURACY/LOSS =====
# ******************************************************
# Accuracy
plt.figure(figsize=(8,5))
plt.plot(train_model.history["accuracy"])
plt.plot(train_model.history["val_accuracy"])
plt.title("Accuracy")
plt.legend(["Train","Validation"])
plt.show()

# Loss
plt.figure(figsize=(8,5))
plt.plot(train_model.history["loss"])
plt.plot(train_model.history["val_loss"])
plt.title("Loss")
plt.legend(["Train","Validation"])
plt.show()



# ******************************************************
# ===== STEP 10: SAVE MODEL =====
# ******************************************************
model_path = os.path.join(gv.TRAINED_MODEL_PATH, gv.TRAINED_MODEL_NAME)
model.save(model_path)

gv.start_partition("TRAINED MODEL SAVED")
print("\n Model Saved Successfull..!")
gv.end_partition()



# ******************************************************
# ===== STEP 11: SAVE TEST DATASET =====
# ******************************************************
test_saved_path = os.path.join(gv.DATASET_DIR, "test_saved_data")

if os.path.exists(test_saved_path):
    os.removedirs(test_saved_path)
os.makedirs(test_saved_path)

np.save(f"{test_saved_path}\X_test.npy", X_test)
np.save(f"{test_saved_path}\y_test.npy", y_test)

gv.start_partition("TESTING DATASET SAVED")
print("Test dataset saved successfully..!")
gv.end_partition()
