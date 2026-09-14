# ******************************************************
# ===== STEP 1: IMPORT LIBRARIES =====
# ******************************************************
import os
import cv2
import random
import requests

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import json



# ******************************************************
# ===== STEP 2: SET CONFIG =====
# ******************************************************
# ===== DATASET PATH =====
DATASET_DIR = "C:\\Users\Krishnam Mavani\OneDrive\Documents\Projects\Medical_AI_Assistant\dataset"
SPLITED_DATASET = "C:\\Users\Krishnam Mavani\OneDrive\Documents\Projects\Medical_AI_Assistant\splited_dataset"

SPLITED_TRAIN_DATASET = "C:\\Users\Krishnam Mavani\OneDrive\Documents\Projects\Medical_AI_Assistant\splited_dataset\\train.csv"
SPLITED_VAL_DATASET = "C:\\Users\Krishnam Mavani\OneDrive\Documents\Projects\Medical_AI_Assistant\splited_dataset\val.csv"
SPLITED_TEST_DATASET = "C:\\Users\Krishnam Mavani\OneDrive\Documents\Projects\Medical_AI_Assistant\splited_dataset\\test.csv"

CARDIOMEGALY = "CARDIOMEGALY"
COVID19 = "COVID19"
NORMAL = "NORMAL"
PNEUMONIA = "PNEUMONIA"
TUBERCULOSIS = "TUBERCULOSIS"
CATEGORIES = [CARDIOMEGALY, COVID19, NORMAL, PNEUMONIA, TUBERCULOSIS]
# CATEGORIES = [CARDIOMEGALY, COVID19]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ===== PREPROCESS VARIABLES =====
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 1
SEED = 42
LEARNING_RATE = 1e-4


# ===== GRAPH PATH =====
GRAPH_DIR = "C:\\Users\Krishnam Mavani\OneDrive\Documents\Projects\Medical_AI_Assistant\graphs"
# ACC_GRAPH = "accuracy_graph.png"
# LOSS_GRAPH = "loss_graph.png"
# CM_GRAPH = "confusion_matrix.png"
# AUG_GRAPH = "augmentation_examples.png"
# TRAINNING_ACC = "Training_and_Validation_Accuracy.png"


# ===== MODEL PATH =====

BEST_MODEL_PATH = "C:\\Users\Krishnam Mavani\OneDrive\Documents\Projects\Medical_AI_Assistant\models\\best_model.h5"
FINAL_BEST_MODEL_PATH = "C:\\Users\Krishnam Mavani\OneDrive\Documents\Projects\Medical_AI_Assistant\models\final_model.h5"
TRAINED_MODEL_PATH = "C:\\Users\Krishnam Mavani\OneDrive\Documents\Projects\Medical_AI_Assistant\models\medical_ai_cnn.keras"
CLASS_INDEX_PATH = "C:\\Users\Krishnam Mavani\OneDrive\Documents\Projects\Medical_AI_Assistant\models\class_indices.json"

# ===== LOG DIRECTORY PATH =====
LOG_DIR = "logs"



# ******************************************************
# ===== STEP 3: SET SEEDS =====
# ******************************************************
def set_seeds(seed=SEED):
      random.seed(seed)
      np.random.seed(seed)
      # tf.random.seed(seed)



# ******************************************************
# ===== STEP 3: SCAN DATASET & CHECK CURRPTED FILES =====
# ******************************************************
def scan_dataset(dataset_dir = DATASET_DIR):
      corrupted_files = []

      images_data = []
      images_label = []
      
      print(dataset_dir)
      for label, cate_name in enumerate(CATEGORIES):
            folder_path = os.path.join(f"{DATASET_DIR}\{cate_name}")
            print(f"Working in the '{folder_path}' directory")
            for img_name in os.listdir(folder_path):
                  img_path = os.path.join(folder_path, img_name)

                  img = cv2.imread(img_path)
                  if img is None:
                        corrupted_files.append({"File":img_path, "Label": {label}})
                        continue
                  images_data.append(img)
                  images_label.append(label)

      print(f"Found {len(CATEGORIES)} classes: {CATEGORIES}")
      print(f"Total valid images: {len(images_label)}")
      # print(f"Readable Files Sample:\n{images_label[:5]}")
      print(f"Corrupt/unreadable files skipped: {len(corrupted_files)}")
      if corrupted_files:
            print("Example corrupt files:", corrupted_files[:5])

      X = np.array(images_data)
      y = np.array(images_label)

      return X, y, CATEGORIES, corrupted_files



# ******************************************************
# ===== STEP 4: SPLIT DATASET =====
# ******************************************************
def split_train_validation_test(X, y, seed=SEED):
      train_ratio = TRAIN_RATIO
      val_ratio = VAL_RATIO
      test_ratio = TEST_RATIO

      X_train, y_train, X_temp, y_temp = train_test_split(
            X,
            y,
            train_size=train_ratio,
            random_state=seed
      )

      # from temp (temp_df) to val & test
      val_relative_ratio = val_ratio / (val_ratio + test_ratio)
      X_val, y_val, X_test, y_test = train_test_split(
            X_temp,
            y_temp,
            train_size=val_relative_ratio,
            random_state=seed
      )


      test_df = pd.DataFrame({"File":X_test,"Label":y_test})
      val_df = pd.DataFrame({"File":X_val,"Label":y_val})
      train_df = pd.DataFrame({"File":X_train,"Label":y_train})
      print(type(test_df))
      if os.path.exists(SPLITED_DATASET):
             os.removedirs(SPLITED_DATASET)      
      os.makedirs(SPLITED_DATASET)

      np.save(f"{SPLITED_TEST_DATASET}\X_test.npy")
      np.save(f"{SPLITED_TEST_DATASET}\y_test.npy")
      np.save(f"{SPLITED_VAL_DATASET}\X_val.npy")
      np.save(f"{SPLITED_VAL_DATASET}\y_val.npy")
      np.save(f"{SPLITED_TRAIN_DATASET}\X_train.npy")
      np.save(f"{SPLITED_TRAIN_DATASET}\y_train.npy")

      print(f"Train Len: {len(train_df)} | Validation Len: {len(val_df)} | Test Len: {len(test_df)}")
      print(f"Datasets Saved at:\n  Train DF: {SPLITED_TRAIN_DATASET}\n  Val DF: {SPLITED_VAL_DATASET}\n  Test DF: {SPLITED_TEST_DATASET}")

      return train_df, val_df, test_df



# ******************************************************
# ===== STEP 5: LOAD SPLITED DATASET FUNCTION =====
# ******************************************************
def load_splitted_dataset(path):
      return pd.read_csv(path)



# ******************************************************
# ===== STEP 6: OPENCV PRE-PROCESSING IMAGES =====
# ******************************************************
def preprocessing_images(img):
      img_size = IMAGE_SIZE

      if len(img.shape) == 2: # grayscale image shape is 2 | 2 channel (BLACK, WHITE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
      elif img.shape[2] == 4: # 4 channel (BLUR, GREEN, RED, ALPHA)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

      #resize
      img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

      # low quality image read.
      img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

      # contrast enhancment on L channel (0 black to 100 white)
      lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
      l, a, b = cv2.split(lab)
      clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
      l = clahe.apply(l)
      lab = cv2.merge((l, a, b))
      img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

      # normalize
      img = img.astype("float32") / 255.0



# ******************************************************
# ===== STEP 7: PRE-PROCESSING FOR AUGMENTATION =====
# ******************************************************
def preprocessing_for_augmentation(img):
      img = img.astype("uint8")
      img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      processed = preprocessing_images(img_bgr)
      processed_rgb = cv2.cvtColor((processed * 255).astype("uint8"), cv2.COLOR_BGR2RGB)

      return processed_rgb.astype("float32") / 255.0



# ******************************************************
# ===== STEP 8: LOAD IMAGE FOR TESTING =====
# ******************************************************
def load_image_for_testing_from_url_or_path(path_or_url):
      if path_or_url.startswith("https://") or path_or_url.startswith("http://"):
            response = requests.get(
                  path_or_url,
                  timeout=20
            )

            if response.status_code != 200:
                  raise Exception(f"Could Not Download Image From: {path_or_url}")

            img_ary  = np.frombuffer(
                  response.content,
                  np.uint8
            )

            web_img = cv2.imdecode(
                  img_ary,
                  cv2.IMREAD_COLOR
            )

            if web_img is None:
                  raise Exception(f"OpenCV Could Not Read Image From: {path_or_url}")

            img = cv2.cvtColor(web_img, cv2.COLOR_BGR2RGB)

      else:
            img = cv2.imread(path_or_url)

      if img is None:
            raise Exception(f"Could Not Read Image From: {path_or_url}")


      return img



# ******************************************************
# ===== STEP 9: SAVE CLASS INDEX MAPPING =====
# ******************************************************
def save_class_indices_map(class_indexes, path=CLASS_INDEX_PATH):
      inv_map = {v: k for k, v in class_indexes.items()}
      with open(path, "w") as f:
            json.dump(inv_map, f, indent=2)



# ******************************************************
# ===== STEP 10: LOAD CLASS INDEX MAPPING =====
# ******************************************************
def load_class_indices(path=CLASS_INDEX_PATH):
      with open(path, "r") as f:
            inv_map = json.load(f)

      return {int(k): v for k, v in inv_map.items()}



# ******************************************************
# ===== STEP 11: VISUALIZING FUNCTIONS =====
# ******************************************************
# ===== GRID IMAGES =====
def plot_image_grid(imgs, titels=None, cols=5, main_title=""):
      rows = (len(imgs) + cols - 1) // cols

      plt.figure(figsize=(cols*2.5, rows*2.5))
      for i, img in enumerate(imgs):
            plt.subplot(rows, cols, i+1)

            if img.max() <= 1.0:
                  plt.imshow(cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_BGR2RGB))
            else:
                  plt.imshow(cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2RGB))
            if titels:
                  plt.title(titels[i], fontsize=9)
            plt.axis("off")

      plt.suptitle(main_title)
      plt.tight_layout()
      plt.show()


# ===== ORIGINAL V/S PREPROCESSED =====
def plot_original_vs_processed(original_bgr, processed_bgr, title="Original vs Preprocessed"):
      plt.figure(figsize=(8, 4))

      plt.subplot(1, 2, 1)
      plt.imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
      plt.title("Original")
      plt.axis("off")

      plt.subplot(1, 2, 2)
      if processed_bgr.max() <= 1.0:
            disp = (processed_bgr * 255).astype("uint8")
      else:
            disp = processed_bgr.astype("uint8")
      plt.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
      plt.title("Preprocessed")
      plt.axis("off")

      plt.suptitle(title)
      plt.tight_layout()
      plt.show()


# ===== ORIGINAL V/S PREPROCESSED =====
def draw_data_plot(size, data, titels, fig_name, kind):
      plt.figure(figsize=size)
      data.plot(kind=kind)
      plt.title(titels[0])
      plt.xlabel(titels[1])
      plt.ylabel(titels[2])
      plt.tight_layout()
      plt.savefig(os.path.join(GRAPH_DIR, fig_name))
      plt.show()

def draw_histogram_plot(size, data, titels, fig_name, bin):
      plt.figure(figsize=size)
      plt.hist(data, bins=30)
      plt.title(titels[0])
      plt.xlabel(titels[1])
      plt.ylabel(titels[2])
      plt.tight_layout()
      plt.savefig(os.path.join(GRAPH_DIR, fig_name))
      plt.show()



# ******************************************************
# ===== STEP 12: PARTITIONS PRINT FUNCTIONS =====
# ******************************************************
# ===== START PARTITION =====
def start_partition(title):
      print("\n")
      print("="*100)
      print("-"*35, end=" ")
      print(f"{title}", end=" ")
      print("-"*35)
      print("="*100)


# ===== END PARTITION =====
def end_partition():
      print("*"*100)
      print("\n")
