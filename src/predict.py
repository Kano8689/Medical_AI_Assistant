# ****************************************
# ===== STEP 1: IMPORT LIBRARIES =====
# ****************************************
import os
import json
import requests
import urllib.request
import cv2
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from global_variables import TRAINED_MODEL_PATH, CLASS_INDEX_FILE, IMAGE_SIZE, CATEGORIES
from global_variables import start_partition, end_partition 



# ****************************************
# ===== STEP 2: LOAD SAVED MODEL =====
# ****************************************
# Loads the BEST model (saved by ModelCheckpoint during training)
model = tf.keras.models.load_model(TRAINED_MODEL_PATH)
start_partition("MODEL LOADED")
print(model)
end_partition()

# Load class index mapping saved by train.py (index -> class name)
class_index_path = os.path.join(TRAINED_MODEL_PATH, CLASS_INDEX_FILE)
if os.path.exists(class_index_path):
    with open(class_index_path, "r") as f:
        index_to_class = {int(k): v for k, v in json.load(f).items()}
else:
    # fallback: assume the order in global_variables.py matches training order
    index_to_class = {i: name for i, name in enumerate(CATEGORIES)}



# ****************************************
# ===== STEP 3: ASK FOR IMAGE PATH (local path OR website URL) =====
# ****************************************
predict_image_path = input("Enter Your Image Path or URL to Classify: ").strip()



# ****************************************
# ===== STEP 4: LOAD IMAGE (downloads it first if it's a URL) =====
# ****************************************
def load_image(path_or_url):
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        response = requests.get(
            path_or_url,
            timeout=20
        )

        if response.status_code != 200:
            raise Exception(f"Could not download image from {path_or_url}.")
        
        img_ary = np.frombuffer(
            response.content,
            np.uint8
        )

        img = cv2.imdecode(
            img_ary,
            cv2.IMREAD_COLOR
        )

        if img is None:
            raise Exception(f"OpenCV Could Not Read Image From: {path_or_url}")

        print(f"Image is downloaded successfully.")

    else:
        img = cv2.imread(path_or_url)

        if img is None:
            raise Exception(f"Could Not Read Image From: {path_or_url}")


    return img

img = load_image(predict_image_path)
original_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cv2.imshow(original_img_rgb)



# ****************************************
# ===== STEP 5: PREPROCESS IMAGE =====
# ****************************************
img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=2.0)
l = clahe.apply(l)

lab = cv2.merge((l, a, b))
img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

preprocessed_display = img.copy()  # uint8, for display

img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)



# ****************************************
# ===== STEP 6: SHOW ORIGINAL vs PREPROCESSED IMAGE =====
# ****************************************
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_img_rgb)
plt.title("Original X-ray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(preprocessed_display)
plt.title("Preprocessed X-ray")
plt.axis("off")

plt.tight_layout()
plt.show()



# ****************************************
# ===== STEP 7: PREDICT =====
# ****************************************
prediction = model.predict(img)
pred_index = int(np.argmax(prediction[0]))
confidence = float(prediction[0][pred_index])

answer = index_to_class[pred_index]


 
# ****************************************
# ===== STEP 8: DISPLAY FINAL RESULT =====
# ****************************************
start_partition("FINAL RESULT")
print(f"Predicted Class: {answer}")
print(f"Confidence: {confidence * 100:.2f}%")
end_partition()

# Show the original image again with the predicted result as the title
plt.figure(figsize=(5, 5))
plt.imshow(original_img_rgb)
plt.title(f"Prediction: {answer} ({confidence * 100:.2f}%)")
plt.axis("off")
plt.show()