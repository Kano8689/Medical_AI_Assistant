# ****************************************
# ===== STEP 1: IMPORT LIBRARIES =====
# ****************************************
import global_variables as gv

import os
import cv2
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input



# ****************************************
# ===== STEP 2: LOAD SAVED MODEL =====
# ****************************************
model_path = os.path.join(gv.TRAINED_MODEL_PATH, gv.TRAINED_MODEL_NAME)
model = tf.keras.models.load_model(model_path)
gv.start_partition("MODEL LOADED")
print(model)
gv.end_partition()



# ****************************************
# ===== STEP 3: ASK FOR IMAGE PATH =====
# ****************************************
predict_image_path = input("Enter Your Image Path to Classify: ")



# ****************************************
# ===== STEP 4:  LOAD IMAGE =====
# ****************************************
img = cv2.imread(predict_image_path)
if img is None:
      print("ERROR: Unable to load Image!")
      exit()
print("Image Loaded Successfully!")



# ****************************************
# ===== STEP 4:  SHOW LOADED IMAGE =====
# ****************************************
plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original X-ray")
plt.axis("off")
plt.show()


# ****************************************
# ===== STEP 5: PREPROCESS IMAGE =====
# ****************************************
Img_Size = gv.IMAGE_SIZE

img = cv2.resize(img, (Img_Size, Img_Size))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = preprocess_input(img.astype(np.float32))
img = np.expand_dims(img,axis=0)



# ****************************************
# ===== STEP 6: PREDICT =====
# ****************************************
prediction = model.predict(img)
probability = prediction[0][0]

if prediction[0][0] > 0.5:
  answer = "PNEUMONIA"
  confidence = probability
else:
  answer = "NORMAL"
  confidence = 1 - probability



# ****************************************
# ===== STEP 7: DISPLAY =====
# ****************************************
gv.start_partition("FINAL RESULT")
print(f"Predicted Class: {answer}")
print(f"Confidence: {confidence * 100:.2f}%")
gv.end_partition()
