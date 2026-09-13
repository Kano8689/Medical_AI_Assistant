# **************************************
# ===== STEP 1: IMPORT LIBRARIES =====
# **************************************
import global_variables as gv

import os 
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
      accuracy_score,
      precision_score,
      recall_score,
      f1_score,
      confusion_matrix,
      classification_report
)



# **************************************
# ===== STEP 2: LOAD SAVED MODEL =====
# **************************************
model_path = os.path.join(gv.TRAINED_MODEL_PATH, gv.TRAINED_MODEL_NAME)
model = tf.keras.models.load_model(model_path)
gv.start_partition("MODEL LOADED")
print(model)
gv.end_partition()



# **************************************
# ===== STEP 3: LOAD TEST DATASET =====
# **************************************
test_saved_path = os.path.join(gv.DATASET_DIR, "test_saved_data")

X_test = np.load(os.path.join(test_saved_path, "X_test.npy"))
y_test = np.load(os.path.join(test_saved_path, "y_test.npy"))

gv.start_partition("TEST DATASET")
print(f"X_test Shape: {X_test.shape}")
print(f"y_test Shape: {y_test.shape}")
gv.end_partition()



# **************************************
# ===== STEP 4:  MAKE PREDICTIONS =====
# **************************************
predictions = model.predict(X_test)
y_pred = (predictions >= 0.5).astype(int).flatten()

gv.start_partition("PREDICTIONS")
print(f"Predictions Shape: {predictions.shape}")
print(f"Predicted Labels Shape: {y_pred.shape}")
gv.end_partition()



#**************************************
# ===== STEP 5: EVALUATION METRICS =====
# **************************************


gv.start_partition("EVALUATION METRICS")
#**************************************
# ===== STEP 5: ACCURACY SCORE =====
# **************************************
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")

plt.figure(figsize=(4,4))

plt.bar(["Accuracy"], [accuracy], color="green")

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Accuracy Score")

plt.text(0, accuracy + 0.02, f"{accuracy:.4f}", ha="center")

plt.tight_layout()
plt.show()



#**************************************
# ===== STEP 5: PRECISION =====
# **************************************
precision = precision_score(y_test, y_pred)
print(f"Precision Score: {precision:.4f}")

plt.figure(figsize=(4,4))

plt.bar(["Precision"], [precision], color="royalblue")

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Precision Score")

plt.text(0, precision + 0.02, f"{precision:.4f}", ha="center")

plt.tight_layout()
plt.show()



#**************************************
# ===== STEP 5: RECALL SCORE =====
# **************************************
recall = recall_score(y_test, y_pred)
print(f"Recall Score: {recall:.4f}")
plt.figure(figsize=(4,4))

plt.bar(["Recall"], [recall])

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Recall Score")

plt.text(0, recall + 0.02, f"{recall:.4f}", ha="center")

plt.tight_layout()
plt.show()



#**************************************
# ===== STEP 5: F1 SCORE =====
# **************************************
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")

#**************************************
# ===== STEP 5: CONFUSION MATRIX =====
# **************************************
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d",
    xticklabels = gv.CATEGORIES,
    yticklabels = gv.CATEGORIES
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#**************************************
# ===== STEP 5: CLASSIFICATION REPORT =====
# **************************************
cr = classification_report(y_test, y_pred)
print(f"Classification Report:\n{cr}")
print(classification_report(
      y_test,
      y_pred,
      target_names=gv.CATEGORIES
))


gv.end_partition()
