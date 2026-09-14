# **************************************
# ===== STEP 1: IMPORT LIBRARIES =====
# **************************************
import global_variables as gv

import os
import json
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
      classification_report,
      roc_curve,
      auc
)
from sklearn.preprocessing import label_binarize


# **************************************
# ===== STEP 2: LOAD SAVED MODEL =====
# **************************************
# Evaluate the BEST model (same one predict.py uses), not the final-epoch model.
model_path = os.path.join(gv.TRAINED_MODEL_PATH, gv.BEST_MODEL_NAME)
model = tf.keras.models.load_model(model_path)
gv.start_partition("MODEL LOADED")
print(model)
gv.end_partition()

# Load class index mapping saved by train.py
class_index_path = os.path.join(gv.TRAINED_MODEL_PATH, gv.CLASS_INDEX_FILE)
with open(class_index_path, "r") as f:
    index_to_class = {int(k): v for k, v in json.load(f).items()}
class_names = [index_to_class[i] for i in range(len(index_to_class))]
NUM_CLASSES = len(class_names)


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
# ===== STEP 4: MAKE PREDICTIONS =====
# **************************************
predictions = model.predict(X_test)

if predictions.shape[-1] == 1:
    # Binary model
    y_pred = (predictions >= 0.5).astype(int).flatten()
    avg_method = "binary"
else:
    # Multi-class model - use argmax, and use "weighted" averaging
    # for precision/recall/f1 since classes are likely imbalanced.
    y_pred = np.argmax(predictions, axis=1)
    avg_method = "weighted"

gv.start_partition("PREDICTIONS")
print(f"Predictions Shape: {predictions.shape}")
print(f"Predicted Labels Shape: {y_pred.shape}")
gv.end_partition()


# **************************************
# ===== STEP 5: EVALUATION METRICS =====
# **************************************
gv.start_partition("EVALUATION METRICS")

# ----- ACCURACY -----
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")

plt.figure(figsize=(4, 4))
plt.bar(["Accuracy"], [accuracy], color="green")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Accuracy Score")
plt.text(0, accuracy + 0.02, f"{accuracy:.4f}", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(gv.GRAPH_DIR, gv.ACC_GRAPH))
plt.show()

# ----- PRECISION -----
precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
print(f"Precision Score: {precision:.4f}")

plt.figure(figsize=(4, 4))
plt.bar(["Precision"], [precision], color="royalblue")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Precision Score")
plt.text(0, precision + 0.02, f"{precision:.4f}", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(gv.GRAPH_DIR, gv.PRECISION_GRAPH))
plt.show()

# ----- RECALL -----
recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
print(f"Recall Score: {recall:.4f}")

plt.figure(figsize=(4, 4))
plt.bar(["Recall"], [recall], color="orange")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Recall Score")
plt.text(0, recall + 0.02, f"{recall:.4f}", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(gv.GRAPH_DIR, gv.RECALL_GRAPH))
plt.show()

# ----- F1 SCORE (graph was missing before - added here) -----
f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
print(f"F1 Score: {f1:.4f}")

plt.figure(figsize=(4, 4))
plt.bar(["F1 Score"], [f1], color="purple")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("F1 Score")
plt.text(0, f1 + 0.02, f"{f1:.4f}", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(gv.GRAPH_DIR, gv.F1_GRAPH))
plt.show()

# ----- CONFUSION MATRIX -----
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(gv.GRAPH_DIR, gv.CM_GRAPH))
plt.show()

# ----- CLASSIFICATION REPORT -----
cr = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
print(f"Classification Report:\n{cr}")

with open(os.path.join(gv.GRAPH_DIR, "classification_report.txt"), "w") as f:
    f.write(cr)


# **************************************
# ===== STEP 6: ROC-AUC CURVE =====
# **************************************
# Only meaningful when the model outputs per-class probabilities
# (multi-class softmax). Skipped for binary sigmoid models.
if predictions.shape[-1] > 1:
    y_test_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))

    plt.figure(figsize=(8, 6))
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cname} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (One-vs-Rest per Class)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(gv.GRAPH_DIR, "roc_curve.png"))
    plt.show()

gv.end_partition()