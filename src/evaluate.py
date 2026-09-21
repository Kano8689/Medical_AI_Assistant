# **************************************
# ===== STEP 1: IMPORT LIBRARIES =====
# **************************************
import os
import json
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

from global_variables import GRAPH_DIR, TRAINED_MODEL_PATH, CLASS_INDEX_FILE, DATASET_DIR, ACC_GRAPH, PRECISION_GRAPH, RECALL_GRAPH, F1_GRAPH, CM_GRAPH, TEST_SAVED_PATH, MODEL_DIR
from global_variables import start_partition, end_partition

# **************************************
# ===== STEP 2: LOAD SAVED MODEL =====
# **************************************
# Evaluate the BEST model (same one predict.py uses), not the final-epoch model.
model = tf.keras.models.load_model(TRAINED_MODEL_PATH)
start_partition("MODEL LOADED")
print(model)
end_partition()

# Load class index mapping saved by train.py
class_index_path = os.path.join(MODEL_DIR, CLASS_INDEX_FILE)
with open(class_index_path, "r") as f:
    index_to_class = {int(k): v for k, v in json.load(f).items()}
class_names = [index_to_class[i] for i in range(len(index_to_class))]
NUM_CLASSES = len(class_names)


# **************************************
# ===== STEP 3: LOAD TEST DATASET =====
# **************************************
X_test = np.load(os.path.join(TEST_SAVED_PATH, "X_test.npy"))
y_test = np.load(os.path.join(TEST_SAVED_PATH, "y_test.npy"))

start_partition("TEST DATASET")
print(f"X_test Shape: {X_test.shape}")
print(f"y_test Shape: {y_test.shape}")
end_partition()


# **************************************
# ===== STEP 4: MAKE PREDICTIONS =====
# **************************************
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
avg_method = "weighted"

start_partition("PREDICTIONS")
print(f"Predictions Shape: {predictions.shape}")
print(f"Predicted Labels Shape: {y_pred.shape}")
end_partition()


# **************************************
# ===== STEP 5: EVALUATION METRICS =====
# **************************************
def plot_metric(bar, metrics, color, title, graph):
    plt.figure(figsize=(4, 4))
    plt.bar([bar], [metrics], color=color)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(title)
    plt.text(0, metrics + 0.02, f"{metrics:.4f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"evaluation_{graph}"))
    plt.show()

start_partition("EVALUATION METRICS")

# ----- ACCURACY -----
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")
plot_metric("Accuracy", accuracy, "green", "Accuracy Score", ACC_GRAPH)

# ----- PRECISION -----
precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
print(f"Precision Score: {precision:.4f}")
plot_metric("Precision", precision, "royalblue", "Precision Score", PRECISION_GRAPH)

# ----- RECALL -----
recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
print(f"Recall Score: {recall:.4f}")
plot_metric("Recall", recall, "orange", "Recall Score", RECALL_GRAPH)

# ----- F1 SCORE (graph was missing before - added here) -----
f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
print(f"F1 Score: {f1:.4f}")
plot_metric("F1 Score", f1, "purple", "F1 Score", F1_GRAPH)

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
plt.savefig(os.path.join(GRAPH_DIR, f"evaluation_{CM_GRAPH}"))
plt.show()

# ----- CLASSIFICATION REPORT -----
cr = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
print(f"Classification Report:\n{cr}")

with open(os.path.join(GRAPH_DIR, "classification_report.txt"), "w") as f:
    f.write(cr)


# **************************************
# ===== STEP 6: ROC-AUC CURVE =====
# **************************************
# Only meaningful when the model outputs per-class probabilities
# (multi-class softmax). Skipped for binary sigmoid models.
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
plt.savefig(os.path.join(GRAPH_DIR, "roc_curve.png"))
plt.show()

end_partition()