# **************************************
# ===== STEP 1: IMPORT LIBRARIES =====
# **************************************
# import global_variables as gv

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

from global_variables import GRAPH_DIR, TRAINED_MODEL_PATH, MODEL_DIR, CLASS_INDEX_FILE, ACC_GRAPH, PRECISION_GRAPH, RECALL_GRAPH, F1_GRAPH, CM_GRAPH, TEST_SAVED_PATH
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

if predictions.shape[-1] == 1:
    # Binary model
    y_pred = (predictions >= 0.5).astype(int).flatten()
    avg_method = "binary"
else:
    # Multi-class model - use argmax, and use "weighted" averaging
    # for precision/recall/f1 since classes are likely imbalanced.
    y_pred = np.argmax(predictions, axis=1)
    avg_method = "weighted"

start_partition("PREDICTIONS")
print(f"Predictions Shape: {predictions.shape}")
print(f"Predicted Labels Shape: {y_pred.shape}")
end_partition()


# **************************************
# ===== STEP 5: EVALUATION METRICS =====
# **************************************
def plot_metric_bar(metric_name, value, color, filename):
    plt.figure(figsize=(4, 4))
    plt.bar([metric_name], [value], color=color)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"{metric_name} Score")
    plt.text(0, value + 0.02, f"{value:.4f}", ha="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"evaluation_{filename}"))
    plt.show()

start_partition("EVALUATION METRICS")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")
plot_metric_bar("Accuracy", accuracy, "green", ACC_GRAPH)

precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
print(f"Precision Score: {precision:.4f}")
plot_metric_bar("Precision", precision, "royalblue", PRECISION_GRAPH)

recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
print(f"Recall Score: {recall:.4f}")
plot_metric_bar("Recall", recall, "orange", RECALL_GRAPH)

f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
print(f"F1 Score: {f1:.4f}")
plot_metric_bar("F1 Score", f1, "purple", F1_GRAPH)

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")

plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 12})
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, f"evaluation_{CM_GRAPH}"))
plt.show()

cr = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
print(f"\nClassification Report:\n{cr}")

with open(os.path.join(GRAPH_DIR, "classification_report.txt"), "w") as f:
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
    plt.savefig(os.path.join(GRAPH_DIR, "roc_curve.png"))
    plt.show()

end_partition()