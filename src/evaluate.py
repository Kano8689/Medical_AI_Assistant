# ******************************************************
# ===== 1. IMPORT LIBRARIES =====
# ******************************************************
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

from global_variables import (
    GRAPH_DIR, TRAINED_MODEL_PATH, CLASS_INDEX_FILE,
    SPLITED_SAVED_PATH, MODEL_DIR, start_partition, end_partition
)

model = tf.keras.models.load_model(TRAINED_MODEL_PATH)

with open(os.path.join(MODEL_DIR, CLASS_INDEX_FILE), "r") as f:
    index_to_class = {int(k): v for k, v in json.load(f).items()}
class_names = [index_to_class[i] for i in range(len(index_to_class))]
NUM_CLASSES = len(class_names)

X_test = np.load(os.path.join(SPLITED_SAVED_PATH, "X_test.npy"))
y_test = np.load(os.path.join(SPLITED_SAVED_PATH, "y_test.npy"))

start_partition("EVALUATION ON TEST DATA")
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")



# ******************************************************
# ===== 2. Confusion Matrix =====
# ******************************************************
# cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "confusion_matrix.png"))
plt.close()



# ******************************************************
# ===== 3. Classification Report =====
# ******************************************************
cr = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
print("\nClassification Report:\n", cr)
with open(os.path.join(GRAPH_DIR, "classification_report.txt"), "w") as f:
    f.write(cr)



# ******************************************************
# ===== 4. ROC-AUC Curves =====
# ******************************************************
y_test_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
plt.figure(figsize=(9, 7))
for i, cname in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], predictions[:, i])
    plt.plot(fpr, tpr, label=f"{cname} (AUC = {auc(fpr, tpr):.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curves")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "roc_curve.png"))
plt.close()
end_partition()
