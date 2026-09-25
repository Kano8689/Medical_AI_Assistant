# ******************************************************
# ===== STEP 1: IMPORT LIBRARIES =====
# ******************************************************
import os
import json
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from global_variables import (
    GRAPH_DIR, TRAINED_MODEL_PATH, CLASS_INDEX_FILE,
    TEST_SAVED_PATH, MODEL_DIR, CATEGORIES,
    ACC_GRAPH, PRECISION_GRAPH, RECALL_GRAPH, F1_GRAPH, CM_GRAPH,
    start_partition, end_partition
)



# ******************************************************
# ===== STEP 2: LOAD MODEL =====
# ******************************************************
model = tf.keras.models.load_model(TRAINED_MODEL_PATH)
start_partition("MODEL LOADED")
print(f"Model: {TRAINED_MODEL_PATH}")
end_partition()

class_index_path = os.path.join(MODEL_DIR, CLASS_INDEX_FILE)
with open(class_index_path, "r") as f:
    index_to_class = {int(k): v for k, v in json.load(f).items()}
class_names = [index_to_class[i] for i in range(len(index_to_class))]
NUM_CLASSES = len(class_names)



# ******************************************************
# ===== STEP 3: LOAD TEST DATA =====
# ******************************************************
X_test = np.load(os.path.join(TEST_SAVED_PATH, "X_test.npy"))
y_test = np.load(os.path.join(TEST_SAVED_PATH, "y_test.npy"))

start_partition("TEST DATASET")
print(f"X_test Shape: {X_test.shape}")
print(f"y_test Shape: {y_test.shape}")
end_partition()



# ******************************************************
# ===== STEP 4: PREDICTIONS =====
# ******************************************************
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
avg_method = "weighted"

start_partition("PREDICTIONS")
print(f"Predictions Shape: {predictions.shape}")
print(f"Predicted Labels Shape: {y_pred.shape}")
end_partition()



# ******************************************************
# ===== STEP 5: EVALUATION METRICS =====
# ******************************************************
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



# ******************************************************
# ===== STEP 6: ROC-AUC CURVE =====
# ******************************************************
y_test_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))

plt.figure(figsize=(10, 8))
for i, cname in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], predictions[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{cname} (AUC = {roc_auc:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], "k--", label="Random guess", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (One-vs-Rest per Class)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "roc_curve.png"))
plt.show()

end_partition()



# ******************************************************
# ===== STEP 7: PER-CLASS ANALYSIS =====
# ******************************************************
start_partition("PER-CLASS PERFORMANCE")
for i, class_name in enumerate(class_names):
    class_mask = (y_test == i)
    class_correct = (y_pred == i) & class_mask
    class_total = np.sum(class_mask)
    class_tp = np.sum(class_correct)
    
    if class_total > 0:
        class_accuracy = class_tp / class_total
        print(f"{class_name}: {class_tp}/{class_total} correct ({class_accuracy:.2%})")
end_partition()