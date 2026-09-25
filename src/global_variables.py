import os

# ******************************************************
# ===== LOG PARTITION =====
# ******************************************************
def start_partition(title):
    print("\n")
    print("="*100)
    print("-"*35, end=" ")
    print(f"{title}", end=" ")
    print("-"*35)
    print("="*100)

def end_partition():
    print("*"*100)
    print("\n")



# ******************************************************
# ===== DATASET PATH =====
# ******************************************************
CURRENT_FILE = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_FILE)
BASE_DIR = os.path.dirname(SRC_DIR)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
LOG_DIR = os.path.join(BASE_DIR, "logs")
TEST_SAVED_PATH = os.path.join(BASE_DIR, "test_saved_data")
SPLITED_DATASET = os.path.join(BASE_DIR, "splited_dataset")
EXTERNAL_TEST_DIR = os.path.join(BASE_DIR, "external_test_dataset")

# Disease categories
NORMAL = "NORMAL"
PNEUMONIA = "PNEUMONIA"
TUBERCULOSIS = "TUBERCULOSIS"
CARDIOMEGALY = "CARDIOMEGALY"

CATEGORIES = [NORMAL, PNEUMONIA, TUBERCULOSIS, CARDIOMEGALY]
NUM_CLASSES = len(CATEGORIES)



# ******************************************************
# ===== PREPROCESS VARIABLES =====
# ******************************************************
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
L2_REG = 5e-4
LEARNING_RATE = 1e-4
LABEL_SMOOTHING = 0.1
DROPOUT_HEAD = 0.5

TEST_SPLIT = 0.20
VAL_FROM_TEMP_SPLIT = 0.50
SPLIT_BY_PATIENT = False

EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

AUGMENTATION_ROTATION = 15
AUGMENTATION_ZOOM = 0.15
AUGMENTATION_SHIFT = 0.1
AUGMENTATION_FLIP = True



# ******************************************************
# ===== GRAPH PATH =====
# ******************************************************
ACC_GRAPH = "accuracy_graph.png"
LOSS_GRAPH = "loss_graph.png"
PRECISION_GRAPH = "precision_graph.png"
RECALL_GRAPH = "recall_graph.png"
F1_GRAPH = "f1_graph.png"
CM_GRAPH = "confusion_matrix.png"
AUG_GRAPH = "augmentation_examples.png"
CLASS_DIST_GRAPH = "class_distribution.png"
ORIGINAL_IMAGES_GRAPH = "original_vs_preprocessed.png"
TRAINNING_ACC = "Training_and_Validation_Accuracy.png"



# ******************************************************
# ===== TRAIN MODEL PATH =====
# ******************************************************
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
TRAINED_MODEL_PATH = os.path.join(MODEL_DIR, "medical_ai_cnn.keras")



# ******************************************************
# ===== LOG DIRECTORY PATH =====
# ******************************************************
CLASS_INDEX_FILE = "class_indices.json"
