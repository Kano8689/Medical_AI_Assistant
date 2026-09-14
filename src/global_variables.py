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
DATASET_DIR = r"C:\\Users\Krishnam Mavani\Documents\Projects\Medical_AI_Assistant\dataset"
SPLITED_DATASET = r"C:\\Users\Krishnam Mavani\Documents\Projects\Medical_AI_Assistant\splited_dataset"

SPLITED_TRAIN_DATASET = r"C:\\Users\Krishnam Mavani\Documents\Projects\Medical_AI_Assistant\splited_dataset"
SPLITED_VAL_DATASET = r"C:\\Users\Krishnam Mavani\Documents\Projects\Medical_AI_Assistant\splited_dataset"
SPLITED_TEST_DATASET = r"C:\\Users\Krishnam Mavani\Documents\Projects\Medical_AI_Assistant\splited_dataset"

TEST_SAVED_PATH = r"C:\\Users\Krishnam Mavani\Documents\Projects\Medical_AI_Assistant\test_saved_data"

CARDIOMEGALY = "CARDIOMEGALY"
COVID19 = "COVID19"
NORMAL = "NORMAL"
PNEUMONIA = "PNEUMONIA"
TUBERCULOSIS = "TUBERCULOSIS"

CATEGORIES = [
      NORMAL,
      PNEUMONIA,
      COVID19,
      TUBERCULOSIS,
      CARDIOMEGALY,
]
NUM_CLASSES = len(CATEGORIES)

# ******************************************************
# ===== PREPROCESS VARIABLES =====
# ******************************************************
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30

# ----- TRAIN / VAL / TEST SPLIT -----
TEST_SPLIT = 0.30
VAL_FROM_TEMP_SPLIT = 0.50

# ----- CALLBACK SETTINGS -----
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7
LEARNING_RATE = 0.001

# ******************************************************
# ===== GRAPH PATH =====
# ******************************************************
GRAPH_DIR = r"C:\\Users\Krishnam Mavani\Documents\Projects\Medical_AI_Assistant\graphs"
ACC_GRAPH = "accuracy_graph.png"
LOSS_GRAPH = "loss_graph.png"
PRECISION_GRAPH = "precision_graph.png"
RECALL_GRAPH = "recall_graph.png"
F1_GRAPH = "f1_graph.png"
CM_GRAPH = "confusion_matrix.png"
AUG_GRAPH = "augmentation_examples.png"
CLASS_DIST_GRAPH = "class_distribution.png"
ORIGINAL_IMAGES_GRAPH = "original_images.png"
PREPROCESSED_IMAGES_GRAPH = "preprocessed_images.png"
TRAINNING_ACC = "Training_and_Validation_Accuracy.png"


# ******************************************************
# ===== TRAIN MODEL PATH =====
# ******************************************************
BEST_MODEL_PATH = r"C:\Users\Krishnam Mavani\Documents\Projects\Medical_AI_Assistant\models\\best_model.keras"
FINAL_BEST_MODEL_PATH = r"C:\\Users\Krishnam Mavani\Documents\Projects\Medical_AI_Assistant\models\final_model.keras"
TRAINED_MODEL_PATH = r"C:\Users\Krishnam Mavani\Documents\Projects\Medical_AI_Assistant\models\medical_ai_cnn.keras"

# ******************************************************
# ===== LOG DIRECTORY PATH =====
# ******************************************************
LOG_DIR = "logs"
CLASS_INDEX_FILE = r"C:\\Users\Krishnam Mavani\Documents\Projects\Medical_AI_Assistant\models\\class_indices.json"