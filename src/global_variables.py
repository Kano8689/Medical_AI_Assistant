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
DATASET_DIR = "D:\Data Science Projects\Medical_AI_Assistant\dataset"
DATASET_NAME = "chest_xray"
NORMAL_NAME = "NORMAL"
PNEUMONIA_NAME = "PNEUMONIA"
CATEGORIES = [NORMAL_NAME, PNEUMONIA_NAME]

# ******************************************************
# ===== PREPROCESS VARIABLES =====
# ******************************************************
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 1

# ******************************************************
# ===== GRAPH PATH =====
# ******************************************************
GRAPH_PATH = "D:\Data Science Projects\Medical_AI_Assistant\graphs"
ACC_GRAPH = "accuracy_graph.png"
LOSS_GRAPH = "loss_graph.png"
CM_GRAPH = "confusion_matrix.png"
AUG_GRAPH = "augmentation_examples.png"
TRAINNING_ACC = "Training_and_Validation_Accuracy.png"


# ******************************************************
# ===== TRAIN MODEL PATH =====
# ******************************************************
TRAINED_MODEL_PATH = "D:\Data Science Projects\Medical_AI_Assistant\models"
TRAINED_MODEL_NAME = "medical_ai_cnn.keras"

# ******************************************************
# ===== LOG DIRECTORY PATH =====
# ******************************************************

LOG_DIR = "logs"



