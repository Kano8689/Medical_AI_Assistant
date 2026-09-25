# ******************************************************
# ===== 1. IMPORT LIBRARIES =====
# ******************************************************
import os
import cv2
import numpy as np
import requests



# ******************************************************
# ===== 2. LOG PARTITION HELPERS =====
# ******************************************************
def start_partition(title: str):
    print("\n" + "=" * 80)
    print("-"*20 + title.upper() + "-"*20)
    print("=" * 80)

def end_partition():
    print("*" * 80 + "\n")



# ******************************************************
# ===== 3. PATH CONFIGURATIONS =====
# ******************************************************
CURRENT_FILE = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_FILE)
BASE_DIR = os.path.dirname(SRC_DIR)
# BASE_DIR = r"C:\Users\Krishnam Mavani\Documents\Projects\Medical_AI_Assistant"

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SPLITED_SAVED_PATH = os.path.join(BASE_DIR, "splited_saved_data")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SPLITED_SAVED_PATH, exist_ok=True)



# ******************************************************
# ===== 4. CATEGORIES & HYPERPARAMETERS =====
# ******************************************************
CARDIOMEGALY = "CARDIOMEGALY"
COVID19 = "COVID19"
NORMAL = "NORMAL"
PNEUMONIA = "PNEUMONIA"
TUBERCULOSIS = "TUBERCULOSIS"

CATEGORIES = [CARDIOMEGALY, COVID19, NORMAL, PNEUMONIA, TUBERCULOSIS]
NUM_CLASSES = len(CATEGORIES)
CLASS_INDEX_FILE = "class_indices.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
L2_REG = 1e-4

TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
TRAINED_MODEL_PATH = os.path.join(MODEL_DIR, "medical_ai_cnn.keras")



# ******************************************************
# ===== 5. TRAINNING PARAMETERS =====
# ******************************************************
ROTATION_RANGE = 12
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
ZOOM_RANGE = 0.1
HORIZONTAL_FLIP = True
FILL_MODE = "nearest"

LOSS_FUNCTION = "categorical_crossentropy"



# ******************************************************
# ===== 6. REUSABLE PREPROCESSING FUNCTIONS =====
# ******************************************************
def download_or_read_image(path_or_url: str) -> np.ndarray:
    """Handles reading local files or fetching images securely from web URLs."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(path_or_url, timeout=15, headers=headers)
        if response.status_code != 200:
            raise Exception(f"HTTP Error {response.status_code}: Could not fetch image URL.")
        img_array = np.frombuffer(response.content, np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise Exception("OpenCV failed to decode image from URL stream.")
    else:
        img_bgr = cv2.imread(path_or_url)
        if img_bgr is None:
            raise Exception(f"Failed to read local image at path: {path_or_url}")
    return img_bgr

def process_single_image(img_bgr: np.ndarray):
    """
    Applies resizing, CLAHE contrast enhancement on the L-channel of LAB space,
    and normalization. Returns displayable RGB and model input array.
    """
    resized = cv2.resize(img_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)
    
    lab_merged = cv2.merge((l_enhanced, a_chan, b_chan))
    preprocessed_rgb = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)
    
    model_input = preprocessed_rgb.astype("float32") / 255.0
    return preprocessed_rgb, model_input
