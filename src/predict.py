# ****************************************
# ===== STEP 1: IMPORT LIBRARIES =====
# ****************************************
import os
import json
import requests
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

# Tkinter Imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Project-specific imports
from global_variables import TRAINED_MODEL_PATH, CLASS_INDEX_FILE, IMAGE_SIZE, CATEGORIES
from global_variables import start_partition, end_partition


# ****************************************
# ===== STEP 2: LOAD SAVED MODEL =====
# ****************************************
model = tf.keras.models.load_model(TRAINED_MODEL_PATH)
start_partition("MODEL LOADED")
print(model)
end_partition()

# Load class index mapping saved by train.py (index -> class name)
class_index_path = os.path.join(TRAINED_MODEL_PATH, CLASS_INDEX_FILE)
if os.path.exists(class_index_path):
    with open(class_index_path, "r") as f:
        index_to_class = {int(k): v for k, v in json.load(f).items()}
else:
    # Fallback: assume the order in global_variables.py matches training order
    index_to_class = {i: name for i, name in enumerate(CATEGORIES)}


# ****************************************
# ===== STEP 3: IMAGE PROCESSING HELPER FUNCTIONS =====
# ****************************************
def download_or_read_image(path_or_url):
    """Downloads or reads an image from a URL or local file path."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        response = requests.get(path_or_url, timeout=20)
        if response.status_code != 200:
            raise Exception(f"Could not download image from {path_or_url}.")
        
        img_ary = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_ary, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception(f"OpenCV could not read image from URL: {path_or_url}")
    else:
        img = cv2.imread(path_or_url)
        if img is None:
            raise Exception(f"Could not read image from path: {path_or_url}")
            
    return img


def preprocess_image(cv2_img):
    """Applies resizing, LAB color-space conversion, and CLAHE enhancement."""
    resized = cv2.resize(cv2_img, (IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # CLAHE processing
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    
    preprocessed_display = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # uint8 for UI display

    # Model input preparation
    model_input = preprocessed_display.astype("float32") / 255.0
    model_input = np.expand_dims(model_input, axis=0)

    return preprocessed_display, model_input


def cv2_to_photoimage(cv2_rgb_img, max_size=(250, 250)):
    """Helper to scale an RGB NumPy image to a Tkinter ImageTk.PhotoImage."""
    pil_img = Image.fromarray(cv2_rgb_img)
    pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(pil_img)


# ****************************************
# ===== STEP 4: TKINTER UI APPLICATION CLASS =====
# ****************************************
class XRayClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical X-Ray Classification Dashboard")
        self.root.geometry("850x650")

        # Global variables for active images
        self.current_original_rgb = None
        self.current_preprocessed_rgb = None
        self.current_model_input = None

        # Build UI layout grid
        self._build_ui()

    def _build_ui(self):
        # Configure Main Window 2x2 Grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # ----------------------------------------------------
        # TOP-LEFT FRAME: Input Method & Controls
        # ----------------------------------------------------
        tl_frame = ttk.LabelFrame(self.root, text="1. Input Selection", padding=10)
        tl_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Radio buttons for source selection
        self.source_var = tk.StringVar(value="url")
        rb_url = ttk.Radiobutton(
            tl_frame, 
            text="Web URL", 
            variable=self.source_var, 
            value="url", 
            command=self._on_source_change
        )
        rb_local = ttk.Radiobutton(
            tl_frame, 
            text="Local Drive", 
            variable=self.source_var, 
            value="local", 
            command=self._on_source_change
        )
        rb_url.pack(anchor="w", pady=2)
        rb_local.pack(anchor="w", pady=2)

        # Path/URL Entry Box
        ttk.Label(tl_frame, text="Image Path / URL:").pack(anchor="w", pady=(10, 2))
        
        path_input_subframe = ttk.Frame(tl_frame)
        path_input_subframe.pack(fill="x", pady=2)

        self.path_entry = ttk.Entry(path_input_subframe)
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.browse_btn = ttk.Button(path_input_subframe, text="Browse", command=self._browse_file, state="disabled")
        self.browse_btn.pack(side="right")

        # Action Buttons (Load & Reset)
        btn_frame = ttk.Frame(tl_frame)
        btn_frame.pack(fill="x", pady=15)

        self.load_btn = ttk.Button(btn_frame, text="Download / Load Image", command=self.load_image_action)
        self.load_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.reset_btn = ttk.Button(btn_frame, text="Reset", command=self.reset_all)
        self.reset_btn.pack(side="right", fill="x", expand=True)

        # ----------------------------------------------------
        # BOTTOM-LEFT FRAME: Original Image & Predict Trigger
        # ----------------------------------------------------
        bl_frame = ttk.LabelFrame(self.root, text="2. Selected Image", padding=10)
        bl_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        self.lbl_original_img = ttk.Label(bl_frame, text="No image loaded", anchor="center")
        self.lbl_original_img.pack(fill="both", expand=True, pady=5)

        self.predict_btn = ttk.Button(
            bl_frame, 
            text="Predict", 
            command=self.predict_action, 
            state="disabled"
        )
        self.predict_btn.pack(fill="x", pady=5)

        # ----------------------------------------------------
        # TOP-RIGHT FRAME: Prediction Results & Class Probabilities
        # ----------------------------------------------------
        tr_frame = ttk.LabelFrame(self.root, text="3. Prediction Results", padding=10)
        tr_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.lbl_predicted_class = ttk.Label(
            tr_frame, 
            text="Predicted Class: --", 
            font=("Arial", 12, "bold")
        )
        self.lbl_predicted_class.pack(anchor="w", pady=2)

        self.lbl_confidence = ttk.Label(
            tr_frame, 
            text="Accuracy / Confidence: --", 
            font=("Arial", 10, "italic")
        )
        self.lbl_confidence.pack(anchor="w", pady=(0, 10))

        ttk.Label(tr_frame, text="Class Probabilities:").pack(anchor="w", pady=(5, 2))
        
        # Text view to display probabilities breakdown
        self.txt_probabilities = tk.Text(tr_frame, height=8, width=30, wrap="none")
        self.txt_probabilities.pack(fill="both", expand=True)

        # ----------------------------------------------------
        # BOTTOM-RIGHT FRAME: Preprocessed/Augmented View
        # ----------------------------------------------------
        br_frame = ttk.LabelFrame(self.root, text="4. Preprocessed X-Ray", padding=10)
        br_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

        self.lbl_preprocessed_img = ttk.Label(br_frame, text="Awaiting prediction...", anchor="center")
        self.lbl_preprocessed_img.pack(fill="both", expand=True, pady=5)

    def _on_source_change(self):
        """Enables/Disables the file browser button based on radio selection."""
        if self.source_var.get() == "local":
            self.browse_btn.config(state="normal")
        else:
            self.browse_btn.config(state="disabled")

    def _browse_file(self):
        """Opens native file explorer to pick local images."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, file_path)

    def load_image_action(self):
        """Step 4 execution inside UI: Reads/Downloads the image."""
        path_or_url = self.path_entry.get().strip()
        if not path_or_url:
            messagebox.showwarning("Warning", "Please enter an image URL or local path.")
            return

        try:
            cv2_bgr = download_or_read_image(path_or_url)
            self.current_original_rgb = cv2.cvtColor(cv2_bgr, cv2.COLOR_BGR2RGB)

            # Display thumbnail in Bottom-Left Frame
            photo = cv2_to_photoimage(self.current_original_rgb)
            self.lbl_original_img.config(image=photo, text="")
            self.lbl_original_img.image = photo  # Keep reference

            # Enable predict button
            self.predict_btn.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Error Loading Image", str(e))

    def predict_action(self):
        """Step 5-8 execution inside UI: Preprocesses and Predicts."""
        if self.current_original_rgb is None:
            return

        try:
            # Step 5: Preprocess Image
            cv2_bgr = cv2.cvtColor(self.current_original_rgb, cv2.COLOR_RGB2BGR)
            self.current_preprocessed_rgb, self.current_model_input = preprocess_image(cv2_bgr)

            # Display Preprocessed Image in Bottom-Right Frame
            prep_photo = cv2_to_photoimage(self.current_preprocessed_rgb)
            self.lbl_preprocessed_img.config(image=prep_photo, text="")
            self.lbl_preprocessed_img.image = prep_photo

            # Step 7: Predict
            predictions = model.predict(self.current_model_input)[0]
            pred_index = int(np.argmax(predictions))
            confidence = float(predictions[pred_index])
            predicted_class = index_to_class.get(pred_index, f"Class {pred_index}")

            # Step 8: Update Prediction Details UI Top-Right Frame
            self.lbl_predicted_class.config(text=f"Predicted Class: {predicted_class}")
            self.lbl_confidence.config(text=f"Accuracy / Confidence: {confidence * 100:.2f}%")

            # Render detailed class probabilities breakdown
            self.txt_probabilities.delete("1.0", tk.END)
            for idx, prob in enumerate(predictions):
                c_name = index_to_class.get(idx, f"Class {idx}")
                self.txt_probabilities.insert(tk.END, f"{c_name}: {prob * 100:.2f}%\n")

            # Log to terminal (retaining original function calls)
            start_partition("FINAL RESULT")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence * 100:.2f}%")
            end_partition()

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def reset_all(self):
        """Clears all inputs, images, and labels."""
        self.path_entry.delete(0, tk.END)
        self.current_original_rgb = None
        self.current_preprocessed_rgb = None
        self.current_model_input = None

        self.lbl_original_img.config(image="", text="No image loaded")
        self.lbl_original_img.image = None

        self.lbl_preprocessed_img.config(image="", text="Awaiting prediction...")
        self.lbl_preprocessed_img.image = None

        self.lbl_predicted_class.config(text="Predicted Class: --")
        self.lbl_confidence.config(text="Accuracy / Confidence: --")
        self.txt_probabilities.delete("1.0", tk.END)

        self.predict_btn.config(state="disabled")


# ****************************************
# ===== STEP 5: MAIN EXECUTION =====
# ****************************************
if __name__ == "__main__":
    root = tk.Tk()
    app = XRayClassifierApp(root)
    root.mainloop()