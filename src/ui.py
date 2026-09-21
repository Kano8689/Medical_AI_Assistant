import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import requests
from io import BytesIO
import cv2
import numpy as np
import os

# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_TITLE = "Medical AI Image Classification"
WINDOW_SIZE = "1400x850"

# Change these according to your model
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA",
    "COVID19",
    "TUBERCULOSIS",
    "CARDIOMEGALY"
]

# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "models/model.keras"

model = None


def load_model():
    """
    Load TensorFlow/Keras model.
    """

    global model

    try:
        from tensorflow.keras.models import load_model as keras_load_model

        if not os.path.exists(MODEL_PATH):
            print("Model file not found.")
            print("UI will run in demo mode.")

            model = None
            return

        model = keras_load_model(MODEL_PATH)

        print("Model loaded successfully.")

    except Exception as e:

        print("Could not load model:")
        print(e)

        model = None


# ============================================================
# GLOBAL VARIABLES
# ============================================================

original_image = None
preprocessed_image = None

original_photo = None
preprocessed_photo = None

current_image_path = None


# ============================================================
# IMAGE LOADING
# ============================================================

def load_local_image():
    """
    Open image from local computer.
    """

    global original_image
    global current_image_path

    file_path = filedialog.askopenfilename(
        title="Select Medical Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("JPEG Files", "*.jpg *.jpeg"),
            ("PNG Files", "*.png"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        return

    try:

        image = Image.open(file_path).convert("RGB")

        original_image = image
        current_image_path = file_path

        source_entry.delete(0, tk.END)
        source_entry.insert(0, file_path)

        source_type_var.set("Local Image")

        display_original_image()

        preprocess_image()

        clear_prediction()

    except Exception as e:

        messagebox.showerror(
            "Image Error",
            f"Could not open image.\n\n{e}"
        )


def load_url_image():
    """
    Download image from internet URL.
    """

    global original_image
    global current_image_path

    url = source_entry.get().strip()

    if not url:

        messagebox.showwarning(
            "URL Required",
            "Please enter an image URL."
        )

        return

    if not url.startswith(("http://", "https://")):

        messagebox.showerror(
            "Invalid URL",
            "Please enter a valid HTTP/HTTPS image URL."
        )

        return

    try:

        status_var.set("Downloading image...")
        root.update_idletasks()

        response = requests.get(
            url,
            timeout=20
        )

        response.raise_for_status()

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        original_image = image

        current_image_path = None

        display_original_image()

        preprocess_image()

        clear_prediction()

        status_var.set("Internet image loaded successfully.")

    except Exception as e:

        status_var.set("")

        messagebox.showerror(
            "Download Error",
            f"Could not download image.\n\n{e}"
        )


# ============================================================
# IMAGE DISPLAY
# ============================================================

def resize_for_display(image, max_width=500, max_height=350):

    image = image.copy()

    image.thumbnail(
        (max_width, max_height),
        Image.Resampling.LANCZOS
    )

    return image


def display_original_image():

    global original_photo

    if original_image is None:
        return

    display_img = resize_for_display(
        original_image,
        500,
        350
    )

    original_photo = ImageTk.PhotoImage(
        display_img
    )

    original_image_label.config(
        image=original_photo,
        text=""
    )

    original_info_label.config(
        text=(
            f"Original Size: "
            f"{original_image.width} × "
            f"{original_image.height}"
        )
    )


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image():

    global preprocessed_image
    global preprocessed_photo

    if original_image is None:
        return

    try:

        # PIL → NumPy
        img = np.array(original_image)

        # RGB → BGR for OpenCV
        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )

        # Resize
        resized = cv2.resize(
            img,
            IMAGE_SIZE
        )

        # Normalize
        normalized = resized.astype(
            np.float32
        ) / 255.0

        # Save preprocessed representation
        preprocessed_image = normalized

        # Convert back for displaying
        display_img = (
            normalized * 255
        ).astype(np.uint8)

        display_img = cv2.cvtColor(
            display_img,
            cv2.COLOR_BGR2RGB
        )

        display_pil = Image.fromarray(
            display_img
        )

        preprocessed_photo = ImageTk.PhotoImage(
            display_pil
        )

        preprocessed_image_label.config(
            image=preprocessed_photo,
            text=""
        )

        preprocessing_info_label.config(
            text=(
                f"Input: "
                f"{original_image.width} × "
                f"{original_image.height}\n"
                f"Model Input: "
                f"{IMAGE_SIZE[0]} × "
                f"{IMAGE_SIZE[1]}\n"
                f"Channels: RGB\n"
                f"Normalization: 0–1"
            )
        )

    except Exception as e:

        messagebox.showerror(
            "Preprocessing Error",
            str(e)
        )


# ============================================================
# PREDICTION
# ============================================================

def predict_image():

    if original_image is None:

        messagebox.showwarning(
            "No Image",
            "Please load an image first."
        )

        return

    if preprocessed_image is None:

        preprocess_image()

    try:

        status_var.set("Running prediction...")
        root.update_idletasks()

        # ====================================================
        # REAL MODEL PREDICTION
        # ====================================================

        if model is not None:

            input_data = np.expand_dims(
                preprocessed_image,
                axis=0
            )

            predictions = model.predict(
                input_data,
                verbose=0
            )[0]

            # Softmax output
            if predictions.ndim == 0:

                predictions = np.array(
                    [predictions]
                )

            # If model outputs logits
            if not np.isclose(
                np.sum(predictions),
                1.0,
                atol=0.01
            ):

                exp_predictions = np.exp(
                    predictions -
                    np.max(predictions)
                )

                predictions = (
                    exp_predictions /
                    np.sum(exp_predictions)
                )

        # ====================================================
        # DEMO MODE
        # ====================================================

        else:

            # Demo probabilities.
            # Remove this section once your real model is loaded.

            predictions = np.array([
                0.08,
                0.68,
                0.12,
                0.07,
                0.05
            ])

        # ====================================================
        # FIND PREDICTED CLASS
        # ====================================================

        predicted_index = np.argmax(
            predictions
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = (
            predictions[predicted_index] * 100
        )

        # ====================================================
        # UPDATE UI
        # ====================================================

        predicted_class_value.config(
            text=predicted_class
        )

        confidence_value.config(
            text=f"{confidence:.2f}%"
        )

        update_class_results(
            predictions
        )

        status_var.set(
            "Prediction completed successfully."
        )

    except Exception as e:

        status_var.set("")

        messagebox.showerror(
            "Prediction Error",
            str(e)
        )


# ============================================================
# RESULT SECTION
# ============================================================

def update_class_results(predictions):

    # Remove old results
    for widget in class_results_frame.winfo_children():

        widget.destroy()

    for i, class_name in enumerate(CLASS_NAMES):

        probability = float(
            predictions[i] * 100
        )

        # Row
        row = tk.Frame(
            class_results_frame,
            bg="white"
        )

        row.pack(
            fill="x",
            pady=6
        )

        # Class name
        name_label = tk.Label(
            row,
            text=class_name,
            font=("Segoe UI", 10, "bold"),
            bg="white",
            anchor="w",
            width=18
        )

        name_label.pack(
            side="left"
        )

        # Progress bar
        progress = ttk.Progressbar(
            row,
            orient="horizontal",
            length=220,
            mode="determinate",
            maximum=100
        )

        progress["value"] = probability

        progress.pack(
            side="left",
            padx=10
        )

        # Percentage
        percentage_label = tk.Label(
            row,
            text=f"{probability:.2f}%",
            font=("Segoe UI", 10),
            bg="white",
            width=10,
            anchor="e"
        )

        percentage_label.pack(
            side="left"
        )


def clear_prediction():

    predicted_class_value.config(
        text="—"
    )

    confidence_value.config(
        text="—"
    )

    for widget in class_results_frame.winfo_children():

        widget.destroy()


# ============================================================
# SOURCE TYPE
# ============================================================

def source_type_changed():

    selected = source_type_var.get()

    if selected == "Internet URL":

        browse_button.config(
            state="disabled"
        )

        load_button.config(
            text="Load URL",
            state="normal"
        )

        source_entry.config(
            state="normal"
        )

    else:

        browse_button.config(
            state="normal"
        )

        load_button.config(
            text="Load Image",
            state="normal"
        )


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    WINDOW_TITLE
)

root.geometry(
    WINDOW_SIZE
)

root.minsize(
    1100,
    700
)

root.configure(
    bg="#EEF2F7"
)


# ============================================================
# STYLE
# ============================================================

style = ttk.Style()

try:
    style.theme_use("clam")
except:
    pass

style.configure(
    "TProgressbar",
    thickness=10
)

style.configure(
    "Predict.TButton",
    font=("Segoe UI", 11, "bold"),
    padding=10
)


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    root,
    bg="#172554",
    height=70
)

header.pack(
    fill="x"
)

header.pack_propagate(False)


title_label = tk.Label(
    header,
    text="Medical AI Image Classification",
    font=("Segoe UI", 22, "bold"),
    fg="white",
    bg="#172554"
)

title_label.pack(
    side="left",
    padx=25,
    pady=15
)


status_var = tk.StringVar(
    value="Ready"
)

status_label = tk.Label(
    header,
    textvariable=status_var,
    font=("Segoe UI", 10),
    fg="#D1D5DB",
    bg="#172554"
)

status_label.pack(
    side="right",
    padx=25
)


# ============================================================
# MAIN CONTAINER
# ============================================================

main_frame = tk.Frame(
    root,
    bg="#EEF2F7"
)

main_frame.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=15
)


# ============================================================
# 2 x 2 GRID
# ============================================================

main_frame.grid_columnconfigure(
    0,
    weight=1
)

main_frame.grid_columnconfigure(
    1,
    weight=1
)

main_frame.grid_rowconfigure(
    0,
    weight=1
)

main_frame.grid_rowconfigure(
    1,
    weight=1
)


# ============================================================
# CARD FUNCTION
# ============================================================

def create_card(parent, title):

    card = tk.Frame(
        parent,
        bg="white",
        bd=1,
        relief="solid"
    )

    title_label = tk.Label(
        card,
        text=title,
        font=("Segoe UI", 14, "bold"),
        fg="#172554",
        bg="white"
    )

    title_label.pack(
        anchor="w",
        padx=18,
        pady=(15, 10)
    )

    return card


# ============================================================
# TOP LEFT
# IMAGE INPUT
# ============================================================

input_card = create_card(
    main_frame,
    "1. Image Input"
)

input_card.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(0, 8),
    pady=(0, 8)
)


# Radio buttons

source_type_var = tk.StringVar(
    value="Local Image"
)

radio_frame = tk.Frame(
    input_card,
    bg="white"
)

radio_frame.pack(
    fill="x",
    padx=18,
    pady=5
)


local_radio = tk.Radiobutton(
    radio_frame,
    text="Local Image",
    variable=source_type_var,
    value="Local Image",
    command=source_type_changed,
    font=("Segoe UI", 10),
    bg="white",
    activebackground="white"
)

local_radio.pack(
    side="left",
    padx=(0, 20)
)


url_radio = tk.Radiobutton(
    radio_frame,
    text="Internet URL",
    variable=source_type_var,
    value="Internet URL",
    command=source_type_changed,
    font=("Segoe UI", 10),
    bg="white",
    activebackground="white"
)

url_radio.pack(
    side="left"
)


# Source entry

tk.Label(
    input_card,
    text="Image Path / URL",
    font=("Segoe UI", 10, "bold"),
    bg="white"
).pack(
    anchor="w",
    padx=18,
    pady=(15, 5)
)


entry_frame = tk.Frame(
    input_card,
    bg="white"
)

entry_frame.pack(
    fill="x",
    padx=18
)


source_entry = tk.Entry(
    entry_frame,
    font=("Segoe UI", 10),
    relief="solid",
    bd=1
)

source_entry.pack(
    side="left",
    fill="x",
    expand=True,
    ipady=8
)


browse_button = tk.Button(
    entry_frame,
    text="Browse",
    command=load_local_image,
    font=("Segoe UI", 10, "bold"),
    bg="#2563EB",
    fg="white",
    activebackground="#1D4ED8",
    activeforeground="white",
    relief="flat",
    padx=15
)

browse_button.pack(
    side="left",
    padx=(8, 0),
    ipady=3
)


load_button = tk.Button(
    input_card,
    text="Load Image",
    command=lambda: (
        load_local_image()
        if source_type_var.get() == "Local Image"
        else load_url_image()
    ),
    font=("Segoe UI", 10, "bold"),
    bg="#0F766E",
    fg="white",
    activebackground="#115E59",
    activeforeground="white",
    relief="flat",
    padx=20,
    pady=8
)

load_button.pack(
    anchor="w",
    padx=18,
    pady=15
)


# Information

input_info = tk.Label(
    input_card,
    text=(
        "Supported formats: JPG, JPEG, PNG, BMP, WEBP\n"
        "For URL input, provide a direct image URL."
    ),
    font=("Segoe UI", 9),
    fg="#6B7280",
    bg="white",
    justify="left"
)

input_info.pack(
    anchor="w",
    padx=18,
    pady=5
)


# ============================================================
# TOP RIGHT
# ORIGINAL IMAGE
# ============================================================

original_card = create_card(
    main_frame,
    "2. Original Image"
)

original_card.grid(
    row=0,
    column=1,
    sticky="nsew",
    padx=(8, 0),
    pady=(0, 8)
)


original_image_label = tk.Label(
    original_card,
    text="No image loaded",
    font=("Segoe UI", 11),
    fg="#9CA3AF",
    bg="#F8FAFC",
    width=55,
    height=14
)

original_image_label.pack(
    fill="both",
    expand=True,
    padx=18,
    pady=5
)


original_info_label = tk.Label(
    original_card,
    text="",
    font=("Segoe UI", 9),
    fg="#6B7280",
    bg="white"
)

original_info_label.pack(
    pady=5
)


predict_button = tk.Button(
    original_card,
    text="▶  Predict Image Class",
    command=predict_image,
    font=("Segoe UI", 12, "bold"),
    bg="#EA580C",
    fg="white",
    activebackground="#C2410C",
    activeforeground="white",
    relief="flat",
    padx=30,
    pady=10
)

predict_button.pack(
    pady=(5, 15)
)


# ============================================================
# BOTTOM LEFT
# PREPROCESSED IMAGE
# ============================================================

preprocess_card = create_card(
    main_frame,
    "3. Preprocessed Image"
)

preprocess_card.grid(
    row=1,
    column=0,
    sticky="nsew",
    padx=(0, 8),
    pady=(8, 0)
)


preprocessed_image_label = tk.Label(
    preprocess_card,
    text="Preprocessed image will appear here",
    font=("Segoe UI", 11),
    fg="#9CA3AF",
    bg="#F8FAFC",
    width=55,
    height=10
)

preprocessed_image_label.pack(
    fill="both",
    expand=True,
    padx=18,
    pady=5
)


preprocessing_info_label = tk.Label(
    preprocess_card,
    text="",
    font=("Segoe UI", 9),
    fg="#6B7280",
    bg="white",
    justify="left"
)

preprocessing_info_label.pack(
    pady=10
)


# ============================================================
# BOTTOM RIGHT
# PREDICTION RESULTS
# ============================================================

result_card = create_card(
    main_frame,
    "4. Prediction Results"
)

result_card.grid(
    row=1,
    column=1,
    sticky="nsew",
    padx=(8, 0),
    pady=(8, 0)
)


# Predicted class section

prediction_summary = tk.Frame(
    result_card,
    bg="#F8FAFC",
    bd=1,
    relief="solid"
)

prediction_summary.pack(
    fill="x",
    padx=18,
    pady=5
)


# Predicted class

tk.Label(
    prediction_summary,
    text="Predicted Class",
    font=("Segoe UI", 10),
    fg="#6B7280",
    bg="#F8FAFC"
).grid(
    row=0,
    column=0,
    sticky="w",
    padx=15,
    pady=(12, 3)
)


predicted_class_value = tk.Label(
    prediction_summary,
    text="—",
    font=("Segoe UI", 20, "bold"),
    fg="#0F766E",
    bg="#F8FAFC"
)

predicted_class_value.grid(
    row=1,
    column=0,
    sticky="w",
    padx=15,
    pady=(0, 12)
)


# Confidence

tk.Label(
    prediction_summary,
    text="Confidence",
    font=("Segoe UI", 10),
    fg="#6B7280",
    bg="#F8FAFC"
).grid(
    row=0,
    column=1,
    sticky="w",
    padx=15,
    pady=(12, 3)
)


confidence_value = tk.Label(
    prediction_summary,
    text="—",
    font=("Segoe UI", 20, "bold"),
    fg="#2563EB",
    bg="#F8FAFC"
)

confidence_value.grid(
    row=1,
    column=1,
    sticky="w",
    padx=15,
    pady=(0, 12)
)


prediction_summary.grid_columnconfigure(
    0,
    weight=1
)

prediction_summary.grid_columnconfigure(
    1,
    weight=1
)


# ============================================================
# CLASS PROBABILITIES
# ============================================================

tk.Label(
    result_card,
    text="Class Probability",
    font=("Segoe UI", 11, "bold"),
    fg="#374151",
    bg="white"
).pack(
    anchor="w",
    padx=18,
    pady=(10, 5)
)


# Scrollable result frame

result_canvas = tk.Canvas(
    result_card,
    bg="white",
    highlightthickness=0
)

result_scrollbar = ttk.Scrollbar(
    result_card,
    orient="vertical",
    command=result_canvas.yview
)

result_canvas.configure(
    yscrollcommand=result_scrollbar.set
)

result_canvas.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(18, 0),
    pady=(0, 10)
)

result_scrollbar.pack(
    side="right",
    fill="y",
    padx=(0, 18),
    pady=(0, 10)
)


class_results_frame = tk.Frame(
    result_canvas,
    bg="white"
)


result_canvas.create_window(
    (0, 0),
    window=class_results_frame,
    anchor="nw"
)


def update_scroll_region(event=None):

    result_canvas.configure(
        scrollregion=result_canvas.bbox("all")
    )


class_results_frame.bind(
    "<Configure>",
    update_scroll_region
)


# ============================================================
# INITIALIZE
# ============================================================

load_model()

source_type_changed()

root.mainloop()