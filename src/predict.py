import os
import json
import requests
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

import customtkinter as ctk
from tkinter import filedialog, messagebox

from global_variables import (
    TRAINED_MODEL_PATH,
    CLASS_INDEX_FILE,
    MODEL_DIR,
    IMAGE_SIZE,
    CATEGORIES,
    start_partition,
    end_partition
)

# =====================================================
# LOAD MODEL
# =====================================================
model = tf.keras.models.load_model(TRAINED_MODEL_PATH)

start_partition("MODEL LOADED")
print(model)
end_partition()

class_index_path = os.path.join(MODEL_DIR, CLASS_INDEX_FILE)

if os.path.exists(class_index_path):
    with open(class_index_path, "r") as f:
        index_to_class = {int(k): v for k, v in json.load(f).items()}
else:
    index_to_class = {i: c for i, c in enumerate(CATEGORIES)}


# =====================================================
# IMAGE FUNCTIONS
# =====================================================
def download_or_read_image(path_or_url):
    if path_or_url.startswith(("http://", "https://")):
        response = requests.get(path_or_url, timeout=20)

        if response.status_code != 200:
            raise Exception("Unable to download image.")

        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise Exception("OpenCV could not decode image.")
    else:
        img = cv2.imread(path_or_url)

        if img is None:
            raise Exception("Could not read image.")

    return img


def preprocess_image(cv2_img):
    img = cv2.resize(cv2_img, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0)
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    model_input = processed.astype("float32") / 255.0
    model_input = np.expand_dims(model_input, axis=0)

    return processed, model_input


# =====================================================
# UI
# =====================================================
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")


class MedicalAIDashboard(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Medical AI Assistant")
        self.geometry("1100x780")
        self.configure(fg_color="#F4F6F9")

        self.current_original = None
        self.current_processed = None
        self.current_input = None

        self.create_header()
        self.create_body()

    # -------------------------------------------------
    def create_header(self):
        frame = ctk.CTkFrame(
            self,
            fg_color="white",
            corner_radius=0,
            height=70
        )
        frame.pack(fill="x")

        ctk.CTkLabel(
            frame,
            text="💙 Medical AI Assistant",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(side="left", padx=20, pady=18)

        ctk.CTkLabel(
            frame,
            text="AI Powered Chest X-ray Classification",
            font=ctk.CTkFont(size=12),
            text_color="gray40"
        ).pack(side="left")

        ctk.CTkLabel(
            frame,
            text="Early Detection • Better Care",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#2563EB"
        ).pack(side="right", padx=20)

    # -------------------------------------------------
    def create_body(self):

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=15, pady=15)

        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)
        body.grid_rowconfigure(1, weight=1)

        self.create_upload_card(body)
        self.create_category_card(body)
        self.create_preview_card(body)
        self.create_result_card(body)

    # -------------------------------------------------
    def create_upload_card(self, parent):

        card = ctk.CTkFrame(parent, fg_color="white", corner_radius=12)
        card.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        ctk.CTkLabel(
            card,
            text="Upload Medical Image",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 5))

        ctk.CTkLabel(
            card,
            text="Choose local image or paste URL",
            font=ctk.CTkFont(size=12),
            text_color="gray50"
        ).pack()

        self.path_entry = ctk.CTkEntry(
            card,
            placeholder_text="Image path or URL..."
        )
        self.path_entry.pack(fill="x", padx=20, pady=15)

        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=20)

        ctk.CTkButton(
            row,
            text="Browse",
            width=90,
            command=self.browse
        ).pack(side="left")

        ctk.CTkButton(
            row,
            text="Load Image",
            command=self.load_image
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            row,
            text="Reset",
            fg_color="gray70",
            hover_color="gray50",
            command=self.reset
        ).pack(side="right")

    # -------------------------------------------------
    def create_category_card(self, parent):

        card = ctk.CTkFrame(parent, fg_color="#EFF6FF", corner_radius=12)
        card.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)

        ctk.CTkLabel(
            card,
            text="Supported Diseases",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#1E3A8A"
        ).pack(anchor="w", padx=20, pady=15)

        for c in CATEGORIES:
            item = ctk.CTkFrame(card, fg_color="white", corner_radius=8)
            item.pack(fill="x", padx=20, pady=4)

            ctk.CTkLabel(
                item,
                text="• " + c,
                font=ctk.CTkFont(size=13, weight="bold")
            ).pack(anchor="w", padx=10, pady=8)

    # -------------------------------------------------
    def create_preview_card(self, parent):

        card = ctk.CTkFrame(parent, fg_color="white", corner_radius=12)
        card.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        ctk.CTkLabel(
            card,
            text="Original & Preprocessed Image",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=20, pady=10)

        container = ctk.CTkFrame(card, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10)

        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)

        self.lbl_original = ctk.CTkLabel(
            container,
            text="Original",
            fg_color="#F1F5F9",
            corner_radius=8
        )
        self.lbl_original.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.lbl_processed = ctk.CTkLabel(
            container,
            text="Preprocessed",
            fg_color="#F1F5F9",
            corner_radius=8
        )
        self.lbl_processed.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.predict_btn = ctk.CTkButton(
            card,
            text="Run AI Prediction",
            height=40,
            state="disabled",
            fg_color="#16A34A",
            hover_color="#15803D",
            command=self.predict
        )
        self.predict_btn.pack(fill="x", padx=20, pady=15)

    # -------------------------------------------------
    def create_result_card(self, parent):

        card = ctk.CTkFrame(parent, fg_color="white", corner_radius=12)
        card.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)

        ctk.CTkLabel(
            card,
            text="Prediction Result",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=20, pady=10)

        self.result_box = ctk.CTkFrame(card, fg_color="#ECFDF5")
        self.result_box.pack(fill="x", padx=20)

        self.lbl_class = ctk.CTkLabel(
            self.result_box,
            text="Predicted Class : --",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#166534"
        )
        self.lbl_class.pack(anchor="w", padx=15, pady=(10, 5))

        self.lbl_conf = ctk.CTkLabel(
            self.result_box,
            text="Confidence : --",
            font=ctk.CTkFont(size=13),
            text_color="#15803D"
        )
        self.lbl_conf.pack(anchor="w", padx=15, pady=(0, 10))

        ctk.CTkLabel(
            card,
            text="All Class Probabilities",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=20, pady=10)

        self.scroll = ctk.CTkScrollableFrame(card, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=15, pady=(0, 10))

    # =====================================================
    # BUTTON FUNCTIONS
    # =====================================================
    def browse(self):

        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )

        if path:
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, path)

    def load_image(self):

        path = self.path_entry.get().strip()

        if path == "":
            messagebox.showwarning("Warning", "Enter image path or URL.")
            return

        try:
            bgr = download_or_read_image(path)

            self.current_original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            self.current_processed, self.current_input = preprocess_image(bgr)

            self.show_image(self.current_original, self.lbl_original)
            self.show_image(self.current_processed, self.lbl_processed)

            self.predict_btn.configure(state="normal")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_image(self, rgb, label):

        img = Image.fromarray(rgb)

        ctk_img = ctk.CTkImage(
            light_image=img,
            dark_image=img,
            size=(190, 180)
        )

        label.configure(image=ctk_img, text="")
        label.image = ctk_img

    def predict(self):

        if self.current_input is None:
            return

        predictions = model.predict(self.current_input, verbose=0)[0]

        pred_index = int(np.argmax(predictions))
        confidence = float(predictions[pred_index])

        predicted_class = index_to_class[pred_index]

        self.lbl_class.configure(
            text=f"Predicted Class : {predicted_class}"
        )

        self.lbl_conf.configure(
            text=f"Confidence : {confidence*100:.2f}%"
        )

        for w in self.scroll.winfo_children():
            w.destroy()

        for i, prob in enumerate(predictions):

            row = ctk.CTkFrame(self.scroll, fg_color="transparent")
            row.pack(fill="x", pady=4)

            ctk.CTkLabel(
                row,
                text=index_to_class[i],
                width=110,
                anchor="w",
                font=ctk.CTkFont(weight="bold")
            ).pack(side="left")

            bar = ctk.CTkProgressBar(
                row,
                progress_color="#2563EB" if i == pred_index else "#94A3B8"
            )
            bar.set(float(prob))
            bar.pack(side="left", fill="x", expand=True, padx=8)

            ctk.CTkLabel(
                row,
                text=f"{prob*100:.2f}%",
                width=55
            ).pack(side="right")

        start_partition("FINAL RESULT")
        print(f"Predicted Class : {predicted_class}")
        print(f"Confidence : {confidence*100:.2f}%")
        end_partition()

    def reset(self):

        self.path_entry.delete(0, "end")

        self.current_original = None
        self.current_processed = None
        self.current_input = None

        self.lbl_original.configure(image=None, text="Original")
        self.lbl_processed.configure(image=None, text="Preprocessed")

        self.lbl_class.configure(text="Predicted Class : --")
        self.lbl_conf.configure(text="Confidence : --")

        for w in self.scroll.winfo_children():
            w.destroy()

        self.predict_btn.configure(state="disabled")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    app = MedicalAIDashboard()
    app.mainloop()