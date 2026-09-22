# predict.py
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

import customtkinter as ctk
from tkinter import filedialog, messagebox

from global_variables import (
    TRAINED_MODEL_PATH, CLASS_INDEX_FILE, IMAGE_SIZE, CATEGORIES, MODEL_DIR,
    start_partition, end_partition
)

# ****************************************
# ===== STEP 2: LOAD MODEL =====
# ****************************************
model = tf.keras.models.load_model(TRAINED_MODEL_PATH)
start_partition("MODEL LOADED")
print(model)
end_partition()

class_index_path = os.path.join(MODEL_DIR, CLASS_INDEX_FILE)
with open(class_index_path, "r") as f:
    index_to_class = {int(k): v for k, v in json.load(f).items()}

# ****************************************
# ===== STEP 3: IMAGE PROCESSING HELPERS =====
# ****************************************
def download_or_read_image(path_or_url):
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
    resized = cv2.resize(cv2_img, (IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    
    preprocessed_display = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    model_input = preprocessed_display.astype("float32") / 255.0
    model_input = np.expand_dims(model_input, axis=0)

    return preprocessed_display, model_input

# ****************************************
# ===== STEP 4: MODERN UI DASHBOARD =====
# ****************************************
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

class MedicalAIDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Medical AI Assistance")
        self.geometry("1100x780")
        self.configure(fg_color="#F4F6F9")

        self.current_original_rgb = None
        self.current_preprocessed_rgb = None
        self.current_model_input = None

        self._build_header()
        self._build_main_grid()

    def _build_header(self):
        header_frame = ctk.CTkFrame(self, fg_color="#FFFFFF", corner_radius=0, height=70)
        header_frame.pack(fill="x", side="top")

        title_label = ctk.CTkLabel(
            header_frame, text="💙  Medical AI Assistance",
            font=ctk.CTkFont(family="Arial", size=22, weight="bold"),
            text_color="#1E293B"
        )
        title_label.pack(side="left", padx=25, pady=10)

        subtitle_label = ctk.CTkLabel(
            header_frame, text="AI-Powered Medical Image Analysis for a Healthier Tomorrow",
            font=ctk.CTkFont(family="Arial", size=12),
            text_color="#64748B"
        )
        subtitle_label.pack(side="left", pady=10)

        tagline_label = ctk.CTkLabel(
            header_frame, text="🛡️ Early Detection  |  Better Care",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            text_color="#2563EB"
        )
        tagline_label.pack(side="right", padx=25, pady=10)

    def _build_main_grid(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=20, pady=15)

        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)
        container.grid_rowconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=1)

        self._build_top_left_upload(container)
        self._build_top_right_categories(container)
        self._build_bottom_left_preview(container)
        self._build_bottom_right_results(container)

    def _build_top_left_upload(self, parent):
        card = ctk.CTkFrame(parent, fg_color="#FFFFFF", corner_radius=12)
        card.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        title = ctk.CTkLabel(card, text="Upload a Medical Image",
                            font=ctk.CTkFont(size=18, weight="bold"),
                            text_color="#0F172A")
        title.pack(pady=(15, 2))

        sub = ctk.CTkLabel(card, text="Select a local file or paste an image URL below.",
                          font=ctk.CTkFont(size=12), text_color="#64748B")
        sub.pack(pady=(0, 10))

        input_box = ctk.CTkFrame(card, fg_color="#F8FAFC", corner_radius=8,
                                border_width=1, border_color="#E2E8F0")
        input_box.pack(fill="x", padx=20, pady=5)

        self.path_entry = ctk.CTkEntry(input_box,
            placeholder_text="Paste Image URL or File Path here...",
            height=35, fg_color="#FFFFFF")
        self.path_entry.pack(side="left", fill="x", expand=True, padx=10, pady=10)

        browse_btn = ctk.CTkButton(input_box, text="Browse", width=80,
                                  command=self._browse_file,
                                  fg_color="#64748B", hover_color="#475569")
        browse_btn.pack(side="right", padx=(0, 10))

        action_frame = ctk.CTkFrame(card, fg_color="transparent")
        action_frame.pack(fill="x", padx=20, pady=10)

        load_btn = ctk.CTkButton(action_frame, text="☁️ Download / Load Image",
                                height=38, command=self.load_image_action,
                                fg_color="#2563EB", hover_color="#1D4ED8")
        load_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        reset_btn = ctk.CTkButton(action_frame, text="🔄 Reset", width=90, height=38,
                                 command=self.reset_all,
                                 fg_color="#E2E8F0", text_color="#334155",
                                 hover_color="#CBD5E1")
        reset_btn.pack(side="right")

    def _build_top_right_categories(self, parent):
        card = ctk.CTkFrame(parent, fg_color="#EFF6FF", corner_radius=12)
        card.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        title = ctk.CTkLabel(card, text="Supported Categories",
                            font=ctk.CTkFont(size=16, weight="bold"),
                            text_color="#1E3A8A")
        title.pack(anchor="w", padx=20, pady=(15, 10))

        cat_frame = ctk.CTkFrame(card, fg_color="transparent")
        cat_frame.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        for idx, cat_name in enumerate(CATEGORIES):
            item = ctk.CTkFrame(cat_frame, fg_color="#FFFFFF", corner_radius=8)
            item.pack(fill="x", pady=3)

            lbl_name = ctk.CTkLabel(item, text=f"•  {cat_name}",
                                   font=ctk.CTkFont(size=13, weight="bold"),
                                   text_color="#1E293B")
            lbl_name.pack(side="left", padx=12, pady=6)

    def _build_bottom_left_preview(self, parent):
        card = ctk.CTkFrame(parent, fg_color="#FFFFFF", corner_radius=12)
        card.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        title = ctk.CTkLabel(card, text="Input & Processed Image",
                            font=ctk.CTkFont(size=16, weight="bold"),
                            text_color="#0F172A")
        title.pack(anchor="w", padx=20, pady=(12, 5))

        img_container = ctk.CTkFrame(card, fg_color="transparent")
        img_container.pack(fill="both", expand=True, padx=15, pady=5)
        img_container.grid_columnconfigure(0, weight=1)
        img_container.grid_columnconfigure(1, weight=1)
        img_container.grid_rowconfigure(0, weight=1)

        self.lbl_orig_img = ctk.CTkLabel(img_container, text="Original Image",
                                        fg_color="#F1F5F9", corner_radius=8)
        self.lbl_orig_img.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.lbl_prep_img = ctk.CTkLabel(img_container, text="Preprocessed (CLAHE)",
                                        fg_color="#F1F5F9", corner_radius=8)
        self.lbl_prep_img.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.predict_btn = ctk.CTkButton(
            card, text="⚡ Run Analysis & Predict", height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.predict_action, state="disabled",
            fg_color="#16A34A", hover_color="#15803D"
        )
        self.predict_btn.pack(fill="x", padx=20, pady=(5, 15))

    def _build_bottom_right_results(self, parent):
        card = ctk.CTkFrame(parent, fg_color="#FFFFFF", corner_radius=12)
        card.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

        title = ctk.CTkLabel(card, text="Prediction Result",
                            font=ctk.CTkFont(size=16, weight="bold"),
                            text_color="#0F172A")
        title.pack(anchor="w", padx=20, pady=(12, 5))

        self.result_card = ctk.CTkFrame(card, fg_color="#F0FDF4", corner_radius=10,
                                       border_width=1, border_color="#DCFCE7")
        self.result_card.pack(fill="x", padx=20, pady=5)

        self.lbl_predicted_class = ctk.CTkLabel(self.result_card,
            text="Predicted Class: --",
            font=ctk.CTkFont(size=18, weight="bold"), text_color="#166534")
        self.lbl_predicted_class.pack(anchor="w", padx=15, pady=(10, 2))

        self.lbl_confidence = ctk.CTkLabel(self.result_card,
            text="Confidence: --",
            font=ctk.CTkFont(size=13), text_color="#15803D")
        self.lbl_confidence.pack(anchor="w", padx=15, pady=(0, 10))

        prob_title = ctk.CTkLabel(card, text="All Class Probabilities",
                                 font=ctk.CTkFont(size=14, weight="bold"),
                                 text_color="#334155")
        prob_title.pack(anchor="w", padx=20, pady=(10, 5))

        self.prob_scroll_frame = ctk.CTkScrollableFrame(card, fg_color="transparent")
        self.prob_scroll_frame.pack(fill="both", expand=True, padx=15, pady=(0, 10))

    def _browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, file_path)

    def load_image_action(self):
        path_or_url = self.path_entry.get().strip()
        if not path_or_url:
            messagebox.showwarning("Warning", "Please enter an image URL or local path.")
            return

        try:
            cv2_bgr = download_or_read_image(path_or_url)
            self.current_original_rgb = cv2.cvtColor(cv2_bgr, cv2.COLOR_BGR2RGB)
            self.current_preprocessed_rgb, self.current_model_input = preprocess_image(cv2_bgr)
            self._render_image(self.current_original_rgb, self.lbl_orig_img)
            self._render_image(self.current_preprocessed_rgb, self.lbl_prep_img)
            self.predict_btn.configure(state="normal")
        except Exception as e:
            messagebox.showerror("Error Loading Image", str(e))

    def _render_image(self, rgb_array, label):
        pil_img = Image.fromarray(rgb_array)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(190, 180))
        label.configure(image=ctk_img, text="")

    def predict_action(self):
        if self.current_model_input is None:
            return

        try:
            predictions = model.predict(self.current_model_input)[0]
            pred_index = int(np.argmax(predictions))
            confidence = float(predictions[pred_index])
            predicted_class = index_to_class.get(pred_index, f"Class {pred_index}")

            self.lbl_predicted_class.configure(text=f"Predicted: {predicted_class}")
            self.lbl_confidence.configure(text=f"Model Confidence: {confidence * 100:.2f}%")

            for widget in self.prob_scroll_frame.winfo_children():
                widget.destroy()

            for idx, prob in enumerate(predictions):
                c_name = index_to_class.get(idx, f"Class {idx}")
                
                row_frame = ctk.CTkFrame(self.prob_scroll_frame, fg_color="transparent")
                row_frame.pack(fill="x", pady=4)

                lbl_cname = ctk.CTkLabel(row_frame, text=c_name, width=110,
                                        anchor="w", font=ctk.CTkFont(size=12, weight="bold"))
                lbl_cname.pack(side="left")

                progress = ctk.CTkProgressBar(row_frame, height=10,
                                             fg_color="#E2E8F0",
                                             progress_color="#2563EB" if idx == pred_index else "#94A3B8")
                progress.set(prob)
                progress.pack(side="left", fill="x", expand=True, padx=8)

                lbl_val = ctk.CTkLabel(row_frame, text=f"{prob * 100:.2f}%",
                                      width=55, anchor="e",
                                      font=ctk.CTkFont(size=11))
                lbl_val.pack(side="right")

            start_partition("FINAL RESULT")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence * 100:.2f}%")
            end_partition()

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def reset_all(self):
        self.path_entry.delete(0, "end")
        self.current_original_rgb = None
        self.current_preprocessed_rgb = None
        self.current_model_input = None

        self.lbl_orig_img.configure(image=None, text="Original Image")
        self.lbl_prep_img.configure(image=None, text="Preprocessed (CLAHE)")

        self.lbl_predicted_class.configure(text="Predicted Class: --")
        self.lbl_confidence.configure(text="Confidence: --")

        for widget in self.prob_scroll_frame.winfo_children():
            widget.destroy()

        self.predict_btn.configure(state="disabled")

# ****************************************
# ===== STEP 5: MAIN EXECUTION =====
# ****************************************
if __name__ == "__main__":
    app = MedicalAIDashboard()
    app.mainloop()