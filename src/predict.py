# ******************************************************
# ===== 1. IMPORT LIBRARIES =====
# ******************************************************
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog, messagebox

from global_variables import TRAINED_MODEL_PATH, CLASS_INDEX_FILE, MODEL_DIR, download_or_read_image, process_single_image

model = tf.keras.models.load_model(TRAINED_MODEL_PATH)

with open(os.path.join(MODEL_DIR, CLASS_INDEX_FILE), "r") as f:
    index_to_class = {int(k): v for k, v in json.load(f).items()}

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")



# ******************************************************
# ===== .  =====
# ******************************************************
class MedicalAIDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Medical AI Diagnostics Tool")
        self.geometry("1050x750")

        self.current_model_input = None
        self._build_ui()

    def _build_ui(self):
        header = ctk.CTkFrame(self, height=60, fg_color="#1E293B")
        header.pack(fill="x", side="top")
        ctk.CTkLabel(header, text="🩺 Medical AI Diagnostic Assistant", font=ctk.CTkFont(size=20, weight="bold"), text_color="#FFFFFF").pack(side="left", padx=20)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=15, pady=15)

        input_card = ctk.CTkFrame(body, fg_color="#FFFFFF", corner_radius=10)
        input_card.pack(fill="x", pady=10, padx=10)

        self.path_entry = ctk.CTkEntry(input_card, placeholder_text="Paste Image URL or Local File Path...", width=600)
        self.path_entry.pack(side="left", padx=15, pady=15, fill="x", expand=True)

        ctk.CTkButton(input_card, text="Browse", command=self._browse_file, width=90).pack(side="left", padx=5)
        ctk.CTkButton(input_card, text="Load Image", command=self.load_image_action, width=100, fg_color="#2563EB").pack(side="left", padx=15)

        display_frame = ctk.CTkFrame(body, fg_color="transparent")
        display_frame.pack(fill="both", expand=True, pady=10)

        preview_card = ctk.CTkFrame(display_frame, fg_color="#FFFFFF", corner_radius=10)
        preview_card.pack(side="left", fill="both", expand=True, padx=5)

        self.lbl_image = ctk.CTkLabel(preview_card, text="No Image Loaded", fg_color="#F1F5F9", corner_radius=8)
        self.lbl_image.pack(fill="both", expand=True, padx=15, pady=15)

        self.predict_btn = ctk.CTkButton(preview_card, text="Run AI Prediction", command=self.predict_action, state="disabled", fg_color="#16A34A", height=40)
        self.predict_btn.pack(fill="x", padx=15, pady=(0, 15))

        result_card = ctk.CTkFrame(display_frame, fg_color="#FFFFFF", corner_radius=10)
        result_card.pack(side="right", fill="both", expand=True, padx=5)

        self.lbl_result = ctk.CTkLabel(result_card, text="Diagnosis: --", font=ctk.CTkFont(size=18, weight="bold"), text_color="#1E293B")
        self.lbl_result.pack(anchor="w", padx=20, pady=(20, 5))

        self.lbl_confidence = ctk.CTkLabel(result_card, text="Confidence: --", font=ctk.CTkFont(size=14), text_color="#64748B")
        self.lbl_confidence.pack(anchor="w", padx=20, pady=(0, 15))

        self.prob_container = ctk.CTkScrollableFrame(result_card, fg_color="transparent")
        self.prob_container.pack(fill="both", expand=True, padx=15, pady=10)

    def _browse_file(self):
        fpath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if fpath:
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, fpath)

    def load_image_action(self):
        path_or_url = self.path_entry.get().strip()
        if not path_or_url:
            messagebox.showwarning("Input Error", "Please provide a valid file path or image URL.")
            return

        try:
            img_bgr = download_or_read_image(path_or_url)
            preprocessed_rgb, model_input = process_single_image(img_bgr)
            
            self.current_model_input = np.expand_dims(model_input, axis=0)
            
            pil_img = Image.fromarray(preprocessed_rgb)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(260, 260))
            self.lbl_image.configure(image=ctk_img, text="")
            self.predict_btn.configure(state="normal")
        except Exception as e:
            messagebox.showerror("Loading Error", str(e))

    def predict_action(self):
        if self.current_model_input is None:
            return

        try:
            preds = model.predict(self.current_model_input)[0]
            pred_idx = int(np.argmax(preds))
            confidence = float(preds[pred_idx])
            pred_class = index_to_class.get(pred_idx, f"Class {pred_idx}")

            self.lbl_result.configure(text=f"Diagnosis: {pred_class}")
            self.lbl_confidence.configure(text=f"Confidence: {confidence * 100:.2f}%")

            for widget in self.prob_container.winfo_children():
                widget.destroy()

            for i, prob in enumerate(preds):
                cname = index_to_class.get(i, f"Class {i}")
                row = ctk.CTkFrame(self.prob_container, fg_color="transparent")
                row.pack(fill="x", pady=5)

                ctk.CTkLabel(row, text=cname, width=120, anchor="w", font=ctk.CTkFont(weight="bold")).pack(side="left")
                pbar = ctk.CTkProgressBar(row, height=12)
                pbar.set(prob)
                pbar.pack(side="left", fill="x", expand=True, padx=10)
                ctk.CTkLabel(row, text=f"{prob * 100:.1f}%", width=50).pack(side="right")

        except Exception as e:
            messagebox.showerror("Inference Error", str(e))



# ******************************************************
# ===== .  =====
# ******************************************************
if __name__ == "__main__":
    app = MedicalAIDashboard()
    app.mainloop()
