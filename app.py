
# Paste the complete app.py code I provided below this line.

import os
import csv
from datetime import datetime
from threading import Lock

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import timm

from torchvision.models import mobilenet_v3_large
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================
# 1. Configuration
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.txt")
EFFNET_PATH = os.path.join(BASE_DIR, "efficientnet_emotion.pth")
MOBILE_PATH = os.path.join(BASE_DIR, "mobilenetv3_emotion.pth")

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

NUM_CLASSES = len(class_names)


# ============================================
# 2. Reconstruct EfficientNet-B0
# ============================================

model_effnet = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=NUM_CLASSES
)

effnet_state = torch.load(
    EFFNET_PATH,
    map_location=DEVICE,
    weights_only=True
)
model_effnet.load_state_dict(effnet_state)
model_effnet = model_effnet.to(DEVICE).eval()


# ============================================
# 3. Reconstruct MobileNetV3-Large
# ============================================

model_mobile = mobilenet_v3_large(weights=None)

in_features = model_mobile.classifier[3].in_features
model_mobile.classifier[3] = nn.Linear(
    in_features,
    NUM_CLASSES
)

mobile_state = torch.load(
    MOBILE_PATH,
    map_location=DEVICE,
    weights_only=True
)
model_mobile.load_state_dict(mobile_state)
model_mobile = model_mobile.to(DEVICE).eval()


# ============================================
# 4. Image Preprocessing
# ============================================
# Must match the normalization used during training.

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])


# ============================================
# 5. Prediction Logging
# ============================================
# Local Space files are not guaranteed to persist.
# Use an external database for durable production logs.

LOG_FILE = os.path.join(BASE_DIR, "prediction_logs.csv")
log_lock = Lock()

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "model",
            "predicted_emotion",
            "confidence_percent"
        ])


def log_prediction(model_name, emotion, confidence):
    with log_lock:
        with open(
            LOG_FILE, "a", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                model_name,
                emotion,
                round(confidence, 2)
            ])


# ============================================
# 6. Prediction Function
# ============================================

def predict_emotion(img, model_choice):

    if img is None:
        return "Please upload a face image."

    if model_choice == "EfficientNet-B0":
        model = model_effnet
    elif model_choice == "MobileNetV3-Large":
        model = model_mobile
    else:
        return "Please select a valid model."

    img = img.convert("RGB")
    img_array = np.array(img)

    tensor = transform(image=img_array)["image"]
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = probabilities.max(dim=1)

    emotion = class_names[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    log_prediction(model_choice, emotion, confidence_pct)

    return (
        f"Predicted Emotion: {emotion.capitalize()}\n"
        f"Selected Model: {model_choice}\n"
        f"Confidence: {confidence_pct:.2f}%\n\n"
        "Confidence is an estimated class probability, "
        "not a guarantee that the prediction is correct."
    )


# ============================================
# 7. Gradio Interface
# ============================================

demo = gr.Interface(
    fn=predict_emotion,
    inputs=[
        gr.Image(type="pil", label="Upload Face Image"),
        gr.Radio(
            choices=["EfficientNet-B0", "MobileNetV3-Large"],
            value="EfficientNet-B0",
            label="Select Model"
        )
    ],
    outputs=gr.Textbox(label="Prediction Results", lines=5),
    title="Facial Emotion Recognition",
    description=(
        "Upload a facial image and choose one of two trained "
        "deep learning models to predict one of seven "
        "facial-expression categories."
    ),
    flagging_mode="never"
)


# ============================================
# 8. Launch
# ============================================

if __name__ == "__main__":
    demo.launch()
