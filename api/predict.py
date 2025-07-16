import sys
import os
import io
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification
import torch
import numpy as np

# Load model and processor globally (cold start optimization)
model_name = "prithivMLmods/Alphabet-Sign-Language-Detection"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def predict_sign(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
        class_idx = np.argmax(probs)
        labels = {
            0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
            10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
            20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
        }
        predicted_class = labels.get(class_idx, "unknown")
        return predicted_class
    except Exception as e:
        return "error"

def handler(request):
    if request.method != "POST":
        return {
            "statusCode": 405,
            "body": "Method Not Allowed"
        }
    try:
        # Vercel Python API: request.body is bytes
        image_bytes = request.body
        prediction = predict_sign(image_bytes)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": f'{{"predicted_sign": "{prediction}"}}'
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": f"Error: {str(e)}"
        } 