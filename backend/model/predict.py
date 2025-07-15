import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch
import io
import numpy as np

# Load model and processor
model_name = "prithivMLmods/Alphabet-Sign-Language-Detection"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def predict_sign(image_bytes: bytes) -> str:
    """Predicts sign language alphabet for image bytes (for FastAPI compatibility)."""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Process image with the transformer model
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
        
        # Get predicted class index
        class_idx = np.argmax(probs)
        
        # Map class index to letter (0-25 for A-Z)
        labels = {
            0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
            10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
            20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
        }
        
        predicted_class = labels.get(class_idx, "unknown")
        confidence = float(np.max(probs))
        
        # Print debug info
        print(f"Predicted class index: {class_idx}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.3f}")
        print(f"All probabilities: {probs}")
        
        return predicted_class
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "error"

def sign_language_classification(image):
    """Predicts sign language alphabet category for an image (for Gradio interface)."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "A", "1": "B", "2": "C", "3": "D", "4": "E", "5": "F", "6": "G", "7": "H", "8": "I", "9": "J",
        "10": "K", "11": "L", "12": "M", "13": "N", "14": "O", "15": "P", "16": "Q", "17": "R", "18": "S", "19": "T",
        "20": "U", "21": "V", "22": "W", "23": "X", "24": "Y", "25": "Z"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=sign_language_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Alphabet Sign Language Detection",
    description="Upload an image to classify it into one of the 26 sign language alphabet categories."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()

