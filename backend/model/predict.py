import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'asl_model.h5')
model = tf.keras.models.load_model(MODEL_PATH)

# Map class indices to ASL letters (A-Z + del, nothing, space)
# Order must match train_gen.class_indices from training
CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# Preprocess image to model input
INPUT_SIZE = (64, 64)  # Update if your model uses a different size

def predict_sign(image_bytes: bytes) -> str:
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize to model input size
        image = image.resize(INPUT_SIZE)
        
        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        preds = model.predict(img_array, verbose=0)
        
        # Get predicted class index
        class_idx = np.argmax(preds, axis=1)[0]
        
        # Get prediction confidence
        confidence = float(np.max(preds))
        
        # Get predicted class name
        predicted_class = CLASS_NAMES[class_idx]
        
        # Print debug info
        print(f"Predicted class index: {class_idx}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.3f}")
        print(f"All probabilities: {preds[0]}")
        
        return predicted_class
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "error"

