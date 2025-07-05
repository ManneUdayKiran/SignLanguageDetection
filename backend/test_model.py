import tensorflow as tf
import numpy as np
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Model and class configuration
MODEL_PATH = os.path.join('model', 'asl_model.h5')
TEST_DATA_DIR = "model/asl_alphabet_test/asl_alphabet_test"  # Update this path
IMG_SIZE = (64, 64)

# Class names matching your training data
CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def load_test_images():
    """Load test images from the single folder with filename-based class labels"""
    images = []
    true_labels = []
    predicted_labels = []
    confidences = []
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(TEST_DATA_DIR, ext)))
        image_files.extend(glob.glob(os.path.join(TEST_DATA_DIR, ext.upper())))
    
    print(f"Found {len(image_files)} test images")
    
    if len(image_files) == 0:
        print(f"❌ No images found in {TEST_DATA_DIR}")
        return None, None, None, None
    
    # Load model
    model = load_model()
    if model is None:
        return None, None, None, None
    
    for img_path in image_files:
        try:
            # Extract class from filename (e.g., "A_test.jpg" -> "A")
            filename = os.path.basename(img_path)
            class_name = filename.split('_')[0].upper()  # Get the letter before underscore
            
            # Skip if class not in our expected classes
            if class_name not in CLASS_NAMES:
                print(f"⚠️ Skipping {filename} - class '{class_name}' not in expected classes")
                continue
            
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            image = image.resize(IMG_SIZE)
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            confidence = float(np.max(prediction))
            predicted_class = CLASS_NAMES[predicted_class_idx]
            true_class_idx = CLASS_NAMES.index(class_name)
            
            # Store results
            images.append(img_array[0])
            true_labels.append(true_class_idx)
            predicted_labels.append(predicted_class_idx)
            confidences.append(confidence)
            
            print(f"📸 {filename}: True={class_name}, Predicted={predicted_class}, Confidence={confidence:.3f}")
            
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
    
    return np.array(true_labels), np.array(predicted_labels), np.array(confidences), images

def evaluate_predictions(true_labels, predicted_labels, confidences):
    """Evaluate model predictions"""
    if true_labels is None or len(true_labels) == 0:
        print("❌ No valid predictions to evaluate")
        return
    
    # Calculate accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    print(f"\n📊 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(true_labels, predicted_labels, 
                              target_names=CLASS_NAMES, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(CLASS_NAMES)))
    print(f"\n📈 Confusion Matrix Shape: {cm.shape}")
    
    return cm

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - ASL Alphabet Classification')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("📊 Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()

def analyze_predictions(true_labels, predicted_labels, confidences):
    """Analyze prediction confidence and errors"""
    print("\n🔍 Prediction Analysis:")
    
    # Confidence analysis
    print(f"Average confidence: {np.mean(confidences):.4f}")
    print(f"Min confidence: {np.min(confidences):.4f}")
    print(f"Max confidence: {np.max(confidences):.4f}")
    
    # Find most confident and least confident predictions
    most_confident_idx = np.argmax(confidences)
    least_confident_idx = np.argmin(confidences)
    
    print(f"\nMost confident prediction:")
    print(f"  True: {CLASS_NAMES[true_labels[most_confident_idx]]}")
    print(f"  Predicted: {CLASS_NAMES[predicted_labels[most_confident_idx]]}")
    print(f"  Confidence: {confidences[most_confident_idx]:.4f}")
    
    print(f"\nLeast confident prediction:")
    print(f"  True: {CLASS_NAMES[true_labels[least_confident_idx]]}")
    print(f"  Predicted: {CLASS_NAMES[predicted_labels[least_confident_idx]]}")
    print(f"  Confidence: {confidences[least_confident_idx]:.4f}")
    
    # Error analysis
    errors = predicted_labels != true_labels
    error_indices = np.where(errors)[0]
    
    if len(error_indices) > 0:
        print(f"\n❌ Top 5 Most Common Errors:")
        error_pairs = []
        for idx in error_indices:
            true_class = CLASS_NAMES[true_labels[idx]]
            pred_class = CLASS_NAMES[predicted_labels[idx]]
            error_pairs.append((true_class, pred_class))
        
        from collections import Counter
        error_counts = Counter(error_pairs)
        for (true, pred), count in error_counts.most_common(5):
            print(f"  {true} → {pred}: {count} times")
    else:
        print("🎉 No errors found!")

def main():
    """Main evaluation function"""
    print("🚀 Starting ASL Model Evaluation")
    print("=" * 50)
    
    # Load test data and make predictions
    true_labels, predicted_labels, confidences, images = load_test_images()
    
    if true_labels is None:
        print("❌ Failed to load test data")
        return
    
    # Evaluate predictions
    cm = evaluate_predictions(true_labels, predicted_labels, confidences)
    
    if cm is not None:
        # Plot confusion matrix
        plot_confusion_matrix(cm, CLASS_NAMES)
        
        # Analyze predictions
        analyze_predictions(true_labels, predicted_labels, confidences)
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main() 