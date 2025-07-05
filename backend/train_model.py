# Indian Sign Language Dataset Training
# Complete training code

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import os
import kagglehub

# Download the dataset
print("�� Downloading Indian Sign Language Dataset...")
vaishnaviasonawane_indian_sign_language_dataset_path = kagglehub.dataset_download('vaishnaviasonawane/indian-sign-language-dataset')
print(f"✅ Dataset downloaded to: {vaishnaviasonawane_indian_sign_language_dataset_path}")

# Configuration
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

print(f"\n⚙️ Configuration:")
print(f"  Image size: {IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")

# Find the data directory
def find_data_directory():
    """Find the main data directory with class folders"""
    for root, dirs, files in os.walk(vaishnaviasonawane_indian_sign_language_dataset_path):
        if len(dirs) > 5:  # Likely class directories
            return root
    return None

data_dir = find_data_directory()
print(f"📁 Data directory: {data_dir}")

# Create data generators
def create_data_generators():
    """Create training and validation data generators"""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"\n📊 Data Generators:")
    print(f"  Training samples: {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Number of classes: {train_gen.num_classes}")
    print(f"  Class indices: {train_gen.class_indices}")
    
    return train_gen, val_gen

# Create the model
def create_model(num_classes):
    """Create a CNN model for sign language classification"""
    
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

# Training function
def train_model():
    """Train the model"""
    
    # Create data generators
    train_gen, val_gen = create_data_generators()
    
    # Create model
    model = create_model(train_gen.num_classes)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n��️ Model Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_indian_sign_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    print(f"\n🚀 Starting Training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history, train_gen, val_gen

# Plot training history
def plot_training_history(history):
    """Plot training and validation metrics"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# Evaluate model
def evaluate_model(model, val_gen, class_names):
    """Evaluate the trained model"""
    
    print(f"\n🔍 Evaluating Model...")
    
    # Get predictions
    predictions = model.predict(val_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"\n📊 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(true_classes, predicted_classes, 
                              target_names=class_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Indian Sign Language Classification')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, predictions, predicted_classes, true_classes, cm

# Main training execution
def main():
    """Main training function"""
    
    print("🚀 Indian Sign Language Model Training")
    print("=" * 50)
    
    # Train the model
    model, history, train_gen, val_gen = train_model()
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    class_names = list(train_gen.class_indices.keys())
    accuracy, predictions, predicted_classes, true_classes, cm = evaluate_model(
        model, val_gen, class_names
    )
    
    # Save final model
    model.save('indian_sign_language_model.h5')
    print(f"\n✅ Model saved as 'indian_sign_language_model.h5'")
    print(f"✅ Best model saved as 'best_indian_sign_model.h5'")
    
    # Print final results
    print(f"\n🎉 Training Complete!")
    print(f"📊 Final Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📁 Model files saved:")
    print(f"  - indian_sign_language_model.h5 (final model)")
    print(f"  - best_indian_sign_model.h5 (best validation accuracy)")
    print(f"  - training_history.png (training plots)")
    print(f"  - confusion_matrix.png (confusion matrix)")

# Run the training
if __name__ == "__main__":
    main()