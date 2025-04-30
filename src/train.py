import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os

Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
to_categorical = tf.keras.utils.to_categorical

# Check if material dataset exists
if not os.path.exists("dataset/X_material.npy") or not os.path.exists("dataset/Y_material.npy"):
    raise FileNotFoundError("❌ Material dataset missing! Run preprocess.py first.")

# Load preprocessed material data
X_material = np.load("dataset/X_material.npy")
Y_material = np.load("dataset/Y_material.npy")
material_classes = np.load("dataset/material_classes.npy")

# Convert labels to categorical
Y_material = to_categorical(Y_material, num_classes=len(material_classes))

# Split material dataset
X_train_mat, X_test_mat, Y_train_mat, Y_test_mat = train_test_split(X_material, Y_material, test_size=0.2, random_state=42)

# Define CNN model function
def create_model(num_classes):
    model = tf.keras.models.Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train material classification model
print("🔹 Training Material Classification Model...")
material_model = create_model(len(material_classes))
material_model.fit(X_train_mat, Y_train_mat, epochs=10, validation_data=(X_test_mat, Y_test_mat))
material_model.save("models/material_model.h5")
print("✅ Material Model Training Completed!")

# Check if container dataset exists BEFORE loading
if os.path.exists("dataset/X_container.npy") and os.path.exists("dataset/Y_container.npy"):
    X_container = np.load("dataset/X_container.npy")
    Y_container = np.load("dataset/Y_container.npy")

    # If the dataset is empty, skip training
    if len(X_container) == 0 or len(Y_container) == 0:
        print("⚠️ No valid container dataset found. Skipping container model training.")
    else:
        print("🔹 Training Container Classification Model...")
        container_classes = np.load("dataset/container_classes.npy")
        Y_container = to_categorical(Y_container, num_classes=len(container_classes))
        X_train_con, X_test_con, Y_train_con, Y_test_con = train_test_split(X_container, Y_container, test_size=0.2, random_state=42)

        container_model = create_model(len(container_classes))
        container_model.fit(X_train_con, Y_train_con, epochs=10, validation_data=(X_test_con, Y_test_con))
        container_model.save("models/container_model.h5")
        print("✅ Container Model Training Completed!")
else:
    print("⚠️ No container dataset found. Skipping container model training.")
