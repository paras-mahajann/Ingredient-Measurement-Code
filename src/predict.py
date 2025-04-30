import numpy as np
import cv2
import tensorflow as tf

# Load trained models
material_model = tf.keras.models.load_model("models/ingredient_model.h5")
container_model = tf.keras.models.load_model("models/container_model.h5")

# Load class names
material_classes = np.load("dataset/material_classes.npy")
container_classes = np.load("dataset/container_classes.npy")

# Density values (g/cm³)
material_density = {
    "Sugar": 0.85,
    "Salt": 1.20,
    "Turmeric": 0.45
}

# Container volumes (cm³)
container_volume = {
    "Teaspoon": 5,
    "Tablespoon": 15,
    "Cup": 240
}

def predict_material_and_container(image):
    """Predict material and container from an image."""
    image = cv2.resize(image, (100, 100)) / 255.0
    image = np.expand_dims(image, axis=0)
    
    material_pred = material_model.predict(image)
    container_pred = container_model.predict(image)
    
    material_label = material_classes[np.argmax(material_pred)]
    container_label = container_classes[np.argmax(container_pred)]
    
    return material_label, container_label

def calculate_weight(material, container):
    """Calculate weight using density and volume."""
    if material in material_density and container in container_volume:
        return material_density[material] * container_volume[container]
    return None

def predict_from_image(image_path):
    """Predict from a static image."""
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Error: Could not read the image.")
        return
    
    material, container = predict_material_and_container(image)
    weight = calculate_weight(material, container)
    
    print(f"🧪 Material: {material}\n🥄 Container: {container}\n⚖️ Estimated Weight: {weight:.2f} g")

def predict_from_webcam():
    """Live prediction using webcam."""
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        material, container = predict_material_and_container(frame)
        weight = calculate_weight(material, container)
        
        label = f"{material}, {container}, {weight:.2f} g"
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Live Prediction", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Predict from an image")
    print("2. Predict live using webcam")
    choice = input("Enter choice: ")
    
    if choice == "1":
        image_path = input("Enter image path: ")
        predict_from_image(image_path)
    elif choice == "2":
        predict_from_webcam()
    else:
        print("❌ Invalid choice!")
