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
    "Tablespoon":7,
    "Cup": 240
}

def predict_material_and_container(image):
    """Predict material and container from an image."""
    image = cv2.resize(image, (100, 100)) / 255.0
    image = np.expand_dims(image, axis=0)
    
    material_pred = material_model.predict(image,verbose =0)
    container_pred = container_model.predict(image,verbose = 0)
    
    material_label = material_classes[np.argmax(material_pred)]
    container_label = container_classes[np.argmax(container_pred)]
    container_conf = np.max(container_pred)
    
    return material_label, container_label, container_conf

def calculate_weight(material, container):
    """Calculate weight using density and volume."""
    if material in material_density and container in container_volume:
        return material_density[material] * container_volume[container]
    return None

# ===================== Draw Bounding Box =====================
def draw_box(frame, text, color=(0, 255, 0)):
    """Draw a centered bounding box with label."""
    h, w, _ = frame.shape
    box_size = int(min(h, w) * 0.5)
    x1, y1 = (w - box_size) // 2, (h - box_size) // 2
    x2, y2 = x1 + box_size, y1 + box_size

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1 + 10, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame

def predict_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Error: Could not read the image.")
        return

    material, container, conf = predict_material_and_container(image)

    if conf > 0.70:
        weight = calculate_weight(material, container)
        print(f"🧪 Material: {material}\n🥄 Container: {container}\n⚖️ Estimated Weight: {weight:.2f} g")
        image = draw_box(image, f"{material}, {container}, {weight:.2f} g")
    else:
        print("⏳ No confident container detected.")
        image = draw_box(image, "Detecting container...", color=(0, 255, 255))

    cv2.imshow("Prediction Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict_from_webcam():
    """Live prediction using webcam."""
    cap = cv2.VideoCapture(0)
    detected_container = False
    container_name = None

    
    print("📷 Starting webcam... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)
        
        material, container, conf = predict_material_and_container(frame)

        if not detected_container:

            frame = draw_box(frame, "🔍 Detecting container...", color=(0, 255, 255))
            cv2.putText(frame, f"Confidence: {conf:.2f}", (50, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            # Wait until a confident container is detected
            if conf > 0.65:  # Adjust confidence threshold as needed
                detected_container = True
                container_name = container
                print(f"✅ Container detected: {container_name}")
        else:
            # Once container is detected, start showing predictions
            weight = calculate_weight(material, container_name)
            label = f"{material}, {container_name}, {weight:.2f} g"
            frame = draw_box(frame, label, color=(0, 255, 0))

           
            
        cv2.imshow("AI Ingredient Measurement", frame)

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