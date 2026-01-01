import numpy as np
import cv2
from picamera2 import Picamera2
import subprocess
import threading
import platform
import time
import os
import tensorflow as tf

# ===================== PLATFORM CHECK =====================
IS_PI = platform.system() == "Linux"

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

FRAME_SKIP = 1

DETECT_CONF = 0.60
LOST_CONF = 0.40
STABLE_FRAMES = 5
ING_STABLE_FRAMES = 4
last_spoken_text = None

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
    return 0.0

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


    """Live prediction using webcam (continuous container detection)"""

    cap = cv2.VideoCapture(0)

    # ----- State Machine Variables (LOCAL) -----
    FRAME_SKIP = 2
    frame_count = 0

    DETECT_CONF = 0.60
    LOST_CONF = 0.40
    STABLE_FRAMES = 2

    state = "SEARCHING"
    stable_count = 0
    container_name = None
    ING_STABLE_FRAMES = 2
    ingredient_name = None
    ingredient_stable_count = 0

    # ------------------------------------------

    print("📷 Starting webcam... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # ⏭ Skip frames for FPS optimization
        if frame_count % FRAME_SKIP != 0:
            cv2.imshow("AI Ingredient Measurement (Webcam)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        material, container, conf = predict_material_and_container(frame)

        # ================= STATE MACHINE =================
        if state == "SEARCHING":
            frame = draw_box(frame, "🔍 Searching for container...", (0, 255, 255))
            cv2.putText(frame, f"Conf: {conf:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if conf >= DETECT_CONF:
                stable_count += 1
                if stable_count >= STABLE_FRAMES:
                    container_name = container
                    state = "LOCKED"
                    stable_count = 0
                    print(f"✅ Container locked: {container_name}")
            else:
                stable_count = 0

        elif state == "LOCKED":
            # 🔄 Check if container is removed
            if conf < LOST_CONF or container != container_name:
                print("❌ Container removed → restarting detection")
                state = "SEARCHING"
                container_name = None
                stable_count = 0
                frame = draw_box(frame, "🔄 Container removed", (0, 0, 255))
            else:
                 # 🧪 INGREDIENT STABILIZATION
                if ingredient_name == material:
                    ingredient_stable_count += 1
                else:
                    ingredient_name = material
                    ingredient_stable_count = 1

                # ✅ Ingredient stable → SHOW WEIGHT
                if ingredient_stable_count >= ING_STABLE_FRAMES:
                    weight = calculate_weight(ingredient_name, container_name)

                    label = (
                        f"Ingredient: {ingredient_name} | "
                        f"Container: {container_name} | "
                        f"Weight: {weight:.2f} g"
                    )

                    frame = draw_box(frame, label, (0, 255, 0))

                    # OPTIONAL console print
                    print(f"🧪 {ingredient_name} | 🥄 {container_name} | ⚖️ {weight:.2f} g")

                else:
                    frame = draw_box(frame, "🔄 Stabilizing ingredient...", (255, 255, 0))

        # ================================================
        cv2.putText(
    frame,
    f"State: {state}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)


        cv2.imshow("AI Ingredient Measurement (Webcam)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def predict_from_webcam():
    cap = cv2.VideoCapture(0)

    
    frame_count = 0
    state = "SEARCHING"
    stable_count = 0
    container_name = None

    ingredient_name = None
    ingredient_stable_count = 0

    print("📷 Starting webcam... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        display_text = "⏳ Processing..."
        
        if frame_count % FRAME_SKIP != 0:
            frame = draw_box(frame, "⏳ Processing...")
            cv2.imshow("AI Ingredient Measurement", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        material, container, conf = predict_material_and_container(frame)

        display_text = "🔍 Searching for container..."

        # ================= STATE MACHINE =================
        if state == "SEARCHING":
            if conf >= DETECT_CONF:
                stable_count += 1
                display_text = "⏳ Locking container..."
                if stable_count >= STABLE_FRAMES:
                    container_name = container
                    state = "LOCKED"
                    stable_count = 0
                    ingredient_name = None
                    ingredient_stable_count = 0
                    last_spoken_text = None
                    print(f"✅ Container locked: {container_name}")
            else:
                stable_count = 0

        elif state == "LOCKED":
            if conf < LOST_CONF or container != container_name:
                print("❌ Container removed")
                # last_spoken_text = None
                state = "SEARCHING"
                container_name = None
                ingredient_name = None
                ingredient_stable_count = 0
                display_text = "🔄 Container removed"
            else:
                if material == ingredient_name:
                    ingredient_stable_count += 1
                else:
                    ingredient_name = material
                    ingredient_stable_count = 1

                if ingredient_stable_count >= ING_STABLE_FRAMES:
                    weight = calculate_weight(ingredient_name, container_name)

                    display_text = f"{ingredient_name}, {container_name}, {weight:.2f} g"

                    speak_text = f"{ingredient_name} weight is {weight} grams"

                    if speak_text != last_spoken_text:
                        speak_async(speak_text)
                        last_spoken_text = speak_text

                else:
                    display_text = "🔄 Stabilizing ingredient..."

        # ================================================

        frame = draw_box(frame, display_text)

        cv2.putText(frame, f"State: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("AI Ingredient Measurement", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def predict_from_picamera():
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()

    
    frame_count = 0

    

    state = "SEARCHING"
    last_spoken_text = None

    stable_count = 0
    container_name = None

    ingredient_name = None
    ingredient_stable_count = 0

    print("📷 Starting Pi Camera... Press 'q' to quit.")
    time.sleep(2)  # Camera warm-up

    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            frame = draw_box(frame, "⏳ Processing...")
            cv2.imshow("AI Ingredient Measurement", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        material, container, conf = predict_material_and_container(frame)

        display_text = "🔍 Searching for container..."

        # ================= STATE MACHINE =================
        if state == "SEARCHING":
            display_text = "🔍 Searching for container..."
            if conf >= DETECT_CONF:
                stable_count += 1
                display_text = "⏳ Locking container..."
                if stable_count >= STABLE_FRAMES:
                    container_name = container
                    state = "LOCKED"
                    stable_count = 0
                    ingredient_name = None
                    ingredient_stable_count = 0
                    last_spoken_text = None
                    print(f"✅ Container locked: {container_name}")
            else:
                stable_count = 0

        elif state == "LOCKED":
            if conf < LOST_CONF or container != container_name:
                state = "SEARCHING"
                container_name = None
                ingredient_name = None
                ingredient_stable_count = 0
                display_text = "🔄 Container removed"
            else:
                if material == ingredient_name:
                    ingredient_stable_count += 1
                else:
                    ingredient_name = material
                    ingredient_stable_count = 1

                if ingredient_stable_count >= ING_STABLE_FRAMES:
                    weight = calculate_weight(ingredient_name, container_name)
                    display_text = f"{ingredient_name}, {container_name}, {weight} grams"
                    
                    if display_text != last_spoken_text:
                        speak_async(display_text)
                        last_spoken_text = display_text
                else:
                    display_text = "🔄 Stabilizing ingredient..."

        # ================================================

        frame = draw_box(frame, display_text)

        cv2.putText(frame, f"State: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("AI Ingredient Measurement", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()


def speak_async(text):
    global last_spoken_text

    if text == last_spoken_text:
        return

    last_spoken_text = text

    def _speak():
        try:
            if IS_PI:
                subprocess.run(
                    ["espeak-ng", "-s", "140", "-v", "en", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.say(text)
                engine.runAndWait()
        except Exception as e:
            print("🔇 TTS Error:", e)

    threading.Thread(target=_speak, daemon=True).start()

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Predict from an image")
    print("2. Predict live using webcam")
    print("3. Predict live using PI-cam")
    choice = input("Enter choice: ")

    if choice == "1":
        image_path = input("Enter image path: ")
        predict_from_image(image_path)
    elif choice == "2":
        predict_from_webcam()
    elif choice == "3":
        predict_from_picamera()
    else:
        print("❌ Invalid choice!")