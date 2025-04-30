import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque

# Use GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✅ Running on GPU")
else:
    print("⚠️ Running on CPU")

# Load Trained Model
model = keras.models.load_model("models/ingredient_model.h5")

# Define class labels
classes = ["Sugar", "Salt", "Turmeric"]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Maintain a queue for frame averaging (smoothing predictions)
prediction_queue = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize and preprocess frame
    img = cv2.resize(frame, (100, 100))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict Ingredient
    prediction = model.predict(img)[0]  # Get the first element from batch
    confidence = np.max(prediction)  # Get highest probability
    index = np.argmax(prediction)  # Get predicted class index
    predicted_ingredient = classes[index]

    # Add to queue for averaging
    prediction_queue.append(predicted_ingredient)
    most_common_prediction = max(set(prediction_queue), key=prediction_queue.count)

    # Display Bounding Box & Prediction
    h, w, _ = frame.shape
    cv2.rectangle(frame, (50, 50), (w - 50, h - 50), (0, 255, 0), 2)  # Bounding Box
    cv2.putText(frame, f"{most_common_prediction} ({confidence*100:.2f}%)", 
                (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show Frame
    cv2.imshow("Live Ingredient Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
