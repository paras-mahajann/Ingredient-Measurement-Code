from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import os

app = Flask(__name__)

# Load trained models
material_model = tf.keras.models.load_model("models/ingredient_model.h5")
container_model = tf.keras.models.load_model("models/container_model.h5")

# Load class names
material_classes = np.load("dataset/material_classes.npy")
container_classes = np.load("dataset/container_classes.npy")

# Image processing function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)  # Reshape for model

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    file_path = "temp.jpg"
    file.save(file_path)

    # Process image and predict
    img = preprocess_image(file_path)

    material_pred = material_model.predict(img)
    container_pred = container_model.predict(img)

    material = material_classes[np.argmax(material_pred)]
    container = container_classes[np.argmax(container_pred)]

    # Hardcoded densities (g/cm³) & container volumes (cm³)
    densities = {"Sugar": 0.85, "Salt": 1.2, "Turmeric": 0.56}
    volumes = {"Teaspoon": 5, "Tablespoon": 15, "Cup": 240}

    # Calculate weight
    weight = densities.get(material, 1) * volumes.get(container, 1)

    return jsonify({
        "material": material,
        "container": container,
        "estimated_weight": f"{weight:.2f} g"
    })

if __name__ == "__main__":
    app.run(debug=True)
