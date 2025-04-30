import numpy as np
import os
import cv2

dataset_dir = "dataset/"  # Path to your dataset
image_size = (100, 100)  # Resize images to this size

# Define material and container classes
materials = ["Sugar", "Salt", "Turmeric"]
containers = ["Teaspoon", "Tablespoon", "Cup"]

X_material, Y_material = [], []
X_container, Y_container = [], []



for material in materials:
    category_path = os.path.join(dataset_dir, material)

    if not os.path.exists(category_path):
        print(f"⚠️ Warning: {category_path} not found.")
        continue  # Skip missing folders

    files = os.listdir(category_path)
    print(f"📂 Found {len(files)} files in {category_path}")

    # Check if images are readable
    for file in files[:5]:  # Check only first 5 images
        image_path = os.path.join(category_path, file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to read: {image_path}")
        else:
            print(f"✅ Successfully read: {image_path} | Shape: {image.shape}")


# Process material images
for label, category in enumerate(materials):
    category_path = os.path.join(dataset_dir, category)
    
    if not os.path.exists(category_path):
        print(f"Warning: {category_path} not found.")
        continue
    
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"❌ Skipping unreadable file: {img_path}")
            continue
        
        img = cv2.resize(img, image_size)  # Resize image
        img = img / 255.0  # Normalize
        X_material.append(img)
        Y_material.append(label)

# Process container images
# Process container images
# Process container images

process_containers = True  # Change to True once container dataset is ready

if process_containers:
    for label, container in enumerate(containers):
        container_path = os.path.join(dataset_dir, container)
        if not os.path.exists(container_path):
            print(f"⚠️ Warning: {container_path} not found.")
            continue  

        for file in os.listdir(container_path):
            image_path = os.path.join(container_path, file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Skipping unreadable file: {image_path}")
                continue


            image = cv2.resize(image, image_size)
            image = image / 255.0
            X_container.append(image)
            Y_container.append(label)


print(f"Total material images processed: {len(X_material)}")

# Convert to numpy arrays
X_material, Y_material = np.array(X_material), np.array(Y_material)
X_container, Y_container = np.array(X_container), np.array(Y_container)



# Save processed data
np.save("dataset/X_material.npy", X_material)
np.save("dataset/Y_material.npy", Y_material)
np.save("dataset/material_classes.npy", np.array(materials))

np.save("dataset/X_container.npy", X_container)
np.save("dataset/Y_container.npy", Y_container)
np.save("dataset/container_classes.npy", np.array(containers))

print(f"X_container size: {len(X_container)}, Y_container size: {len(Y_container)}")

print("✅ Preprocessing complete. Now run train.py to train the models.")
