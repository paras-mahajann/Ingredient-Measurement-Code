#!/bin/bash

echo "🔧 Setting up Ingredient Measurement Detection on Raspberry Pi"

# Update system
sudo apt update && sudo apt upgrade -y

# Enable camera
sudo raspi-config nonint do_camera 0

# Install system dependencies
sudo apt install -y \
    python3-pip \
    python3-opencv \
    python3-picamera2 \
    libatlas-base-dev \
    libopenblas-dev

# Upgrade pip
pip3 install --upgrade pip

# Install Python dependencies
pip3 install -r requirements.txt

echo "✅ Setup complete!"
echo "📷 Reboot recommended before running Pi Camera"
