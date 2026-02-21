#!/bin/bash
# Colab Training Script for KeyGenius Fingering Model
echo "===================================================="
echo "          KeyGenius Colab Trainer                  "
echo "===================================================="

# 1. SETUP ENVIRONMENT
echo "[1/4] Setting up environment..."
pip install torch torchvision torchaudio numpy scipy pillow oemer --quiet

# 2. DRIVE CHECK
echo "[2/4] Checking Google Drive..."
if [ ! -d "/content/drive/MyDrive" ]; then
    echo "⚠️  Google Drive not found at /content/drive/MyDrive"
    echo "Please run this in a Colab cell first to mount your drive:"
    echo "from google.colab import drive; drive.mount('/content/drive')"
    echo "Continuing without drive backup..."
    DRIVE_AVAILABLE=false
else
    echo "✅ Google Drive mounted."
    DRIVE_AVAILABLE=true
fi

# 3. RUN TRAINING (RIGHT HAND)
echo "[3/4] Starting Right Hand Training..."
cd Backend || exit
python3 train.py

# Back up right hand model
if [ "$DRIVE_AVAILABLE" = true ] && [ -f "best_model_right.pth" ]; then
    cp best_model_right.pth /content/drive/MyDrive/best_model_right.pth
    echo "✅ Right hand model backed up to Drive."
fi

# 4. RUN TRAINING (LEFT HAND)
echo "[4/4] Starting Left Hand Training..."
# Temporarily modify train.py for left hand
sed -i 's/HAND = 0/HAND = 1/' train.py
python3 train.py

# Back up left hand model
if [ "$DRIVE_AVAILABLE" = true ] && [ -f "best_model_left.pth" ]; then
    cp best_model_left.pth /content/drive/MyDrive/best_model_left.pth
    echo "✅ Left hand model backed up to Drive."
fi

echo "===================================================="
echo "✅ ALL DONE! Check your Google Drive for .pth files."
echo "===================================================="

