#!/bin/bash
# Colab Training Script for KeyGenius Fingering Model

# 1. Setup Environment
echo "Setting up environment..."
pip install torch torchvision torchaudio
pip install numpy scipy pillow oemer
# Note: oemer might have many dependencies, you might need --use-feature=fast-deps or similar

# 2. Mount Google Drive
echo "Mounting Google Drive..."
python3 -c "from google.colab import drive; drive.mount('/content/drive')"

# 3. Copy Data from Drive (if needed) or assuming files are uploaded
# Assuming the user has a zip of the Backend folder on Drive
# unzip /content/drive/MyDrive/KeyGenius_Backend.zip -d /content/Backend

# 4. Run Training for Right Hand
echo "Starting Right Hand Training..."
cd Backend
python3 train.py --hand 0

# 5. Copy best model to Google Drive
echo "Saving models to Drive..."
cp best_model_right.pth /content/drive/MyDrive/best_model_right.pth

# 6. Run Training for Left Hand
echo "Starting Left Hand Training..."
# Temporarily modify train.py for left hand if you don't have CLI args
sed -i 's/HAND = 0/HAND = 1/' train.py
python3 train.py
cp best_model_left.pth /content/drive/MyDrive/best_model_left.pth

echo "Training complete! Models saved to Google Drive."
