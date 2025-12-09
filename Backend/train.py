import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
import os

# Set CUDA debugging flag for better error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from CNN_RNN import CNN_RNN
from Datasets import ImageDataset, NoteDataset, image_transforms, val_transforms

# ------------------------------
# DEVICE
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------------
# HYPERPARAMETERS
# ------------------------------
MAX_NOTES = 350
RNN_SEQ_LEN = 50  
NUM_HAND_CLASSES = 11
NUM_NOTE_CLASSES = 128
LR = 1e-3
EPOCHS = 6000
BATCH_SIZE = 16
VAL_SPLIT = 0.2

# RNN parameters
RNN_HIDDEN_SIZE = 64
RNN_LAYERS = 3

# ------------------------------
# DATASETS & LOADERS
# ------------------------------

# Create and split image dataset
img_dataset = ImageDataset(transforms=image_transforms)
train_size = int((1 - VAL_SPLIT) * len(img_dataset))
val_size = len(img_dataset) - train_size
train_img_dataset, val_img_dataset = random_split(img_dataset, [train_size, val_size])

# Update validation dataset transforms
val_img_dataset.dataset.transforms = val_transforms

# Create dataloaders with error handling
train_cnn_loader = DataLoader(
    train_img_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # Set to 0 for debugging
    pin_memory=False
)

val_cnn_loader = DataLoader(
    val_img_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

dataset_RNN = DataLoader(
    NoteDataset(),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

# ------------------------------
# MODEL & OPTIMIZER
# ------------------------------
cnn_params = {
    "input_channels": 3,
    "max_sequence": MAX_NOTES,
    "num_hand_classes": NUM_HAND_CLASSES,
    "num_note_classes": NUM_NOTE_CLASSES,
    "note_weight": 1.0,
    "time_stamp_weight": 0.01,
    "hand_weight": 1.0,
    "rnn_weight": 1.0
}

rnn_params = {
    "hidden_size": RNN_HIDDEN_SIZE,
    "num_layers": RNN_LAYERS,
    "output_size": NUM_NOTE_CLASSES  # Must match the actual number of classes
}

model = CNN_RNN(cnn_params=cnn_params, rnn_params=rnn_params).to(device)
print(f"Model on device: {next(model.parameters()).device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def downsample(seq_tensor, target_len=RNN_SEQ_LEN):
    """Downsample sequences to target length"""
    if len(seq_tensor.shape) == 2:
        seq_tensor = seq_tensor.unsqueeze(-1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, L, F = seq_tensor.shape
    if L <= target_len:
        return seq_tensor.squeeze(-1) if squeeze_output else seq_tensor
    
    idx = torch.linspace(0, L-1, steps=target_len).long()
    result = seq_tensor[:, idx, :]
    return result.squeeze(-1) if squeeze_output else result

def validate():
    """Validation loop"""
    model.eval()
    val_loss = 0.0
    count = 0
    with torch.no_grad():
        for images, labels in val_cnn_loader:
            try:
                images = images.to(device)
                labels = labels.to(device)
                loss, _ = model(images, labels)
                val_loss += loss.item()
                count += 1
            except Exception as e:
                print(f"Validation error: {e}")
                continue
    return val_loss / max(count, 1)

def check_labels(labels, name="labels"):
    """Debug function to check label ranges"""
    if len(labels.shape) == 3:  # CNN labels [B, 3, MAX_NOTES]
        hands = labels[:, 0, :]
        notes = labels[:, 1, :]
        print(f"{name} - Hands: min={hands[hands!=999].min():.0f}, max={hands[hands!=999].max():.0f}")
        print(f"{name} - Notes: min={notes[notes!=999].min():.0f}, max={notes[notes!=999].max():.0f}")
    else:  # RNN labels [B, seq_len]
        print(f"{name} - RNN: min={labels[labels!=999].min():.0f}, max={labels[labels!=999].max():.0f}")

# ------------------------------
# TRAINING LOOP
# ------------------------------
best_val_loss = float('inf')
train_losses = []
val_losses = []

print("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    cnn_total_loss = 0.0
    rnn_total_loss = 0.0
    cnn_count = 0
    rnn_count = 0

    # --- Train on images (CNN) ---
    for batch_idx, (images, labels) in enumerate(train_cnn_loader):
        try:
            images = images.to(device)
            labels = labels.to(device)
            
            # Debug: check label ranges (uncomment for debugging)
            # if epoch == 0 and batch_idx == 0:
            #     check_labels(labels, "CNN labels")

            optimizer.zero_grad()
            loss, logits = model(images, labels)
            
            if torch.isnan(loss):
                print(f"NaN CNN loss at epoch {epoch}, batch {batch_idx}")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            cnn_total_loss += loss.item()
            cnn_count += 1
            
        except RuntimeError as e:
            print(f"CNN Training error at epoch {epoch}, batch {batch_idx}: {e}")
            if "out of range" in str(e) or "Assertion" in str(e):
                check_labels(labels, f"Problematic CNN batch {batch_idx}")
            continue

    # --- Train on sequences (RNN) ---
    for batch_idx, (seq_inputs, seq_labels) in enumerate(dataset_RNN):
        try:
            seq_inputs = seq_inputs.to(device)
            seq_labels = seq_labels.to(device)
            
            # Downsample
            seq_inputs_ds = downsample(seq_inputs, target_len=RNN_SEQ_LEN)
            seq_labels_ds = downsample(seq_labels, target_len=RNN_SEQ_LEN)
            
            # Debug: check label ranges (uncomment for debugging)
            # if epoch == 0 and batch_idx == 0:
            #     check_labels(seq_labels_ds, "RNN labels")
            
            # Ensure labels are in valid range
            seq_labels_long = seq_labels_ds.long()
            seq_labels_long[seq_labels_long == 999] = 0  # Replace padding with valid index
            
            optimizer.zero_grad()
            rnn_logits = model.forward_rnn(seq_inputs_ds)
            
            # Compute loss with ignore_index for padding
            rnn_loss = F.cross_entropy(
                rnn_logits.reshape(-1, NUM_NOTE_CLASSES),
                seq_labels_long.reshape(-1),
                ignore_index=0  # Now we use 0 as ignore index since we replaced 999
            )
            
            if torch.isnan(rnn_loss):
                print(f"NaN RNN loss at epoch {epoch}, batch {batch_idx}")
                continue

            rnn_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            rnn_total_loss += rnn_loss.item()
            rnn_count += 1
            
        except RuntimeError as e:
            print(f"RNN Training error at epoch {epoch}, batch {batch_idx}: {e}")
            if "out of range" in str(e) or "Assertion" in str(e):
                check_labels(seq_labels_ds, f"Problematic RNN batch {batch_idx}")
                print(f"RNN output size: {rnn_logits.shape[-1]}, Num classes: {NUM_NOTE_CLASSES}")
            continue

    # --- Validation & Logging ---
    if cnn_count > 0 or rnn_count > 0:
        val_loss = validate()
        avg_cnn_loss = cnn_total_loss / max(cnn_count, 1)
        avg_rnn_loss = rnn_total_loss / max(rnn_count, 1)
        
        train_losses.append(avg_cnn_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            print(f"📉 Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")

        # Logging
        if epoch % 10 == 0:
            print("=" * 80)
            print(f"🎵 EPOCH {epoch:4d}/{EPOCHS} 🎵")
            print(f"📊 CNN Loss: {avg_cnn_loss:.4f} ({cnn_count} batches)")
            print(f"📊 RNN Loss: {avg_rnn_loss:.4f} ({rnn_count} batches)")
            print(f"📊 Val Loss: {val_loss:.4f}")
            print(f"⚡ LR: {optimizer.param_groups[0]['lr']:.6f}")
            print("=" * 80)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, 'best_cnn_rnn_model.pth')
            print(f"💾 Saved best model with val_loss: {val_loss:.4f}")

# ------------------------------
# SAVE FINAL CHECKPOINT
# ------------------------------
torch.save({
    'epoch': EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_val_loss': val_loss if 'val_loss' in locals() else float('inf'),
    'train_losses': train_losses,
    'val_losses': val_losses
}, 'final_cnn_rnn_checkpoint.pth')

print("✅ Training complete! Final checkpoint saved.")
print(f"📈 Best validation loss: {best_val_loss:.4f}")