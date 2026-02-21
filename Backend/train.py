import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from Datasets import FingeringDataset
from model import FingeringTransformer, compute_accuracy, compute_per_finger_accuracy

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "Music_Data/FingeringFiles"
HAND = 0  # 0 = right, 1 = left
HAND_NAME = "right" if HAND == 0 else "left"

MAX_SEQ_LEN = 256
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 100
PATIENCE = 15

# Transformer Params
INPUT_DIM = 18
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6
DIM_FF = 1024
DROPOUT = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Training: {HAND_NAME} hand transformer")

# ============================================================
# DATA
# ============================================================
train_dataset = FingeringDataset(DATA_DIR, hand=HAND, max_seq_len=MAX_SEQ_LEN, split='train')
val_dataset = FingeringDataset(DATA_DIR, hand=HAND, max_seq_len=MAX_SEQ_LEN, split='val')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Calculate class weights for imbalance
finger_counts = {i: 0 for i in range(1, 6)}
for i in train_dataset.indices:
    seq = train_dataset.sequences[i]
    for n in seq:
        if 1 <= n['finger'] <= 5:
            finger_counts[n['finger']] += 1

total = sum(finger_counts.values())
class_weights = torch.zeros(6)
for i in range(1, 6):
    class_weights[i] = total / (5 * finger_counts[i])
class_weights[0] = 0.0 # padding
print(f"Class weights: {class_weights}")

# ============================================================
# MODEL
# ============================================================
model = FingeringTransformer(
    input_dim=INPUT_DIM,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FF,
    dropout=DROPOUT,
    num_fingers=6,
    class_weights=class_weights.to(device)
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# ============================================================
# TRAINING HOOKS
# ============================================================
def train_epoch():
    model.train()
    total_loss, total_acc, num_batches = 0, 0, 0
    
    for features, fingers, mask in train_loader:
        features = features.to(device)
        fingers = fingers.to(device)
        mask = mask.to(device)
        
        # Transformer pad mask is (mask == 0)
        pad_mask = (mask == 0)
        
        optimizer.zero_grad()
        emissions, loss = model(features, fingers=fingers, mask=mask, src_key_padding_mask=pad_mask)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # For accuracy, use generate (CRF decode)
        with torch.no_grad():
            preds = model.generate(features, src_key_padding_mask=pad_mask, mask=mask)
            total_acc += compute_accuracy(preds, fingers, mask)
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def validate():
    model.eval()
    total_loss, total_acc, num_batches = 0, 0, 0
    all_per_finger = {i: [] for i in range(1, 6)}
    
    with torch.no_grad():
        for features, fingers, mask in val_loader:
            features = features.to(device)
            fingers = fingers.to(device)
            mask = mask.to(device)
            pad_mask = (mask == 0)
            
            emissions, loss = model(features, fingers=fingers, mask=mask, src_key_padding_mask=pad_mask)
            preds = model.generate(features, src_key_padding_mask=pad_mask, mask=mask)
            
            total_loss += loss.item()
            total_acc += compute_accuracy(preds, fingers, mask)
            
            per_finger = compute_per_finger_accuracy(preds, fingers, mask)
            for f in range(1, 6):
                all_per_finger[f].append(per_finger[f])
            
            num_batches += 1
    
    avg_per_finger = {f: np.mean(all_per_finger[f]) for f in range(1, 6)}
    return total_loss / num_batches, total_acc / num_batches, avg_per_finger


# ============================================================
# MAIN
# ============================================================
best_val_acc = 0
patience_counter = 0

print("\n" + "=" * 60)
print("Starting Transformer Training...")
print("=" * 60 + "\n")

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch()
    val_loss, val_acc, per_finger = validate()
    scheduler.step()
    
    pf = " ".join([f"{f}:{per_finger[f]*100:.0f}%" for f in range(1, 6)])
    print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% | {pf}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'per_finger': per_finger,
            'config': {
                'input_dim': INPUT_DIM,
                'd_model': D_MODEL,
                'nhead': NHEAD,
                'num_layers': NUM_LAYERS
            }
        }, f'best_model_{HAND_NAME}.pth')
        print(f"         💾 New best! {val_acc*100:.1f}%")
    else:
        patience_counter += 1
    
    if patience_counter >= PATIENCE:
        print(f"\n⚠️ Early stopping at epoch {epoch+1}")
        break

print("\n" + "=" * 60)
print(f"✅ Done! Best: {best_val_acc*100:.1f}%")
print("=" * 60)