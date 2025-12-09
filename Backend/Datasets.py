from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
from PIL import Image

MAX_NOTES = 350

# -------------------------
# Step 1: Load and parse text files
# -------------------------

folder = Path("Music_Data/FingeringFiles")
txt_files = list(folder.glob("*.txt"))
if not txt_files:
    raise FileNotFoundError(f"No .txt files found in {folder.absolute()}")

y_labels_cnn_notes = []
y_labels_cnn_hand = []
y_labels_cnn_time_stamp = []
y_labels_rnn = []

for file_path in txt_files:
    lines = file_path.read_text().strip().split("\n")

    notes, hands, times, rnns = [], [], [], []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) > 6:
            notes.append(parts[3])
            hands.append(parts[6])
            times.append(parts[2])
            # For RNN, we'll also use notes (not the last column)
            rnns.append(parts[3])  # Changed to use notes for RNN too
    
    y_labels_cnn_notes.append(notes)
    y_labels_cnn_hand.append(hands)
    y_labels_cnn_time_stamp.append(times)
    y_labels_rnn.append(rnns)

# Check if we loaded any valid data
if not y_labels_cnn_notes:
    raise ValueError("No valid data loaded from fingering files")

# -------------------------
# Step 2: Padding with 999 placeholders
# -------------------------
for i in range(len(y_labels_cnn_hand)):
    n = MAX_NOTES - len(y_labels_cnn_hand[i])
    if n > 0:
        pad = [999] * n
        y_labels_cnn_hand[i] += pad
        y_labels_cnn_notes[i] += pad
        y_labels_cnn_time_stamp[i] += [0] * n  # Use 0 for time padding
        y_labels_rnn[i] += pad

# -------------------------
# Step 3: Build mappings
# -------------------------

all_notes_flat = [note for seq in y_labels_cnn_notes for note in seq if note != 999]
unique_notes = sorted(list(set(all_notes_flat)))  # Sort for consistent mapping
note_map = {note: idx for idx, note in enumerate(unique_notes)}

print(f"Found {len(unique_notes)} unique notes")
print(f"Note mapping size: {len(note_map)}")

# Ensure we don't exceed NUM_NOTE_CLASSES
NUM_NOTE_CLASSES = 128
if len(note_map) > NUM_NOTE_CLASSES:
    print(f"WARNING: Found {len(note_map)} unique notes but NUM_NOTE_CLASSES is {NUM_NOTE_CLASSES}")
    print(f"Clamping note indices to 0-{NUM_NOTE_CLASSES-1}")

# -------------------------
# Step 4: Convert data to numeric format
# -------------------------
for i in range(len(y_labels_cnn_notes)):
    for z in range(len(y_labels_cnn_hand[i])):
        # Convert hand (0-10 for valid hands, 999 for padding)
        if y_labels_cnn_hand[i][z] != 999:
            try:
                hand_val = abs(int(y_labels_cnn_hand[i][z]))
                # Clamp to valid range
                y_labels_cnn_hand[i][z] = min(hand_val, 10)
            except:
                y_labels_cnn_hand[i][z] = 999

        # Convert timestamp
        if y_labels_cnn_time_stamp[i][z] != 0:
            try:
                y_labels_cnn_time_stamp[i][z] = float(y_labels_cnn_time_stamp[i][z])
            except:
                y_labels_cnn_time_stamp[i][z] = 0.0

        # Convert note for CNN
        val_note = y_labels_cnn_notes[i][z]
        if val_note == 999:
            y_labels_cnn_notes[i][z] = 999
        else:
            note_idx = note_map.get(val_note, 0)
            # Clamp to valid range
            y_labels_cnn_notes[i][z] = min(note_idx, NUM_NOTE_CLASSES - 1)

        # Convert note for RNN (same as CNN notes)
        val_rnn = y_labels_rnn[i][z]
        if val_rnn == 999:
            y_labels_rnn[i][z] = 999
        else:
            rnn_idx = note_map.get(val_rnn, 0)
            # Clamp to valid range
            y_labels_rnn[i][z] = min(rnn_idx, NUM_NOTE_CLASSES - 1)

# -------------------------
# Step 5: Verify data ranges
# -------------------------
def verify_data_ranges():
    """Verify all data is within expected ranges"""
    for i in range(len(y_labels_cnn_hand)):
        for j in range(len(y_labels_cnn_hand[i])):
            # Check hand values
            hand_val = y_labels_cnn_hand[i][j]
            if hand_val != 999 and (hand_val < 0 or hand_val > 10):
                print(f"Invalid hand value: {hand_val}")
                y_labels_cnn_hand[i][j] = 999
            
            # Check note values
            note_val = y_labels_cnn_notes[i][j]
            if note_val != 999 and (note_val < 0 or note_val >= NUM_NOTE_CLASSES):
                print(f"Invalid note value: {note_val}")
                y_labels_cnn_notes[i][j] = 999
            
            # Check RNN values
            rnn_val = y_labels_rnn[i][j]
            if rnn_val != 999 and (rnn_val < 0 or rnn_val >= NUM_NOTE_CLASSES):
                print(f"Invalid RNN value: {rnn_val}")
                y_labels_rnn[i][j] = 999

verify_data_ranges()

# -------------------------
# Step 6: Define Transforms
# -------------------------

# Training transforms
image_transforms = transforms.Compose([
    transforms.RandomCrop((224, 224), padding=10),
    transforms.RandomRotation(degrees=2, fill=255),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.002, 0.01), value=0.5)
])

# Validation transforms (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------
# Step 7: Define Datasets
# -------------------------

class ImageDataset(Dataset):
    def __init__(self, transforms=None):
        self.images = list(Path("Music_Data/Scores").glob("*.jpg"))
        if not self.images:
            raise FileNotFoundError("No images found in Music_Data/Scores/")
        self.hand = y_labels_cnn_hand
        self.notes = y_labels_cnn_notes
        self.time = y_labels_cnn_time_stamp
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        idx = index % len(self.hand)
        
        # Prepare labels
        labels = torch.zeros((3, MAX_NOTES), dtype=torch.float32)
        labels[0, :] = torch.tensor(self.hand[idx][:MAX_NOTES], dtype=torch.float32)
        labels[1, :] = torch.tensor(self.notes[idx][:MAX_NOTES], dtype=torch.float32)
        labels[2, :] = torch.tensor(self.time[idx][:MAX_NOTES], dtype=torch.float32)
        
        # Verify label ranges
        assert (labels[0][labels[0] != 999] >= 0).all() and (labels[0][labels[0] != 999] <= 10).all(), f"Hand values out of range"
        assert (labels[1][labels[1] != 999] >= 0).all() and (labels[1][labels[1] != 999] < NUM_NOTE_CLASSES).all(), f"Note values out of range"

        # Load and transform image
        img = Image.open(self.images[index]).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        return img, labels

class NoteDataset(Dataset):
    def __init__(self):
        self.hand = y_labels_cnn_hand
        self.notes = y_labels_cnn_notes
        self.time = y_labels_cnn_time_stamp
        self.labels = y_labels_rnn  # These are also note indices now

    def __len__(self):
        return len(self.hand)

    def __getitem__(self, index):
        # Create input features
        x = torch.zeros((MAX_NOTES, 3), dtype=torch.float32)
        x[:, 0] = torch.tensor(self.hand[index][:MAX_NOTES], dtype=torch.float32)
        x[:, 1] = torch.tensor(self.notes[index][:MAX_NOTES], dtype=torch.float32)
        x[:, 2] = torch.tensor(self.time[index][:MAX_NOTES], dtype=torch.float32)
        
        # Labels for RNN (note predictions)
        labels = torch.tensor(self.labels[index][:MAX_NOTES], dtype=torch.float32)
        
        # Verify ranges
        assert (labels[labels != 999] >= 0).all() and (labels[labels != 999] < NUM_NOTE_CLASSES).all(), f"RNN labels out of range"
        
        return x, labels

# -------------------------
# Debug check
# -------------------------
if __name__ == "__main__":
    print(f"✅ Loaded {len(txt_files)} fingering files")
    print(f"✅ {len(y_labels_cnn_notes)} label groups prepared")

    try:
        img_ds = ImageDataset(transforms=image_transforms)
        print(f"✅ {len(img_ds)} images loaded")
        
        # Test loading a sample
        sample_img, sample_labels = img_ds[0]
        print(f"Sample image shape: {sample_img.shape}")
        print(f"Sample labels shape: {sample_labels.shape}")
        print(f"Hand range: [{sample_labels[0][sample_labels[0]!=999].min():.0f}, {sample_labels[0][sample_labels[0]!=999].max():.0f}]")
        print(f"Note range: [{sample_labels[1][sample_labels[1]!=999].min():.0f}, {sample_labels[1][sample_labels[1]!=999].max():.0f}]")
    except Exception as e:
        print(f"⚠️ {e}")

    note_ds = NoteDataset()
    sample_x, sample_y = note_ds[0]
    print(f"RNN input shape: {sample_x.shape}, labels shape: {sample_y.shape}")
    print(f"RNN label range: [{sample_y[sample_y!=999].min():.0f}, {sample_y[sample_y!=999].max():.0f}]")