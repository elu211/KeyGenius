"""
KeyGenius Inference - predict fingerings from sheet music.
Handles multiple pages.
"""
import torch
import numpy as np
from pathlib import Path
from model import FingeringTransformer
from fast_oemer_extract import extract_from_image, extract_from_pages, extract_from_folder, NOTE_TO_MIDI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path, hand='right'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = FingeringTransformer(
        input_dim=18,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,  # No dropout at inference
        num_fingers=6,
        class_weights=None
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Val accuracy was: {checkpoint.get('val_acc', 'N/A')}")
    
    return model


def notes_to_features(notes):
    """
    Convert note names to 17-dim model features.
    
    Features:
    0: midi_norm
    1: duration
    2: delta_time
    3: interval_prev
    4: interval_next
    5: direction
    6: is_chord
    7: chord_size_norm
    8: chord_position
    9: pattern_scale
    10: pattern_arpeggio
    11: pattern_repeat
    12: pattern_repeat
    13-17: prev_finger one-hot
    18: black_key
    """
    features = []
    midis = [NOTE_TO_MIDI.get(n, 60) for n in notes]
    n = len(notes)
    
    for i, note_name in enumerate(notes):
        midi = midis[i]
        
        # Basic features
        midi_norm = (midi - 21) / 87.0
        duration = 0.5  # Placeholder
        delta_time = 0.2  # Placeholder
        
        # Intervals
        interval_prev = (midi - midis[i-1]) / 24.0 if i > 0 else 0.0
        interval_next = (midis[i+1] - midi) / 24.0 if i < n - 1 else 0.0
        interval_prev = np.clip(interval_prev, -1, 1)
        interval_next = np.clip(interval_next, -1, 1)
        
        # Direction
        if i > 0:
            direction = 1.0 if midi > midis[i-1] else (-1.0 if midi < midis[i-1] else 0.0)
        else:
            direction = 0.0
        
        # Chord detection (simple: same x position = chord)
        # At inference we don't have timing, so set to 0
        is_chord = 0.0
        chord_size_norm = 0.2
        chord_position = 0.5
        
        # Pattern detection
        if i >= 2:
            recent = [midis[j] - midis[j-1] for j in range(max(1, i-3), i+1)]
            steps = sum(1 for iv in recent if abs(iv) in [1, 2])
            arps = sum(1 for iv in recent if abs(iv) in [3, 4, 5])
            repeats = sum(1 for iv in recent if iv == 0)
            pattern_scale = steps / len(recent)
            pattern_arpeggio = arps / len(recent)
            pattern_repeat = repeats / len(recent)
        else:
            pattern_scale = 0.0
            pattern_arpeggio = 0.0
            pattern_repeat = 0.0
        
        # Previous finger (unknown at inference, use zeros)
        prev_finger_onehot = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Black key feature
        def is_black(m):
            return (m % 12) in [1, 3, 6, 8, 10]
        black_key = 1.0 if is_black(midi) else 0.0

        feature_vec = [
            midi_norm,
            duration,
            delta_time,
            interval_prev,
            interval_next,
            direction,
            is_chord,
            black_key,
            chord_size_norm,
            chord_position,
            pattern_scale,
            pattern_arpeggio,
            pattern_repeat,
            *prev_finger_onehot
        ]
        
        features.append(feature_vec)
    
    return np.array(features, dtype=np.float32)


def predict_batch(model, features, max_seq=200):
    """Predict fingerings using CRF Viterbi decoding."""
    n = len(features)
    all_fingers = []
    
    for i in range(0, n, max_seq):
        batch = features[i:i+max_seq]
        seq_len = len(batch)
        
        # Pad to max_seq
        if seq_len < max_seq:
            batch = np.pad(batch, ((0, max_seq - seq_len), (0, 0)), constant_values=0)
        
        mask = np.zeros(max_seq, dtype=np.float32)
        mask[:seq_len] = 1.0
        
        with torch.no_grad():
            feat_t = torch.from_numpy(batch).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
            pad_mask = (mask_t == 0)
            
            # Use CRF Viterbi decoding
            preds = model.generate(feat_t, src_key_padding_mask=pad_mask, mask=mask_t)
            
            all_fingers.extend(preds[0, :seq_len].cpu().numpy().tolist())
    
    return all_fingers, [0.9] * len(all_fingers) # Mock confidences for now


def infer(img_input, checkpoint_path):
    """
    Main inference function.
    
    Args:
        img_input: single image path, list of paths, or folder path
        checkpoint_path: path to model checkpoint
    
    Returns:
        notes: ['C4', 'D4', ...] 
        coords: [(x, y, page), ...]
        fingers: [1, 2, 3, ...]
        confidences: [0.95, 0.88, ...]
    """
    # Extract notes
    print("Extracting notes from image...")
    
    if isinstance(img_input, list):
        notes, coords, bboxes = extract_from_pages(img_input)
    elif Path(img_input).is_dir():
        notes, coords, bboxes = extract_from_folder(img_input)
    else:
        notes, coords, bboxes = extract_from_image(img_input)
        coords = [(x, y, 0) for x, y in coords]
    
    if len(notes) == 0:
        print("No notes found!")
        return [], [], []
    
    print(f"Found {len(notes)} notes")
    
    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path)
    
    # Convert to features
    features = notes_to_features(notes)
    
    # Predict
    print("Predicting fingerings with CRF...")
    fingers, confidences = predict_batch(model, features)
    
    return notes, coords, fingers, confidences

def adjust_fingering_coords(coords, notes, hand='right'):
    """
    Adjust note coordinates to move them to a good position for finger numbers.
    RH: Usually above the note. LH: Usually below.
    """
    adjusted = []
    offset = -35 if hand == 'right' else 35
    for (x, y, p) in coords:
        # Move up/down depending on hand
        adjusted.append((x, y + offset, p))
    return adjusted


if __name__ == "__main__":
    import sys
    
    checkpoint = "best_model_right.pth"
    
    if not Path(checkpoint).exists():
        print(f"ERROR: Model not found at {checkpoint}")
        print("Train the model first with: python train.py")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python infer.py image.jpg")
        print("  python infer.py page1.jpg page2.jpg page3.jpg")
        print("  python infer.py ./folder_with_pages/")
        sys.exit(1)
    
    # Handle input
    if len(sys.argv) == 2:
        img_input = sys.argv[1]
    else:
        img_input = sys.argv[1:]
    
    # Run
    notes, coords, fingers = infer(img_input, checkpoint)
    
    print(f"\n{'='*50}")
    print(f"Results: {len(notes)} notes")
    print(f"{'='*50}\n")
    
    for i in range(min(20, len(notes))):
        x, y, page = coords[i]
        print(f"  {notes[i]:4s} @ ({x:4d}, {y:4d}) page {page} -> finger {fingers[i]}")
    
    if len(notes) > 20:
        print(f"  ... and {len(notes) - 20} more")