import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from collections import Counter

NOTE_TO_MIDI = {}
for octave in range(0, 9):
    for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
        midi_num = octave * 12 + i + 12
        NOTE_TO_MIDI[f"{note}{octave}"] = midi_num
        flat_map = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'}
        if note in flat_map:
            NOTE_TO_MIDI[f"{flat_map[note]}{octave}"] = midi_num


def parse_fingering_file(filepath):
    lines = Path(filepath).read_text().strip().split("\n")
    notes = []
    for line in lines:
        if not line.strip() or line.startswith("//"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            start_time = float(parts[1])
            end_time = float(parts[2])
            note_name = parts[3]
            hand = int(parts[6])
            finger_raw = parts[7]
            if '_' in finger_raw:
                finger = int(finger_raw.split('_')[0])
            else:
                finger = int(finger_raw)
            finger = abs(finger)
            if finger < 1 or finger > 5:
                continue
            if note_name not in NOTE_TO_MIDI:
                continue
            notes.append({
                'start': start_time,
                'end': end_time,
                'note': note_name,
                'midi': NOTE_TO_MIDI[note_name],
                'hand': hand,
                'finger': finger,
                'duration': end_time - start_time
            })
        except (ValueError, IndexError):
            continue
    return notes


def extract_sequences(notes, hand, max_seq_len=200):
    hand_notes = sorted(
        [n for n in notes if n['hand'] == hand],
        key=lambda x: x['start']
    )
    if len(hand_notes) == 0:
        return []
    
    sequences = []
    stride = max(max_seq_len // 2, 1)
    for i in range(0, len(hand_notes), stride):
        seq = hand_notes[i:i + max_seq_len]
        if len(seq) >= 10:
            sequences.append(seq)
        if i + max_seq_len >= len(hand_notes):
            break
    return sequences


def encode_sequence(seq):
    """
    17 features per note:
    0: midi_norm
    1: duration
    2: delta_time
    3: interval_prev
    4: interval_next (lookahead)
    5: direction
    6: is_chord
    7: is_chord
    8: black_key
    9: chord_size_norm
    10: chord_position
    11: pattern_scale
    12: pattern_arpeggio
    13: pattern_repeat
    14-18: prev_finger one-hot
    """
    features = []
    fingers = []
    
    midis = [n['midi'] for n in seq]
    starts = [n['start'] for n in seq]
    ends = [n['end'] for n in seq]
    
    # Detect chords
    chord_groups = []
    current_chord = [0]
    for i in range(1, len(seq)):
        if starts[i] < ends[i-1] - 0.01:  # Overlapping = chord
            current_chord.append(i)
        else:
            chord_groups.append(current_chord)
            current_chord = [i]
    chord_groups.append(current_chord)
    
    # Map note index to chord info
    note_chord_info = {}
    for chord in chord_groups:
        chord_size = len(chord)
        sorted_by_pitch = sorted(chord, key=lambda x: midis[x])
        for pos, idx in enumerate(sorted_by_pitch):
            note_chord_info[idx] = {
                'is_chord': chord_size > 1,
                'chord_size': chord_size,
                'chord_position': pos / max(chord_size - 1, 1) if chord_size > 1 else 0.5
            }
    
    prev_finger = None
    
    for i, note in enumerate(seq):
        midi = note['midi']
        start = note['start']
        dur = note['duration']
        finger = note['finger']
        
        # Basic features
        midi_norm = (midi - 21) / 87.0
        duration = min(dur, 2.0)
        
        # Delta time
        delta = min(start - starts[i-1], 2.0) if i > 0 else 0.0
        
        # Intervals
        interval_prev = (midi - midis[i-1]) / 24.0 if i > 0 else 0.0
        interval_next = (midis[i+1] - midi) / 24.0 if i < len(seq) - 1 else 0.0
        interval_prev = np.clip(interval_prev, -1, 1)
        interval_next = np.clip(interval_next, -1, 1)
        
        # Direction
        if i > 0:
            direction = 1.0 if midi > midis[i-1] else (-1.0 if midi < midis[i-1] else 0.0)
        else:
            direction = 0.0
        
        # Chord info
        chord_info = note_chord_info.get(i, {'is_chord': False, 'chord_size': 1, 'chord_position': 0.5})
        is_chord = 1.0 if chord_info['is_chord'] else 0.0
        chord_size_norm = min(chord_info['chord_size'], 5) / 5.0
        chord_position = chord_info['chord_position']
        
        # Pattern detection (look at previous 4 notes)
        if i >= 2:
            recent_intervals = [midis[j] - midis[j-1] for j in range(max(1, i-3), i+1)]
            steps = sum(1 for iv in recent_intervals if abs(iv) in [1, 2])
            arps = sum(1 for iv in recent_intervals if abs(iv) in [3, 4, 5])
            repeats = sum(1 for iv in recent_intervals if iv == 0)
            n = len(recent_intervals)
            pattern_scale = steps / n if n > 0 else 0
            pattern_arpeggio = arps / n if n > 0 else 0
            pattern_repeat = repeats / n if n > 0 else 0
        else:
            pattern_scale = 0.0
            pattern_arpeggio = 0.0
            pattern_repeat = 0.0
        
        # Previous finger one-hot
        prev_finger_onehot = [0.0] * 5
        if prev_finger is not None and 1 <= prev_finger <= 5:
            prev_finger_onehot[prev_finger - 1] = 1.0
        
        # Black key feature
        def is_black(midi):
            return (midi % 12) in [1, 3, 6, 8, 10]
        black_key = 1.0 if is_black(midi) else 0.0

        feature_vec = [
            midi_norm,
            duration,
            delta,
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
        fingers.append(finger)
        prev_finger = finger
    
    return np.array(features, dtype=np.float32), np.array(fingers, dtype=np.int64)


class FingeringDataset(Dataset):
    def __init__(self, data_dir, hand=0, max_seq_len=200, split='train', val_ratio=0.2):
        self.max_seq_len = max_seq_len
        self.hand = hand
        
        finger_files = sorted(Path(data_dir).glob("*.txt"))
        print(f"Found {len(finger_files)} fingering files")
        
        # Group files by PIECE ID (e.g., "001", "002")
        piece_to_files = {}
        for f in finger_files:
            piece_id = f.stem.split("-")[0]  # "001-1_fingering" -> "001"
            if piece_id not in piece_to_files:
                piece_to_files[piece_id] = []
            piece_to_files[piece_id].append(f)
        
        piece_ids = sorted(piece_to_files.keys())
        print(f"Found {len(piece_ids)} unique pieces")
        
        # Split by PIECE, not by file
        np.random.seed(42)
        piece_indices = np.random.permutation(len(piece_ids))
        split_idx = int(len(piece_indices) * (1 - val_ratio))
        
        if split == 'train':
            selected_pieces = [piece_ids[i] for i in piece_indices[:split_idx]]
        else:
            selected_pieces = [piece_ids[i] for i in piece_indices[split_idx:]]
        
        print(f"{split.capitalize()}: {len(selected_pieces)} pieces")
        
        # Get files for selected pieces
        selected_files = []
        for piece_id in selected_pieces:
            selected_files.extend(piece_to_files[piece_id])
        
        print(f"  {len(selected_files)} files")
        
        # Extract sequences
        all_sequences = []
        for f in selected_files:
            notes = parse_fingering_file(f)
            sequences = extract_sequences(notes, hand, max_seq_len)
            all_sequences.extend(sequences)
        
        self.sequences = all_sequences
        print(f"  {len(self.sequences)} sequences for {'right' if hand == 0 else 'left'} hand")
        
        # Print distribution
        all_fingers = []
        for seq in self.sequences:
            all_fingers.extend([n['finger'] for n in seq])
        dist = Counter(all_fingers)
        print(f"  Finger dist: {dict(sorted(dist.items()))}")
        
        # Chord stats
        chord_count = sum(1 for seq in self.sequences for n in seq 
                         if any(abs(n['start'] - n2['start']) < 0.02 and n != n2 
                               for n2 in seq))
        total_notes = len(all_fingers)
        print(f"  Chord notes: ~{chord_count}/{total_notes}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        features, fingers = encode_sequence(seq)
        
        seq_len = len(features)
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
            fingers = np.pad(fingers, (0, pad_len), mode='constant')
        
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:seq_len] = 1.0
        
        return (
            torch.from_numpy(features),
            torch.from_numpy(fingers),
            torch.from_numpy(mask),
        )


if __name__ == "__main__":
    data_dir = "Music_Data/FingeringFiles"
    
    print("=" * 60)
    print("Checking for data leakage...")
    print("=" * 60)
    
    train_ds = FingeringDataset(data_dir, hand=0, split='train')
    val_ds = FingeringDataset(data_dir, hand=0, split='val')
    
    print(f"\nTrain sequences: {len(train_ds)}")
    print(f"Val sequences: {len(val_ds)}")
    
    # Check feature shape
    f, fingers, mask = train_ds[0]
    print(f"\nFeature shape: {f.shape}")
    print(f"First 5 fingers: {fingers[:5].tolist()}")