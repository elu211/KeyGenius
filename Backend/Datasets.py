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

def encode_sequence(seq, is_training=False):
    """
    15 Technique-Aware features:
    0: midi_norm
    1: duration
    2: delta_time (gap from prev note)
    3: rel_interval (midi - prev_midi) - CRITICAL for reach
    4: interval_next
    5: direction
    6: is_chord
    7: black_key
    8: chord_size_norm
    9: chord_position
    10: pattern_scale
    11: pattern_arpeggio
    12: pattern_repeat
    13: time_since_phrase_start
    14: local_pitch_variance (detecting jumps)
    """
    features = []
    fingers = []
    
    midis = [n['midi'] for n in seq]
    starts = [n['start'] for n in seq]
    ends = [n['end'] for n in seq]
    
    # --- DATA AUGMENTATION ---
    if is_training:
        # Transposition range +/- 5 semitones
        if np.random.random() > 0.4:
            trans = np.random.randint(-5, 6)
            midis = [m + trans for m in midis]
        
        # Time jitter
        if np.random.random() > 0.4:
            jitter = 1.0 + (np.random.random() - 0.5) * 0.15
            starts = [s * jitter for s in starts]
            ends = [e * jitter for e in ends]

    # Chord & Neighborhood Processing
    note_info = []
    for i in range(len(seq)):
        # Reach/Interval
        rel_interval = (midis[i] - midis[i-1]) / 12.0 if i > 0 else 0.0
        
        # Local variance
        local_range = midis[max(0, i-2):min(len(midis), i+3)]
        l_var = (max(local_range) - min(local_range)) / 24.0 if len(local_range) > 1 else 0.0
        
        # Basic movement
        is_black = 1.0 if (midis[i] % 12) in [1, 3, 6, 8, 10] else 0.0
        
        # Chord logic
        chord_peers = [j for j in range(len(seq)) if abs(starts[i] - starts[j]) < 0.02]
        is_chord = 1.0 if len(chord_peers) > 1 else 0.0
        chord_pos = 0.5
        if len(chord_peers) > 1:
            sorted_peers = sorted(chord_peers, key=lambda x: midis[x])
            chord_pos = sorted_peers.index(i) / (len(chord_peers)-1)

        note_info.append({
            'rel_iv': np.clip(rel_interval, -1.5, 1.5),
            'l_var': l_var,
            'is_black': is_black,
            'is_chord': is_chord,
            'chord_pos': chord_pos,
            'chord_size': min(len(chord_peers), 5) / 5.0
        })

    for i, note in enumerate(seq):
        midi = midis[i]
        start = starts[i]
        dur = ends[i] - starts[i]
        
        # Features
        feature_vec = [
            (midi - 21) / 87.0,
            min(dur, 2.0),
            min(start - starts[i-1], 2.0) if i > 0 else 0.0,
            note_info[i]['rel_iv'],
            (midis[i+1] - midi) / 12.0 if i < len(seq)-1 else 0.0,
            1.0 if (i > 0 and midi > midis[i-1]) else (-1.0 if (i > 0 and midi < midis[i-1]) else 0.0),
            note_info[i]['is_chord'],
            note_info[i]['is_black'],
            note_info[i]['chord_size'],
            note_info[i]['chord_pos'],
            0.0, 0.0, 0.0, # Pattern placeholders
            min(start - starts[0], 10.0) / 10.0,
            note_info[i]['l_var']
        ]
        
        # Update pattern logic
        if i >= 2:
            ivs = [abs(midis[j] - midis[j-1]) for j in range(max(1, i-3), i+1)]
            feature_vec[10] = sum(1 for v in ivs if 1 <= v <= 2) / len(ivs) # Scale
            feature_vec[11] = sum(1 for v in ivs if 3 <= v <= 5) / len(ivs) # Arp
            feature_vec[12] = sum(1 for v in ivs if v == 0) / len(ivs)     # Repeat

        features.append(feature_vec)
        fingers.append(note['finger'])
    
    return np.array(features, dtype=np.float32), np.array(fingers, dtype=np.int64)

class FingeringDataset(Dataset):
    def __init__(self, data_dir, hand=0, max_seq_len=200, split='train', val_ratio=0.15):
        self.max_seq_len = max_seq_len
        self.hand = hand
        self.split = split
        
        # Keep piece-wise split for validation integrity
        files = sorted(Path(data_dir).glob("*.txt"))
        piece_map = {}
        for f in files:
            pid = f.stem.split('-')[0]
            if pid not in piece_map: piece_map[pid] = []
            piece_map[pid].append(f)
        
        pids = sorted(piece_map.keys())
        np.random.seed(42)
        idx = np.random.permutation(len(pids))
        split_pt = int(len(pids) * (1 - val_ratio))
        
        sel_pids = [pids[i] for i in idx[:split_pt]] if split == 'train' else [pids[i] for i in idx[split_pt:]]
        
        self.sequences = []
        for pid in sel_pids:
            for f in piece_map[pid]:
                notes = sorted([n for n in parse_fingering_file(f) if n['hand'] == hand], key=lambda x: x['start'])
                # Sliding window sequences
                for i in range(0, len(notes), max_seq_len // 2):
                    s = notes[i:i + max_seq_len]
                    if len(s) >= 8: self.sequences.append(s)
        
        print(f"{split.upper()} set: {len(self.sequences)} sequences ({hand})")

    def __len__(self): return len(self.sequences)
    
    def __getitem__(self, idx):
        features, fingers = encode_sequence(self.sequences[idx], is_training=(self.split == 'train'))
        seq_len = len(features)
        
        # Padding
        if seq_len < self.max_seq_len:
            pad = self.max_seq_len - seq_len
            features = np.pad(features, ((0, pad), (0, 0)))
            fingers = np.pad(fingers, (0, pad))
        
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:seq_len] = 1.0
        return torch.from_numpy(features), torch.from_numpy(fingers), torch.from_numpy(mask)

def parse_fingering_file(p):
    notes = []
    lines = p.read_text().split("\n")
    for l in lines:
        if "//" in l or not l.strip(): continue
        pts = l.split()
        if len(pts) < 8: continue
        try:
            f_raw = pts[7].split('_')[0]
            notes.append({
                'start': float(pts[1]), 'end': float(pts[2]), 'midi': NOTE_TO_MIDI.get(pts[3], 60),
                'hand': int(pts[6]), 'finger': abs(int(f_raw))
            })
        except: continue
    return notes