"""
Extract notes from sheet music images.
Handles multiple pages.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from scipy import ndimage
from pathlib import Path
from oemer import MODULE_PATH
from oemer.inference import inference

# Note to MIDI
NOTE_TO_MIDI = {}
for octave in range(0, 9):
    for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
        midi_num = octave * 12 + i + 12
        NOTE_TO_MIDI[f"{note}{octave}"] = midi_num
        flat_map = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'}
        if note in flat_map:
            NOTE_TO_MIDI[f"{flat_map[note]}{octave}"] = midi_num


def extract_from_image(img_path):
    """
    Extract notes from a single image.
    
    Returns:
        notes: list of note names ['C4', 'D4', ...]
        coords: list of (x, y) tuples
        bboxes: list of [x1, y1, x2, y2]
    """
    staff_pred, _ = inference(os.path.join(MODULE_PATH, "checkpoints/unet_big"), img_path, use_tf=False)
    seg_pred, _ = inference(os.path.join(MODULE_PATH, "checkpoints/seg_net"), img_path, manual_th=None, use_tf=False)
    
    staff_map = (staff_pred == 1).astype(np.uint8)
    notehead_map = (seg_pred == 2).astype(np.uint8)
    
    staves = _find_staves(staff_map)
    labeled, n = ndimage.label(notehead_map)
    
    raw = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < 15 or len(ys) > 3000:
            continue
        
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        note_name, hand = _y_to_note(cy, staves)
        raw.append((cx, note_name, hand, (cx, cy), [x1, y1, x2, y2]))
    
    # Sort left to right
    raw.sort(key=lambda x: x[0])
    
    notes = [r[1] for r in raw]
    hands = [r[2] for r in raw]
    coords = [r[3] for r in raw]
    bboxes = [r[4] for r in raw]
    
    return notes, hands, coords, bboxes


def extract_from_pages(img_paths):
    """
    Extract notes from multiple page images.
    
    Args:
        img_paths: list of image paths (in page order)
    
    Returns:
        all_notes: list of note names
        all_coords: list of (x, y, page) tuples
        all_bboxes: list of [x1, y1, x2, y2, page]
    """
    all_notes = []
    all_coords = []
    all_bboxes = []
    
    for page_num, img_path in enumerate(img_paths):
        print(f"Processing page {page_num + 1}/{len(img_paths)}: {img_path}")
        
        notes, coords, bboxes = extract_from_image(img_path)
        
        # Add page number
        for note, (x, y), bbox in zip(notes, coords, bboxes):
            all_notes.append(note)
            all_coords.append((x, y, page_num))
            all_bboxes.append(bbox + [page_num])
        
        print(f"  Found {len(notes)} notes")
    
    print(f"Total: {len(all_notes)} notes across {len(img_paths)} pages")
    return all_notes, all_coords, all_bboxes


def extract_from_folder(folder_path, pattern="*.jpg"):
    """
    Extract notes from all images in a folder.
    
    Args:
        folder_path: path to folder with page images
        pattern: glob pattern for images (default *.jpg)
    
    Returns:
        Same as extract_from_pages
    """
    folder = Path(folder_path)
    img_paths = sorted(folder.glob(pattern))
    
    if not img_paths:
        # Try other formats
        for ext in ['*.png', '*.jpeg', '*.tiff']:
            img_paths = sorted(folder.glob(ext))
            if img_paths:
                break
    
    if not img_paths:
        raise ValueError(f"No images found in {folder_path}")
    
    return extract_from_pages([str(p) for p in img_paths])


def _find_staves(staff_map):
    row_sums = staff_map.sum(axis=1)
    if row_sums.max() == 0:
        return []
    
    threshold = row_sums.max() * 0.2
    line_rows = np.where(row_sums > threshold)[0]
    
    if len(line_rows) == 0:
        return []
    
    staff_lines = []
    start = line_rows[0]
    for i in range(1, len(line_rows)):
        if line_rows[i] - line_rows[i-1] > 3:
            staff_lines.append((start + line_rows[i-1]) // 2)
            start = line_rows[i]
    staff_lines.append((start + line_rows[-1]) // 2)
    
    if len(staff_lines) < 5:
        return []
    
    gaps = [staff_lines[i+1] - staff_lines[i] for i in range(len(staff_lines)-1)]
    median_gap = np.median(gaps)
    
    staves = []
    current = [staff_lines[0]]
    for i in range(1, len(staff_lines)):
        if staff_lines[i] - staff_lines[i-1] > median_gap * 2:
            if len(current) >= 5:
                staves.append(current[:5])
            current = [staff_lines[i]]
        else:
            current.append(staff_lines[i])
            if len(current) == 5:
                staves.append(current)
                current = []
    if len(current) >= 5:
        staves.append(current[:5])
    
    return staves


def _y_to_note(y, staves):
    if not staves:
        return 'C4', 0
    
    best = None
    idx = 0
    min_d = float('inf')
    for i, lines in enumerate(staves):
        d = abs(y - (lines[0] + lines[4]) / 2)
        if d < min_d:
            min_d = d
            best = lines
            idx = i
    
    unit = np.mean([best[j+1] - best[j] for j in range(4)])
    pos = int(round((best[4] - y) / (unit / 2)))
    
    all_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    if idx % 2 == 0:
        base_idx, base_oct = 2, 4
    else:
        base_idx, base_oct = 4, 2
    
    note_idx = (base_idx + pos) % 7
    octave = base_oct + (base_idx + pos) // 7
    hand = idx % 2  # 0 for even (Treble), 1 for odd (Bass)
    
    return f"{all_notes[note_idx]}{octave}", hand


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python oemer_extract.py image.jpg           # single image")
        print("  python oemer_extract.py page1.jpg page2.jpg # multiple pages")
        print("  python oemer_extract.py ./folder/           # folder of pages")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    if Path(arg).is_dir():
        notes, coords, bboxes = extract_from_folder(arg)
    elif len(sys.argv) > 2:
        notes, coords, bboxes = extract_from_pages(sys.argv[1:])
    else:
        notes, coords, bboxes = extract_from_image(arg)
        # Add page 0 for single image
        coords = [(x, y, 0) for x, y in coords]
        bboxes = [b + [0] for b in bboxes]
    
    print(f"\nNotes: {notes[:10]}...")
    print(f"Coords: {coords[:5]}...")
    print(f"Bboxes: {bboxes[:3]}...")

# notes, coords, fingers, conf = infer("/Users/anandkashyap/Documents/GitHub/KeyGenius/Backend/Music_Data/Scores/001_Bach_Invention_No1_C_page_1.jpg", "/Users/anandkashyap/Documents/GitHub/KeyGenius/Backend/best_model_right.pth")
