import sys
import cv2
import numpy as np
import os
from PIL import Image, ImageOps

# Ensure Backend is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../Backend'))
sys.path.append(BACKEND_DIR)

import inference as op

def pil_to_opencv(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def opencv_to_pil(cv_image):
    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

def overlay_transparent_image(background_img, overlay_img_rgba, x_offset, y_offset, hand=0):
    bg_h, bg_w = background_img.shape[:2]

    # Dynamically scale number based on image width
    scale = bg_w / 1080 * 0.35
    h, w = overlay_img_rgba.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    overlay = cv2.resize(overlay_img_rgba, (new_w, new_h))

    # Center the overlay on the requested coordinate
    x_start = x_offset - new_w // 2
    y_start = y_offset - new_h // 2

    # Bounds checking
    if x_start < 0 or y_start < 0 or x_start + new_w > bg_w or y_start + new_h > bg_h:
        return background_img

    b, g, r, a = cv2.split(overlay)
    
    # Optional: Tint left hand numbers differently (e.g., slightly blue-ish) if you want to distinguish them
    if hand == 1:
        # Just an example: shift LH colors to distinguish if needed
        # b = cv2.add(b, 30) 
        pass

    overlay_rgb = cv2.merge((b, g, r))
    mask = a.astype(float) / 255.0
    inv_mask = 1.0 - mask

    roi = background_img[y_start:y_start+new_h, x_start:x_start+new_w]
    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * inv_mask + overlay_rgb[:, :, c] * mask

    background_img[y_start:y_start+new_h, x_start:x_start+new_w] = roi
    return background_img

def main():
    if len(sys.argv) < 3:
        print("Usage: process_image.py <input_path> <output_path>", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Model paths
    rh_path = os.path.join(BACKEND_DIR, "best_model_right.pth")
    lh_path = os.path.join(BACKEND_DIR, "best_model_left.pth")
    
    # Fallbacks
    if not os.path.exists(rh_path): rh_path = os.path.join(BACKEND_DIR, "final_model_right.pth")
    if not os.path.exists(lh_path): lh_path = os.path.join(BACKEND_DIR, "final_model_left.pth")

    try:
        print(f"--- KeyGenius Dual Inference ---")
        print(f"Input: {input_path}")
        
        # 1. Run dual-hand inference
        notes, hands, raw_coords, fingers, confs = op.infer_dual(input_path, rh_path, lh_path)
        
        if not notes:
            print("No notes detected to process.", file=sys.stderr)
            # Just copy the original if failed
            with Image.open(input_path) as img:
                img.save(output_path)
            return

        # 2. Adjust coordinates (RH goes UP, LH goes DOWN)
        coords = op.adjust_fingering_coords(raw_coords, hands)

        with Image.open(input_path) as img:
            bg = pil_to_opencv(img)
            numbers_dir = os.path.join(CURRENT_DIR, "numbers")

            processed_count = 0
            for i in range(len(fingers)):
                f_num = fingers[i]
                if f_num < 1 or f_num > 5:
                    continue
                
                num_img_path = os.path.join(numbers_dir, f"{f_num}_small.png")
                overlay = cv2.imread(num_img_path, cv2.IMREAD_UNCHANGED)
                
                if overlay is not None:
                    x, y, _ = coords[i]
                    # Pass the hand index to the overlay function
                    bg = overlay_transparent_image(bg, overlay, x, y, hand=hands[i])
                    processed_count += 1

            # Save result
            output_image = opencv_to_pil(bg)
            output_image.save(output_path)
            print(f"✅ Success! Processed {processed_count} notes with dual models.")
            print(f"Output saved to: {output_path}")

    except Exception as exc:
        print(f"❌ Error during dual processing: {str(exc)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
