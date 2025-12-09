from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path

folder = Path("Music_Data/ScorePDF")
folder_new = Path("Music_Data/Scores")

for item in folder.iterdir():
    print(f"Checking: {item.name}")
    if item.suffix.lower() == '.pdf':
        print(f"📄 Processing PDF: {item.name}")
        try:
            images = convert_from_path(item)
            print(f"Got {len(images)} pages")
            for i, image in enumerate(images):
                output_path = folder_new / f"{item.stem}_page_{i}.jpg"

                print(output_path)
                image.save(output_path, "JPEG")
                print(f"Saved: {output_path}")
        except Exception as e:
            print(f"⚠️ Error processing {item.name}: {e}")

print("✅ Loop finished!")
