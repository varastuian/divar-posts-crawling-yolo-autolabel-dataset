import os
import time
from pathlib import Path

# Define base directory
base_dir = "dataset/final"

# Process only dataset1 to dataset6
for i in range(1, 7):
    dataset_path = Path(base_dir) / f"dataset{i}"
    
    if not dataset_path.exists():
        continue

    # Iterate over files
    for file in dataset_path.iterdir():
        if file.is_file():
            # Get file timestamp (modification time)
            # timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(file.stat().st_mtime))
            timestamp = os.path.getmtime(file)

            # Get file extension
            ext = file.suffix.lower()
            
            # Find matching image and text file
            if ext in [".jpg", ".png", ".jpeg"]:  # Add more image formats if needed
                txt_file = file.with_suffix(".txt")
                
                # Rename image
                new_img_name = f"{timestamp}{ext}"
                new_img_path = dataset_path / new_img_name
                file.rename(new_img_path)

                # Rename corresponding text file
                if txt_file.exists():
                    new_txt_name = f"{timestamp}.txt"
                    new_txt_path = dataset_path / new_txt_name
                    txt_file.rename(new_txt_path)

                print(f"Renamed {file.name} and {txt_file.name} to {timestamp}")

print("Renaming complete!")




#-------------------------------------------------#

import os
import shutil
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS

def rename_and_copy_images(src_dirs, dest_dir, car_model):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        def get_image_timestamp(image_path):
            image = Image.open(image_path)
            exif_data = image._getexif()
            if exif_data:
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    if tag_name == 'DateTimeOriginal':
                        return datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
            return None
    image_counter = 1

    for src_dir in src_dirs:
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    new_filename = f"{car_model}_{image_counter}({timestamp}).jpg"
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_dir, new_filename)
                    
                    shutil.copy2(src_file, dest_file)
                    image_counter += 1

# Example usage
src_dirs = [
    'dataset\final\dataset1',
    'dataset\final\dataset2',
    'dataset\final\dataset3',
    'dataset\final\dataset4',
    'dataset\final\dataset5',
    'dataset\final\dataset6',
]
dest_dir = 'dataset\final\merged'

rename_and_copy_images(src_dirs, dest_dir)