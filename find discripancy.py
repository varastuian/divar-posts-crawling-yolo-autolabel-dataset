import os

# Paths
root_dir = "dataset/dataset_1746269488"
car_dir = os.path.join(root_dir, "car")
color_dir = os.path.join(root_dir, "color")

# Get image names (without extension)
images = {os.path.splitext(f)[0] for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png'))}

# Get annotation names from both folders
annotations = set()
for folder in [car_dir, color_dir]:
    if os.path.exists(folder):
        annotations.update(os.path.splitext(f)[0] for f in os.listdir(folder) if f.lower().endswith('.txt'))

# Find discrepancies
missing_annotations = images - annotations
orphan_annotations = annotations - images

print("Images with no annotation:")
for name in sorted(missing_annotations):
    print(f"  {name}")

print("\nAnnotations with no image:")
for name in sorted(orphan_annotations):
    print(f"  {name}")

    wZWXKz76_elantra_0_1746270266
    wZWXKz76_elantra_0_1746270266
