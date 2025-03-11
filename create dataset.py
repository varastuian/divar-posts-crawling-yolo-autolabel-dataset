# import os
# import shutil
# import random
# from sklearn.model_selection import train_test_split

# # Define paths
# dataset_path = r"C:\Users\varas\OneDrive\Documents\Codes\mohi\dataset 1221\dataset\final\finalMenrged"
# output_path = r"C:\Users\varas\OneDrive\Documents\Codes\mohi\dataset 1221\dataset\nahayi\yolo_dataset"

# # Create directories
# os.makedirs(os.path.join(output_path, "images", "train"), exist_ok=True)
# os.makedirs(os.path.join(output_path, "images", "val"), exist_ok=True)
# os.makedirs(os.path.join(output_path, "labels", "train"), exist_ok=True)
# os.makedirs(os.path.join(output_path, "labels", "val"), exist_ok=True)

# # Get list of files
# image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
# txt_files = [f for f in os.listdir(dataset_path) if f.endswith('.txt')]

# # Ensure that each image has a corresponding txt file
# image_files = [f for f in image_files if f.replace('.jpg', '.txt') in txt_files]

# # Split dataset into train and validation sets
# train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)

# # Function to copy files
# def copy_files(files, subset):
#     for file in files:
#         # Copy image
#         shutil.copy(os.path.join(dataset_path, file), os.path.join(output_path, "images", subset, file))
#         # Copy corresponding label
#         txt_file = file.replace('.jpg', '.txt')
#         shutil.copy(os.path.join(dataset_path, txt_file), os.path.join(output_path, "labels", subset, txt_file))

# # Copy files to respective directories
# copy_files(train_images, "train")
# copy_files(val_images, "val")

# print("Dataset has been successfully organized into YOLO format.")


import os
from collections import defaultdict

# Define the path to the dataset
dataset_path = r"C:\Users\varas\OneDrive\Documents\Codes\mohi\dataset 1221\dataset\final\finalMenrged"

# Initialize a dictionary to store the count of each car name
car_counts = defaultdict(int)

# Iterate through all files in the dataset
for file_name in os.listdir(dataset_path):
    if file_name.endswith('.jpg') or file_name.endswith('.txt'):
        # Extract the car name (part before the first underscore)
        car_name = file_name.split('_')[0]
        # Update the count for this car name
        car_counts[car_name] += 1

# Print the results
print("Car Name Counts:")
for car_name, count in car_counts.items():
    print(f"{car_name}: {count} images")