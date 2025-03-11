import os
import shutil
folders = []
for i in range(7, 16):
    # if i == 3:
    #     continue
    folders.append(r"C:\Users\varas\OneDrive\Documents\Codes\mohi\dataset 1221\dataset\dataset"+str(i))

output_folder = r"C:\Users\varas\OneDrive\Documents\Codes\mohi\dataset 1221\dataset\final\files7_15"
os.makedirs(output_folder, exist_ok=True)  


for folder_path in folders:
    if os.path.isdir(folder_path): 
        print(f"Processing folder: {folder_path}")

        files = os.listdir(folder_path)
        images = [f for f in files if f != 'detected_dataset' and f.endswith(".jpg")]
        txt_files = [f for f in files if f.endswith(".txt")]
        images_with_txt = [img for img in images if os.path.splitext(img)[0] + '.txt' in txt_files]
        for img in images:
            if img not in images_with_txt:
                os.remove(os.path.join(folder_path, img))
                print(f"Removed {img} (no corresponding .txt file)")
                # Process each image and its corresponding txt file
        for image in images:
            base_name = os.path.splitext(image)[0]  # Get base name without extension
            txt_file = f"{base_name}.txt"  # Corresponding txt file

            if txt_file in txt_files:  # Ensure txt file exists
                image_path = os.path.join(folder_path, image)
                txt_path = os.path.join(folder_path, txt_file)

                # Extract car name from the base name (before the first underscore)
                car_name = base_name.split('_')[0]

                # Get the last modification time of the image
                timestamp = int(os.path.getmtime(image_path))



                # # Check if the image and txt file names are duplicates
                # new_image_name = f"{car_name}_{timestamp}.jpg"
                # new_txt_name = f"{car_name}_{timestamp}.txt"

                # # If both image and txt file already exist with the same base name, add "r"
                # if os.path.exists(os.path.join(output_folder, new_image_name)) and os.path.exists(os.path.join(output_folder, new_txt_name)):
                # new_image_name = f"{car_name}_{timestamp}_r.jpg"
                # new_txt_name = f"{car_name}_{timestamp}r.txt"

                # New paths for renamed files in the output folder
                new_image_path = os.path.join(output_folder, base_name+".jpg")
                new_txt_path = os.path.join(output_folder, base_name+".txt")

                # Copy the files to the output folder with the new names
                shutil.copy(image_path, new_image_path)
                shutil.copy(txt_path, new_txt_path)

                print(f"Copied: {image} & {txt_file} → {base_name} & {base_name}")

