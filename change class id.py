import os

# Directory containing the text files
directory = r"C:\Users\varas\OneDrive\Documents\Codes\mohi\dataset 1221\dataset\nahayi\yolo_dataset\labels\val"

ikco_models = ["rira", "206", "207", "405", "pars", "samand", "dena", "runna", "tara", "haima", "arisun"]
saipa_models = ["sahand", "tondar-90", "pride", "tiba", "quick", "saina", "shahin", "zamyad", "atlas"]

interestCars = ikco_models + saipa_models

# Create a dictionary mapping car models to their respective class IDs
car_classes = {model: idx for idx, model in enumerate(interestCars)}

# Iterate through all files in the directory
for filename in os.listdir(directory):
    # Check if the file ends with '.txt'
    if filename.endswith(".txt"):
        # Check if the filename contains any key from the car_classes dictionary
        for model, class_id in car_classes.items():
            if model in filename:
                filepath = os.path.join(directory, filename)
                
                # Read the content of the file
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                
                # Modify the content if the class ID differs from the expected value
                modified_lines = []
                for line in lines:
                    parts = line.split()
                    if parts:  # Check if the line is not empty
                        current_class_id = parts[0]
                        # If the current class ID differs from the expected value, update it
                        if current_class_id != str(class_id):
                            parts[0] = str(class_id)
                            modified_line = ' '.join(parts) + '\n'
                            modified_lines.append(modified_line)
                        else:
                            modified_lines.append(line)
                    else:
                        modified_lines.append(line)  # Keep empty lines as is
                
                # Write the modified content back to the file
                with open(filepath, 'w') as file:
                    file.writelines(modified_lines)
                
                break  # Stop checking other models once a match is found

print("Modification complete.")