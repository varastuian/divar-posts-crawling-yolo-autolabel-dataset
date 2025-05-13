import os
import random
import pandas as pd

def get_image_paths(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def generate_siamese_pairs(front_folder, other_folder, output_csv="pairs.csv", num_positive=1000, num_negative=1000):
    front_imgs = get_image_paths(front_folder)
    other_imgs = get_image_paths(other_folder)

    if len(front_imgs) < 2:
        print("❌ Not enough front images.")
        return
    if len(other_imgs) < 1:
        print("❌ No 'other' images.")
        return

    pairs = []

    # --- Positive (wanted) pairs: front vs front ---
    for _ in range(num_positive):
        img1, img2 = random.sample(front_imgs, 2)
        pairs.append([img1, img2, "1"])

    # --- Negative (unwanted) pairs: front vs other ---
    for _ in range(num_negative):
        img1 = random.choice(front_imgs)
        img2 = random.choice(other_imgs)
        pairs.append([img1, img2, "0"])

    # --- Shuffle and save ---
    random.shuffle(pairs)
    df = pd.DataFrame(pairs, columns=["img1", "img2", "label"])
    df.to_csv(output_csv, index=False)
    print(f"✅ Dataset created: {output_csv} with {len(pairs)} pairs")

# --- Usage ---
if __name__ == "__main__":
    generate_siamese_pairs(
        front_folder="Siamese data/405/front",
        other_folder="Siamese data/405/other",
        output_csv="Siamese data/405/pairs_405.csv",
        num_positive=1000,
        num_negative=1000
    )
