import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset ,DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import shutil
class SiameseDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img1 = Image.open(row['img1']).convert('RGB')
        img2 = Image.open(row['img2']).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor(float(row['label']), dtype=torch.float32)
        return img1, img2, label


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        base = models.resnet18(pretrained=True)
        base.fc = nn.Identity()  # Remove final layer
        self.encoder = base
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward_once(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        euclidean = torch.nn.functional.pairwise_distance(out1, out2)
        loss = (1 - label) * 0.5 * torch.pow(euclidean, 2) + \
               label * 0.5 * torch.pow(torch.clamp(self.margin - euclidean, min=0.0), 2)
        return loss.mean()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = SiameseDataset("z:/pairs_all.csv", transform=transform)
loader = DataLoader(dataset, shuffle=True, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)

model_path = "siamese_model_1.pth"
if os.path.exists(model_path):
    # --- Load Model ---
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loded")

else:
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    epochs = 12
    for epoch in range(epochs):
        total_loss = 0
        for img1, img2, label in loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


    torch.save(model.state_dict(),model_path)
    print("Model saved")



# --- Transformation ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# # --- Load Anchor (front-view reference image) ---
# anchor_img = Image.open("Siamese data/405/front/AaYAWYvl_405_3_1746441644.jpg").convert("RGB")
anchor_img = Image.open("z:/front/AaYcUJvb_405_1_1746441709.jpg").convert("RGB")
anchor_tensor = transform(anchor_img).unsqueeze(0).to(device)
anchor_embedding = model.forward_once(anchor_tensor)

# anchor_folder = "Siamese data/405/front"
# anchor_embeddings = []

# for fname in os.listdir(anchor_folder):
#     if not fname.lower().endswith(('.jpg', '.png')):
#         continue
#     img = Image.open(os.path.join(anchor_folder, fname)).convert("RGB")
#     img_tensor = transform(img).unsqueeze(0).to(device)
#     emb = model.forward_once(img_tensor)
#     anchor_embeddings.append(emb)

# Stack and average
# anchor_embeddings = torch.cat(anchor_embeddings, dim=0)

# --- Directory Setup ---
input_folder = r"z:\dataset_fluence"
unwanted_folder = os.path.join(input_folder, "unwanted")
os.makedirs(unwanted_folder, exist_ok=True)

# --- Threshold (adjust based on validation) ---
threshold = 0.8
front_count = 0
rear_count = 0
# --- Predict and move ---
for filename in os.listdir(input_folder):
    filepath = os.path.join(input_folder, filename)

    if not filename.lower().endswith(('.jpg', '.png')):
        continue  # skip non-images

    base = os.path.splitext(filename)[0]
    img_path = os.path.join(input_folder, filename)
    car_label_path = os.path.join(input_folder, base + ".txt")
    color_label_path = os.path.join(input_folder, base + "c.txt")



    try:
        img = Image.open(filepath).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        img_embedding = model.forward_once(img_tensor)

        min_distance = torch.nn.functional.pairwise_distance(anchor_embedding, img_embedding).item()
        # distances = torch.nn.functional.pairwise_distance(anchor_embeddings, img_embedding.repeat(anchor_embeddings.size(0), 1))
        # min_distance = distances.min().item()
        if min_distance >= threshold:
            rear_count += 1
            print(f"{filename}: REAR (distance={min_distance:.2f})")
            shutil.move(filepath, os.path.join(unwanted_folder, filename))
            shutil.move(car_label_path, os.path.join(unwanted_folder, os.path.basename(car_label_path)))
            shutil.move(color_label_path, os.path.join(unwanted_folder, os.path.basename(color_label_path)))
        else:
            front_count += 1
            # print(f"{filename}: FRONT (distance={distance:.2f})")

    except Exception as e:
        print(f"Error with {filename}: {e}")

print(f"\n✅ Done. FRONT: {front_count}, REAR: {rear_count}")
