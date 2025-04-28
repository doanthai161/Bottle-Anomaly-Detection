import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from autoencoder import ConvAutoencoder
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
IMG_SIZE = 256
CATEGORY = "bottle"  

data_dir = f"data/{CATEGORY}/{CATEGORY}/train"

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ConvAutoencoder().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

os.makedirs("models", exist_ok=True)

print(f"train Autoencoder on '{CATEGORY}'...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        loss = F.mse_loss(outputs, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.6f}")

save_path = f"models/autoencoder_{CATEGORY}.pth"
torch.save(model.state_dict(), save_path)
print(f"save at: {save_path}")
