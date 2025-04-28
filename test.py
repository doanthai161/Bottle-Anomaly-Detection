import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from autoencoder import ConvAutoencoder
import torch.nn.functional as F
from sklearn.metrics import classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CATEGORY = "bottle"
THRESHOLD = 0.01  

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

data_dir = f"data/{CATEGORY}/{CATEGORY}/test"
test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = ConvAutoencoder().to(DEVICE)
model.load_state_dict(torch.load(f"models/autoencoder_{CATEGORY}.pth", map_location=DEVICE))
model.eval()

y_true = []
y_pred = []
losses = []

with torch.no_grad():
    for img, label in test_loader:
        img = img.to(DEVICE)
        output = model(img)
        loss = F.mse_loss(output, img).item()
        losses.append(loss)

        y_true.append(label.item())  
        y_pred.append(1 if loss > THRESHOLD else 0)

print("phân loại:")
print(classification_report(y_true, y_pred, target_names=test_dataset.classes))
