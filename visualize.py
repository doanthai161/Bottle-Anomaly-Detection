import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from utils import load_model, create_heatmap, overlay_heatmap_on_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = "data/bottle/bottle/test/broken_large/001.png"  

def visualize_anomaly(image_path, category="bottle"):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    model = load_model(f"models/autoencoder_{category}.pth", device=DEVICE)
    with torch.no_grad():
        output = model(input_tensor)

    input_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    output_np = output.squeeze().permute(1, 2, 0).cpu().numpy()
    heatmap = create_heatmap(input_tensor, output)
    overlay = overlay_heatmap_on_image(input_np * 255, heatmap) #scale img 0-255

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(input_np)
    axs[0].set_title("Ảnh gốc")
    axs[1].imshow(output_np)
    axs[1].set_title("Ảnh tái tạo")
    axs[2].imshow(overlay)
    axs[2].set_title("Heatmap lỗi")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
visualize_anomaly(image_path)