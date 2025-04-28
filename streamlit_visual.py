import streamlit as st
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from utils import load_model, create_heatmap, overlay_heatmap_on_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(page_title="Visualize Anomaly Detection", layout="centered")
st.title("Visualize Anomaly Detection with Autoencoder")

uploaded_file = st.file_uploader("Upload một ảnh lỗi hoặc ảnh bình thường", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Ảnh gốc đã upload", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    category = "bottle"  # có t
    model = load_model(f"models/autoencoder_{category}.pth", device=DEVICE)

    with torch.no_grad():
        output = model(input_tensor)

    input_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    output_np = output.squeeze().permute(1, 2, 0).cpu().numpy()
    heatmap = create_heatmap(input_tensor, output)
    overlay = overlay_heatmap_on_image(input_np * 255, heatmap)

    st.subheader("So sánh ảnh:")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(input_np)
    axs[0].set_title("Ảnh gốc")
    axs[0].axis("off")

    axs[1].imshow(output_np)
    axs[1].set_title("Ảnh tái tạo")
    axs[1].axis("off")

    axs[2].imshow(overlay)
    axs[2].set_title("Heatmap lỗi")
    axs[2].axis("off")

    st.pyplot(fig)
