import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from utils import load_model, create_heatmap, overlay_heatmap_on_image

st.set_page_config(page_title="Anomaly Detection", layout="centered")
st.title("Anomaly Detection with Autoencoder")

uploaded_file = st.file_uploader("upload image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="root img", use_column_width=True)

    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),])
    input_tensor = transform(image).unsqueeze(0)

    model = load_model("models/autoencoder_bottle.pth", device="cpu")
    with torch.no_grad():
        output = model(input_tensor)
        loss = torch.nn.functional.mse_loss(output, input_tensor).item()

    st.write(f"Reconstruction Loss (MSE): `{loss:.4f}`")
    if loss > 0.01:
        st.error("Phát hiện lỗi trên sản phẩm")
    else:
        st.success("Sản phẩm bình thường.")

    input_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    heatmap = create_heatmap(input_tensor, output)
    overlay = overlay_heatmap_on_image(input_np * 255, heatmap)

    st.image(Image.fromarray(overlay), caption="error heatmap", use_column_width=True)