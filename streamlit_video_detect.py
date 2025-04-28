import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import tempfile
import cv2
import numpy as np
from utils import load_model, create_heatmap, overlay_heatmap_on_image

st.set_page_config(page_title="Video Anomaly Detection")
st.title("Anomaly Detection on video with Autoencoder")

uploaded_video = st.file_uploader("Upload video (.mp4)", type=["mp4"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stframe = st.empty()
    model = load_model("models/autoencoder_bottle.pth", device="cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    st.write("processing...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)

        heatmap = create_heatmap(input_tensor, output)
        input_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        overlay = overlay_heatmap_on_image(input_np * 255, heatmap)

        stframe.image(overlay, channels="RGB", use_column_width=True)

    cap.release()
    st.success("Processing complete!")
