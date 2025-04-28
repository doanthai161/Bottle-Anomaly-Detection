import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from utils import load_model, create_heatmap, overlay_heatmap_on_image

VIDEO_PATH = "input_video.mp4"
OUTPUT_PATH = "output_video.mp4"
MODEL_PATH = "models/autoencoder_bottle.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model(MODEL_PATH, device=DEVICE)

cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)

    heatmap = create_heatmap(input_tensor, output)
    input_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    overlay = overlay_heatmap_on_image(input_np * 255, heatmap)

    out.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)) 

cap.release()
out.release()
cv2.destroyAllWindows()
print("done processing")
