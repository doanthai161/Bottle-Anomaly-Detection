import torch
import torch.nn.functional as F
import numpy as np
import cv2
from autoencoder import ConvAutoencoder


def reconstruction_loss(input_tensor, output_tensor):
    return F.mse_loss(output_tensor, input_tensor, reduction='none')


def create_heatmap(input_tensor, output_tensor):
    diff = torch.abs(input_tensor - output_tensor)
    diff = diff.squeeze().permute(1, 2, 0).cpu().numpy()
    heatmap = np.mean(diff, axis=2)  
    return heatmap


def overlay_heatmap_on_image(image_np, heatmap, alpha=0.6):
    heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap_norm = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_np.astype(np.uint8), 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def load_model(path, device='cpu'):
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model