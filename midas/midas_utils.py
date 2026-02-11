import torch
import cv2
import numpy as np


class MiDaSDepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        self.device = torch.device("cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device).eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform

    def estimate_depth(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(image_rgb).to(self.device)

        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth = pred.cpu().numpy()

        # Percentile normalization (stable)
        p5, p95 = np.percentile(depth, [5, 95])
        depth = np.clip((depth - p5) / (p95 - p5 + 1e-6), 0, 1)

        return depth
