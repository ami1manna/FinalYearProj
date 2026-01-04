import torch
import cv2
import numpy as np


class MiDaSDepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        self.device = torch.device("cpu")

        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def estimate_depth(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(image_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_min = depth_map.min()
        depth_max = depth_map.max()

        return (depth_map - depth_min) / (depth_max - depth_min + 1e-6)
