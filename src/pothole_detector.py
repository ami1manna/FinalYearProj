import cv2
from ultralytics import YOLO


class PotholeDetector:
    def __init__(self, model_path: str, conf: float = 0.25):
        self.model = YOLO(model_path)
        self.conf = conf
        print("Model classes:", self.model.names)

    # For IMAGE
    def detect(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or invalid path")

        results = self.model.predict(
            source=image,
            conf=self.conf,
            device="cpu",
            verbose=False
        )

        return image.copy(), results[0]

    # ✅ For VIDEO (REAL-TIME)
    def detect_frame(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            device="cpu",
            verbose=False
        )
        return frame.copy(), results[0]
