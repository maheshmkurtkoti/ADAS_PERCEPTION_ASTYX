import cv2
import numpy as np
from ultralytics import YOLO

class CameraDetector:
    def __init__(self, model_name = "yolov8s.pt",conf=0.2):
        """
        model_name: yolov8n.pt(fast)/yolov8s.pt)(better)
        conf: confidence threshold
        """
        self.model = YOLO(model_name)
        self.conf = conf

        #COCO vehicle classes
        self.vehicle_classes ={
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

    def detect(self, image):
        """
        image: BGR Immage (cv2)
        returns: list of detections
        """
        results = self.model(image,verbose =False)[0]

        detections = []

        if results.boxes is None:
            return detections
        
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for box,score,cls in zip(boxes,scores, classes):
            if score < self.conf:
                continue
            if cls not in self.vehicle_classes:
                continue

            x1,y1,x2,y2 = box

            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(score),
                "class_id": int(cls),
                "class_name": self.vehicle_classes[cls],
                "center": [int((x1 + x2)/2), int(( y1 + y2)/2)]})
            
            return detections
        



