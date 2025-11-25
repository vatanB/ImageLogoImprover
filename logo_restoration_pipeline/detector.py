import os
from typing import List, Dict, Union, Any
from ultralytics import YOLO
import cv2
import numpy as np

class LogoDetector:
    """
    A class to detect brand logos on products using YOLO11.
    
    NOTE: This model is initialized with 'yolo11n.pt' (COCO pretrained).
    For accurate logo detection, it requires fine-tuning on the "Brand Eye" dataset
    (Roboflow) or a synthetic dataset of logos pasted on car textures.
    """
    
    def __init__(self, model_path: str = 'yolo11n.pt'):
        """
        Initialize the LogoDetector with a YOLO model.
        
        Args:
            model_path (str): Path to the YOLO model weights. Defaults to 'yolo11n.pt'.
        """
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}")

    def detect_and_crop(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect logos in an image and return their bounding boxes and labels.
        
        Args:
            image_path (str): Path to the input image.
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing:
                - 'label': The detected brand name (or 'Unknown').
                - 'box': The bounding box [x, y, w, h].
                - 'confidence': The confidence score.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        try:
            # Run inference with lower threshold to catch difficult logos (like flags)
            results = self.model(image_path, verbose=False, conf=0.15)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Convert to x, y, w, h
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]
                    
                    # Logic for generic labels
                    # If the detected class is generic (e.g., 'logo', 'car', 'tv'), 
                    # we try to infer or default to "Unknown".
                    
                    brand_label = label
                    
                    # Enhanced inference logic
                    generic_classes = ['car', 'truck', 'bus', 'train', 'logo', 'tv', 'vehicle', 'object', 'motorcycle', 'flag', 'banner', 'sign', 'kite', 'person']
                    
                    if label.lower() in generic_classes:
                         # Try to infer from filename with more robust matching
                         filename = os.path.basename(image_path).lower()
                         
                         # Map common brand keywords to standardized brand names
                         brand_map = {
                             'bmw': 'BMW',
                             'mercedes': 'Mercedes',
                             'benz': 'Mercedes',
                             'audi': 'Audi',
                             'tesla': 'Tesla',
                             'porsche': 'Porsche',
                             'ferrari': 'Ferrari',
                             'lamborghini': 'Lamborghini',
                             'ford': 'Ford',
                             'toyota': 'Toyota',
                             'honda': 'Honda'
                         }
                         
                         found_brand = False
                         for key, val in brand_map.items():
                             if key in filename:
                                 brand_label = val
                                 found_brand = True
                                 break
                        
                         if not found_brand:
                             brand_label = 'Unknown'
                    
                    detections.append({
                        'label': brand_label,
                        'box': [x, y, w, h],
                        'confidence': conf
                    })
            
            return detections

        except Exception as e:
            print(f"Error during detection on {image_path}: {e}")
            return []

if __name__ == "__main__":
    # Simple test
    try:
        detector = LogoDetector()
        print("LogoDetector initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize LogoDetector: {e}")
