"""
SAM 3 Logo Detector - for integration with pipeline
"""
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
import numpy as np

class SAM3LogoDetector:
    """Logo detector using SAM 3 with text prompts."""
    
    def __init__(self):
        """Initialize SAM 3 model."""
        print("Loading SAM 3 model...")
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
    
    def detect_and_crop(self, image_path: str, text_prompt: str = "logo") -> list:
        """
        Detect logos using SAM 3 text prompting.
        
        Args:
            image_path: Path to image
            text_prompt: Text description (default: "logo")
            
        Returns:
            List of detections with format:
            [{'label': 'logo', 'box': [x, y, w, h], 'confidence': float}]
        """
        # Load image
        image = Image.open(image_path)
        inference_state = self.processor.set_image(image)
        
        # Detect with text prompt
        output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        
        # Convert to pipeline format
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        detections = []
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            # Convert box format from [x1, y1, x2, y2] to [x, y, w, h]
            box_np = box.cpu().numpy().astype(int)
            x1, y1, x2, y2 = box_np
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            
            detections.append({
                'label': 'logo',
                'box': [x, y, w, h],
                'confidence': float(score)
            })
            print(f"  - Detected 'logo' with confidence {float(score):.2f}")
        
        return detections
