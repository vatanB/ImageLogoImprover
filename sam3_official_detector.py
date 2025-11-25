"""
SAM 3 Logo Detector - Official Implementation
Uses text prompts like "logo" to find and segment logos.
"""

import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import numpy as np
import cv2
import os

def detect_logos_sam3(image_path, text_prompt="logo", output_dir="output"):
    """
    Detect logos using SAM 3 with text prompts.
    
    Args:
        image_path: Path to input image
        text_prompt: Text description (e.g., "logo", "BMW logo")
        output_dir: Directory to save results
        
    Returns:
        List of detections with masks, boxes, scores
    """
    
    print("Loading SAM 3 model...")
    print("Note: First run will download checkpoint from HuggingFace")
    print("Make sure you have requested access to: https://huggingface.co/facebook/sam3")
    print("And logged in with: huggingface-cli login")
    
    try:
        # Load the model
        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        
        # Load image
        print(f"Processing: {image_path}")
        image = Image.open(image_path)
        inference_state = processor.set_image(image)
        
        # Prompt with text
        print(f"Searching for: '{text_prompt}'")
        output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        
        # Get results
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        print(f"\n✓ Found {len(masks)} logo(s)")
        
        # Prepare for visualization
        detections = []
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            detections.append({
                'mask': mask.cpu().numpy(),
                'box': box.cpu().numpy().astype(int).tolist(),
                'score': float(score)
            })
            print(f"  Logo {i+1}: Box={detections[i]['box']}, Score={score:.3f}")
        
        # Visualize
        os.makedirs(output_dir, exist_ok=True)
        image_np = np.array(image)
        result_img = image_np.copy()
        
        for i, det in enumerate(detections):
            mask = det['mask'].astype(bool)
            box = det['box']
            color = (0, 255, 0)  # Green
            
            # Overlay mask
            result_img[mask] = (result_img[mask] * 0.6 + np.array(color) * 0.4).astype(np.uint8)
            
            # Draw box
            cv2.rectangle(result_img, (box[0], box[1]), (box[2], box[3]), color, 3)
            cv2.putText(result_img, f"Logo {i+1}: {det['score']:.2f}",
                       (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        output_path = os.path.join(output_dir, "sam3_official_detection.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        
        print(f"✓ Saved to: {output_path}")
        
        return detections
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Request access to https://huggingface.co/facebook/sam3")
        print("2. Run: huggingface-cli login")
        print("3. Paste your HuggingFace token")
        raise

if __name__ == "__main__":
    image_path = "input/BMW_24V_Drift_Kart_Licensed_Electric_Ride_on_Drift_Kart_[BDM0978]_-_AI_Background_Square_1.jpg"
    
    # Detect logos with text prompt
    detections = detect_logos_sam3(image_path, text_prompt="logo")
