import os
import sys
import cv2
import numpy as np
from unittest.mock import MagicMock, patch

# Add pipeline to path
sys.path.append("logo_restoration_pipeline")

# Mock the genai client before importing generator if possible, 
# or patch the restore_logo function.
# We will patch detector and generator to ensure the test runs without 
# needing a real YOLO model download (which might fail or take time) 
# and without real API keys.

def mock_detect_and_crop(self, image_path):
    print(f"[MOCK] Detecting logo in {image_path}...")
    # Return a dummy detection centered in the image
    return [{
        'label': 'BMW', # Matches our dummy filename
        'box': [270, 270, 100, 100], # x, y, w, h (centered 100x100 box)
        'confidence': 0.99
    }]

def mock_restore_logo(original_img_path, mask_path, reference_logo_path, brand_name, output_path):
    print(f"[MOCK] Generating logo for {brand_name}...")
    # Just copy the reference logo to the output path, resized to match the mask area roughly?
    # Or just create a green square to show it "generated" something.
    
    # Read reference
    ref = cv2.imread(reference_logo_path)
    if ref is None:
        # Create dummy if missing
        ref = np.full((100, 100, 3), (0, 255, 0), dtype=np.uint8)
        
    # Resize to a reasonable patch size (e.g. same as detection box 100x100)
    # The generator usually outputs the full image size? 
    # Wait, the generator prompt says "Project... into the white masked area".
    # And the blender blends `generated_patch` into `original_img`.
    # If `generated_patch` is the full image returned by Gemini, it works.
    # So our mock should return a full image.
    
    original = cv2.imread(original_img_path)
    if original is None:
        original = np.zeros((640, 640, 3), dtype=np.uint8)
        
    # Draw the reference logo onto the original image at the center
    # This simulates the "restored" image
    h, w = original.shape[:2]
    # Center
    cx, cy = w // 2, h // 2
    # Logo size
    lw, lh = 100, 100
    x = cx - lw // 2
    y = cy - lh // 2
    
    # Resize ref
    ref_resized = cv2.resize(ref, (lw, lh))
    
    # Place it
    output_img = original.copy()
    output_img[y:y+lh, x:x+lw] = ref_resized
    
    # Draw a green border to indicate "Generated"
    cv2.rectangle(output_img, (x, y), (x+lw, y+lh), (0, 255, 0), 2)
    
    cv2.imwrite(output_path, output_img)
    return output_path

def run_mock_test():
    print("Running Mock Pipeline Test...")
    
    # Patch the detector and generator
    with patch('detector.LogoDetector.detect_and_crop', side_effect=mock_detect_and_crop, autospec=True) as mock_det, \
         patch('generator.restore_logo', side_effect=mock_restore_logo) as mock_gen:
        
        # Import main after patching? No, main imports them.
        # We need to patch where they are used in main.
        # Or patch the classes/functions in the modules themselves.
        
        import main
        
        # We also need to mock LogoDetector.__init__ to avoid loading YOLO model
        with patch('detector.LogoDetector.__init__', return_value=None) as mock_init:
             main.main()

if __name__ == "__main__":
    run_mock_test()
