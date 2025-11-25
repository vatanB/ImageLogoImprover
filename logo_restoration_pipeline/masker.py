import cv2
import numpy as np
import os

def create_clinical_mask(image_shape: tuple, box: list, output_path: str = None) -> str:
    """
    Create a clinical elliptical mask for the detected logo.
    
    Args:
        image_shape (tuple): The shape of the original image (height, width, channels).
        box (list): The bounding box [x, y, w, h].
        output_path (str, optional): Path to save the mask image. If None, returns the mask array.
        
    Returns:
        str: Path to the saved mask image.
    """
    try:
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        x, y, w, h = box
        
        # Calculate center and axes for ellipse
        center = (x + w // 2, y + h // 2)
        
        # Apply 10% reduction in mask size (0.9 * w) to ensure we only target the logo interior
        # We apply this reduction to both axes to maintain aspect ratio relative to the box
        # The prompt specifies 0.9 * w, let's apply it to both axes for the ellipse
        axes = (int((w * 0.9) / 2), int((h * 0.9) / 2))
        
        angle = 0
        startAngle = 0
        endAngle = 360
        
        # Draw the ellipse
        # 255 is white (foreground), 0 is black (background)
        cv2.ellipse(mask, center, axes, angle, startAngle, endAngle, 255, -1)
        
        # Apply dilation to the mask to slightly expand the area for better blending
        # This helps in reducing the "halo" effect or sharp edges during Poisson blending
        # We use a small kernel for subtle dilation
        kernel_size = int(max(w, h) * 0.02) # 2% of the object size
        kernel_size = max(3, kernel_size) # Minimum 3x3
        if kernel_size % 2 == 0: kernel_size += 1 # Ensure odd kernel size
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        if output_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, mask)
            return output_path
        else:
            # If no output path is provided, we might want to return the array or save to a temp file
            # For this pipeline, we are expected to save to disk.
            # Let's generate a default path if none provided, or raise error.
            # But the signature says return str, so let's save to a temp file if needed or assume caller provides path.
            # For safety, let's raise if no path provided as per "Output: A binary image ... saved to disk" requirement.
             raise ValueError("output_path must be provided to save the mask.")

    except Exception as e:
        raise RuntimeError(f"Failed to create clinical mask: {e}")

if __name__ == "__main__":
    # Test
    try:
        # Create a dummy image shape and box
        shape = (500, 500, 3)
        test_box = [100, 100, 200, 100] # x, y, w, h
        output = "test_mask.png"
        create_clinical_mask(shape, test_box, output)
        print(f"Mask created at {output}")
        # Clean up
        if os.path.exists(output):
            os.remove(output)
    except Exception as e:
        print(f"Test failed: {e}")
