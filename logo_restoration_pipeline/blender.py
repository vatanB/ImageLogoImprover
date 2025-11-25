import cv2
import numpy as np
import os

def seamless_merge(original_img_path: str, generated_patch_path: str, mask_path: str, output_path: str) -> str:
    """
    Seamlessly merge the generated logo patch into the original image using Poisson blending.
    
    Args:
        original_img_path (str): Path to the original image.
        generated_patch_path (str): Path to the generated patch image.
        mask_path (str): Path to the binary mask.
        output_path (str): Path to save the final blended image.
        
    Returns:
        str: Path to the saved blended image.
    """
    try:
        # Load images
        src = cv2.imread(generated_patch_path)
        dst = cv2.imread(original_img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if src is None:
            raise FileNotFoundError(f"Generated patch not found at {generated_patch_path}")
        if dst is None:
            raise FileNotFoundError(f"Original image not found at {original_img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found at {mask_path}")

        # Resize src to match dst if needed
        # The prompt says: "Ensure generated_patch is resized to match the original image dimensions exactly."
        # This implies the generated patch is the full image size (which it should be if generated from the original + mask).
        # But if it's a crop, we'd need to place it.
        # Given the generator takes the full original image and mask, the output should be full size.
        # However, let's ensure dimensions match.
        
        if src.shape != dst.shape:
            src = cv2.resize(src, (dst.shape[1], dst.shape[0]))
            
        # Ensure mask is uint8 and single channel (already loaded as grayscale)
        # But seamlessClone needs mask to be same size as src/dst?
        # Actually, seamlessClone takes: src, dst, mask, center, flags.
        # The mask should be the size of src (or at least cover the area).
        # If src and dst are full size, mask should be too.
        
        # Find the center of the white area in the mask to use as the center point for cloning
        # Or we can just use the center of the image if the mask is in place.
        # The prompt says: "Center point of the mask (x + w//2, y + h//2)".
        # This implies we might need the bounding box again, OR we calculate it from the mask moments.
        # Let's calculate from mask moments to be safe and self-contained.
        
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center = (cX, cY)
        else:
            # Fallback to image center if mask is empty (shouldn't happen)
            center = (dst.shape[1] // 2, dst.shape[0] // 2)
            
        # Perform Poisson Blending
        # We use cv2.MIXED_CLONE for better texture preservation and reduced color bleeding.
        # MIXED_CLONE assumes that the texture of the source image should be preserved, 
        # but the gradients should be a mix of source and destination.
        # This is often better for logos on textured surfaces than NORMAL_CLONE.
        
        flags = cv2.MIXED_CLONE
        
        blended = cv2.seamlessClone(src, dst, mask, center, flags)
        
        # Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, blended)
        
        return output_path

    except Exception as e:
        raise RuntimeError(f"Failed to blend images: {e}")

if __name__ == "__main__":
    print("Blender module ready.")
