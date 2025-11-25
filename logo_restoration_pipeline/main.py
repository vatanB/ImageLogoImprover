import os
import glob
import logging
from dotenv import load_dotenv

# Import modules
from detector import LogoDetector
from masker import create_clinical_mask
from generator import restore_logo
from blender import seamless_merge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
INPUT_DIR = "./input"
OUTPUT_DIR = "./output"
ASSETS_DIR = "./assets"

# Brand Assets Map (Example)
# In a real scenario, this might be loaded from a config file or database
BRAND_ASSETS = {
    'bmw': os.path.join(ASSETS_DIR, 'bmw_logo.webp'),
    'mercedes': os.path.join(ASSETS_DIR, 'mercedes_logo.png'),
    'audi': os.path.join(ASSETS_DIR, 'audi_logo.png'),
    # Add more brands as needed
}

def main():
    """
    Main orchestrator for the Logo Restoration Pipeline.
    """
    logger.info("Starting Logo Restoration Pipeline...")
    
    # 1. Initialize Detector
    try:
        detector = LogoDetector() # Defaults to yolo11n.pt
        logger.info("LogoDetector initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return

    # 2. Scan Input Directory
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.[jJ][pP][gG]")) + \
                  glob.glob(os.path.join(INPUT_DIR, "*.[pP][nN][gG]")) + \
                  glob.glob(os.path.join(INPUT_DIR, "*.[jJ][pP][eE][gG]"))
    
    if not image_paths:
        logger.warning(f"No images found in {INPUT_DIR}")
        return

    logger.info(f"Found {len(image_paths)} images to process.")

    # 3. Process Images
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        logger.info(f"Processing {filename}...")
        
        try:
            # A. Detect Logo
            detections = detector.detect_and_crop(img_path)
            
            if not detections:
                logger.info(f"No logos detected in {filename}. Skipping.")
                continue
                
            for i, detection in enumerate(detections):
                label = detection['label']
                box = detection['box']
                confidence = detection['confidence']
                
                logger.info(f"  - Detected '{label}' with confidence {confidence:.2f}")
                
                # Determine Brand and Reference Asset
                brand_key = label.lower()
                
                # The detector now has enhanced inference logic, so 'label' should be more accurate.
                # We still check if we have the asset for the detected brand.
                
                if brand_key not in BRAND_ASSETS:
                    logger.warning(f"    - Brand '{label}' detected but asset not found in BRAND_ASSETS. Skipping.")
                    continue
                
                reference_logo_path = BRAND_ASSETS.get(brand_key)
                if not reference_logo_path or not os.path.exists(reference_logo_path):
                    logger.warning(f"    - Reference asset for '{brand_key}' not found at {reference_logo_path}. Skipping.")
                    continue

                # B. Generate Clinical Mask (Now with Dilation)
                mask_filename = f"mask_{filename}_{i}.png"
                mask_path = os.path.join(OUTPUT_DIR, "masks", mask_filename)
                
                import cv2
                temp_img = cv2.imread(img_path)
                if temp_img is None:
                     logger.error(f"    - Failed to read image {img_path}. Skipping.")
                     continue
                
                create_clinical_mask(temp_img.shape, box, mask_path)
                logger.info(f"    - Clinical Mask (dilated) generated at {mask_path}")
                
                
                # C. Restore Logo (Gemini 3.0 Pro with cropped region)
                final_filename = f"restored_{filename}".replace(f"_{i}", "")  # One file per image
                final_path = os.path.join(OUTPUT_DIR, final_filename)
                
                # Ensure directories exist
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                
                # restore_logo now crops, enhances, and pastes back - returns full image
                restore_logo(img_path, mask_path, reference_logo_path, brand_key, box, final_path)
                logger.info(f"    - Logo enhanced and integrated at {final_path}")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue

    logger.info("=== Pipeline Execution Completed Successfully ===")
    logger.info(f"Outputs saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
