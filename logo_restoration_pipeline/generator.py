import os
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def restore_logo(original_img_path: str, mask_path: str, reference_logo_path: str, brand_name: str, box: list, output_path: str = None) -> str:
    """
    Restore the logo using Gemini 3.0 Pro Image.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"========== LOGO RESTORATION DEBUG ==========")
        logger.info(f"Original image: {original_img_path}")
        logger.info(f"Reference logo: {reference_logo_path}")
        logger.info(f"Bounding box: {box}")
        
        # Initialize Client
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        # Check for API Key
        api_key = os.getenv("GOOGLE_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if api_key:
             logger.info(f"Using API Key authentication")
             client = genai.Client(api_key=api_key, vertexai=False)
        elif project_id:
             logger.info(f"Using Vertex AI authentication (project: {project_id})")
             client = genai.Client(vertexai=True, project=project_id, location=location)
        else:
             raise ValueError("No valid authentication found. Set GOOGLE_GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT.")

        model_id = "gemini-3-pro-image-preview"
        logger.info(f"Using model: {model_id}")
        
        # STEP 1: Crop to bounding box from ORIGINAL IMAGE
        full_image = Image.open(original_img_path)
        logger.info(f"Full image size: {full_image.size}")
        
        x, y, w, h = box
        logger.info(f"Cropping region: x={x}, y={y}, w={w}, h={h}")
        
        cropped_logo = full_image.crop((x, y, x+w, y+h))
        logger.info(f"Cropped logo size: {cropped_logo.size}")
        
        # SAVE cropped input for review
        debug_dir = "./output/debug"
        os.makedirs(debug_dir, exist_ok=True)
        cropped_input_path = os.path.join(debug_dir, f"cropped_input_{brand_name}.png")
        cropped_logo.save(cropped_input_path)
        logger.info(f"SAVED cropped input to: {cropped_input_path}")
        
        # STEP 2: Load reference logo
        reference_logo = Image.open(reference_logo_path)
        logger.info(f"Reference logo size: {reference_logo.size}")
        
        # STEP 3: Create context-aware prompt
        prompt = f"""Enhance ONLY the clarity and sharpness of the {brand_name} logo in this image.

CRITICAL REQUIREMENTS:
- Preserve the EXACT angle and perspective of the logo
- Maintain the surface material and texture (plastic, metal, sticker, etc.)
- Keep ALL existing lighting, shadows, reflections, and highlights
- Do NOT replace or regenerate the logo - only sharpen and clarify what's already there
- Keep the exact size, position, and 3D appearance
- Match the logo design and colors to the reference image
- Preserve any wear, scratches, or real-world imperfections in a natural way

The reference image shows the correct {brand_name} logo design and brand colors.
The original cropped image shows the actual logo on the product - enhance its clarity while keeping its real-world appearance intact.
Output should be the same size as the input cropped logo."""
        
        logger.info(f"PROMPT: {prompt}")
        
        # STEP 4: Call Gemini API with improved config
        logger.info(f"Calling Gemini API...")
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt, cropped_logo, reference_logo],
            config=types.GenerateContentConfig(
                temperature=0.4,  # Lower temperature for more faithful reproduction
                image_config=types.ImageConfig(
                    aspect_ratio="1:1",  # Use 1:1 for logos (they're typically square-ish)
                    image_size="2K"
                )
            )
        )
        logger.info(f"Gemini API call completed")
        
        # STEP 5: Extract response
        if response.text:
             logger.warning(f"Model returned text: {response.text}")
             raise RuntimeError(f"Model returned text instead of image: {response.text}")
             
        generated_image_bytes = None
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    generated_image_bytes = part.inline_data.data
                    break
        
        if generated_image_bytes:
            from PIL import Image as PILImage
            import io
            
            # Decode to image
            enhanced_logo = PILImage.open(io.BytesIO(generated_image_bytes))
            logger.info(f"Enhanced logo size from Gemini: {enhanced_logo.size}")
            
            # SAVE Gemini output for review
            gemini_output_path = os.path.join(debug_dir, f"gemini_output_{brand_name}.png")
            enhanced_logo.save(gemini_output_path)
            logger.info(f"SAVED Gemini output to: {gemini_output_path}")
            
            # STEP 6: Resize to match original crop size
            logger.info(f"Resizing enhanced logo from {enhanced_logo.size} to ({w}, {h})")
            enhanced_logo = enhanced_logo.resize((w, h), Image.Resampling.LANCZOS)
            
            # STEP 7: Paste back into full image
            logger.info(f"Pasting enhanced logo back at position ({x}, {y})")
            result_image = full_image.copy()
            result_image.paste(enhanced_logo, (x, y))
            
            # SAVE final result
            if output_path:
                result_image.save(output_path)
                logger.info(f"SAVED final result to: {output_path}")
                logger.info(f"========== END DEBUG ==========")
                return output_path
            else:
                 raise ValueError("output_path is required")
        else:
            raise RuntimeError("No image generated in response.")

    except Exception as e:
        logger.error(f"Error in restore_logo: {e}")
        raise RuntimeError(f"Failed to generate logo: {e}")

if __name__ == "__main__":
    print("Generator module ready. Requires API credentials and valid inputs to run.")
