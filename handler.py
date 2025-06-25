import runpod
import base64
import requests
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import cv2
import json
import os
import logging
import time
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from environment variable first, then fallback
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN', 'r8_6cksfxEmLxWlYxjW4K1FEbnZMEEmlQw2UeNNY')

def find_input_data(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Extract input data from nested structure"""
    if isinstance(job_input, dict):
        if 'input' in job_input:
            return find_input_data(job_input['input'])
        elif 'image' in job_input or 'mask_type' in job_input:
            return job_input
    return job_input

def create_gradient_mask(shape: Tuple[int, int], center: Tuple[int, int], radius: int) -> np.ndarray:
    """Create a gradient mask for smooth blending"""
    mask = np.zeros(shape, dtype=np.float32)
    y, x = np.ogrid[:shape[0], :shape[1]]
    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Create smooth gradient
    mask = 1 - np.clip(distance / radius, 0, 1)
    mask = np.power(mask, 2)  # Smooth falloff
    
    return mask

def enhance_wedding_ring_image(image_input: str, mask_type: str = "none") -> Tuple[str, str]:
    """Enhance wedding ring image with adjusted background preservation"""
    try:
        # Decode base64 image
        if image_input.startswith('data:'):
            image_input = image_input.split(',')[1]
        
        image_data = base64.b64decode(image_input)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Create numpy array for processing
        img_array = np.array(image)
        original_array = img_array.copy()
        
        # Detect ring region (center area)
        height, width = img_array.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Convert to LAB for better color manipulation
        lab_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        
        # Create mask for ring area (more focused)
        ring_mask = create_gradient_mask((height, width), (center_x, center_y), min(width, height) // 3)
        
        # Separate enhancement for ring and background
        # Ring area - stronger enhancement
        ring_l = l_channel.copy()
        ring_enhancement = 1.15  # Reduced from 1.25
        ring_l = cv2.multiply(ring_l, ring_enhancement)
        ring_l = np.clip(ring_l, 0, 255)
        
        # Background area - minimal enhancement
        bg_enhancement = 1.05  # Reduced from 1.15
        bg_l = cv2.multiply(l_channel, bg_enhancement)
        bg_l = np.clip(bg_l, 0, 255)
        
        # Blend ring and background
        enhanced_l = ring_l * ring_mask + bg_l * (1 - ring_mask)
        enhanced_l = enhanced_l.astype(np.uint8)
        
        # Merge channels
        enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to PIL
        enhanced_image = Image.fromarray(enhanced_rgb)
        
        # Apply subtle adjustments
        # Reduce brightness increase
        enhancer = ImageEnhance.Brightness(enhanced_image)
        enhanced_image = enhancer.enhance(1.03)  # Reduced from 1.08
        
        # Reduce contrast for softer look
        enhancer = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = enhancer.enhance(1.05)  # Reduced from 1.12
        
        # Very subtle color enhancement
        enhancer = ImageEnhance.Color(enhanced_image)
        enhanced_image = enhancer.enhance(1.02)  # Reduced from 1.05
        
        # Apply very light sharpening only to ring area
        # Create sharpened version
        sharpened = enhanced_image.filter(ImageFilter.UnsharpMask(radius=1, percent=30, threshold=3))
        
        # Blend sharpened ring area with original background
        sharpened_array = np.array(sharpened)
        enhanced_array = np.array(enhanced_image)
        
        # Use ring mask for selective sharpening
        final_array = sharpened_array * ring_mask[:, :, np.newaxis] + enhanced_array * (1 - ring_mask[:, :, np.newaxis])
        final_array = final_array.astype(np.uint8)
        
        # Final image
        final_image = Image.fromarray(final_array)
        
        # Save enhanced image
        enhanced_buffer = BytesIO()
        final_image.save(enhanced_buffer, format='PNG', quality=95)
        enhanced_base64 = base64.b64encode(enhanced_buffer.getvalue()).decode('utf-8')
        
        # For thumbnail, use the same image (no separate processing needed)
        thumbnail_base64 = enhanced_base64
        
        logger.info("Image enhancement completed successfully")
        
        return enhanced_base64, thumbnail_base64
        
    except Exception as e:
        logger.error(f"Enhancement error: {str(e)}")
        raise

def remove_background_with_replicate(image_base64: str, mask_type: str) -> Optional[str]:
    """Remove background using Replicate API"""
    try:
        headers = {
            'Authorization': f'Token {REPLICATE_API_TOKEN}',
            'Content-Type': 'application/json',
        }
        
        # Ensure proper base64 format
        if image_base64.startswith('data:'):
            image_base64 = image_base64.split(',')[1]
        
        # Add data URI prefix for Replicate
        image_with_prefix = f"data:image/png;base64,{image_base64}"
        
        # Create prediction
        create_url = "https://api.replicate.com/v1/predictions"
        
        # Set model based on mask type
        if mask_type == "person":
            model_version = "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003"
            model_input = {
                "image": image_with_prefix,
                "model": "u2netp",
                "return_mask": False,
                "alpha_matting": True,
                "alpha_matting_foreground_threshold": 240,
                "alpha_matting_background_threshold": 10,
                "alpha_matting_erode_size": 10
            }
        else:  # general or none
            model_version = "lucataco/remove-bg:95fcc2a26d3899cd6c2691c900465aaeff466285a65c14638cc5f36f34befaf1"
            model_input = {"image": image_with_prefix}
        
        create_data = {
            "version": model_version,
            "input": model_input
        }
        
        logger.info(f"Creating prediction with {mask_type} model...")
        response = requests.post(create_url, json=create_data, headers=headers)
        response.raise_for_status()
        
        prediction_id = response.json()['id']
        get_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
        
        # Poll for result
        max_attempts = 30
        for attempt in range(max_attempts):
            time.sleep(2)
            
            response = requests.get(get_url, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            status = result['status']
            
            logger.info(f"Attempt {attempt + 1}: Status = {status}")
            
            if status == 'succeeded':
                output = result.get('output')
                if output:
                    # Download the result image
                    img_response = requests.get(output)
                    img_response.raise_for_status()
                    
                    # Convert to base64
                    result_base64 = base64.b64encode(img_response.content).decode('utf-8')
                    logger.info("Background removal successful")
                    return result_base64
                else:
                    logger.error("No output in successful prediction")
                    return None
                    
            elif status == 'failed':
                error = result.get('error', 'Unknown error')
                logger.error(f"Prediction failed: {error}")
                return None
        
        logger.error("Timeout waiting for prediction")
        return None
        
    except Exception as e:
        logger.error(f"Replicate API error: {str(e)}")
        return None

def detect_wedding_rings(image: Image.Image) -> bool:
    """Simple detection to check if image likely contains wedding rings"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles (rings)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )
        
        # If circles detected, likely wedding rings
        return circles is not None
        
    except Exception as e:
        logger.error(f"Ring detection error: {str(e)}")
        return True  # Default to true to process image

def handler(job):
    """Main handler function"""
    start_time = time.time()
    
    try:
        # Extract input from nested structure
        job_input = job.get('input', {})
        input_data = find_input_data(job_input)
        
        logger.info(f"Processing with input keys: {list(input_data.keys())}")
        
        # Get parameters
        image_input = input_data.get('image')
        mask_type = input_data.get('mask_type', 'none').lower()
        
        # Validate inputs
        if not image_input:
            raise ValueError("No image provided")
        
        # Log processing start
        logger.info(f"Starting enhancement with mask_type: {mask_type}")
        
        # Enhance image
        enhanced_base64, thumbnail_base64 = enhance_wedding_ring_image(image_input, mask_type)
        
        # Handle background removal if needed
        masked_base64 = None
        if mask_type != "none":
            logger.info(f"Removing background with mask_type: {mask_type}")
            masked_base64 = remove_background_with_replicate(enhanced_base64, mask_type)
            
            if not masked_base64:
                logger.warning("Background removal failed, using enhanced image")
                masked_base64 = enhanced_base64
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response - MUST return {"output": {...}}
        result = {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "mask_type": mask_type,
                "processing_time": f"{processing_time:.2f}s",
                "has_mask": mask_type != "none",
                "success": True
            }
        }
        
        # Add masked image if background was removed
        if masked_base64 and mask_type != "none":
            result["output"]["masked_image"] = masked_base64
        
        logger.info(f"Successfully processed in {processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        error_response = {
            "output": {
                "error": str(e),
                "success": False,
                "processing_time": f"{time.time() - start_time:.2f}s"
            }
        }
        return error_response

# RunPod handler
runpod.serverless.start({"handler": handler})
