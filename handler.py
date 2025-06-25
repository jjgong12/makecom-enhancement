import runpod
import base64
import requests
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "Enhancement_V68_SIMPLIFIED"
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN', 'r8_6cksfxEmLxWlYxjW4K1FEbnZMEEmlQw2UeNNY')

def decode_base64_image(base64_str):
    """Decode base64 image with padding fix"""
    try:
        # Remove data URI prefix if present
        if 'data:' in base64_str and 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]
        
        # Fix padding
        base64_str = base64_str.strip()
        padding = 4 - len(base64_str) % 4
        if padding != 4:
            base64_str += '=' * padding
            
        return base64.b64decode(base64_str)
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise

def enhance_image(image_data):
    """Simple enhancement for wedding ring images"""
    try:
        # Open image
        image = Image.open(BytesIO(image_data)).convert('RGB')
        logger.info(f"Image size: {image.size}")
        
        # Convert to numpy for LAB processing
        img_array = np.array(image)
        
        # LAB color space for better brightness control
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance L channel (brightness)
        l = cv2.multiply(l, 1.1)
        l = np.clip(l, 0, 255).astype(np.uint8)
        
        # Merge and convert back
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        enhanced_image = Image.fromarray(enhanced_rgb)
        
        # PIL adjustments
        enhancer = ImageEnhance.Brightness(enhanced_image)
        enhanced_image = enhancer.enhance(1.03)
        
        enhancer = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = enhancer.enhance(1.05)
        
        enhancer = ImageEnhance.Color(enhanced_image)
        enhanced_image = enhancer.enhance(1.02)
        
        # Light sharpening
        enhanced_image = enhanced_image.filter(ImageFilter.UnsharpMask(radius=1, percent=30, threshold=3))
        
        # Save to base64
        output_buffer = BytesIO()
        enhanced_image.save(output_buffer, format='PNG', quality=95)
        enhanced_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        return enhanced_base64.rstrip('=')
        
    except Exception as e:
        logger.error(f"Enhancement error: {str(e)}")
        raise

def remove_background_replicate(image_base64, mask_type):
    """Remove background using Replicate API"""
    try:
        if mask_type == "none":
            return None
            
        headers = {
            'Authorization': f'Token {REPLICATE_API_TOKEN}',
            'Content-Type': 'application/json',
        }
        
        # Prepare image with data URI
        image_with_prefix = f"data:image/png;base64,{image_base64}"
        
        # Select model
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
        else:
            model_version = "lucataco/remove-bg:95fcc2a26d3899cd6c2691c900465aaeff466285a65c14638cc5f36f34befaf1"
            model_input = {"image": image_with_prefix}
        
        # Create prediction
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            json={"version": model_version, "input": model_input},
            headers=headers
        )
        response.raise_for_status()
        
        prediction_id = response.json()['id']
        
        # Poll for result
        for attempt in range(30):
            time.sleep(2)
            
            result = requests.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers=headers
            ).json()
            
            if result['status'] == 'succeeded':
                output_url = result.get('output')
                if output_url:
                    img_data = requests.get(output_url).content
                    return base64.b64encode(img_data).decode('utf-8').rstrip('=')
                return None
                
            elif result['status'] == 'failed':
                logger.error(f"Replicate failed: {result.get('error')}")
                return None
        
        return None
        
    except Exception as e:
        logger.error(f"Replicate error: {str(e)}")
        return None

def handler(job):
    """Main handler function"""
    start_time = time.time()
    
    try:
        # Extract input - simplified search
        job_input = job.get('input', {})
        
        # Find image in common locations
        image_input = None
        mask_type = 'none'
        
        # Direct check
        if isinstance(job_input, dict):
            image_input = job_input.get('image') or job_input.get('image_base64') or job_input.get('base64_image')
            mask_type = job_input.get('mask_type', 'none').lower()
        elif isinstance(job_input, str):
            image_input = job_input
        
        # Check nested input
        if not image_input and isinstance(job_input, dict) and 'input' in job_input:
            nested = job_input['input']
            if isinstance(nested, dict):
                image_input = nested.get('image') or nested.get('image_base64') or nested.get('base64_image')
                mask_type = nested.get('mask_type', mask_type).lower()
        
        if not image_input:
            raise ValueError("No image provided")
        
        logger.info(f"Processing with mask_type: {mask_type}")
        
        # Decode and enhance image
        image_data = decode_base64_image(image_input)
        enhanced_base64 = enhance_image(image_data)
        
        # Optional background removal
        masked_base64 = None
        if mask_type != "none":
            masked_base64 = remove_background_replicate(enhanced_base64, mask_type)
            if not masked_base64:
                masked_base64 = enhanced_base64
        
        # Prepare response
        result = {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": enhanced_base64,  # Same as enhanced for simplicity
                "mask_type": mask_type,
                "processing_time": f"{time.time() - start_time:.2f}s",
                "has_mask": mask_type != "none",
                "success": True,
                "version": VERSION
            }
        }
        
        if masked_base64 and mask_type != "none":
            result["output"]["masked_image"] = masked_base64
        
        logger.info(f"Completed in {time.time() - start_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "success": False,
                "processing_time": f"{time.time() - start_time:.2f}s",
                "version": VERSION
            }
        }

# RunPod handler
runpod.serverless.start({"handler": handler})
