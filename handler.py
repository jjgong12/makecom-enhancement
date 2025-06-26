import runpod
import os
import json
import time
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "V78-FixedDictError"

def find_input_data(data):
    """Find actual image data - returns the base64 string, not dict"""
    
    # Log structure for debugging
    logger.info(f"Input structure: {json.dumps(data, indent=2)[:500]}...")
    
    # Direct access attempts
    if isinstance(data, dict):
        # Check for direct image data keys
        image_keys = ['enhanced_image', 'image', 'image_data', 'base64_image', 
                     'imageBase64', 'image_base64', 'base64']
        
        for key in image_keys:
            if key in data and data[key]:
                # Return the actual string, not the dict
                return data[key]
        
        # Check for 'input' key
        if 'input' in data:
            return find_input_data(data['input'])
        
        # Common RunPod paths
        common_paths = [
            ['job', 'input'],
            ['data', 'input'],
            ['payload', 'input'],
            ['body', 'input'],
            ['request', 'input']
        ]
        
        for path in common_paths:
            current = data
            for key in path:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    break
            else:
                # If we got through the whole path, check if it's a string or dict
                if isinstance(current, str):
                    return current
                elif isinstance(current, dict):
                    # Extract image data from the dict
                    for img_key in image_keys:
                        if img_key in current and current[img_key]:
                            return current[img_key]
    
    # If data is already a string, return it
    elif isinstance(data, str):
        return data
    
    # Recursive search with proper extraction
    def recursive_search(obj):
        if isinstance(obj, dict):
            # Check image keys first
            image_keys = ['enhanced_image', 'image', 'image_data', 'base64_image', 
                         'imageBase64', 'image_base64', 'base64']
            
            for key in image_keys:
                if key in obj and obj[key] and isinstance(obj[key], str):
                    return obj[key]
            
            # Check numeric keys (Make.com)
            for key in obj:
                if key.isdigit():
                    result = recursive_search(obj[key])
                    if result:
                        return result
            
            # Check other common keys
            for key in ['input', 'data', 'output', 'payload']:
                if key in obj:
                    result = recursive_search(obj[key])
                    if result:
                        return result
        
        return None
    
    result = recursive_search(data)
    logger.info(f"Found data: {str(result)[:100]}..." if result else "No data found")
    return result

def decode_base64_safe(base64_str: str) -> bytes:
    """Safely decode base64 with automatic padding correction"""
    if not isinstance(base64_str, str):
        raise ValueError(f"Expected string, got {type(base64_str)}")
    
    if base64_str.startswith('data:'):
        base64_str = base64_str.split(',')[1]
    
    base64_str = base64_str.strip()
    padding = 4 - len(base64_str) % 4
    if padding != 4:
        base64_str += '=' * padding
    
    return base64.b64decode(base64_str)

def detect_ring_color(image: Image.Image) -> str:
    """Ultra-conservative yellow gold detection - only pure gold colors"""
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Focus on center 50%
    center_y, center_x = height // 2, width // 2
    crop_size = min(height, width) // 2
    
    y1 = max(0, center_y - crop_size // 2)
    y2 = min(height, center_y + crop_size // 2)
    x1 = max(0, center_x - crop_size // 2)
    x2 = min(width, center_x + crop_size // 2)
    
    center_region = img_array[y1:y2, x1:x2]
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
    
    # Calculate average values
    avg_hue = np.mean(hsv[:, :, 0])
    avg_saturation = np.mean(hsv[:, :, 1])
    avg_value = np.mean(hsv[:, :, 2])
    
    # RGB analysis
    r_mean = np.mean(center_region[:, :, 0])
    g_mean = np.mean(center_region[:, :, 1])
    b_mean = np.mean(center_region[:, :, 2])
    
    # Normalize RGB values
    max_rgb = max(r_mean, g_mean, b_mean)
    if max_rgb > 0:
        r_norm = r_mean / max_rgb
        g_norm = g_mean / max_rgb
        b_norm = b_mean / max_rgb
    else:
        r_norm = g_norm = b_norm = 1.0
    
    # Calculate color ratios
    rg_ratio = r_mean / (g_mean + 1)  # Red to Green ratio
    rb_ratio = r_mean / (b_mean + 1)  # Red to Blue ratio
    gb_ratio = g_mean / (b_mean + 1)  # Green to Blue ratio
    
    # ULTRA-CONSERVATIVE YELLOW GOLD - Only pure gold colors
    # Must have ALL conditions met for yellow gold
    is_pure_gold = (
        avg_hue >= 25 and avg_hue <= 32 and  # Very narrow hue range for pure gold
        avg_saturation > 80 and  # Very high saturation required
        avg_value > 120 and avg_value < 200 and  # Not too bright, not too dark
        gb_ratio > 1.4 and  # Strong green/blue ratio
        r_mean > 180 and g_mean > 140 and  # High red and green values
        b_mean < 100  # Low blue for pure gold
    )
    
    if is_pure_gold:
        return "옐로우골드"
    
    # Rose gold detection - clear pink/red tones
    elif rg_ratio > 1.2 and rb_ratio > 1.3 and avg_hue < 15:
        return "로즈골드"
    
    # White gold - cool metallic
    elif avg_saturation < 50 and avg_value > 180 and b_norm > r_norm:
        return "화이트골드"
    
    # DEFAULT: 무도금화이트 for everything else
    else:
        return "무도금화이트"

def apply_color_enhancement_simple(image: Image.Image, detected_color: str) -> Image.Image:
    """Simple color-specific enhancement - no blue boost"""
    
    if detected_color == "무도금화이트":
        # Pure white enhancement - PIL only
        # Step 1: Strong brightness increase
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.2)
        
        # Step 2: Moderate saturation reduction (not too much)
        color = ImageEnhance.Color(image)
        image = color.enhance(0.5)  # Changed from 0.2 to 0.5 to maintain some color
        
        # Step 3: Slight contrast adjustment
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.05)
        
        # NO BLUE BOOST - no LAB conversion
        
    elif detected_color == "옐로우골드":
        # Pure gold enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(1.1)
        
    elif detected_color == "로즈골드":
        # Rose gold enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.05)
        
        # Slight warm tone
        img_array = np.array(image)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.03, 0, 255)
        image = Image.fromarray(img_array)
        
    elif detected_color == "화이트골드":
        # White gold enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.1)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.9)
    
    return image

def process_enhancement(job):
    """Enhancement processing - stable version"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    
    try:
        # Find image data - now properly extracts string from dict
        image_data = find_input_data(job)
        
        if not image_data:
            logger.error("No image data found in input")
            return {
                "output": {
                    "error": "No image data found",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Ensure we have a string
        if not isinstance(image_data, str):
            logger.error(f"Image data is not a string: {type(image_data)}")
            return {
                "output": {
                    "error": f"Image data must be a string, got {type(image_data)}",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Decode image
        image_bytes = decode_base64_safe(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            else:
                image = image.convert('RGB')
        
        logger.info(f"Image loaded: {image.size}")
        
        # Detect color
        detected_color = detect_ring_color(image)
        logger.info(f"Detected color: {detected_color}")
        
        # Basic enhancement - V56 style
        # 1. Brightness
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.12)
        
        # 2. Contrast
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.08)
        
        # 3. Color
        color = ImageEnhance.Color(image)
        image = color.enhance(1.05)
        
        # 4. Apply color-specific enhancement
        image = apply_color_enhancement_simple(image, detected_color)
        
        # 5. Light sharpening
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.3)
        
        # Save to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=True, quality=95)
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        enhanced_base64_no_padding = enhanced_base64.rstrip('=')
        
        logger.info("Enhancement completed successfully")
        
        return {
            "output": {
                "enhanced_image": f"data:image/png;base64,{enhanced_base64_no_padding}",
                "detected_color": detected_color,
                "original_size": list(image.size),
                "version": VERSION,
                "status": "success"
            }
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        error_trace = traceback.format_exc()
        logger.error(error_trace)
        
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": VERSION,
                "traceback": error_trace
            }
        }

# RunPod handler
runpod.serverless.start({"handler": process_enhancement})
