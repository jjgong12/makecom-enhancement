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

VERSION = "V82-GoogleFix"

# Global cache to prevent duplicate processing
PROCESSED_IMAGES = {}

def find_input_data(data):
    """Find input data - optimized for speed"""
    
    # Fast return if already a string
    if isinstance(data, str):
        return data
    
    # Direct key access without logging
    if isinstance(data, dict):
        # Priority 1: Direct image keys
        image_keys = ['enhanced_image', 'image', 'image_data', 'base64_image', 
                     'imageBase64', 'image_base64', 'base64']
        
        for key in image_keys:
            if key in data and isinstance(data[key], str) and len(data[key]) > 100:
                return data[key]
        
        # Priority 2: Check 'input' key
        if 'input' in data:
            if isinstance(data['input'], str):
                return data['input']
            elif isinstance(data['input'], dict):
                # Check image keys in input
                for key in image_keys:
                    if key in data['input'] and isinstance(data['input'][key], str):
                        return data['input'][key]
        
        # Priority 3: Check numeric keys (Make.com) - limit to single digits
        for i in range(10):  # Only check 0-9
            key = str(i)
            if key in data:
                result = find_input_data(data[key])
                if result:
                    return result
        
        # Priority 4: Specific paths only
        # Direct path checking without loop
        if 'job' in data and isinstance(data['job'], dict) and 'input' in data['job']:
            result = find_input_data(data['job']['input'])
            if result:
                return result
        
        # Make.com specific path
        if '4' in data and isinstance(data['4'], dict):
            if 'data' in data['4'] and isinstance(data['4']['data'], dict):
                if 'output' in data['4']['data'] and isinstance(data['4']['data']['output'], dict):
                    if 'output' in data['4']['data']['output'] and isinstance(data['4']['data']['output']['output'], dict):
                        if 'enhanced_image' in data['4']['data']['output']['output']:
                            return data['4']['data']['output']['output']['enhanced_image']
    
    # Limited recursive search
    def quick_search(obj, depth=0):
        if depth > 3:  # Limit depth to 3
            return None
            
        if isinstance(obj, str) and len(obj) > 100:
            return obj
            
        if isinstance(obj, dict):
            # Only check specific keys
            for key in ['enhanced_image', 'image', 'image_data', 'input', 'data', 'output']:
                if key in obj:
                    if isinstance(obj[key], str) and len(obj[key]) > 100:
                        return obj[key]
                    else:
                        result = quick_search(obj[key], depth + 1)
                        if result:
                            return result
        
        return None
    
    result = quick_search(data)
    
    if not result:
        logger.error("No image data found!")
        
    return result

def decode_base64_safe(base64_str: str) -> bytes:
    """Safely decode base64 with automatic padding correction"""
    if not isinstance(base64_str, str):
        raise ValueError(f"Expected string, got {type(base64_str)}")
    
    # Remove data URL prefix if present
    if base64_str.startswith('data:'):
        base64_str = base64_str.split(',')[1]
    
    # Clean and add padding
    base64_str = base64_str.strip()
    padding = 4 - len(base64_str) % 4
    if padding != 4:
        base64_str += '=' * padding
    
    return base64.b64decode(base64_str)

def detect_ring_color(image: Image.Image) -> str:
    """Improved color detection - better white gold vs unplated white distinction"""
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
    
    # Convert to HSV
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
    rg_ratio = r_mean / (g_mean + 1)
    rb_ratio = r_mean / (b_mean + 1)
    gb_ratio = g_mean / (b_mean + 1)
    
    # Color variance (how different are RGB channels)
    rgb_variance = np.var([r_mean, g_mean, b_mean])
    
    # ULTRA-CONSERVATIVE YELLOW GOLD
    if (avg_hue >= 25 and avg_hue <= 32 and
        avg_saturation > 80 and
        avg_value > 120 and avg_value < 200 and
        gb_ratio > 1.4 and
        r_mean > 180 and g_mean > 140 and
        b_mean < 100):
        return "옐로우골드"
    
    # ROSE GOLD
    elif rg_ratio > 1.2 and rb_ratio > 1.3 and avg_hue < 15:
        return "로즈골드"
    
    # WHITE GOLD vs UNPLATED WHITE distinction
    # White gold: has some warmth/color, metallic sheen
    # Unplated white: very low saturation, high brightness, almost pure white
    elif avg_saturation < 15 and avg_value > 200 and rgb_variance < 50:
        # Very low saturation + very high brightness + low variance = unplated white
        return "무도금화이트"
    
    else:
        # Everything else is white gold (has some color/warmth)
        return "화이트골드"

def apply_color_enhancement_simple(image: Image.Image, detected_color: str) -> Image.Image:
    """Simple color-specific enhancement - ULTRA WHITE for unplated"""
    
    if detected_color == "무도금화이트":
        # ULTRA WHITE - maximum whiteness
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.35)  # Even brighter
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.1)  # Almost no color (10% only)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(0.9)  # Softer contrast
        
        # Heavy whitening
        img_array = np.array(image)
        # Mix 50% with pure white
        img_array = img_array * 0.5 + 255 * 0.5
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Additional brightness boost
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.1)
        
    elif detected_color == "옐로우골드":
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(1.1)
        
    elif detected_color == "로즈골드":
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.05)
        
        img_array = np.array(image)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.03, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
    elif detected_color == "화이트골드":
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.1)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.9)
    
    return image

def calculate_image_hash(image: Image.Image) -> str:
    """Calculate a simple hash to detect duplicate images"""
    # Resize to small size for fast comparison
    small = image.resize((8, 8), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())
    avg = sum(sum(pixel) for pixel in pixels) / len(pixels) / 3
    
    # Create binary hash
    hash_str = ""
    for pixel in pixels:
        if sum(pixel) / 3 > avg:
            hash_str += "1"
        else:
            hash_str += "0"
    
    return hash_str

def process_enhancement(job):
    """Enhancement processing - with Google Script compatibility"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    
    try:
        # Find image data - fast version
        image_data = find_input_data(job)
        
        if not image_data:
            return {
                "output": {
                    "error": "No image data found",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Ensure we have a string
        if not isinstance(image_data, str):
            return {
                "output": {
                    "error": f"Image data must be a string, got {type(image_data)}",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Check for duplicate processing
        image_preview = image_data[:100]  # First 100 chars as quick check
        current_time = time.time()
        
        # Clean old entries (older than 60 seconds)
        global PROCESSED_IMAGES
        PROCESSED_IMAGES = {k: v for k, v in PROCESSED_IMAGES.items() 
                          if current_time - v < 60}
        
        # Check if recently processed
        if image_preview in PROCESSED_IMAGES:
            logger.warning("Duplicate image detected, skipping processing")
            return {
                "output": {
                    "error": "Duplicate processing detected",
                    "status": "duplicate",
                    "version": VERSION
                }
            }
        
        # Mark as processed
        PROCESSED_IMAGES[image_preview] = current_time
        
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
        
        # Detect color with improved logic
        detected_color = detect_ring_color(image)
        logger.info(f"Detected color: {detected_color}")
        
        # Basic enhancement
        # 1. Brightness
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.12)
        
        # 2. Contrast
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.08)
        
        # 3. Color
        color = ImageEnhance.Color(image)
        image = color.enhance(1.05)
        
        # 4. Apply color-specific enhancement (ULTRA WHITE for unplated)
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
        
        # IMPORTANT: Return ONLY base64 without data URL prefix for Google Script
        return {
            "output": {
                "enhanced_image": enhanced_base64_no_padding,  # NO prefix!
                "enhanced_image_with_prefix": f"data:image/png;base64,{enhanced_base64_no_padding}",  # For other uses
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
