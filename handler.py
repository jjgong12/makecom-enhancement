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

VERSION = "V95-1.8PercentWhiteOverlay"

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

def find_filename(data, depth=0):
    """Extract filename from input data - IMPROVED for Make.com"""
    if depth > 5:  # Prevent infinite recursion
        return None
        
    if isinstance(data, dict):
        # Check common filename keys - EXPANDED LIST
        filename_keys = ['filename', 'file_name', 'name', 'fileName', 'file', 
                        'originalName', 'original_name', 'image_name', 'imageName']
        
        # Log the keys at current level for debugging
        if depth == 0:
            logger.info(f"Top level keys: {list(data.keys())[:20]}")  # First 20 keys
        
        for key in filename_keys:
            if key in data and isinstance(data[key], str):
                logger.info(f"Found filename at key '{key}': {data[key]}")
                return data[key]
        
        # Deep recursive search through ALL keys
        for key, value in data.items():
            if isinstance(value, dict):
                result = find_filename(value, depth + 1)
                if result:
                    return result
            elif isinstance(value, list) and len(value) > 0:
                for item in value:
                    if isinstance(item, dict):
                        result = find_filename(item, depth + 1)
                        if result:
                            return result
    
    return None

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

def detect_if_unplated_white(filename: str) -> bool:
    """Check if filename indicates unplated white (contains 'c')"""
    if not filename:
        logger.warning("No filename found, defaulting to standard enhancement")
        return False
    
    # Convert to lowercase for case-insensitive check
    filename_lower = filename.lower()
    logger.info(f"Checking filename pattern: {filename_lower}")
    
    # Check for ac_, bc_, dc_, etc. patterns
    import re
    # Pattern: any letter followed by c_ or just c_
    pattern1 = re.search(r'[a-z]?c_', filename_lower)
    # Pattern: any letter followed by c. or just c.
    pattern2 = re.search(r'[a-z]?c\.', filename_lower)
    # Pattern: c followed by number
    pattern3 = re.match(r'^c\d', filename_lower)
    
    is_unplated = bool(pattern1 or pattern2 or pattern3)
    
    logger.info(f"Pattern check - ac_/bc_/c_: {bool(pattern1)}, c.: {bool(pattern2)}, c+digit: {bool(pattern3)}")
    logger.info(f"Is unplated white: {is_unplated}")
    
    return is_unplated

def apply_color_enhancement_simple(image: Image.Image, is_unplated_white: bool, filename: str) -> Image.Image:
    """Simple enhancement - 1.8% WHITE OVERLAY ONLY FOR UNPLATED WHITE (filename with 'c')"""
    
    logger.info(f"Applying enhancement - Filename: {filename}, Is unplated white: {is_unplated_white}")
    
    if is_unplated_white:
        # UPDATED TO 1.8% WHITE EFFECT FOR V95!
        logger.info("Applying unplated white enhancement (1.8% white overlay)")
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.5)  # Keep more color
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.0)  # No contrast change
        
        # UPDATED: 1.8% white mixing (was 1.5%)
        img_array = np.array(image)
        img_array = img_array * 0.982 + 255 * 0.018  # 1.8% white overlay
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Very tiny additional boost
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.01)  # Minimal boost
        
    else:
        # For all other colors - NO white overlay, just slight enhancement
        logger.info("Standard enhancement (no white overlay)")
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.06)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(1.05)
    
    return image

def apply_center_focus(image: Image.Image) -> Image.Image:
    """Apply subtle center brightening to focus on ring"""
    width, height = image.size
    
    # Create radial gradient
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Distance from center
    distance = np.sqrt(X**2 + Y**2)
    
    # Create center focus mask (brighter in center, normal at edges)
    focus_mask = 1 + 0.04 * np.exp(-distance**2 * 0.8)
    focus_mask = np.clip(focus_mask, 1.0, 1.04)
    
    # Apply focus
    img_array = np.array(image)
    for i in range(3):
        img_array[:, :, i] = np.clip(img_array[:, :, i] * focus_mask, 0, 255)
    
    return Image.fromarray(img_array.astype(np.uint8))

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
    """Enhancement processing - with improved filename detection"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    logger.info(f"Input data type: {type(job)}")
    
    try:
        # Find filename FIRST - IMPROVED
        filename = find_filename(job)
        if filename:
            logger.info(f"Successfully extracted filename: {filename}")
        else:
            logger.warning("Could not extract filename from input")
            # Log the structure for debugging
            if isinstance(job, dict):
                logger.info(f"Job structure keys: {list(job.keys())[:10]}")
        
        # Find image data
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
        image_preview = image_data[:100]
        current_time = time.time()
        
        # Clean old entries
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
        
        # Check if unplated white based on filename
        is_unplated_white = detect_if_unplated_white(filename)
        detected_type = "무도금화이트" if is_unplated_white else "기타색상"
        logger.info(f"Final detection - Type: {detected_type}, Filename: {filename}")
        
        # Basic enhancement
        # 1. Brightness
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)
        
        # 2. Contrast
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.05)
        
        # 3. Color
        color = ImageEnhance.Color(image)
        image = color.enhance(1.03)
        
        # 4. Apply color-specific enhancement (1.8% white overlay only for 'c' filenames)
        image = apply_color_enhancement_simple(image, is_unplated_white, filename)
        
        # 5. Apply center focus
        image = apply_center_focus(image)
        
        # 6. Light sharpening
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.2)
        
        # Save to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=True, quality=95)
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        enhanced_base64_no_padding = enhanced_base64.rstrip('=')
        
        logger.info("Enhancement completed successfully")
        
        return {
            "output": {
                "enhanced_image": enhanced_base64_no_padding,
                "enhanced_image_with_prefix": f"data:image/png;base64,{enhanced_base64_no_padding}",
                "detected_type": detected_type,
                "filename": filename,
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

def handler(event):
    """RunPod handler function"""
    return process_enhancement(event)

# RunPod handler
runpod.serverless.start({"handler": handler})
