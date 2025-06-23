import os
import sys
import base64
import io
import time
import re
import traceback
import json
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "v8"

# Import availability checks
NUMPY_AVAILABLE = False
PIL_AVAILABLE = False
CV2_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] NumPy not available")

try:
    from PIL import Image, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] PIL not available")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] OpenCV not available")

try:
    import runpod
    RUNPOD_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] RunPod not available")
    RUNPOD_AVAILABLE = False

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif NUMPY_AVAILABLE and isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif NUMPY_AVAILABLE and isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif NUMPY_AVAILABLE and isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif NUMPY_AVAILABLE and isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def find_image_data(data, depth=0, max_depth=3):
    """
    Find image data in nested input structure
    """
    if depth > max_depth:
        return None
    
    logger.info(f"Searching for image at depth {depth}, type: {type(data)}")
    
    # Direct string check
    if isinstance(data, str) and len(data) > 100:
        # Check if it looks like base64
        base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
        if base64_pattern.match(data.replace('\n', '').replace('\r', '')):
            logger.info(f"Found base64-like string at depth {depth}")
            return data
    
    # Dictionary search
    if isinstance(data, dict):
        # Primary image keys
        image_keys = [
            'image', 'image_base64', 'base64', 'img', 'data', 'imageData', 
            'image_data', 'input_image', 'enhanced_image', 'file_content'
        ]
        
        for key in image_keys:
            if key in data and data[key]:
                logger.info(f"Found image in key: {key}")
                return data[key]
        
        # Search nested structures
        for key, value in data.items():
            result = find_image_data(value, depth + 1, max_depth)
            if result:
                return result
    
    # List search
    elif isinstance(data, list):
        for item in data:
            result = find_image_data(item, depth + 1, max_depth)
            if result:
                return result
    
    return None

def decode_base64_image(base64_str):
    """
    Decode base64 image with multiple fallback methods
    """
    if not base64_str:
        raise ValueError("Empty base64 string")
    
    # Clean the string
    base64_str = base64_str.strip()
    
    # Remove data URL prefix if present
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    # Remove whitespace
    base64_str = base64_str.replace('\n', '').replace('\r', '').replace(' ', '')
    
    # Try multiple decoding methods
    methods = [
        lambda x: base64.b64decode(x, validate=True),
        lambda x: base64.b64decode(x + '=='),
        lambda x: base64.b64decode(x + '='),
        lambda x: base64.urlsafe_b64decode(x + '==')
    ]
    
    for i, method in enumerate(methods):
        try:
            image_data = method(base64_str)
            img = Image.open(io.BytesIO(image_data))
            logger.info(f"Base64 decode successful with method {i+1}")
            return img
        except Exception as e:
            logger.warning(f"Decode method {i+1} failed: {str(e)}")
            continue
    
    raise ValueError("All base64 decode methods failed")

def detect_metal_type(image):
    """
    Detect wedding ring metal type from 4 categories
    Based on 38 training data pairs (28 + 10)
    """
    try:
        if not NUMPY_AVAILABLE:
            return "white_gold"
            
        img_array = np.array(image)
        
        # Convert to HSV for better color analysis
        if CV2_AVAILABLE:
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            h_channel = hsv[:, :, 0]
            s_channel = hsv[:, :, 1]
            v_channel = hsv[:, :, 2]
            
            # Calculate color characteristics
            avg_hue = np.mean(h_channel)
            avg_saturation = np.mean(s_channel)
            avg_value = np.mean(v_channel)
            
            # Metal type classification based on training data
            if avg_saturation < 50 and avg_value > 180:
                metal_type = "plain_white"  # 무도금화이트
            elif avg_hue < 30 and avg_saturation > 80:
                metal_type = "yellow_gold"  # 옐로우골드
            elif avg_hue < 20 and avg_saturation > 50:
                metal_type = "rose_gold"    # 로즈골드
            else:
                metal_type = "white_gold"   # 화이트골드
        else:
            # Simple RGB analysis fallback
            avg_color = np.mean(img_array.reshape(-1, 3), axis=0)
            r, g, b = avg_color
            
            if r > 200 and g > 200 and b > 200:
                metal_type = "white_gold"  # 화이트골드
            elif r > g + 20 and r > b + 20:
                if g > b:
                    metal_type = "yellow_gold"  # 옐로우골드
                else:
                    metal_type = "rose_gold"   # 로즈골드
            else:
                metal_type = "plain_white"     # 무도금화이트
        
        logger.info(f"Metal type detected: {metal_type}")
        return metal_type
        
    except Exception as e:
        logger.error(f"Metal detection error: {str(e)}")
        return "white_gold"  # Default fallback

def apply_metal_specific_enhancement(img, metal_type):
    """
    Apply metal-specific color enhancement
    Based on 38 training data pairs (28 + 10)
    """
    try:
        logger.info(f"Applying enhancement for {metal_type}")
        
        # Base enhancement - Image 3 → Image 5 style
        # 1. Slight brightness increase (5%)
        brightness_enhancer = ImageEnhance.Brightness(img)
        img = brightness_enhancer.enhance(1.05)
        
        # 2. Slight saturation decrease for cleaner look (5%)
        color_enhancer = ImageEnhance.Color(img)
        img = color_enhancer.enhance(0.95)
        
        # Metal-specific enhancements
        if metal_type == "yellow_gold":
            # Enhance warm golden tones
            if NUMPY_AVAILABLE:
                img_array = np.array(img)
                img_array[:, :, 0] = np.minimum(255, img_array[:, :, 0] * 1.08).astype(np.uint8)  # More red
                img_array[:, :, 1] = np.minimum(255, img_array[:, :, 1] * 1.05).astype(np.uint8)  # Slight green
                img = Image.fromarray(img_array)
            
            # Additional warmth
            color_enhancer = ImageEnhance.Color(img)
            img = color_enhancer.enhance(1.02)  # Slight saturation increase for gold
            
        elif metal_type == "rose_gold":
            # Enhance pink/warm tones
            if NUMPY_AVAILABLE:
                img_array = np.array(img)
                img_array[:, :, 0] = np.minimum(255, img_array[:, :, 0] * 1.10).astype(np.uint8)  # More red
                img_array[:, :, 2] = np.minimum(255, img_array[:, :, 2] * 1.03).astype(np.uint8)  # Slight blue
                img = Image.fromarray(img_array)
            
            # Enhance contrast for rose gold details
            contrast_enhancer = ImageEnhance.Contrast(img)
            img = contrast_enhancer.enhance(1.05)
            
        elif metal_type == "white_gold":
            # Enhance cool tones and clarity
            contrast_enhancer = ImageEnhance.Contrast(img)
            img = contrast_enhancer.enhance(1.08)
            
            # Enhance sharpness for white gold details
            sharpness_enhancer = ImageEnhance.Sharpness(img)
            img = sharpness_enhancer.enhance(1.12)
            
        elif metal_type == "plain_white":
            # Enhance brightness and clean look
            brightness_enhancer = ImageEnhance.Brightness(img)
            img = brightness_enhancer.enhance(1.08)
            
            # Reduce saturation for cleaner white look
            color_enhancer = ImageEnhance.Color(img)
            img = color_enhancer.enhance(0.90)
        
        # Selective background brightening (for all types)
        if NUMPY_AVAILABLE:
            img_array = np.array(img)
            
            # Create mask for bright background areas (likely background)
            mask = np.all(img_array > 200, axis=-1)
            
            # Apply additional brightening to background areas
            if mask.any():
                for c in range(3):
                    img_array[mask, c] = np.minimum(255, img_array[mask, c] * 1.05).astype(np.uint8)
                
                img = Image.fromarray(img_array)
                logger.info("Background areas brightened")
        
        # Final detail enhancement
        sharpness_enhancer = ImageEnhance.Sharpness(img)
        img = sharpness_enhancer.enhance(1.08)
        
        logger.info(f"Metal-specific enhancement completed for {metal_type}")
        return img
        
    except Exception as e:
        logger.error(f"Enhancement error: {str(e)}")
        return img

def apply_wedding_ring_details(img):
    """
    Apply wedding ring detail enhancement
    Based on 38 training data pairs (28 + 10)
    """
    try:
        # Enhance fine details and textures
        sharpness_enhancer = ImageEnhance.Sharpness(img)
        img = sharpness_enhancer.enhance(1.15)
        
        # Enhance contrast for better detail visibility
        contrast_enhancer = ImageEnhance.Contrast(img)
        img = contrast_enhancer.enhance(1.06)
        
        # Advanced detail enhancement using unsharp mask effect
        if NUMPY_AVAILABLE and CV2_AVAILABLE:
            img_array = np.array(img)
            
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
            
            # Create unsharp mask
            unsharp_mask = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
            
            # Apply mask to enhance details
            for i in range(3):
                img_array[:, :, i] = cv2.addWeighted(
                    img_array[:, :, i], 0.7, 
                    unsharp_mask, 0.3, 0
                )
            
            img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
            logger.info("Advanced detail enhancement applied")
        
        return img
        
    except Exception as e:
        logger.error(f"Detail enhancement error: {str(e)}")
        return img

def image_to_base64(img):
    """
    Convert PIL Image to base64 - MUST REMOVE PADDING for Make.com
    """
    try:
        buffer = io.BytesIO()
        img.save(buffer, format='PNG', quality=95, optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # CRITICAL: Remove padding for Make.com compatibility
        # Google Apps Script will restore padding when needed
        img_base64 = img_base64.rstrip('=')
        
        logger.info(f"Image converted to base64, length: {len(img_base64)}, padding removed")
        return img_base64
        
    except Exception as e:
        logger.error(f"Base64 conversion error: {str(e)}")
        return ""

def handler(job):
    """
    RunPod handler for image enhancement V8 - Complete full version
    Wedding ring enhancement with metal detection and detail enhancement
    """
    start_time = time.time()
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Enhancement Handler {VERSION} - Complete Full Version")
        logger.info(f"Features: Metal detection, Wedding ring enhancement, Detail enhancement")
        logger.info(f"Training: 38 data pairs (28 + 10), 4 metal types")
        logger.info(f"{'='*60}")
        
        # Get input
        job_input = job.get('input', {})
        logger.info(f"Input keys: {list(job_input.keys())}")
        
        # Debug mode
        if job_input.get('debug_mode', False):
            return {
                "output": {
                    "status": "debug_success",
                    "message": f"{VERSION} handler working - Complete full version",
                    "version": VERSION,
                    "features": [
                        "Metal type detection (4 types)",
                        "Metal-specific enhancement",
                        "Wedding ring detail enhancement", 
                        "38 training data pairs",
                        "Image 3 → Image 5 style enhancement",
                        "JSON serialization safe",
                        "Make.com compatible",
                        "Google Apps Script compatible"
                    ]
                }
            }
        
        # Find image data
        image_data_str = find_image_data(job_input)
        if not image_data_str:
            error_msg = f"No image found. Available keys: {list(job_input.keys())}"
            logger.error(error_msg)
            return {
                "output": {
                    "error": error_msg,
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Decode image
        img = decode_base64_image(image_data_str)
        original_size = img.size
        logger.info(f"Original image size: {original_size}")
        
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        # Detect metal type
        metal_type = detect_metal_type(img)
        
        # Apply metal-specific enhancement
        enhanced_img = apply_metal_specific_enhancement(img, metal_type)
        
        # Apply wedding ring detail enhancement
        enhanced_img = apply_wedding_ring_details(enhanced_img)
        
        # Convert to base64
        enhanced_base64 = image_to_base64(enhanced_img)
        
        # Processing info with type safety
        processing_info = {
            "original_size": list(original_size),
            "final_size": list(enhanced_img.size),
            "metal_type": str(metal_type),
            "enhancement_applied": True,
            "detail_enhancement": True,
            "masking_removed": False,
            "processing_time": round(time.time() - start_time, 2),
            "version": VERSION,
            "training_data": "38 pairs (28 + 10)",
            "metal_types": ["yellow_gold", "rose_gold", "white_gold", "plain_white"]
        }
        
        # Convert all numpy types to native Python types
        processing_info = convert_numpy_types(processing_info)
        
        result = {
            "output": {
                "enhanced_image": enhanced_base64,
                "processing_info": processing_info,
                "status": "success"
            }
        }
        
        # Final numpy type conversion for entire result
        result = convert_numpy_types(result)
        
        logger.info(f"Enhancement completed in {processing_info['processing_time']}s")
        logger.info(f"Metal type: {metal_type}")
        logger.info(f"Output structure: {list(result['output'].keys())}")
        
        return result
        
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # Ensure error response is also JSON-safe
        error_result = {
            "output": {
                "error": error_msg,
                "status": "error",
                "version": VERSION,
                "processing_time": round(time.time() - start_time, 2)
            }
        }
        
        return convert_numpy_types(error_result)

# RunPod serverless start
if __name__ == "__main__":
    if RUNPOD_AVAILABLE:
        runpod.serverless.start({"handler": handler})
    else:
        print(f"[{VERSION}] RunPod not available, running in test mode")
        test_job = {
            "input": {
                "debug_mode": True
            }
        }
        result = handler(test_job)
        print(json.dumps(result, indent=2))
