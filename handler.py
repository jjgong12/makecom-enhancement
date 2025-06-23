import os
import sys
import base64
import io
import time
import re
import traceback
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "v10"

# Safe imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] NumPy not available")
    NUMPY_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] PIL not available")
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] OpenCV not available")
    CV2_AVAILABLE = False

try:
    import runpod
    RUNPOD_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] RunPod not available")
    RUNPOD_AVAILABLE = False

def safe_json_convert(obj):
    """
    Safely convert objects to JSON-serializable types
    COMPLETELY AVOIDS np.bool references
    """
    if obj is None:
        return None
    elif isinstance(obj, bool):
        return bool(obj)  # Native Python bool
    elif isinstance(obj, int):
        return int(obj)
    elif isinstance(obj, float):
        return float(obj)
    elif isinstance(obj, str):
        return str(obj)
    elif isinstance(obj, list):
        return [safe_json_convert(item) for item in obj]
    elif isinstance(obj, tuple):
        return [safe_json_convert(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(key): safe_json_convert(value) for key, value in obj.items()}
    elif NUMPY_AVAILABLE and hasattr(np, 'ndarray') and isinstance(obj, np.ndarray):
        return obj.tolist()
    elif NUMPY_AVAILABLE and hasattr(np, 'integer') and isinstance(obj, np.integer):
        return int(obj)
    elif NUMPY_AVAILABLE and hasattr(np, 'floating') and isinstance(obj, np.floating):
        return float(obj)
    elif NUMPY_AVAILABLE and str(type(obj)).startswith("<class 'numpy.bool"):
        return bool(obj)  # Handle any numpy bool type safely
    else:
        # Fallback: convert to string
        return str(obj)

def find_image_data(data, depth=0, max_depth=3):
    """Find image data in nested input structure"""
    if depth > max_depth:
        return None
    
    logger.info(f"Searching for image at depth {depth}, type: {type(data)}")
    
    if isinstance(data, str) and len(data) > 100:
        base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
        if base64_pattern.match(data.replace('\n', '').replace('\r', '')):
            logger.info(f"Found base64-like string at depth {depth}")
            return data
    
    if isinstance(data, dict):
        image_keys = [
            'image', 'image_base64', 'base64', 'img', 'data', 'imageData', 
            'image_data', 'input_image', 'enhanced_image', 'file_content'
        ]
        
        for key in image_keys:
            if key in data and data[key]:
                logger.info(f"Found image in key: {key}")
                return data[key]
        
        for key, value in data.items():
            result = find_image_data(value, depth + 1, max_depth)
            if result:
                return result
    
    elif isinstance(data, list):
        for item in data:
            result = find_image_data(item, depth + 1, max_depth)
            if result:
                return result
    
    return None

def decode_base64_image(base64_str):
    """Decode base64 image with multiple fallback methods"""
    if not base64_str:
        raise ValueError("Empty base64 string")
    
    base64_str = base64_str.strip()
    
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    base64_str = base64_str.replace('\n', '').replace('\r', '').replace(' ', '')
    
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
    """Detect wedding ring metal type from 4 categories"""
    try:
        if not NUMPY_AVAILABLE:
            return "white_gold"
            
        img_array = np.array(image)
        
        if CV2_AVAILABLE:
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            h_channel = hsv[:, :, 0]
            s_channel = hsv[:, :, 1]
            v_channel = hsv[:, :, 2]
            
            avg_hue = np.mean(h_channel)
            avg_saturation = np.mean(s_channel)
            avg_value = np.mean(v_channel)
            
            if avg_saturation < 50 and avg_value > 180:
                metal_type = "plain_white"
            elif avg_hue < 30 and avg_saturation > 80:
                metal_type = "yellow_gold"
            elif avg_hue < 20 and avg_saturation > 50:
                metal_type = "rose_gold"
            else:
                metal_type = "white_gold"
        else:
            avg_color = np.mean(img_array.reshape(-1, 3), axis=0)
            r, g, b = avg_color
            
            if r > 200 and g > 200 and b > 200:
                metal_type = "white_gold"
            elif r > g + 20 and r > b + 20:
                if g > b:
                    metal_type = "yellow_gold"
                else:
                    metal_type = "rose_gold"
            else:
                metal_type = "plain_white"
        
        logger.info(f"Metal type detected: {metal_type}")
        return metal_type
        
    except Exception as e:
        logger.error(f"Metal detection error: {str(e)}")
        return "white_gold"

def apply_metal_specific_enhancement(img, metal_type):
    """Apply metal-specific color enhancement"""
    try:
        logger.info(f"Applying enhancement for {metal_type}")
        
        # Base enhancement - Image 3 → Image 5 style
        brightness_enhancer = ImageEnhance.Brightness(img)
        img = brightness_enhancer.enhance(1.05)  # 5% brightness increase
        
        color_enhancer = ImageEnhance.Color(img)
        img = color_enhancer.enhance(0.95)  # 5% saturation decrease
        
        # Metal-specific enhancements
        if metal_type == "yellow_gold" and NUMPY_AVAILABLE:
            img_array = np.array(img)
            img_array[:, :, 0] = np.minimum(255, img_array[:, :, 0] * 1.08).astype(np.uint8)
            img_array[:, :, 1] = np.minimum(255, img_array[:, :, 1] * 1.05).astype(np.uint8)
            img = Image.fromarray(img_array)
            
            color_enhancer = ImageEnhance.Color(img)
            img = color_enhancer.enhance(1.02)
            
        elif metal_type == "rose_gold" and NUMPY_AVAILABLE:
            img_array = np.array(img)
            img_array[:, :, 0] = np.minimum(255, img_array[:, :, 0] * 1.10).astype(np.uint8)
            img_array[:, :, 2] = np.minimum(255, img_array[:, :, 2] * 1.03).astype(np.uint8)
            img = Image.fromarray(img_array)
            
            contrast_enhancer = ImageEnhance.Contrast(img)
            img = contrast_enhancer.enhance(1.05)
            
        elif metal_type == "white_gold":
            contrast_enhancer = ImageEnhance.Contrast(img)
            img = contrast_enhancer.enhance(1.08)
            
            sharpness_enhancer = ImageEnhance.Sharpness(img)
            img = sharpness_enhancer.enhance(1.12)
            
        elif metal_type == "plain_white":
            brightness_enhancer = ImageEnhance.Brightness(img)
            img = brightness_enhancer.enhance(1.08)
            
            color_enhancer = ImageEnhance.Color(img)
            img = color_enhancer.enhance(0.90)
        
        # Selective background brightening
        if NUMPY_AVAILABLE:
            img_array = np.array(img)
            mask = np.all(img_array > 200, axis=-1)
            
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
    """Apply wedding ring detail enhancement"""
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
            
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
            unsharp_mask = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
            
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
    """Convert PIL Image to base64 - REMOVE PADDING for Make.com"""
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
    """RunPod handler for image enhancement V10"""
    start_time = time.time()
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Enhancement Handler {VERSION} - Safe JSON & NumPy Fix")
        logger.info(f"{'='*60}")
        
        job_input = job.get('input', {})
        logger.info(f"Input keys: {list(job_input.keys())}")
        
        if job_input.get('debug_mode', False):
            return {
                "output": {
                    "status": "debug_success",
                    "message": f"{VERSION} handler working - Safe JSON conversion",
                    "version": VERSION,
                    "features": [
                        "Safe JSON conversion (no np.bool)",
                        "Metal type detection (4 types)",
                        "Metal-specific enhancement",
                        "Wedding ring detail enhancement", 
                        "Image 3 → Image 5 style enhancement",
                        "Make.com compatible",
                        "Google Apps Script compatible"
                    ]
                }
            }
        
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
        
        img = decode_base64_image(image_data_str)
        original_size = img.size
        logger.info(f"Original image size: {original_size}")
        
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        metal_type = detect_metal_type(img)
        enhanced_img = apply_metal_specific_enhancement(img, metal_type)
        enhanced_img = apply_wedding_ring_details(enhanced_img)
        enhanced_base64 = image_to_base64(enhanced_img)
        
        # Use safe JSON conversion
        processing_info = {
            "original_size": [original_size[0], original_size[1]],
            "final_size": [enhanced_img.size[0], enhanced_img.size[1]],
            "metal_type": str(metal_type),
            "enhancement_applied": True,
            "detail_enhancement": True,
            "masking_removed": False,
            "processing_time": round(time.time() - start_time, 2),
            "version": VERSION,
            "training_data": "38 pairs (28 + 10)",
            "metal_types": ["yellow_gold", "rose_gold", "white_gold", "plain_white"]
        }
        
        result = {
            "output": {
                "enhanced_image": enhanced_base64,
                "processing_info": safe_json_convert(processing_info),
                "status": "success"
            }
        }
        
        logger.info(f"Enhancement completed in {processing_info['processing_time']}s")
        logger.info(f"Metal type: {metal_type}")
        
        return safe_json_convert(result)
        
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        error_result = {
            "output": {
                "error": error_msg,
                "status": "error",
                "version": VERSION,
                "processing_time": round(time.time() - start_time, 2)
            }
        }
        
        return safe_json_convert(error_result)

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
