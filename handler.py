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
import re

logging.basicConfig(level=logging.INFO)  # Changed to INFO for debugging
logger = logging.getLogger(__name__)

VERSION = "V128-CompleteFix"

def extract_file_number(filename: str) -> str:
    """Extract number from filename - optimized"""
    if not filename:
        return None
    
    match = re.search(r'(\d{3})', filename)
    if match:
        return match.group(1)
    
    match = re.search(r'(\d{2})', filename)
    if match:
        return match.group(1).zfill(3)
    
    return None

def find_input_data_comprehensive(data, depth=0):
    """Comprehensive search for input data with multiple strategies"""
    if depth > 5:  # Prevent infinite recursion
        return None
    
    # If data is already a string, check if it's base64
    if isinstance(data, str):
        if len(data) > 100:  # Likely base64
            return data
    
    if isinstance(data, dict):
        # Extended list of possible image keys
        image_keys = [
            'enhanced_image', 'image', 'image_data', 'base64_image',
            'imageBase64', 'image_base64', 'base64', 'img', 'photo',
            'picture', 'file', 'content', 'data', 'b64', 'base64_data',
            'image_content', 'file_content', 'image_url', 'url'
        ]
        
        # 1. Check direct keys
        for key in image_keys:
            if key in data:
                value = data[key]
                if isinstance(value, str) and len(value) > 100:
                    logger.info(f"Found image data at key: {key}")
                    return value
                elif isinstance(value, dict):
                    result = find_input_data_comprehensive(value, depth + 1)
                    if result:
                        return result
        
        # 2. Check 'input' variations
        input_keys = ['input', 'inputs', 'data', 'payload', 'body', 'request']
        for input_key in input_keys:
            if input_key in data:
                if isinstance(data[input_key], str) and len(data[input_key]) > 100:
                    logger.info(f"Found image data at {input_key}")
                    return data[input_key]
                elif isinstance(data[input_key], dict):
                    for key in image_keys:
                        if key in data[input_key]:
                            value = data[input_key][key]
                            if isinstance(value, str) and len(value) > 100:
                                logger.info(f"Found image data at {input_key}.{key}")
                                return value
        
        # 3. Check numeric keys (Make.com patterns)
        for i in range(20):  # Check more numeric keys
            key = str(i)
            if key in data:
                if isinstance(data[key], str) and len(data[key]) > 100:
                    logger.info(f"Found image data at numeric key: {key}")
                    return data[key]
                elif isinstance(data[key], dict):
                    result = find_input_data_comprehensive(data[key], depth + 1)
                    if result:
                        return result
        
        # 4. Check nested structures - common API patterns
        nested_paths = [
            ['job', 'input'],
            ['job', 'data'],
            ['data', 'image'],
            ['data', 'input'],
            ['payload', 'image'],
            ['body', 'data'],
            ['request', 'data'],
            ['output', 'enhanced_image'],
            ['result', 'image'],
            ['response', 'data'],
            # Make.com specific paths
            ['4', 'data', 'output', 'output', 'enhanced_image'],
            ['3', 'data', 'output', 'enhanced_image'],
            ['2', 'data', 'image'],
            ['1', 'image'],
            ['0', 'data']
        ]
        
        for path in nested_paths:
            obj = data
            try:
                for key in path:
                    obj = obj[key]
                if isinstance(obj, str) and len(obj) > 100:
                    logger.info(f"Found image data at path: {'.'.join(path)}")
                    return obj
                elif isinstance(obj, dict):
                    result = find_input_data_comprehensive(obj, depth + 1)
                    if result:
                        return result
            except:
                continue
        
        # 5. Deep search all string values
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 100:
                # Check if it looks like base64
                if not value.startswith('http') and not '/' in value[:50]:
                    logger.info(f"Found potential image data at key: {key}")
                    return value
            elif isinstance(value, dict):
                result = find_input_data_comprehensive(value, depth + 1)
                if result:
                    return result
            elif isinstance(value, list) and len(value) > 0:
                for item in value:
                    if isinstance(item, dict):
                        result = find_input_data_comprehensive(item, depth + 1)
                        if result:
                            return result
                    elif isinstance(item, str) and len(item) > 100:
                        logger.info(f"Found image data in list at key: {key}")
                        return item
    
    elif isinstance(data, list):
        # Check list items
        for i, item in enumerate(data):
            if isinstance(item, str) and len(item) > 100:
                logger.info(f"Found image data in list at index: {i}")
                return item
            elif isinstance(item, dict):
                result = find_input_data_comprehensive(item, depth + 1)
                if result:
                    return result
    
    return None

def find_filename_comprehensive(data, depth=0):
    """Comprehensive filename search"""
    if depth > 5:
        return None
    
    if isinstance(data, dict):
        # Extended filename keys
        filename_keys = [
            'filename', 'file_name', 'fileName', 'name', 'originalName', 
            'original_name', 'image_name', 'imageName', 'file', 'title',
            'display_name', 'displayName', 'originalFilename', 'original_filename'
        ]
        
        # Check current level
        for key in filename_keys:
            if key in data and isinstance(data[key], str):
                value = data[key]
                if any(p in value.lower() for p in ['ac_', 'bc_', 'a_', 'b_', 'c_']):
                    logger.info(f"Found filename at key: {key} = {value}")
                    return value
                elif '.' in value and len(value) < 100:  # Any filename with extension
                    logger.info(f"Found filename at key: {key} = {value}")
                    return value
        
        # Recursive search
        for key, value in data.items():
            if isinstance(value, dict):
                result = find_filename_comprehensive(value, depth + 1)
                if result:
                    return result
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        result = find_filename_comprehensive(item, depth + 1)
                        if result:
                            return result
    
    return None

def decode_base64_safe(base64_str: str) -> bytes:
    """Safely decode base64 with comprehensive error handling"""
    try:
        # Remove data URL prefix if present
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]
        elif base64_str.startswith('data:'):
            base64_str = base64_str.split(',')[1]
        
        # Clean the string
        base64_str = base64_str.strip()
        
        # Remove any whitespace or newlines
        base64_str = base64_str.replace('\n', '').replace('\r', '').replace(' ', '')
        
        # Add padding if needed
        padding = 4 - len(base64_str) % 4
        if padding != 4:
            base64_str += '=' * padding
        
        # Try to decode
        return base64.b64decode(base64_str)
        
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        # Try without padding
        try:
            return base64.b64decode(base64_str.rstrip('='))
        except:
            # Try with different padding
            for pad in range(4):
                try:
                    return base64.b64decode(base64_str.rstrip('=') + '=' * pad)
                except:
                    continue
            raise ValueError(f"Failed to decode base64: {str(e)}")

def detect_pattern_type(filename: str) -> str:
    """Detect pattern type - optimized"""
    if not filename:
        return "other"
    
    filename_lower = filename.lower()
    
    if 'ac_' in filename_lower or 'bc_' in filename_lower:
        return "ac_bc"
    elif 'a_' in filename_lower and 'ac_' not in filename_lower:
        return "a_only"
    else:
        return "other"

def detect_wedding_ring_fast(image: Image.Image) -> bool:
    """Fast wedding ring detection - returns Python bool"""
    try:
        # Convert center region to grayscale
        width, height = image.size
        center_crop = image.crop((width//3, height//3, 2*width//3, 2*height//3))
        gray = center_crop.convert('L')
        gray_array = np.array(gray)
        
        # Check bright metallic areas
        bright_pixels = np.sum(gray_array > 200)
        total_pixels = gray_array.size
        bright_ratio = bright_pixels / total_pixels
        
        # Convert to Python bool
        return bool(bright_ratio > 0.15)
    except:
        return False

def calculate_quality_metrics(image: Image.Image) -> dict:
    """Calculate quality metrics for second correction decision"""
    img_array = np.array(image)
    
    # Calculate average RGB values - convert to Python float
    r_avg = float(np.mean(img_array[:,:,0]))
    g_avg = float(np.mean(img_array[:,:,1]))
    b_avg = float(np.mean(img_array[:,:,2]))
    
    # Calculate brightness (luminance)
    brightness = (r_avg + g_avg + b_avg) / 3
    
    # Calculate RGB deviation
    rgb_values = [r_avg, g_avg, b_avg]
    rgb_deviation = max(rgb_values) - min(rgb_values)
    
    # Calculate saturation
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    saturation = float(np.mean(img_hsv[:,:,1]) / 255 * 100)
    
    # Cool tone check (B should be higher than R)
    cool_tone_diff = b_avg - r_avg
    
    return {
        "brightness": brightness,
        "rgb": {"r": r_avg, "g": g_avg, "b": b_avg},
        "rgb_deviation": rgb_deviation,
        "saturation": saturation,
        "cool_tone_diff": cool_tone_diff
    }

def needs_second_correction(metrics: dict, pattern_type: str) -> tuple:
    """Determine if second correction is needed"""
    if pattern_type != "ac_bc":
        return False, None
    
    # Quality criteria for unplated white
    reasons = []
    
    if metrics["brightness"] < 235:
        reasons.append("brightness_low")
    
    if metrics["cool_tone_diff"] < 3:
        reasons.append("insufficient_cool_tone")
    
    if metrics["rgb_deviation"] > 5:
        reasons.append("rgb_deviation_high")
    
    if metrics["saturation"] > 3:
        reasons.append("saturation_high")
    
    return len(reasons) > 0, reasons

def apply_second_correction(image: Image.Image, reasons: list) -> Image.Image:
    """Apply second correction based on quality check"""
    logger.info(f"Applying second correction for reasons: {reasons}")
    
    # Enhanced white overlay for unplated white
    if "brightness_low" in reasons:
        white_overlay_percent = 0.20  # Increased from 15%
        img_array = np.array(image)
        img_array = img_array * (1 - white_overlay_percent) + 255 * white_overlay_percent
        image = Image.fromarray(img_array.astype(np.uint8))
    
    # Cool tone enhancement
    if "insufficient_cool_tone" in reasons:
        img_array = np.array(image)
        # Slightly boost blue channel
        img_array[:,:,2] = np.clip(img_array[:,:,2] * 1.02, 0, 255)
        # Slightly reduce red channel
        img_array[:,:,0] = np.clip(img_array[:,:,0] * 0.98, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
    
    # Detail enhancement with edge preservation
    if any(r in reasons for r in ["brightness_low", "saturation_high"]):
        # Apply unsharp mask for detail
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=30, threshold=3))
    
    return image

def apply_enhancement_optimized(image: Image.Image, pattern_type: str, is_wedding_ring: bool) -> Image.Image:
    """Optimized enhancement with pattern-specific settings"""
    
    if pattern_type == "ac_bc":
        # Unplated white enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.02)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.97)
        
        # White overlay
        white_overlay = 0.15 if is_wedding_ring else 0.12
        img_array = np.array(image)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        image = Image.fromarray(img_array.astype(np.uint8))
        
    elif pattern_type == "a_only":
        # a_ pattern enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.025)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.985)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.01)
        
        # Simple center focus for a_ pattern
        width, height = image.size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)
        
        focus_mask = 1 + 0.015 * np.exp(-distance**2 * 1.2)
        focus_mask = np.clip(focus_mask, 1.0, 1.015)
        
        img_array = np.array(image, dtype=np.float32)
        for i in range(3):
            img_array[:, :, i] *= focus_mask
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
    else:
        # Standard enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.02)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.985)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.01)
    
    # Wedding ring focus (simplified)
    if is_wedding_ring:
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.15)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.015)
    
    return image

def resize_to_width_1200(image: Image.Image) -> Image.Image:
    """Resize image to width 1200px maintaining aspect ratio"""
    width, height = image.size
    target_width = 1200
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

def process_enhancement(job):
    """Main enhancement processing with quality check system"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    logger.info(f"Input type: {type(job)}")
    
    if isinstance(job, dict):
        logger.info(f"Input keys: {list(job.keys())[:10]}")
    
    try:
        # Comprehensive filename extraction
        filename = find_filename_comprehensive(job)
        logger.info(f"Filename found: {filename}")
        file_number = extract_file_number(filename) if filename else None
        
        # Comprehensive image data extraction
        image_data = find_input_data_comprehensive(job)
        
        if not image_data:
            logger.error("Failed to find image data after comprehensive search")
            return {
                "output": {
                    "error": "No image data found in input",
                    "status": "error",
                    "version": VERSION,
                    "debug_info": {
                        "input_type": str(type(job)),
                        "keys": list(job.keys())[:20] if isinstance(job, dict) else None
                    }
                }
            }
        
        logger.info(f"Found image data, length: {len(image_data)}")
        
        # Decode image with enhanced error handling
        try:
            image_bytes = decode_base64_safe(image_data)
            image = Image.open(BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Failed to decode/open image: {str(e)}")
            return {
                "output": {
                    "error": f"Failed to decode image: {str(e)}",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Convert to RGB
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            else:
                image = image.convert('RGB')
        
        original_size = image.size
        logger.info(f"Image size: {original_size}")
        
        # Detect pattern type
        pattern_type = detect_pattern_type(filename)
        detected_type = {
            "ac_bc": "무도금화이트",
            "a_only": "a_패턴",
            "other": "기타색상"
        }.get(pattern_type, "기타색상")
        
        logger.info(f"Pattern type: {pattern_type}, Detected type: {detected_type}")
        
        # Fast wedding ring detection - returns Python bool
        is_wedding_ring = detect_wedding_ring_fast(image)
        
        # Basic enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.02)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.02)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(1.01)
        
        # Apply pattern-specific enhancement
        image = apply_enhancement_optimized(image, pattern_type, is_wedding_ring)
        
        # Quality check for second correction (only for ac_bc)
        second_correction_applied = False
        correction_reasons = []
        quality_metrics = None
        
        if pattern_type == "ac_bc":
            quality_metrics = calculate_quality_metrics(image)
            needs_correction, reasons = needs_second_correction(quality_metrics, pattern_type)
            
            if needs_correction:
                image = apply_second_correction(image, reasons)
                second_correction_applied = True
                correction_reasons = reasons
                # Recalculate metrics after correction
                quality_metrics = calculate_quality_metrics(image)
        
        # Final sharpening
        if not is_wedding_ring:
            sharpness = ImageEnhance.Sharpness(image)
            image = sharpness.enhance(1.08)
        
        # Resize to 1200px width
        image = resize_to_width_1200(image)
        logger.info(f"Resized to: {image.size}")
        
        # Save to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=True, quality=95)
        buffered.seek(0)
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com compatibility
        enhanced_base64_no_padding = enhanced_base64.rstrip('=')
        
        # Build enhanced filename
        enhanced_filename = filename
        if filename and file_number:
            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            extension = filename.rsplit('.', 1)[1] if '.' in filename else 'jpg'
            enhanced_filename = f"{base_name}_enhanced.{extension}"
        
        output = {
            "output": {
                "enhanced_image": enhanced_base64_no_padding,
                "enhanced_image_with_prefix": f"data:image/png;base64,{enhanced_base64_no_padding}",
                "detected_type": detected_type,
                "pattern_type": pattern_type,
                "is_wedding_ring": bool(is_wedding_ring),  # Ensure Python bool
                "filename": filename,
                "enhanced_filename": enhanced_filename,
                "file_number": file_number,
                "original_size": list(original_size),
                "final_size": list(image.size),
                "version": VERSION,
                "status": "success",
                "second_correction_applied": bool(second_correction_applied),
                "correction_reasons": correction_reasons
            }
        }
        
        # Add quality metrics for ac_bc patterns
        if quality_metrics:
            output["output"]["quality_check"] = {
                "brightness": round(float(quality_metrics["brightness"]), 1),
                "rgb": {
                    "r": round(float(quality_metrics["rgb"]["r"]), 1),
                    "g": round(float(quality_metrics["rgb"]["g"]), 1),
                    "b": round(float(quality_metrics["rgb"]["b"]), 1)
                },
                "rgb_deviation": round(float(quality_metrics["rgb_deviation"]), 1),
                "saturation": round(float(quality_metrics["saturation"]), 1),
                "cool_tone_diff": round(float(quality_metrics["cool_tone_diff"]), 1)
            }
        
        logger.info("Enhancement completed successfully")
        return output
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

def handler(event):
    """RunPod handler function"""
    return process_enhancement(event)

# RunPod handler
runpod.serverless.start({"handler": handler})
