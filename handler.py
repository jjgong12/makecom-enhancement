import runpod
import os
import json
import time
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import logging
import re

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

VERSION = "V126-TargetRGB-Optimized"

# Target RGB values for unplated white
TARGET_RGB_UNPLATED = (235, 237, 239)

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

def find_input_data_fast(data):
    """Find input data - optimized with limited recursion"""
    if isinstance(data, str):
        return data
    
    if isinstance(data, dict):
        # Direct access to common paths
        image_keys = ['enhanced_image', 'image', 'image_data', 'base64_image', 
                     'imageBase64', 'image_base64', 'base64']
        
        # Check root level
        for key in image_keys:
            if key in data and isinstance(data[key], str) and len(data[key]) > 100:
                return data[key]
        
        # Check input key properly
        if 'input' in data:
            if isinstance(data['input'], str) and len(data['input']) > 100:
                return data['input']
            elif isinstance(data['input'], dict):
                for key in image_keys:
                    if key in data['input'] and isinstance(data['input'][key], str):
                        return data['input'][key]
        
        # Check numeric keys (Make.com) - limited to 0-9
        for i in range(10):
            key = str(i)
            if key in data:
                if isinstance(data[key], str) and len(data[key]) > 100:
                    return data[key]
                elif isinstance(data[key], dict):
                    # Limited recursive search
                    result = _limited_search(data[key], depth=1)
                    if result:
                        return result
        
        # Try common nested paths
        paths = [
            ['job', 'input'],
            ['data', 'image'],
            ['data', 'enhanced_image'],
            ['output', 'enhanced_image'],
            ['4', 'data', 'output', 'output', 'enhanced_image']
        ]
        
        for path in paths:
            obj = data
            try:
                for key in path:
                    obj = obj[key]
                if isinstance(obj, str) and len(obj) > 100:
                    return obj
            except:
                continue
    
    return None

def _limited_search(obj, depth=0):
    """Limited depth search helper"""
    if depth > 2:  # Max depth 2
        return None
    
    if isinstance(obj, str) and len(obj) > 100:
        return obj
    
    if isinstance(obj, dict):
        # Priority keys
        priority_keys = ['enhanced_image', 'image', 'image_data', 'base64_image', 
                        'output', 'data', 'input']
        
        # Check priority keys first
        for key in priority_keys:
            if key in obj:
                if isinstance(obj[key], str) and len(obj[key]) > 100:
                    return obj[key]
                elif isinstance(obj[key], dict):
                    result = _limited_search(obj[key], depth + 1)
                    if result:
                        return result
        
        # Then check other keys
        for key, value in obj.items():
            if key not in priority_keys:
                if isinstance(value, str) and len(value) > 100:
                    return value
                elif isinstance(value, dict) and depth < 2:
                    result = _limited_search(value, depth + 1)
                    if result:
                        return result
    
    return None

def find_filename_fast(data, depth=0):
    """Fast filename extraction with limited recursion"""
    if depth > 3:  # Limit depth
        return None
    
    if isinstance(data, dict):
        # Direct filename keys
        filename_keys = ['filename', 'file_name', 'fileName', 'name', 
                        'originalName', 'original_name', 'image_name']
        
        # Check current level
        for key in filename_keys:
            if key in data and isinstance(data[key], str):
                value = data[key]
                if any(p in value.lower() for p in ['ac_', 'bc_', 'a_', 'b_', 'c_']):
                    return value
        
        # Check common nested locations
        nested_keys = ['input', 'data', 'job', 'output']
        for key in nested_keys:
            if key in data:
                if isinstance(data[key], dict):
                    result = find_filename_fast(data[key], depth + 1)
                    if result:
                        return result
                elif isinstance(data[key], str):
                    # Sometimes the value itself contains the filename
                    if any(p in data[key].lower() for p in ['ac_', 'bc_', 'a_', 'b_', 'c_']):
                        if len(data[key]) < 100:  # Reasonable filename length
                            return data[key]
        
        # Check numeric keys (Make.com)
        for i in range(10):
            key = str(i)
            if key in data and isinstance(data[key], dict):
                result = find_filename_fast(data[key], depth + 1)
                if result:
                    return result
    
    elif isinstance(data, list) and depth < 3:
        # Check list items
        for item in data:
            if isinstance(item, dict):
                result = find_filename_fast(item, depth + 1)
                if result:
                    return result
    
    return None

def decode_base64_safe(base64_str: str) -> bytes:
    """Safely decode base64 - optimized"""
    if base64_str.startswith('data:'):
        base64_str = base64_str.split(',')[1]
    
    base64_str = base64_str.strip()
    
    # Add padding if needed
    padding = 4 - len(base64_str) % 4
    if padding != 4:
        base64_str += '=' * padding
    
    return base64.b64decode(base64_str)

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
    """Fast wedding ring detection - brightness only"""
    # Convert center region to grayscale
    width, height = image.size
    center_crop = image.crop((width//3, height//3, 2*width//3, 2*height//3))
    gray = center_crop.convert('L')
    gray_array = np.array(gray)
    
    # Check bright metallic areas
    bright_pixels = np.sum(gray_array > 200)
    total_pixels = gray_array.size
    bright_ratio = bright_pixels / total_pixels
    
    return bright_ratio > 0.15

def calculate_quality_metrics_fast(image: Image.Image) -> dict:
    """Calculate quality metrics without cv2"""
    img_array = np.array(image)
    
    # Calculate average RGB values
    r_avg = np.mean(img_array[:,:,0])
    g_avg = np.mean(img_array[:,:,1])
    b_avg = np.mean(img_array[:,:,2])
    
    # Calculate brightness (luminance)
    brightness = (r_avg + g_avg + b_avg) / 3
    
    # Calculate RGB deviation
    rgb_values = [r_avg, g_avg, b_avg]
    rgb_deviation = max(rgb_values) - min(rgb_values)
    
    # Calculate saturation approximation (without HSV)
    # Using simplified formula: (max - min) / max
    max_val = max(rgb_values)
    min_val = min(rgb_values)
    saturation = ((max_val - min_val) / max_val * 100) if max_val > 0 else 0
    
    # Cool tone check (B should be higher than R)
    cool_tone_diff = b_avg - r_avg
    
    return {
        "brightness": brightness,
        "rgb": {"r": r_avg, "g": g_avg, "b": b_avg},
        "rgb_deviation": rgb_deviation,
        "saturation": saturation,
        "cool_tone_diff": cool_tone_diff
    }

def apply_target_rgb_correction(image: Image.Image, target_rgb=TARGET_RGB_UNPLATED, strength=0.7):
    """Apply target RGB correction for unplated white"""
    img_array = np.array(image, dtype=np.float32)
    
    # Calculate current average RGB
    current_r = np.mean(img_array[:,:,0])
    current_g = np.mean(img_array[:,:,1])
    current_b = np.mean(img_array[:,:,2])
    
    # Protect highlights (cubic/diamond areas)
    brightness_map = np.mean(img_array, axis=2)
    highlight_mask = brightness_map > 240
    
    # Calculate correction ratios
    r_ratio = target_rgb[0] / current_r if current_r > 0 else 1
    g_ratio = target_rgb[1] / current_g if current_g > 0 else 1
    b_ratio = target_rgb[2] / current_b if current_b > 0 else 1
    
    # Apply correction with strength control
    r_ratio = 1 + (r_ratio - 1) * strength
    g_ratio = 1 + (g_ratio - 1) * strength
    b_ratio = 1 + (b_ratio - 1) * strength
    
    # Apply correction
    img_array[:,:,0] *= r_ratio
    img_array[:,:,1] *= g_ratio
    img_array[:,:,2] *= b_ratio
    
    # Restore highlights
    img_array[highlight_mask, :] = np.array(image)[highlight_mask, :]
    
    # Clip values
    img_array = np.clip(img_array, 0, 255)
    
    return Image.fromarray(img_array.astype(np.uint8))

def apply_enhancement_optimized(image: Image.Image, pattern_type: str, is_wedding_ring: bool) -> Image.Image:
    """Optimized enhancement with pattern-specific settings"""
    
    if pattern_type == "ac_bc":
        # For unplated white - apply target RGB
        logger.info("Applying target RGB correction for unplated white")
        
        # First mild brightness adjustment
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.01)
        
        # Apply target RGB correction
        if is_wedding_ring:
            # Stronger correction for wedding rings
            image = apply_target_rgb_correction(image, strength=0.8)
        else:
            # Standard correction
            image = apply_target_rgb_correction(image, strength=0.7)
        
        # Reduce saturation slightly
        color = ImageEnhance.Color(image)
        image = color.enhance(0.95)
        
    elif pattern_type == "a_only":
        # a_ pattern enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.025)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.985)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.01)
        
        # Simple center focus
        if not is_wedding_ring:
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
        
        # Subtle center brightening for wedding rings
        width, height = image.size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)
        
        focus_mask = 1 + 0.02 * np.exp(-distance**2 * 1.5)
        focus_mask = np.clip(focus_mask, 1.0, 1.02)
        
        img_array = np.array(image, dtype=np.float32)
        for i in range(3):
            img_array[:, :, i] *= focus_mask
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
    
    return image

def resize_to_width_1200(image: Image.Image) -> Image.Image:
    """Resize image to width 1200px"""
    width, height = image.size
    target_width = 1200
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

def process_enhancement(job):
    """Main enhancement processing with target RGB system"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    
    try:
        # Fast filename extraction
        filename = find_filename_fast(job)
        file_number = extract_file_number(filename) if filename else None
        
        if filename:
            logger.info(f"Processing file: {filename}")
        
        # Fast image data extraction
        image_data = find_input_data_fast(job)
        
        if not image_data:
            logger.error(f"Failed to find image data")
            return {
                "output": {
                    "error": "No image data found",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Decode image
        image_bytes = decode_base64_safe(image_data)
        image = Image.open(BytesIO(image_bytes))
        
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
        
        # Fast wedding ring detection
        is_wedding_ring = detect_wedding_ring_fast(image)
        
        # Basic enhancement for all images
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.02)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.02)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(1.01)
        
        # Apply pattern-specific enhancement (includes target RGB for ac_bc)
        image = apply_enhancement_optimized(image, pattern_type, is_wedding_ring)
        
        # Quality check for ac_bc patterns
        quality_metrics = None
        if pattern_type == "ac_bc":
            quality_metrics = calculate_quality_metrics_fast(image)
            
            # Check if additional correction needed
            if quality_metrics["brightness"] < 230:
                # Apply additional white overlay
                img_array = np.array(image, dtype=np.float32)
                img_array = img_array * 0.95 + 255 * 0.05
                image = Image.fromarray(img_array.astype(np.uint8))
                
                # Recalculate metrics
                quality_metrics = calculate_quality_metrics_fast(image)
        
        # Final sharpening
        if not is_wedding_ring:
            sharpness = ImageEnhance.Sharpness(image)
            image = sharpness.enhance(1.08)
        
        # Resize
        image = resize_to_width_1200(image)
        
        # Save to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=True, quality=95)
        buffered.seek(0)
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Remove padding
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
                "is_wedding_ring": is_wedding_ring,
                "filename": filename,
                "enhanced_filename": enhanced_filename,
                "file_number": file_number,
                "original_size": list(original_size),
                "final_size": list(image.size),
                "version": VERSION,
                "status": "success"
            }
        }
        
        # Add quality metrics for ac_bc patterns
        if quality_metrics:
            output["output"]["quality_check"] = {
                "brightness": round(quality_metrics["brightness"], 1),
                "rgb": {
                    "r": round(quality_metrics["rgb"]["r"], 1),
                    "g": round(quality_metrics["rgb"]["g"], 1),
                    "b": round(quality_metrics["rgb"]["b"], 1)
                },
                "rgb_deviation": round(quality_metrics["rgb_deviation"], 1),
                "saturation": round(quality_metrics["saturation"], 1),
                "cool_tone_diff": round(quality_metrics["cool_tone_diff"], 1),
                "target_rgb": TARGET_RGB_UNPLATED
            }
        
        logger.info("Enhancement completed successfully")
        return output
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": VERSION
            }
        }

def handler(event):
    """RunPod handler function"""
    return process_enhancement(event)

# RunPod handler
runpod.serverless.start({"handler": handler})
