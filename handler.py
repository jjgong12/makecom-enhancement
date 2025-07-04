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
import replicate
import requests
import string


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "V139-EnvOptimized"

# ===== REPLICATE INITIALIZATION (환경변수 최적화) =====
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
REPLICATE_CLIENT = None
USE_REPLICATE = False

if REPLICATE_API_TOKEN:
    try:
        REPLICATE_CLIENT = replicate.Client(api_token=REPLICATE_API_TOKEN)
        USE_REPLICATE = True
        logger.info("✅ Replicate client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Replicate client: {e}")
        USE_REPLICATE = False
else:
    logger.warning("⚠️ REPLICATE_API_TOKEN not found in environment variables")

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
        
        # Clean the string - remove all whitespace, newlines, and special chars
        base64_str = base64_str.strip()
        base64_str = ''.join(base64_str.split())  # Remove all whitespace
        base64_str = base64_str.replace('\n', '').replace('\r', '').replace(' ', '').replace('\t', '')
        
        # Remove any non-base64 characters (keep only A-Za-z0-9+/=)
        import string
        valid_chars = string.ascii_letters + string.digits + '+/='
        base64_str = ''.join(c for c in base64_str if c in valid_chars)
        
        # Try decoding with different padding strategies
        # Strategy 1: Try as-is
        try:
            return base64.b64decode(base64_str)
        except:
            pass
        
        # Strategy 2: Remove all padding and add correct padding
        base64_str_no_pad = base64_str.rstrip('=')
        padding = 4 - (len(base64_str_no_pad) % 4)
        if padding != 4:
            base64_str_padded = base64_str_no_pad + ('=' * padding)
        else:
            base64_str_padded = base64_str_no_pad
        
        try:
            return base64.b64decode(base64_str_padded)
        except:
            pass
        
        # Strategy 3: Try with standard base64 (not urlsafe)
        try:
            return base64.b64decode(base64_str_padded, validate=True)
        except:
            pass
        
        # Strategy 4: Try URL-safe base64
        try:
            # Replace standard base64 chars with URL-safe equivalents
            urlsafe_str = base64_str_no_pad.replace('+', '-').replace('/', '_')
            padding = 4 - (len(urlsafe_str) % 4)
            if padding != 4:
                urlsafe_str += '=' * padding
            return base64.urlsafe_b64decode(urlsafe_str)
        except:
            pass
        
        # If all strategies fail, raise error
        raise ValueError("All base64 decoding strategies failed")
        
    except Exception as e:
        logger.error(f"Base64 decode error after all attempts: {str(e)}")
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
    elif 'b_' in filename_lower and 'bc_' not in filename_lower:
        return "b_only"
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

def apply_replicate_enhancement(image: Image.Image, is_wedding_ring: bool, pattern_type: str) -> Image.Image:
    """Apply Replicate API enhancement using pre-initialized client"""
    if not USE_REPLICATE or not REPLICATE_CLIENT:
        logger.error("❌ Replicate not available - API token not configured")
        raise ValueError("Replicate API token not configured. Please set REPLICATE_API_TOKEN environment variable.")
    
    try:
        # Convert image to base64 for Replicate
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        # For wedding rings, use more aggressive enhancement
        if is_wedding_ring:
            logger.info("🔷 Applying Replicate enhancement for wedding ring")
            
            # Use magic-image-refiner for wedding rings
            output = REPLICATE_CLIENT.run(
                "batouresearch/magic-image-refiner:a1ba4c13e7af9ae078be742e276e14bbe4cdcbe43f088ad5b9e2b6cf0f3620a9",
                input={
                    "image": img_data_url,
                    "scale": 2,  # 2x upscale
                    "resemblance": 0.85,  # High resemblance to preserve ring details
                    "prompt": "highly detailed wedding ring, sharp metallic edges, brilliant diamond sparkle, professional jewelry photography, crisp focus"
                }
            )
            
            # Additional pass with swin2sr for extra sharpness
            if output:
                logger.info("🔷 Applying second pass with swin2sr")
                output = REPLICATE_CLIENT.run(
                    "mv-lab/swin2sr:a01b0512004918ca55d02e554914a9eca63909fa83a29ff0f115c78a7045574f",
                    input={
                        "image": output,
                        "task": "real_sr",
                        "scale": 2,
                        "large_model": True
                    }
                )
        
        # For ac_bc pattern (무도금화이트), focus on white metal enhancement
        elif pattern_type == "ac_bc":
            logger.info("⚪ Applying Replicate enhancement for unplated white gold")
            
            output = REPLICATE_CLIENT.run(
                "batouresearch/magic-image-refiner:a1ba4c13e7af9ae078be742e276e14bbe4cdcbe43f088ad5b9e2b6cf0f3620a9",
                input={
                    "image": img_data_url,
                    "scale": 2,
                    "resemblance": 0.8,
                    "prompt": "pure white gold jewelry, clean white metal surface, professional product photography, bright and clean"
                }
            )
        
        # For other patterns, standard enhancement
        else:
            logger.info("🔶 Applying standard Replicate enhancement")
            
            # Use Real-ESRGAN for general enhancement (faster and cheaper)
            output = REPLICATE_CLIENT.run(
                "nightmareai/real-esrgan:350d32041630ffbe63c8352783a26d94126809164e54085352f8326e53999085",
                input={
                    "image": img_data_url,
                    "scale": 2,
                    "face_enhance": False
                }
            )
        
        if output:
            # Convert output back to PIL Image
            if isinstance(output, str):
                # If output is URL
                response = requests.get(output)
                enhanced_image = Image.open(BytesIO(response.content))
            else:
                # If output is base64 or data URL
                if hasattr(output, 'read'):
                    enhanced_image = Image.open(output)
                else:
                    enhanced_image = Image.open(BytesIO(base64.b64decode(output)))
            
            logger.info("✅ Replicate enhancement successful")
            return enhanced_image
        else:
            logger.error("❌ Replicate enhancement failed - no output received")
            raise ValueError("Replicate enhancement failed - no output received")
            
    except Exception as e:
        logger.error(f"❌ Replicate enhancement error: {str(e)}")
        raise ValueError(f"Replicate enhancement failed: {str(e)}")

def auto_white_balance(image: Image.Image) -> Image.Image:
    """Apply automatic white balance correction"""
    img_array = np.array(image, dtype=np.float32)
    
    # Find gray/white areas (R≈G≈B)
    gray_mask = (
        (np.abs(img_array[:,:,0] - img_array[:,:,1]) < 10) & 
        (np.abs(img_array[:,:,1] - img_array[:,:,2]) < 10) &
        (img_array[:,:,0] > 200)  # Bright areas
    )
    
    if np.sum(gray_mask) > 100:  # If enough gray pixels
        # Calculate average RGB in gray areas
        r_avg = np.mean(img_array[gray_mask, 0])
        g_avg = np.mean(img_array[gray_mask, 1])
        b_avg = np.mean(img_array[gray_mask, 2])
        
        # Calculate correction factors
        gray_avg = (r_avg + g_avg + b_avg) / 3
        r_factor = gray_avg / r_avg if r_avg > 0 else 1
        g_factor = gray_avg / g_avg if g_avg > 0 else 1
        b_factor = gray_avg / b_avg if b_avg > 0 else 1
        
        # Apply correction
        img_array[:,:,0] *= r_factor
        img_array[:,:,1] *= g_factor
        img_array[:,:,2] *= b_factor
        
        logger.info(f"White balance correction applied - R:{r_factor:.3f}, G:{g_factor:.3f}, B:{b_factor:.3f}")
    
    img_array = np.clip(img_array, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))

def correct_background_color_subtle(image: Image.Image) -> Image.Image:
    """Subtle background correction - V136 less aggressive"""
    img_array = np.array(image, dtype=np.float32)
    
    # Detect very bright background areas only
    gray = np.mean(img_array, axis=2)
    background_mask = gray > 252  # Increased threshold from 250
    
    # Make background closer to white, but not pure white
    img_array[background_mask] = np.minimum(img_array[background_mask] * 1.01, 255)  # Reduced from 1.02
    
    return Image.fromarray(img_array.astype(np.uint8))

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
    """Determine if second correction is needed - V136 original standards"""
    # Apply ONLY to ac_bc pattern (무도금화이트)
    if pattern_type != "ac_bc":  # Only ac_bc pattern
        return False, None
    
    # V136: Original quality criteria
    reasons = []
    
    if metrics["brightness"] < 241:
        reasons.append("brightness_low")
    
    if metrics["cool_tone_diff"] < 3:
        reasons.append("insufficient_cool_tone")
    
    if metrics["rgb_deviation"] > 5:
        reasons.append("rgb_deviation_high")
    
    if metrics["saturation"] > 2:
        reasons.append("saturation_high")
    
    return len(reasons) > 0, reasons

def apply_second_correction(image: Image.Image, reasons: list) -> Image.Image:
    """Apply second correction based on quality check - V136 original with reduced cool tone"""
    logger.info(f"Applying second correction for reasons: {reasons}")
    
    # Enhanced white overlay for pure white
    if "brightness_low" in reasons:
        white_overlay_percent = 0.18  # Original value
        img_array = np.array(image)
        img_array = img_array * (1 - white_overlay_percent) + 255 * white_overlay_percent
        image = Image.fromarray(img_array.astype(np.uint8))
    
    # Cool tone enhancement - REDUCED
    if "insufficient_cool_tone" in reasons:
        img_array = np.array(image)
        # REDUCED: Slightly boost blue channel
        img_array[:,:,2] = np.clip(img_array[:,:,2] * 1.01, 0, 255)  # Kept reduced
        # REDUCED: Slightly reduce red channel
        img_array[:,:,0] = np.clip(img_array[:,:,0] * 0.99, 0, 255)  # Kept reduced
        image = Image.fromarray(img_array.astype(np.uint8))
    
    # Detail enhancement with edge preservation
    if any(r in reasons for r in ["brightness_low", "saturation_high"]):
        # Apply unsharp mask for detail
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=30, threshold=3))
    
    return image

def apply_center_focus(image: Image.Image, intensity: float = 0.02) -> Image.Image:
    """Apply subtle center focus effect - V136"""
    width, height = image.size
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    
    # Subtle center focus
    focus_mask = 1 + intensity * np.exp(-distance**2 * 1.8)
    focus_mask = np.clip(focus_mask, 1.0, 1.0 + intensity)
    
    img_array = np.array(image, dtype=np.float32)
    for i in range(3):
        img_array[:, :, i] *= focus_mask
    img_array = np.clip(img_array, 0, 255)
    
    return Image.fromarray(img_array.astype(np.uint8))

def apply_enhancement_optimized(image: Image.Image, pattern_type: str, is_wedding_ring: bool) -> Image.Image:
    """Optimized enhancement with pattern-specific settings - V139 with 5% overlay for a/b patterns"""
    
    if pattern_type == "ac_bc":
        # Unplated white enhancement - V136 original
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.04)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.96)  # More desaturated
        
        # V136: Original white overlay for ac_bc - 0.14
        white_overlay = 0.14  # RESTORED
        img_array = np.array(image)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # V136: Original center focus for ac_ pattern - 9%
        width, height = image.size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)
        
        # V136: 9% center focus (RESTORED)
        focus_mask = 1 + 0.09 * np.exp(-distance**2 * 1.5)
        focus_mask = np.clip(focus_mask, 1.0, 1.09)
        
        img_array = np.array(image, dtype=np.float32)
        for i in range(3):
            img_array[:, :, i] *= focus_mask
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # V136: Additional subtle center focus
        image = apply_center_focus(image, 0.02)
        
    elif pattern_type == "a_only":
        # a_ pattern enhancement - V139 UPDATED to 5%
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.04)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.96)  # Same desaturation as ac_bc
        
        # V139: UPDATED white overlay for a_ pattern - 0.05 (5%)
        white_overlay = 0.05  # CHANGED from 0.07 to 0.05
        img_array = np.array(image)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Enhanced sharpness for a_ pattern
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.15)
        
        # V136: Original center focus for a_ pattern - 9%
        width, height = image.size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)
        
        # V136: 9% center focus (RESTORED)
        focus_mask = 1 + 0.09 * np.exp(-distance**2 * 1.2)
        focus_mask = np.clip(focus_mask, 1.0, 1.09)
        
        img_array = np.array(image, dtype=np.float32)
        for i in range(3):
            img_array[:, :, i] *= focus_mask
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # V136: Additional subtle center focus
        image = apply_center_focus(image, 0.02)
        
    elif pattern_type == "b_only":
        # b_ pattern enhancement - V139 NEW with 5% overlay
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.04)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.96)  # Same desaturation as a_
        
        # V139: NEW white overlay for b_ pattern - 0.05 (5%)
        white_overlay = 0.05
        img_array = np.array(image)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Enhanced sharpness for b_ pattern
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.15)
        
        # 9% center focus for b_ pattern
        width, height = image.size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)
        
        focus_mask = 1 + 0.09 * np.exp(-distance**2 * 1.2)
        focus_mask = np.clip(focus_mask, 1.0, 1.09)
        
        img_array = np.array(image, dtype=np.float32)
        for i in range(3):
            img_array[:, :, i] *= focus_mask
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Additional subtle center focus
        image = apply_center_focus(image, 0.02)
        
    else:
        # Standard enhancement (other patterns)
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.025)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.985)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.01)
        
        # V136: Add subtle center focus to other patterns too
        image = apply_center_focus(image, 0.015)
    
    # Wedding ring focus enhancement - V136 REDUCED HIGHLIGHT
    if is_wedding_ring:
        # 1. Highlight Enhancement - ONLY FOR NON-BACKGROUND AREAS
        img_array = np.array(image, dtype=np.float32)
        
        # Detect background (very bright and low saturation)
        gray = np.mean(img_array, axis=2)
        max_rgb = np.max(img_array, axis=2)
        min_rgb = np.min(img_array, axis=2)
        saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
        
        # Background mask: very bright AND low saturation
        is_background = (gray > 245) & (saturation < 0.05)
        
        # Apply bright enhancement only to non-background bright areas
        bright_mask = (img_array > 220) & (~is_background[:,:,np.newaxis])
        img_array[bright_mask] *= 1.10  # Reduced from 1.15
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # 2. Enhanced sharpening (KEPT SAME)
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.25)
        
        # 3. Enhanced contrast (KEPT SAME)
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.025)
        
        # 4. Structure Enhancement (KEPT SAME)
        image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=2))
        
        # 5. Enhanced Center focus - 5% (RESTORED)
        width, height = image.size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)
        
        # Additional 5% for wedding rings (RESTORED)
        center_mask = 1 + 0.05 * np.exp(-distance**2 * 1.8)
        center_mask = np.clip(center_mask, 1.0, 1.05)
        
        # Apply center brightening
        img_array = np.array(image, dtype=np.float32)
        for i in range(3):
            img_array[:, :, i] *= center_mask
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # 6. Enhanced edge darkening (KEPT SAME)
        edge_mask = 0.97 + 0.03 * np.exp(-distance**2 * 0.8)
        edge_mask = np.clip(edge_mask, 0.97, 1.0)
        
        img_array = np.array(image, dtype=np.float32)
        for i in range(3):
            img_array[:, :, i] *= edge_mask
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # 7. Micro Contrast (KEPT SAME)
        gray = image.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges_array = np.array(edges, dtype=np.float32) * 0.12  # 12% micro contrast
        
        img_array = np.array(image, dtype=np.float32)
        for i in range(3):
            img_array[:, :, i] += edges_array
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
    
    return image

def needs_upscaling(image: Image.Image) -> bool:
    """Determine if image needs upscaling"""
    width, height = image.size
    # If width is less than 2400px or height is less than 3200px, needs upscaling
    return width < 2400 or height < 3200

def resize_to_width_1200(image: Image.Image) -> Image.Image:
    """Resize image to width 1200px maintaining aspect ratio"""
    width, height = image.size
    target_width = 1200
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

def process_enhancement(job):
    """Main enhancement processing with quality check system and Replicate integration"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    logger.info(f"Input type: {type(job)}")
    logger.info(f"Replicate available: {USE_REPLICATE}")
    
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
        
        # Apply white balance correction FIRST
        image = auto_white_balance(image)
        
        # Detect pattern type
        pattern_type = detect_pattern_type(filename)
        detected_type = {
            "ac_bc": "무도금화이트(0.14)",
            "a_only": "a_패턴(0.05)",
            "b_only": "b_패턴(0.05)",
            "other": "기타색상"
        }.get(pattern_type, "기타색상")
        
        logger.info(f"Pattern type: {pattern_type}, Detected type: {detected_type}")
        
        # Fast wedding ring detection - returns Python bool
        is_wedding_ring = detect_wedding_ring_fast(image)
        logger.info(f"Wedding ring detected: {is_wedding_ring}")
        
        # Check if upscaling is needed before basic enhancement
        needs_upscale = needs_upscaling(image)
        replicate_applied = False
        
        # Apply Replicate enhancement if available (웨딩링이므로 항상 적용)
        if USE_REPLICATE:
            logger.info(f"Applying Replicate enhancement - Wedding ring: {is_wedding_ring}, Needs upscale: {needs_upscale}")
            image = apply_replicate_enhancement(image, is_wedding_ring, pattern_type)
            replicate_applied = True
        
        # Basic enhancement with original brightness
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.025)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.02)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(1.01)
        
        # Apply pattern-specific enhancement
        image = apply_enhancement_optimized(image, pattern_type, is_wedding_ring)
        
        # Quality check for second correction (only for ac_bc pattern)
        second_correction_applied = False
        correction_reasons = []
        quality_metrics = None
        
        if pattern_type == "ac_bc":  # Only ac_bc pattern
            quality_metrics = calculate_quality_metrics(image)
            needs_correction, reasons = needs_second_correction(quality_metrics, pattern_type)
            
            if needs_correction:
                image = apply_second_correction(image, reasons)
                second_correction_applied = True
                correction_reasons = reasons
                # Recalculate metrics after correction
                quality_metrics = calculate_quality_metrics(image)
        
        # V136: Original final sharpening
        if not is_wedding_ring:
            sharpness = ImageEnhance.Sharpness(image)
            image = sharpness.enhance(1.15)
        
        # Apply SUBTLE background correction for final touch
        image = correct_background_color_subtle(image)
        
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
                "correction_reasons": correction_reasons,
                "white_overlay_info": {
                    "ac_bc": "0.14",
                    "a_only": "0.05",
                    "b_only": "0.05",
                    "other": "none"
                },
                "has_center_focus": True,
                "center_focus_intensity": "9%",
                "white_balance_applied": True,
                "cool_tone_reduced": True,
                "background_correction": "subtle",
                "replicate_enhancement": {
                    "applied": replicate_applied,
                    "upscaling_needed": needs_upscale,
                    "available": USE_REPLICATE,
                    "model_used": "magic-image-refiner + swin2sr" if is_wedding_ring else 
                                  "magic-image-refiner" if pattern_type == "ac_bc" else 
                                  "real-esrgan"
                },
                "wedding_ring_enhancements": {
                    "highlight_enhancement": "10%",
                    "highlight_threshold": "220",
                    "micro_contrast": "12%",
                    "structure_enhancement": "enabled",
                    "enhanced_sharpness": "1.25",
                    "enhanced_contrast": "1.025",
                    "enhanced_center_focus": "5%",
                    "enhanced_edge_darkening": "0.97-1.0"
                } if is_wedding_ring else None
            }
        }
        
        # Add quality metrics for ac_bc pattern only
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
