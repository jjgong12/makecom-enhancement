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

VERSION = "V76-PerfectRecursive"

def decode_base64_safe(base64_str: str) -> bytes:
    """Safely decode base64 with automatic padding correction"""
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

def apply_center_focus(image: Image.Image, strength: float = 0.1) -> Image.Image:
    """Apply subtle center focus vignette"""
    width, height = image.size
    
    # Create radial gradient mask
    mask = Image.new('L', (width, height), 0)
    
    for y in range(height):
        for x in range(width):
            # Distance from center
            dx = (x - width/2) / (width/2)
            dy = (y - height/2) / (height/2)
            dist = np.sqrt(dx**2 + dy**2)
            
            # Smooth falloff
            brightness = int(255 * (1 - strength * min(1, dist**1.5)))
            mask.putpixel((x, y), brightness)
    
    # Apply mask
    mask = mask.filter(ImageFilter.GaussianBlur(radius=50))
    enhancer = ImageEnhance.Brightness(image)
    bright_image = enhancer.enhance(1.0 + strength * 0.3)
    
    return Image.composite(bright_image, image, mask)

def apply_color_enhancement(image: Image.Image, detected_color: str) -> Image.Image:
    """Apply color-specific enhancement"""
    img_array = np.array(image)
    
    if detected_color == "무도금화이트":
        # Strong white enhancement for pure white look
        # Convert to LAB
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Increase lightness more
        lab[:, :, 0] = np.clip(lab[:, :, 0] * 1.15, 0, 255)
        
        # Reduce color channels significantly (pure white)
        lab[:, :, 1] = lab[:, :, 1] * 0.5  # Reduce a channel
        lab[:, :, 2] = lab[:, :, 2] * 0.5  # Reduce b channel
        
        # Convert back
        lab = lab.astype(np.uint8)
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Additional brightness boost
        image = Image.fromarray(img_array)
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.15)
        
        # Reduce saturation to near zero
        color = ImageEnhance.Color(image)
        image = color.enhance(0.3)
        
        # Slight blue boost to remove yellow tint
        img_array = np.array(image)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.08, 0, 255)
        image = Image.fromarray(img_array)
        
    elif detected_color == "옐로우골드":
        # Only for pure gold colors - warm enhancement
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.05, 0, 255)  # Red
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.03, 0, 255)  # Green
        image = Image.fromarray(img_array)
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)
        
    elif detected_color == "로즈골드":
        # Pink tone enhancement
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.06, 0, 255)  # Red
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.96, 0, 255)  # Less blue
        image = Image.fromarray(img_array)
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.05)
        
    elif detected_color == "화이트골드":
        # Cool metallic enhancement
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.03, 0, 255)  # Blue
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 0.98, 0, 255)  # Less red
        image = Image.fromarray(img_array)
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.1)
    
    return image

def find_enhanced_image_data(data, path="root"):
    """Recursively find enhanced_image data from any structure - V76 Perfect"""
    logger.info(f"Searching in path: {path}")
    
    if data is None:
        return None
    
    # If it's a string, it might be our image data
    if isinstance(data, str):
        if len(data) > 100:  # Likely base64 data
            logger.info(f"Found potential image data at {path} (string)")
            return data
        return None
    
    # If it's a dict, check all possible keys
    if isinstance(data, dict):
        # Direct image keys
        image_keys = ['enhanced_image', 'image', 'image_data', 'base64_image', 
                     'imageBase64', 'image_base64', 'base64']
        
        for key in image_keys:
            if key in data and data[key]:
                logger.info(f"Found image data at {path}.{key}")
                return data[key]
        
        # Special handling for numbered keys (like '4')
        for key in data:
            if key.isdigit() or key in ['input', 'data', 'output', 'payload', 'job']:
                result = find_enhanced_image_data(data[key], f"{path}.{key}")
                if result:
                    return result
        
        # Try all other keys
        for key, value in data.items():
            if key not in image_keys:
                result = find_enhanced_image_data(value, f"{path}.{key}")
                if result:
                    return result
    
    # If it's a list, check all items
    if isinstance(data, list):
        for i, item in enumerate(data):
            result = find_enhanced_image_data(item, f"{path}[{i}]")
            if result:
                return result
    
    return None

def process_enhancement(job):
    """Process enhancement request"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    logger.info(f"Input structure: {json.dumps(job, indent=2)[:500]}...")  # Log first 500 chars
    
    try:
        # Find image data using perfect recursive search
        image_data = find_enhanced_image_data(job)
        
        if not image_data:
            # Log the entire structure for debugging
            logger.error(f"No image data found. Full structure: {json.dumps(job, indent=2)}")
            return {
                "output": {
                    "error": "No image data found in input",
                    "status": "error",
                    "version": VERSION,
                    "searched_structure": json.dumps(job, indent=2)[:1000]
                }
            }
        
        logger.info(f"Found image data, length: {len(image_data)}")
        
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
        
        logger.info(f"Original image: {image.size}")
        
        # Detect color FIRST with ultra-conservative logic
        detected_color = detect_ring_color(image)
        logger.info(f"Detected color: {detected_color}")
        
        # Basic enhancement - brighter overall
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.12)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.08)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(1.05)
        
        # Apply color-specific enhancement
        image = apply_color_enhancement(image, detected_color)
        
        # Apply subtle center focus
        image = apply_center_focus(image, strength=0.1)
        
        # Light sharpening
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.4)
        
        # Save to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=True)
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        enhanced_base64_no_padding = enhanced_base64.rstrip('=')
        
        logger.info(f"Enhancement completed successfully")
        
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
        logger.error(traceback.format_exc())
        
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

# RunPod handler
runpod.serverless.start({"handler": process_enhancement})
