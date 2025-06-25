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

VERSION = "V73-ConservativeYellowGold"

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
    """Improved color detection with better yellow/rose gold distinction"""
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
    
    # Improved color detection logic - 무도금화이트 우선, 옐로우골드 보수적
    if avg_saturation < 40 and avg_value > 200:
        # Wider range for 무도금화이트 (includes slightly warm whites)
        return "무도금화이트"
    elif avg_saturation < 50 and avg_value > 180:
        # Medium saturation + high brightness = 화이트골드
        return "화이트골드"
    elif avg_hue >= 20 and avg_hue <= 35 and avg_saturation > 60 and gb_ratio > 1.2:
        # Very strict yellow gold - high saturation + clear yellow hue
        return "옐로우골드"
    elif rg_ratio > 1.2 and rb_ratio > 1.3 and avg_hue < 15:
        # Clear red dominance = 로즈골드
        return "로즈골드"
    elif avg_saturation > 70 and g_norm > 0.9 and r_norm > 0.9:
        # Very saturated warm color = 옐로우골드
        return "옐로우골드"
    else:
        # Default based on characteristics
        if avg_saturation < 45:
            return "무도금화이트"  # Default to 무도금화이트 for low saturation
        elif r_norm > g_norm * 1.1 and r_norm > b_norm * 1.1:
            return "로즈골드"
        else:
            return "화이트골드"

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
        # Warm enhancement for yellow gold
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

def find_input_data(data):
    """Find input data recursively"""
    if isinstance(data, dict):
        # Check for image data keys
        for key in ['image', 'image_data', 'base64_image', 'imageBase64', 'enhanced_image']:
            if key in data and data[key]:
                return data[key]
        
        # Check nested structures
        if 'input' in data:
            result = find_input_data(data['input'])
            if result:
                return result
        
        # Check other common nested keys
        for key in ['job', 'payload', 'data']:
            if key in data:
                result = find_input_data(data[key])
                if result:
                    return result
    
    return data if isinstance(data, str) else None

def process_enhancement(job):
    """Process enhancement request"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    
    try:
        # Find image data using recursive search
        image_data = find_input_data(job)
        
        if not image_data:
            return {
                "output": {
                    "error": "No image data found",
                    "status": "error"
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
        
        logger.info(f"Original image: {image.size}")
        
        # Detect color FIRST
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
                "traceback": traceback.format_exc()
            }
        }

# RunPod handler
runpod.serverless.start({"handler": process_enhancement})
