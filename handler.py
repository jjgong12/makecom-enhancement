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

VERSION = "V70-Optimized"

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
    """Detect ring color with improved accuracy"""
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
    hsv = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
    
    avg_saturation = np.mean(hsv[:, :, 1])
    avg_value = np.mean(hsv[:, :, 2])
    
    # RGB analysis
    r_mean = np.mean(center_region[:, :, 0])
    g_mean = np.mean(center_region[:, :, 1])
    b_mean = np.mean(center_region[:, :, 2])
    
    max_rgb = max(r_mean, g_mean, b_mean)
    if max_rgb > 0:
        r_norm = r_mean / max_rgb
        g_norm = g_mean / max_rgb
        b_norm = b_mean / max_rgb
    else:
        r_norm = g_norm = b_norm = 1.0
    
    # Color detection logic
    if avg_saturation < 25:
        color = "무도금화이트" if avg_value > 200 else "화이트골드"
    elif r_norm > 0.95 and 0.85 < g_norm < 0.95:
        color = "로즈골드" if avg_saturation > 40 else "옐로우골드"
    elif abs(r_norm - g_norm) < 0.1 and abs(g_norm - b_norm) < 0.1:
        color = "화이트골드"
    else:
        warmth = (r_norm + g_norm) / 2 - b_norm
        color = "옐로우골드" if warmth > 0.1 else "화이트골드"
    
    logger.info(f"Detected color: {color}")
    return color

def create_center_focus_mask(image: Image.Image, focus_strength: float = 0.15) -> np.ndarray:
    """Create subtle center focus mask"""
    width, height = image.size
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width / 2, height / 2
    
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2) * 0.7
    
    mask = 1 - (dist / max_dist) * focus_strength
    mask = np.clip(mask, 1 - focus_strength, 1)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    
    return mask

def enhance_wedding_ring_image(image_input: str) -> tuple:
    """Enhanced wedding ring image processing"""
    # Decode base64
    image_bytes = decode_base64_safe(image_input)
    image = Image.open(BytesIO(image_bytes))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Detect color
    detected_color = detect_ring_color(image)
    
    # Basic enhancement
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.15 if detected_color == "무도금화이트" else 1.08)
    
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.05)
    
    color = ImageEnhance.Color(image)
    image = color.enhance(0.7 if detected_color == "무도금화이트" else 1.02)
    
    # Apply center focus
    img_array = np.array(image)
    focus_mask = create_center_focus_mask(image, 0.15)
    for i in range(3):
        img_array[:, :, i] = img_array[:, :, i] * focus_mask
    image = Image.fromarray(img_array.astype(np.uint8))
    
    # Color-specific enhancement
    if detected_color == "무도금화이트":
        # Pure white enhancement
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_cv)
        
        l = cv2.multiply(l, 1.15)
        l = np.clip(l, 0, 255)
        a = cv2.multiply(a, 0.7)
        b = cv2.multiply(b, 0.7)
        
        img_cv = cv2.merge([l, a, b])
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_LAB2RGB)
        image = Image.fromarray(img_rgb)
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.1)
        
        # Remove yellow tint
        img_array = np.array(image)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.05, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
    
    # Final sharpening
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=3))
    
    # Create thumbnail
    thumbnail = create_thumbnail(image)
    
    # Convert to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    enhanced_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8').rstrip('=')
    
    thumb_buffered = BytesIO()
    thumbnail.save(thumb_buffered, format="JPEG", quality=95)
    thumbnail_base64 = base64.b64encode(thumb_buffered.getvalue()).decode('utf-8').rstrip('=')
    
    return enhanced_base64, thumbnail_base64

def create_thumbnail(image: Image.Image) -> Image.Image:
    """Create 400x400 thumbnail"""
    width, height = image.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    
    thumbnail = image.crop((left, top, left + size, top + size))
    return thumbnail.resize((400, 400), Image.Resampling.LANCZOS)

def handler(job):
    """RunPod handler"""
    start_time = time.time()
    
    try:
        job_input = job.get('input', {})
        
        # Find image input
        image_input = None
        if isinstance(job_input, dict):
            image_input = job_input.get('image') or job_input.get('image_base64')
        else:
            image_input = job_input
        
        if not image_input:
            raise ValueError("No image provided")
        
        # Process image
        enhanced_base64, thumbnail_base64 = enhance_wedding_ring_image(image_input)
        
        # Return result
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_time": f"{time.time() - start_time:.2f}s",
                "version": VERSION,
                "success": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "success": False,
                "version": VERSION
            }
        }

runpod.serverless.start({"handler": handler})
