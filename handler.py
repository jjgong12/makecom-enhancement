import runpod
import os
import sys
import json
import time
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import logging
from typing import Dict, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "V69-PureWhite"

def decode_base64_safe(base64_str: str) -> bytes:
    """Safely decode base64 with automatic padding correction"""
    # Remove data URI prefix if present
    if base64_str.startswith('data:'):
        base64_str = base64_str.split(',')[1]
    
    # Remove any whitespace
    base64_str = base64_str.strip()
    
    # Fix padding if necessary
    padding = 4 - len(base64_str) % 4
    if padding != 4:
        base64_str += '=' * padding
    
    try:
        return base64.b64decode(base64_str)
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise ValueError(f"Invalid base64 data: {str(e)}")

def detect_ring_color(image: Image.Image) -> str:
    """Detect ring color with improved accuracy"""
    logger.info("Starting color detection...")
    
    # Convert to numpy array
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Focus on center area (50% of image)
    center_y, center_x = height // 2, width // 2
    crop_size = min(height, width) // 2
    
    y1 = max(0, center_y - crop_size // 2)
    y2 = min(height, center_y + crop_size // 2)
    x1 = max(0, center_x - crop_size // 2)
    x2 = min(width, center_x + crop_size // 2)
    
    center_region = img_array[y1:y2, x1:x2]
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
    
    # Calculate color metrics
    avg_saturation = np.mean(hsv[:, :, 1])
    avg_value = np.mean(hsv[:, :, 2])
    
    # Convert to RGB for color tone analysis
    rgb_center = center_region.astype(float)
    r_mean = np.mean(rgb_center[:, :, 0])
    g_mean = np.mean(rgb_center[:, :, 1])
    b_mean = np.mean(rgb_center[:, :, 2])
    
    # Normalize RGB values
    max_rgb = max(r_mean, g_mean, b_mean)
    if max_rgb > 0:
        r_norm = r_mean / max_rgb
        g_norm = g_mean / max_rgb
        b_norm = b_mean / max_rgb
    else:
        r_norm = g_norm = b_norm = 1.0
    
    logger.info(f"Color metrics - Sat: {avg_saturation:.1f}, Val: {avg_value:.1f}")
    logger.info(f"RGB normalized: R={r_norm:.2f}, G={g_norm:.2f}, B={b_norm:.2f}")
    
    # Detection logic
    if avg_saturation < 25:  # Low saturation
        if avg_value > 200:  # Very bright
            color = "무도금화이트"
        else:
            color = "화이트골드"
    elif r_norm > 0.95 and g_norm > 0.85 and g_norm < 0.95:  # Warm tones
        if avg_saturation > 40:
            color = "로즈골드"
        else:
            color = "옐로우골드"
    elif abs(r_norm - g_norm) < 0.1 and abs(g_norm - b_norm) < 0.1:  # Neutral
        color = "화이트골드"
    else:
        # Default based on warmth
        warmth = (r_norm + g_norm) / 2 - b_norm
        if warmth > 0.1:
            color = "옐로우골드"
        else:
            color = "화이트골드"
    
    logger.info(f"Detected color: {color}")
    return color

def create_center_focus_mask(image: Image.Image, focus_strength: float = 0.3) -> np.ndarray:
    """Create a center focus mask for subtle vignetting"""
    width, height = image.size
    
    # Create radial gradient
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width / 2, height / 2
    
    # Calculate distance from center
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2) * 0.7
    
    # Create smooth gradient mask
    mask = 1 - (dist / max_dist) * focus_strength
    mask = np.clip(mask, 1 - focus_strength, 1)
    
    # Apply gaussian blur for smooth transition
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    
    return mask

def enhance_wedding_ring_image(image_input: Union[str, bytes], mask_type: str = "none") -> Tuple[str, str]:
    """Enhanced wedding ring image processing for V69"""
    try:
        # Handle different input types
        if isinstance(image_input, str):
            if image_input.startswith('http'):
                # URL input - not implemented in this version
                raise ValueError("URL input not supported")
            else:
                # Base64 input
                image_bytes = decode_base64_safe(image_input)
        else:
            # Direct bytes input
            image_bytes = image_input
        
        # Open image
        image = Image.open(BytesIO(image_bytes))
        original_mode = image.mode
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Processing image: {image.size}, mode: {original_mode}")
        
        # Step 1: Detect ring color from original
        detected_color = detect_ring_color(image)
        
        # Step 2: Basic enhancement
        # Overall brightness adjustment
        brightness = ImageEnhance.Brightness(image)
        if detected_color == "무도금화이트":
            image = brightness.enhance(1.15)  # Brighter for pure white
        else:
            image = brightness.enhance(1.08)
        
        # Gentle contrast
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.05)
        
        # Slight color enhancement
        color = ImageEnhance.Color(image)
        if detected_color == "무도금화이트":
            image = color.enhance(0.7)  # Reduce saturation for pure white
        else:
            image = color.enhance(1.02)
        
        # Step 3: Create center focus
        img_array = np.array(image)
        focus_mask = create_center_focus_mask(image, focus_strength=0.2)
        
        # Apply center focus
        for i in range(3):
            img_array[:, :, i] = img_array[:, :, i] * focus_mask
        
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Step 4: Color-specific enhancements
        if detected_color == "무도금화이트":
            # Pure white enhancement
            # Convert to LAB for better white control
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(img_cv)
            
            # Increase lightness significantly
            l = cv2.multiply(l, 1.15)
            l = np.clip(l, 0, 255)
            
            # Reduce color channels for pure white
            a = cv2.multiply(a, 0.7)
            b = cv2.multiply(b, 0.7)
            
            img_cv = cv2.merge([l, a, b])
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_LAB2RGB)
            image = Image.fromarray(img_rgb)
            
            # Additional brightness for pure white
            brightness = ImageEnhance.Brightness(image)
            image = brightness.enhance(1.1)
            
            # Remove any remaining yellow tint
            img_array = np.array(image)
            # Boost blue channel slightly to counteract yellow
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.05, 0, 255)
            image = Image.fromarray(img_array.astype(np.uint8))
            
        elif detected_color == "옐로우골드":
            # Enhance golden tones
            img_array = np.array(image)
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.05, 0, 255)
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.03, 0, 255)
            image = Image.fromarray(img_array.astype(np.uint8))
            
        elif detected_color == "로즈골드":
            # Enhance rose tones
            img_array = np.array(image)
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.08, 0, 255)
            image = Image.fromarray(img_array.astype(np.uint8))
        
        # Step 5: Final sharpening
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=3))
        
        # Create thumbnail
        thumbnail = create_thumbnail(image, detected_color)
        
        # Convert to base64
        # Enhanced image
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Thumbnail
        thumb_buffered = BytesIO()
        thumbnail.save(thumb_buffered, format="JPEG", quality=95)
        thumbnail_base64 = base64.b64encode(thumb_buffered.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        enhanced_base64 = enhanced_base64.rstrip('=')
        thumbnail_base64 = thumbnail_base64.rstrip('=')
        
        return enhanced_base64, thumbnail_base64
        
    except Exception as e:
        logger.error(f"Enhancement error: {str(e)}")
        raise

def create_thumbnail(image: Image.Image, detected_color: str) -> Image.Image:
    """Create a simple 400x400 thumbnail"""
    # Center crop to square
    width, height = image.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    
    thumbnail = image.crop((left, top, right, bottom))
    thumbnail = thumbnail.resize((400, 400), Image.Resampling.LANCZOS)
    
    return thumbnail

def find_input_data(data):
    """Recursively find input data in nested structure"""
    if isinstance(data, dict):
        # Direct input
        if 'image' in data or 'image_url' in data or 'image_base64' in data:
            return data
        
        # Nested input
        if 'input' in data:
            return find_input_data(data['input'])
        
        # Try other keys
        for key in ['job', 'payload', 'data']:
            if key in data:
                result = find_input_data(data[key])
                if result:
                    return result
    
    return data

def handler(job):
    """RunPod handler function"""
    start_time = time.time()
    
    try:
        logger.info(f"[{VERSION}] Job received")
        logger.info(f"Job structure: {json.dumps(job, indent=2)[:500]}...")
        
        # Extract input
        job_input = job.get('input', {})
        input_data = find_input_data(job_input)
        
        logger.info(f"Processing with input keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'non-dict input'}")
        
        # Get image input
        image_input = None
        if isinstance(input_data, dict):
            image_input = input_data.get('image') or input_data.get('image_base64') or input_data.get('image_url')
        else:
            image_input = input_data
        
        if not image_input:
            raise ValueError("No image provided")
        
        # Process image
        enhanced_base64, thumbnail_base64 = enhance_wedding_ring_image(image_input)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return result - V69 no mask data
        result = {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_time": f"{processing_time:.2f}s",
                "version": VERSION,
                "success": True
            }
        }
        
        logger.info(f"Successfully processed in {processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        error_response = {
            "output": {
                "error": str(e),
                "success": False,
                "processing_time": f"{time.time() - start_time:.2f}s",
                "version": VERSION
            }
        }
        return error_response

# RunPod handler
runpod.serverless.start({"handler": handler})
