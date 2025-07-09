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

################################
# ENHANCEMENT HANDLER - 1200x1560
# VERSION: V10.7-Delicate-Ring-Preservation  
################################

VERSION = "V10.7-Delicate-Ring-Preservation"

# ===== REPLICATE INITIALIZATION =====
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

def extract_file_number(filename: str) -> str:
    """Extract number from filename"""
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
    """Find input data - OPTIMIZED VERSION"""
    if isinstance(data, str) and len(data) > 50:
        return data
    
    if isinstance(data, dict):
        # Priority image keys
        priority_keys = ['enhanced_image', 'image', 'image_base64', 'base64', 'img']
        
        # Check priority keys first
        for key in priority_keys:
            if key in data and isinstance(data[key], str) and len(data[key]) > 50:
                logger.info(f"Found base64 at key: {key}")
                return data[key]
        
        # Check nested structures (limited depth)
        for key in ['input', 'data', 'payload']:
            if key in data:
                if isinstance(data[key], str) and len(data[key]) > 50:
                    return data[key]
                elif isinstance(data[key], dict):
                    result = find_input_data_fast(data[key])
                    if result:
                        return result
        
        # Check numeric keys (Make.com)
        for i in range(10):
            key = str(i)
            if key in data and isinstance(data[key], str) and len(data[key]) > 50:
                return data[key]
    
    return None

def find_filename_fast(data):
    """Find filename - OPTIMIZED"""
    if isinstance(data, dict):
        for key in ['filename', 'file_name', 'name']:
            if key in data and isinstance(data[key], str):
                return data[key]
        
        # Check nested only 1 level deep
        for key in ['input', 'data']:
            if key in data and isinstance(data[key], dict):
                for subkey in ['filename', 'file_name', 'name']:
                    if subkey in data[key] and isinstance(data[key][subkey], str):
                        return data[key][subkey]
    
    return None

def decode_base64_fast(base64_str: str) -> bytes:
    """FAST base64 decode - Supports PNG with transparency"""
    try:
        if not base64_str or len(base64_str) < 50:
            raise ValueError("Invalid base64 string")
        
        # Quick clean
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[-1]
        
        # Remove whitespace
        base64_str = ''.join(base64_str.split())
        
        # Keep only valid chars
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        base64_str = ''.join(c for c in base64_str if c in valid_chars)
        
        # Make.com style - try without padding first
        no_pad = base64_str.rstrip('=')
        
        try:
            # Try no padding first (Make.com default)
            decoded = base64.b64decode(no_pad, validate=False)
            return decoded
        except:
            # Try with correct padding
            padding_needed = (4 - len(no_pad) % 4) % 4
            padded = no_pad + ('=' * padding_needed)
            decoded = base64.b64decode(padded, validate=False)
            return decoded
            
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise ValueError(f"Invalid base64 data: {str(e)}")

def detect_pattern_type(filename: str) -> str:
    """Detect pattern type - SIMPLIFIED to ac_ and others"""
    if not filename:
        return "other"
    
    filename_lower = filename.lower()
    
    # Only ac_ is special (무도금화이트)
    if 'ac_' in filename_lower:
        return "ac_pattern"
    else:
        return "other"

def create_background(size, color="#C8C8C8", style="gradient"):
    """Create natural gray background for jewelry - V10.4 BALANCED"""
    width, height = size
    
    if style == "gradient":
        # Create radial gradient background
        background = Image.new('RGB', size, color)
        bg_array = np.array(background, dtype=np.float32)
        
        # Create radial gradient
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width / 2, height / 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max(width, height)
        
        # Very subtle gradient for natural look
        gradient = 1 - (distance * 0.05)  # Only 5% darkening at edges
        gradient = np.clip(gradient, 0.95, 1.0)
        
        # Apply gradient
        bg_array *= gradient[:, :, np.newaxis]
        
        return Image.fromarray(bg_array.astype(np.uint8))
    else:
        # Simple solid color
        return Image.new('RGB', size, color)

def remove_background_with_delicate_settings(image: Image.Image) -> Image.Image:
    """Remove background with DELICATE settings for wedding rings - V10.7"""
    try:
        # Try local rembg FIRST with DELICATE settings
        from rembg import remove, new_session
        
        logger.info("🔷 Removing background with BiRefNet - DELICATE settings for wedding rings")
        
        # Session caching for speed
        if not hasattr(remove_background_with_delicate_settings, 'session'):
            logger.info("Initializing BiRefNet-general session for delicate processing...")
            # Use general model instead of lite for better quality
            remove_background_with_delicate_settings.session = new_session('birefnet-general')
        
        # Convert PIL Image to bytes
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        
        # Remove background with DELICATE settings for rings
        output = remove(
            buffered.getvalue(), 
            session=remove_background_with_delicate_settings.session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=200,  # LOWERED from 240 - more inclusive
            alpha_matting_background_threshold=80,   # RAISED from 50 - less aggressive
            alpha_matting_erode_size=2              # REDUCED from 8 - minimal erosion
        )
        
        # Convert result to PIL Image
        result_image = Image.open(BytesIO(output))
        
        # Apply ring protection
        if result_image.mode == 'RGBA':
            result_image = protect_wedding_ring_edges(result_image)
            result_image = ensure_ring_holes_transparent_delicate(result_image)
        
        logger.info("✅ Background removal successful with delicate settings")
        return result_image
        
    except Exception as e:
        logger.error(f"BiRefNet failed, trying alternative methods: {e}")
        
        # Try alternative local model
        try:
            from rembg import remove, new_session
            
            logger.info("🔷 Trying u2netp model for better ring preservation")
            
            if not hasattr(remove_background_with_delicate_settings, 'u2netp_session'):
                remove_background_with_delicate_settings.u2netp_session = new_session('u2netp')
            
            output = remove(
                buffered.getvalue(),
                session=remove_background_with_delicate_settings.u2netp_session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=180,  # Even more inclusive
                alpha_matting_background_threshold=100,  # Very conservative
                alpha_matting_erode_size=1              # Minimal erosion
            )
            
            result_image = Image.open(BytesIO(output))
            
            if result_image.mode == 'RGBA':
                result_image = protect_wedding_ring_edges(result_image)
                result_image = ensure_ring_holes_transparent_delicate(result_image)
            
            return result_image
            
        except Exception as e2:
            logger.error(f"u2netp also failed: {e2}")
        
        # Fallback: Replicate method with DELICATE settings
        if USE_REPLICATE and REPLICATE_CLIENT:
            try:
                logger.info("🔷 Fallback to Replicate rembg with delicate settings")
                
                # Convert to base64
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                buffered.seek(0)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                img_data_url = f"data:image/png;base64,{img_base64}"
                
                # Use rembg model with DELICATE settings
                output = REPLICATE_CLIENT.run(
                    "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
                    input={
                        "image": img_data_url,
                        "model": "u2netp",  # Better for detailed objects
                        "alpha_matting": True,
                        "alpha_matting_foreground_threshold": 180,  # More inclusive
                        "alpha_matting_background_threshold": 100,   # Conservative
                        "alpha_matting_erode_size": 2,              # Minimal
                        "return_mask": False
                    }
                )
                
                if output:
                    if isinstance(output, str):
                        response = requests.get(output)
                        result_image = Image.open(BytesIO(response.content))
                    else:
                        result_image = Image.open(BytesIO(base64.b64decode(output)))
                    
                    if result_image.mode == 'RGBA':
                        result_image = protect_wedding_ring_edges(result_image)
                        result_image = ensure_ring_holes_transparent_delicate(result_image)
                    
                    return result_image
            except Exception as e3:
                logger.error(f"Replicate also failed: {e3}")
        
        # Final fallback: return original
        logger.warning("All background removal methods failed, returning original")
        return image

def protect_wedding_ring_edges(image: Image.Image) -> Image.Image:
    """Protect wedding ring edges from erosion - NEW in V10.7"""
    if image.mode != 'RGBA':
        return image
    
    logger.info("🛡️ Protecting wedding ring edges")
    
    # Get alpha channel
    r, g, b, a = image.split()
    alpha_array = np.array(a, dtype=np.uint8)
    
    # Detect ring edges (high gradient areas)
    grad_x = cv2.Sobel(alpha_array, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(alpha_array, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Find strong edges (likely ring boundaries)
    strong_edges = gradient_mag > 50
    
    # Dilate edge areas to create protection zone
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    protection_zone = cv2.dilate(strong_edges.astype(np.uint8), kernel, iterations=2)
    
    # Ensure these areas remain opaque
    alpha_array[protection_zone > 0] = np.maximum(alpha_array[protection_zone > 0], 200)
    
    # Smooth the alpha channel slightly
    alpha_array = cv2.GaussianBlur(alpha_array, (3, 3), 1.0)
    
    # Create new image with protected alpha
    a_new = Image.fromarray(alpha_array)
    return Image.merge('RGBA', (r, g, b, a_new))

def ensure_ring_holes_transparent_delicate(image: Image.Image) -> Image.Image:
    """DELICATE ring hole detection - V10.7 for wedding rings"""
    if image.mode != 'RGBA':
        return image
    
    logger.info("🔍 Delicate hole detection V10.7 started")
    
    # Get alpha channel
    r, g, b, a = image.split()
    alpha_array = np.array(a, dtype=np.uint8)
    
    h, w = alpha_array.shape
    
    # Create a copy for modifications
    alpha_modified = alpha_array.copy()
    
    # STAGE 1: Very careful hole detection
    # Only look for very clear holes (very low alpha values)
    hole_mask = np.zeros_like(alpha_array, dtype=bool)
    
    # Only very transparent areas
    for threshold in [30, 50, 70]:
        potential_holes = (alpha_array < threshold)
        hole_mask = hole_mask | potential_holes
    
    # STAGE 2: Minimal morphological operations
    kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Very gentle closing
    hole_mask = cv2.morphologyEx(hole_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_tiny, iterations=1)
    
    # STAGE 3: Find connected components with strict criteria
    num_labels, labels = cv2.connectedComponents(hole_mask)
    
    # STAGE 4: Only process very clear holes
    holes_found = 0
    for label in range(1, num_labels):
        hole_candidate = (labels == label)
        hole_size = np.sum(hole_candidate)
        
        # Skip very small noise
        if hole_size < 50:
            continue
            
        # Get component properties
        coords = np.where(hole_candidate)
        if len(coords[0]) == 0:
            continue
            
        min_y, max_y = coords[0].min(), coords[0].max()
        min_x, max_x = coords[1].min(), coords[1].max()
        center_y = (min_y + max_y) // 2
        center_x = (min_x + max_x) // 2
        
        # Check if it's truly inside the ring (not at edges)
        margin = 0.15  # 15% margin from edges
        if not (margin * h < center_y < (1-margin) * h and margin * w < center_x < (1-margin) * w):
            continue
        
        # Check average alpha value in the region
        avg_alpha = np.mean(alpha_array[hole_candidate])
        
        # Only process if it's really transparent
        if avg_alpha < 50:  # Very transparent
            # Make it fully transparent but don't expand
            alpha_modified[hole_candidate] = 0
            holes_found += 1
            logger.info(f"Found clear hole {holes_found} at ({center_x}, {center_y}), avg_alpha: {avg_alpha:.1f}")
    
    logger.info(f"✅ Delicate hole detection complete - found {holes_found} holes")
    
    # Create new image with corrected alpha
    a_new = Image.fromarray(alpha_modified)
    return Image.merge('RGBA', (r, g, b, a_new))

def composite_with_natural_edge(image, background_color="#C8C8C8"):
    """Natural composite with soft edges - V10.4 BALANCED"""
    if image.mode == 'RGBA':
        # Create background
        background = create_background(image.size, background_color, style="gradient")
        
        # Get channels
        r, g, b, a = image.split()
        
        # Convert to arrays
        fg_array = np.array(image.convert('RGB'), dtype=np.float32)
        bg_array = np.array(background, dtype=np.float32)
        alpha_array = np.array(a, dtype=np.float32) / 255.0
        
        # More aggressive edge softening for natural blend
        alpha_soft = cv2.GaussianBlur(alpha_array, (7, 7), 2.0)  # Increased blur
        
        # Use more soft edge for smoother transition
        alpha_final = alpha_array * 0.6 + alpha_soft * 0.4  # More soft blend
        
        # Apply additional feathering at edges
        alpha_final = cv2.GaussianBlur(alpha_final, (3, 3), 1.0)
        
        # Simple alpha blending
        for i in range(3):
            bg_array[:,:,i] = fg_array[:,:,i] * alpha_final + bg_array[:,:,i] * (1 - alpha_final)
        
        # Convert back
        result = Image.fromarray(bg_array.astype(np.uint8))
        return result
    else:
        # Already has background
        return image

def apply_swinir_enhancement_after_resize(image: Image.Image) -> Image.Image:
    """Apply SwinIR AFTER resize - NEW APPROACH"""
    if not USE_REPLICATE or not REPLICATE_CLIENT:
        return image
    
    try:
        width, height = image.size
        
        # Only apply if image is already resized (smaller than original)
        if width > 1500 or height > 2000:
            logger.info(f"Skipping SwinIR - image too large: {width}x{height}")
            return image
        
        logger.info(f"Applying SwinIR to resized image: {width}x{height}")
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=False)
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        logger.info("🔷 Applying SwinIR (post-resize)")
        
        # Use SwinIR with optimized settings
        output = REPLICATE_CLIENT.run(
            "jingyunliang/swinir:660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a",
            input={
                "image": img_data_url,
                "task_type": "Real-World Image Super-Resolution",
                "scale": 1,  # Keep size, enhance quality only
                "noise_level": 10,
                "jpeg_quality": 50
            }
        )
        
        if output:
            if isinstance(output, str):
                response = requests.get(output)
                enhanced_image = Image.open(BytesIO(response.content))
            else:
                enhanced_image = Image.open(BytesIO(base64.b64decode(output)))
            
            logger.info("✅ SwinIR enhancement successful")
            return enhanced_image
        else:
            return image
            
    except Exception as e:
        logger.warning(f"SwinIR error: {str(e)}")
        return image

def enhance_cubic_details_simple(image: Image.Image) -> Image.Image:
    """Enhanced cubic details with gentle sharpening - V10.4 GENTLE"""
    # Gentle contrast for better cubic visibility
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.08)  # Reduced from 1.10
    
    # Gentle detail enhancement
    image = image.filter(ImageFilter.UnsharpMask(radius=0.5, percent=120, threshold=3))  # Reduced
    
    # Subtle micro-contrast
    contrast2 = ImageEnhance.Contrast(image)
    image = contrast2.enhance(1.03)  # Reduced from 1.04
    
    # Gentle sharpness pass
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.20)  # Reduced from 1.25
    
    return image

def auto_white_balance_fast(image: Image.Image) -> Image.Image:
    """Fast white balance correction"""
    img_array = np.array(image, dtype=np.float32)
    
    # Simplified gray detection
    gray_pixels = img_array[::10, ::10]  # Sample every 10th pixel for speed
    gray_mask = (
        (np.abs(gray_pixels[:,:,0] - gray_pixels[:,:,1]) < 15) & 
        (np.abs(gray_pixels[:,:,1] - gray_pixels[:,:,2]) < 15) &
        (gray_pixels[:,:,0] > 180)
    )
    
    if np.sum(gray_mask) > 10:
        r_avg = np.mean(gray_pixels[gray_mask, 0])
        g_avg = np.mean(gray_pixels[gray_mask, 1])
        b_avg = np.mean(gray_pixels[gray_mask, 2])
        
        gray_avg = (r_avg + g_avg + b_avg) / 3
        
        img_array[:,:,0] *= (gray_avg / r_avg) if r_avg > 0 else 1
        img_array[:,:,1] *= (gray_avg / g_avg) if g_avg > 0 else 1
        img_array[:,:,2] *= (gray_avg / b_avg) if b_avg > 0 else 1
    
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

def apply_center_spotlight(image: Image.Image, intensity: float = 0.025) -> Image.Image:
    """Apply center spotlight - V10.4 REDUCED"""
    width, height = image.size
    
    # Create spotlight mask more efficiently
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width / 2, height / 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max(width, height)
    
    spotlight_mask = 1 + intensity * np.exp(-distance**2 * 3)
    spotlight_mask = np.clip(spotlight_mask, 1.0, 1.0 + intensity)
    
    img_array = np.array(image, dtype=np.float32)
    img_array *= spotlight_mask[:, :, np.newaxis]
    
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

def apply_wedding_ring_enhancement_fast(image: Image.Image) -> Image.Image:
    """Enhanced wedding ring processing with gentle cubic detail - V10.4 GENTLE"""
    # Gentle spotlight
    image = apply_center_spotlight(image, 0.020)  # Reduced from 0.025
    
    # Gentle sharpness for cubic details
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.5)  # Reduced from 1.7
    
    # Gentle contrast
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.04)  # Reduced from 1.05
    
    # Gentle multi-scale unsharp mask
    image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=110, threshold=4))  # Reduced
    
    return image

def apply_enhancement_optimized(image: Image.Image, pattern_type: str) -> Image.Image:
    """Optimized enhancement - 12% white overlay for ac_ (1차) - V10.4 REDUCED"""
    
    # Apply white overlay ONLY to ac_pattern
    if pattern_type == "ac_pattern":
        # Unplated white - 12% white overlay - REDUCED
        white_overlay = 0.12  # Reduced from 0.15
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Minimal brightness for ac_
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.005)  # Very subtle
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.98)  # Slightly desaturated
        
    else:
        # All other patterns - gentle enhancement for gold colors
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)  # Reduced from 1.12
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.99)
        
        # Gentle sharpness
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.4)  # Reduced from 1.6
    
    # Gentle center spotlight
    image = apply_center_spotlight(image, 0.025)  # Reduced from 0.035
    
    # Wedding ring enhancement
    image = apply_wedding_ring_enhancement_fast(image)
    
    return image

def calculate_quality_metrics_fast(image: Image.Image) -> dict:
    """Fast quality metrics - Sample-based"""
    # Sample every 20th pixel for speed
    img_array = np.array(image)[::20, ::20]
    
    r_avg = np.mean(img_array[:,:,0])
    g_avg = np.mean(img_array[:,:,1])
    b_avg = np.mean(img_array[:,:,2])
    
    brightness = (r_avg + g_avg + b_avg) / 3
    rgb_deviation = np.std([r_avg, g_avg, b_avg])
    cool_tone_diff = b_avg - r_avg
    
    return {
        "brightness": brightness,
        "rgb_deviation": rgb_deviation,
        "cool_tone_diff": cool_tone_diff
    }

def resize_to_target_dimensions(image: Image.Image, target_width=1200, target_height=1560) -> Image.Image:
    """Resize image to exact target dimensions maintaining aspect ratio"""
    width, height = image.size
    
    # Calculate the aspect ratios
    img_ratio = width / height
    target_ratio = target_width / target_height
    
    # Expected input is around 2000x2600 (ratio ~0.769)
    expected_ratio = 2000 / 2600
    
    logger.info(f"Input size: {width}x{height}, ratio: {img_ratio:.3f}")
    
    # If close to expected ratio, direct resize
    if abs(img_ratio - expected_ratio) < 0.05:  # 5% tolerance
        logger.info("Direct resize - ratio matches expected")
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Otherwise, resize to fit while maintaining aspect ratio
    if img_ratio > target_ratio:
        # Image is wider - fit to height
        new_height = target_height
        new_width = int(target_height * img_ratio)
    else:
        # Image is taller - fit to width
        new_width = target_width
        new_height = int(target_width / img_ratio)
    
    # Resize maintaining aspect ratio
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Crop to exact dimensions (center crop)
    if new_width != target_width or new_height != target_height:
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        resized = resized.crop((left, top, right, bottom))
        logger.info(f"Center cropped to {target_width}x{target_height}")
    
    return resized

def process_enhancement(job):
    """Main enhancement processing - V10.7 with delicate ring preservation"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    
    try:
        # Fast extraction
        filename = find_filename_fast(job)
        file_number = extract_file_number(filename) if filename else None
        image_data = find_input_data_fast(job)
        
        # Light gray background - BALANCED V10.4
        background_color = '#C8C8C8'  # Darker gray for better contrast
        
        if not image_data:
            return {
                "output": {
                    "error": "No image data found",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Fast decode
        image_bytes = decode_base64_fast(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # STEP 1: BACKGROUND REMOVAL (PNG files only)
        original_mode = image.mode
        has_transparency = image.mode == 'RGBA'
        needs_background_removal = False
        
        if filename and filename.lower().endswith('.png'):
            logger.info("📸 STEP 1: PNG detected - removing background with V10.7 delicate settings")
            image = remove_background_with_delicate_settings(image)
            has_transparency = image.mode == 'RGBA'
            needs_background_removal = True
        
        # Keep transparent version for later
        if has_transparency:
            original_transparent = image.copy()
        
        # Convert to RGB for processing
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                # Temporary white background for processing
                temp_bg = Image.new('RGB', image.size, (255, 255, 255))
                temp_bg.paste(image, mask=image.split()[3])
                image = temp_bg
            else:
                image = image.convert('RGB')
        
        original_size = image.size
        logger.info(f"Original size: {original_size}")
        
        # STEP 2: ENHANCEMENT (GENTLE)
        logger.info("🎨 STEP 2: Applying gentle enhancements")
        
        # Fast white balance
        image = auto_white_balance_fast(image)
        
        # Detect pattern
        pattern_type = detect_pattern_type(filename)
        detected_type = {
            "ac_pattern": "무도금화이트(0.12/0.15)",
            "other": "기타색상(no_overlay)"
        }.get(pattern_type, "기타색상(no_overlay)")
        
        logger.info(f"Pattern: {pattern_type}")
        
        # Enhanced cubic details (gentle)
        image = enhance_cubic_details_simple(image)
        
        # Gentle basic enhancement - V10.4
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)  # Reduced from 1.12
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.05)  # Reduced from 1.06
        
        # Apply pattern-specific enhancement
        image = apply_enhancement_optimized(image, pattern_type)
        
        # RESIZE with proper aspect ratio handling
        logger.info(f"Resizing from {image.size} to 1200x1560")
        image = resize_to_target_dimensions(image, 1200, 1560)
        
        # Apply SwinIR AFTER resize
        swinir_applied = False
        if USE_REPLICATE:
            try:
                logger.info("Applying SwinIR enhancement")
                image = apply_swinir_enhancement_after_resize(image)
                swinir_applied = True
            except Exception as e:
                logger.warning(f"SwinIR failed: {str(e)}")
        
        # Gentle final sharpening - V10.4
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.6)  # Reduced from 1.8
        
        # STEP 3: BACKGROUND COMPOSITE (if transparent)
        if has_transparency and 'original_transparent' in locals():
            logger.info(f"🖼️ STEP 3: Natural background compositing: {background_color}")
            # Apply all enhancements to transparent version
            enhanced_transparent = original_transparent.copy()
            
            # Resize transparent version with proper aspect ratio
            enhanced_transparent = resize_to_target_dimensions(enhanced_transparent, 1200, 1560)
            
            # Apply enhancements to RGBA
            if enhanced_transparent.mode == 'RGBA':
                # Delicate hole detection - V10.7
                enhanced_transparent = ensure_ring_holes_transparent_delicate(enhanced_transparent)
                
                # Split channels
                r, g, b, a = enhanced_transparent.split()
                rgb_image = Image.merge('RGB', (r, g, b))
                
                # Apply same gentle enhancements
                rgb_image = auto_white_balance_fast(rgb_image)
                rgb_image = enhance_cubic_details_simple(rgb_image)
                brightness = ImageEnhance.Brightness(rgb_image)
                rgb_image = brightness.enhance(1.08)  # Reduced
                contrast = ImageEnhance.Contrast(rgb_image)
                rgb_image = contrast.enhance(1.05)  # Reduced
                sharpness = ImageEnhance.Sharpness(rgb_image)
                rgb_image = sharpness.enhance(1.6)  # Reduced from 1.8
                
                # Pattern-specific enhancement
                if pattern_type == "ac_pattern":
                    # 12% white overlay - V10.4
                    white_overlay = 0.12  # Reduced
                    img_array = np.array(rgb_image, dtype=np.float32)
                    img_array = img_array * (1 - white_overlay) + 255 * white_overlay
                    rgb_image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
                
                # Merge back with alpha
                r2, g2, b2 = rgb_image.split()
                enhanced_transparent = Image.merge('RGBA', (r2, g2, b2, a))
            
            # Natural composite with soft edges
            image = composite_with_natural_edge(enhanced_transparent, background_color)
            
            # Final touch after compositing
            sharpness = ImageEnhance.Sharpness(image)
            image = sharpness.enhance(1.10)  # Very subtle
        
        # Save to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=False, quality=95)
        buffered.seek(0)
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        enhanced_base64_no_padding = enhanced_base64.rstrip('=')
        
        # Build filename
        enhanced_filename = filename
        if filename and file_number:
            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            extension = filename.rsplit('.', 1)[1] if '.' in filename else 'jpg'
            enhanced_filename = f"{base_name}_enhanced.{extension}"
        
        # Quality check for ac_pattern (2차 처리)
        if pattern_type == "ac_pattern":
            metrics = calculate_quality_metrics_fast(image)
            if metrics["brightness"] < 235:  # Lowered threshold
                # Apply 15% white overlay as correction - V10.4
                white_overlay = 0.15  # Reduced from 0.18
                img_array = np.array(image, dtype=np.float32)
                img_array = img_array * (1 - white_overlay) + 255 * white_overlay
                img_array = np.clip(img_array, 0, 255)
                image = Image.fromarray(img_array.astype(np.uint8))
                
                # Re-encode
                buffered = BytesIO()
                image.save(buffered, format="PNG", optimize=False, quality=95)
                buffered.seek(0)
                enhanced_base64_no_padding = base64.b64encode(buffered.getvalue()).decode('utf-8').rstrip('=')
        
        output = {
            "output": {
                "enhanced_image": enhanced_base64_no_padding,
                "enhanced_image_with_prefix": f"data:image/png;base64,{enhanced_base64_no_padding}",
                "detected_type": detected_type,
                "pattern_type": pattern_type,
                "is_wedding_ring": True,
                "filename": filename,
                "enhanced_filename": enhanced_filename,
                "file_number": file_number,
                "original_size": list(original_size),
                "final_size": list(image.size),
                "version": VERSION,
                "status": "success",
                "white_overlay": "12% for ac_ (1차), 15% (2차)",
                "brightness_increased": "8% all patterns",
                "contrast_increased": "5%",
                "sharpness_increased": "1.6",
                "spotlight_increased": "2.0%",
                "cubic_enhancement": "Gentle (120% unsharp)",
                "swinir_applied": swinir_applied,
                "swinir_timing": "AFTER resize",
                "png_support": True,
                "has_transparency": has_transparency,
                "background_composite": has_transparency,
                "background_removal": needs_background_removal,
                "background_color": background_color,
                "background_style": "Natural gray (#C8C8C8)",
                "gradient_edge_darkening": "5%",
                "edge_processing": "Natural soft edge (60/40 blend + double feather)",
                "composite_method": "Simple alpha blending",
                "rembg_settings": "DELICATE settings for wedding rings",
                "background_removal_models": [
                    "birefnet-general (primary)",
                    "u2netp (secondary)",
                    "replicate u2netp (fallback)"
                ],
                "delicate_settings": {
                    "foreground_threshold": "180-200 (lowered)",
                    "background_threshold": "80-100 (raised)",
                    "erode_size": "1-2 (minimal)"
                },
                "ring_protection": "Edge protection + gradient detection",
                "hole_detection": "V10.7 Delicate - only very clear holes",
                "hole_threshold": "avg_alpha < 50",
                "resize_method": "Aspect ratio aware with center crop",
                "processing_order": "1.Delicate Background Removal → 2.Gentle Enhancement → 3.Natural Composite",
                "quality": "95",
                "expected_input": "2000x2600 (±30px tolerance)",
                "output_size": "1200x1560",
                "safety_features": "Wedding ring preservation priority"
            }
        }
        
        logger.info("✅ Enhancement completed successfully")
        return output
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        
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
