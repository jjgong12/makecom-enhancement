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
# VERSION: V11.0-Natural-Edge-Processing
################################

VERSION = "V11.0-Natural-Edge-Processing"

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
    """Create natural gray background for jewelry"""
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

def multi_threshold_background_removal(image: Image.Image) -> Image.Image:
    """Remove background with MULTI-THRESHOLD approach - V11.0"""
    try:
        from rembg import remove, new_session
        
        logger.info("🔷 Multi-threshold background removal V11.0")
        
        # Initialize session
        if not hasattr(multi_threshold_background_removal, 'session'):
            logger.info("Initializing BiRefNet-general session...")
            multi_threshold_background_removal.session = new_session('birefnet-general')
        
        # Convert PIL Image to bytes
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        img_data = buffered.getvalue()
        
        # Multi-threshold approach - try different thresholds
        best_result = None
        best_score = -1
        
        threshold_configs = [
            {"fg": 240, "bg": 50, "erode": 0},   # Very conservative
            {"fg": 230, "bg": 60, "erode": 1},   # Conservative
            {"fg": 220, "bg": 70, "erode": 1},   # Balanced
            {"fg": 210, "bg": 80, "erode": 2},   # Standard
            {"fg": 200, "bg": 90, "erode": 2},   # Aggressive
        ]
        
        for config in threshold_configs:
            try:
                output = remove(
                    img_data,
                    session=multi_threshold_background_removal.session,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=config["fg"],
                    alpha_matting_background_threshold=config["bg"],
                    alpha_matting_erode_size=config["erode"]
                )
                
                result_image = Image.open(BytesIO(output))
                
                # Evaluate result quality
                if result_image.mode == 'RGBA':
                    alpha = np.array(result_image.split()[3])
                    
                    # Calculate score based on edge quality
                    edge_quality = calculate_edge_quality(alpha)
                    object_preservation = np.sum(alpha > 200) / alpha.size
                    
                    score = edge_quality * 0.7 + object_preservation * 0.3
                    
                    logger.info(f"Config {config}: score={score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_result = result_image
                        
            except Exception as e:
                logger.warning(f"Threshold {config} failed: {e}")
                continue
        
        if best_result:
            # Apply natural edge processing
            best_result = apply_natural_edge_processing(best_result)
            return best_result
        else:
            logger.warning("All thresholds failed, returning original")
            return image
            
    except Exception as e:
        logger.error(f"Multi-threshold removal failed: {e}")
        return image

def calculate_edge_quality(alpha_channel):
    """Calculate edge quality score"""
    # Detect edges using Sobel
    edges = cv2.Sobel(alpha_channel, cv2.CV_64F, 1, 1, ksize=3)
    edge_magnitude = np.abs(edges)
    
    # Good edges should be smooth and continuous
    edge_smoothness = 1.0 - (np.std(edge_magnitude[edge_magnitude > 10]) / 255.0)
    
    return np.clip(edge_smoothness, 0, 1)

def apply_natural_edge_processing(image: Image.Image) -> Image.Image:
    """Apply natural edge processing to remove black outlines - V11.0"""
    if image.mode != 'RGBA':
        return image
    
    logger.info("🎨 Applying natural edge processing")
    
    r, g, b, a = image.split()
    alpha_array = np.array(a, dtype=np.float32)
    
    # 1. Edge detection
    edges = cv2.Canny(alpha_array.astype(np.uint8), 50, 150)
    edge_mask = edges > 0
    
    # 2. Create feathered edge mask
    kernel_sizes = [3, 5, 7, 9]
    feathered_alpha = alpha_array.copy()
    
    for size in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        dilated_edge = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1)
        
        # Progressive feathering
        blur_size = size * 2 - 1
        edge_alpha = cv2.GaussianBlur(alpha_array, (blur_size, blur_size), size/2)
        
        # Blend based on distance from edge
        weight = 1.0 - (size - 3) / 6.0
        feathered_alpha[dilated_edge > 0] = (
            feathered_alpha[dilated_edge > 0] * weight + 
            edge_alpha[dilated_edge > 0] * (1 - weight)
        )
    
    # 3. Remove dark pixels at edges
    rgb_array = np.array(image.convert('RGB'))
    brightness = np.mean(rgb_array, axis=2)
    
    # Find dark edge pixels
    dark_edges = (edge_mask) & (brightness < 50) & (alpha_array > 100)
    
    # Fade out dark edges
    if np.any(dark_edges):
        feathered_alpha[dark_edges] *= 0.3
    
    # 4. Final smoothing
    feathered_alpha = cv2.bilateralFilter(
        feathered_alpha.astype(np.uint8), 9, 75, 75
    ).astype(np.float32)
    
    # 5. Anti-aliasing
    # Create sub-pixel accurate alpha
    feathered_alpha = cv2.GaussianBlur(feathered_alpha, (3, 3), 0.5)
    
    # Create new image with processed alpha
    a_new = Image.fromarray(feathered_alpha.astype(np.uint8))
    return Image.merge('RGBA', (r, g, b, a_new))

def composite_with_natural_blend(image, background_color="#C8C8C8"):
    """Natural composite with perfect edge blending - V11.0"""
    if image.mode != 'RGBA':
        return image
    
    logger.info("🖼️ Natural blending V11.0")
    
    # Create background
    background = create_background(image.size, background_color, style="gradient")
    
    # Get channels
    r, g, b, a = image.split()
    
    # Convert to arrays
    fg_array = np.array(image.convert('RGB'), dtype=np.float32)
    bg_array = np.array(background, dtype=np.float32)
    alpha_array = np.array(a, dtype=np.float32) / 255.0
    
    # Multi-stage edge softening
    # Stage 1: Edge detection and expansion
    edges = cv2.Canny((alpha_array * 255).astype(np.uint8), 50, 150)
    edge_region = cv2.dilate(edges, np.ones((15, 15)), iterations=1) > 0
    
    # Stage 2: Progressive blur for edges
    alpha_soft1 = cv2.GaussianBlur(alpha_array, (5, 5), 1.5)
    alpha_soft2 = cv2.GaussianBlur(alpha_array, (9, 9), 3.0)
    alpha_soft3 = cv2.GaussianBlur(alpha_array, (15, 15), 5.0)
    
    # Stage 3: Blend different softness levels
    alpha_final = alpha_array.copy()
    
    # Apply progressive softening only to edge regions
    edge_dist = cv2.distanceTransform(
        (1 - edge_region).astype(np.uint8), 
        cv2.DIST_L2, 5
    )
    edge_dist = np.clip(edge_dist / 10.0, 0, 1)  # Normalize to 0-1
    
    # Blend based on distance from edge
    alpha_final = (
        alpha_array * edge_dist +
        alpha_soft1 * (1 - edge_dist) * 0.5 +
        alpha_soft2 * (1 - edge_dist) * 0.3 +
        alpha_soft3 * (1 - edge_dist) * 0.2
    )
    
    # Stage 4: Color spill removal
    # Remove any dark halos
    for i in range(3):
        # Detect dark regions near edges
        dark_mask = (fg_array[:,:,i] < 30) & (edge_region)
        if np.any(dark_mask):
            # Replace dark pixels with nearby bright pixels
            fg_array[dark_mask, i] = cv2.inpaint(
                fg_array[:,:,i].astype(np.uint8),
                dark_mask.astype(np.uint8),
                3, cv2.INPAINT_NS
            )[dark_mask]
    
    # Stage 5: Final composite with color correction
    result = np.zeros_like(bg_array)
    for i in range(3):
        # Premultiplied alpha blending
        result[:,:,i] = (
            fg_array[:,:,i] * alpha_final + 
            bg_array[:,:,i] * (1 - alpha_final)
        )
    
    # Stage 6: Edge color correction
    # Ensure edges match background better
    edge_mask_soft = cv2.GaussianBlur(edges.astype(np.float32), (7, 7), 2.0) / 255.0
    for i in range(3):
        result[:,:,i] = (
            result[:,:,i] * (1 - edge_mask_soft * 0.3) +
            bg_array[:,:,i] * (edge_mask_soft * 0.3)
        )
    
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

def ensure_ring_holes_transparent_multi_threshold(image: Image.Image) -> Image.Image:
    """Multi-threshold hole detection for accuracy - V11.0"""
    if image.mode != 'RGBA':
        return image
    
    logger.info("🔍 Multi-threshold hole detection V11.0")
    
    r, g, b, a = image.split()
    alpha_array = np.array(a, dtype=np.uint8)
    
    h, w = alpha_array.shape
    
    # Multi-threshold detection
    hole_mask_combined = np.zeros_like(alpha_array, dtype=bool)
    
    # Try multiple thresholds
    thresholds = range(10, 100, 10)  # 10, 20, 30, ..., 90
    
    for threshold in thresholds:
        # Find potential holes at this threshold
        potential_holes = alpha_array < threshold
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        potential_holes = cv2.morphologyEx(
            potential_holes.astype(np.uint8), 
            cv2.MORPH_OPEN, kernel
        )
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(potential_holes)
        
        for label in range(1, num_labels):
            component = (labels == label)
            component_size = np.sum(component)
            
            # Size check
            if component_size < 30 or component_size > (h * w * 0.1):
                continue
            
            # Location check - must be internal
            coords = np.where(component)
            if len(coords[0]) == 0:
                continue
                
            min_y, max_y = coords[0].min(), coords[0].max()
            min_x, max_x = coords[1].min(), coords[1].max()
            center_y = (min_y + max_y) // 2
            center_x = (min_x + max_x) // 2
            
            # Check if it's inside (not at edges)
            margin = 0.1
            if not (margin * h < center_y < (1-margin) * h and 
                    margin * w < center_x < (1-margin) * w):
                continue
            
            # Shape check - should be roughly circular
            width = max_x - min_x
            height = max_y - min_y
            aspect_ratio = width / height if height > 0 else 0
            
            if 0.5 < aspect_ratio < 2.0:  # Roughly circular
                # Confidence check
                avg_alpha = np.mean(alpha_array[component])
                if avg_alpha < threshold * 0.8:  # High confidence
                    hole_mask_combined |= component
                    logger.info(f"Found hole at threshold {threshold}, center ({center_x}, {center_y})")
    
    # Apply detected holes
    alpha_modified = alpha_array.copy()
    if np.any(hole_mask_combined):
        # Make holes fully transparent with smooth edges
        hole_mask_float = hole_mask_combined.astype(np.float32)
        hole_mask_smooth = cv2.GaussianBlur(hole_mask_float, (5, 5), 1.0)
        
        alpha_modified = alpha_array * (1 - hole_mask_smooth)
    
    # Create new image
    a_new = Image.fromarray(alpha_modified.astype(np.uint8))
    return Image.merge('RGBA', (r, g, b, a_new))

def apply_swinir_enhancement_after_resize(image: Image.Image) -> Image.Image:
    """Apply SwinIR AFTER resize"""
    if not USE_REPLICATE or not REPLICATE_CLIENT:
        return image
    
    try:
        width, height = image.size
        
        # Only apply if image is already resized
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
        
        output = REPLICATE_CLIENT.run(
            "jingyunliang/swinir:660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a",
            input={
                "image": img_data_url,
                "task_type": "Real-World Image Super-Resolution",
                "scale": 1,
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
    """Enhanced cubic details with gentle sharpening"""
    # Gentle contrast for better cubic visibility
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.08)
    
    # Gentle detail enhancement
    image = image.filter(ImageFilter.UnsharpMask(radius=0.5, percent=120, threshold=3))
    
    # Subtle micro-contrast
    contrast2 = ImageEnhance.Contrast(image)
    image = contrast2.enhance(1.03)
    
    # Gentle sharpness pass
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.20)
    
    return image

def auto_white_balance_fast(image: Image.Image) -> Image.Image:
    """Fast white balance correction"""
    img_array = np.array(image, dtype=np.float32)
    
    # Simplified gray detection
    gray_pixels = img_array[::10, ::10]  # Sample every 10th pixel
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
    """Apply center spotlight"""
    width, height = image.size
    
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width / 2, height / 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max(width, height)
    
    spotlight_mask = 1 + intensity * np.exp(-distance**2 * 3)
    spotlight_mask = np.clip(spotlight_mask, 1.0, 1.0 + intensity)
    
    img_array = np.array(image, dtype=np.float32)
    img_array *= spotlight_mask[:, :, np.newaxis]
    
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

def apply_wedding_ring_enhancement_fast(image: Image.Image) -> Image.Image:
    """Enhanced wedding ring processing"""
    # Gentle spotlight
    image = apply_center_spotlight(image, 0.020)
    
    # Gentle sharpness for cubic details
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.5)
    
    # Gentle contrast
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.04)
    
    # Gentle multi-scale unsharp mask
    image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=110, threshold=4))
    
    return image

def apply_enhancement_optimized(image: Image.Image, pattern_type: str) -> Image.Image:
    """Optimized enhancement - 12% white overlay for ac_"""
    
    # Apply white overlay ONLY to ac_pattern
    if pattern_type == "ac_pattern":
        # Unplated white - 12% white overlay
        white_overlay = 0.12
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Minimal brightness for ac_
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.005)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.98)
        
    else:
        # All other patterns - gentle enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.99)
        
        # Gentle sharpness
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.4)
    
    # Gentle center spotlight
    image = apply_center_spotlight(image, 0.025)
    
    # Wedding ring enhancement
    image = apply_wedding_ring_enhancement_fast(image)
    
    return image

def calculate_quality_metrics_fast(image: Image.Image) -> dict:
    """Fast quality metrics"""
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
    
    # Expected input is around 2000x2600
    expected_ratio = 2000 / 2600
    
    logger.info(f"Input size: {width}x{height}, ratio: {img_ratio:.3f}")
    
    # If close to expected ratio, direct resize
    if abs(img_ratio - expected_ratio) < 0.05:
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
    """Main enhancement processing - V11.0 with natural edge processing"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    
    try:
        # Fast extraction
        filename = find_filename_fast(job)
        file_number = extract_file_number(filename) if filename else None
        image_data = find_input_data_fast(job)
        
        # Light gray background
        background_color = '#C8C8C8'
        
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
            logger.info("📸 STEP 1: PNG detected - multi-threshold background removal")
            image = multi_threshold_background_removal(image)
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
        
        # STEP 2: ENHANCEMENT
        logger.info("🎨 STEP 2: Applying enhancements")
        
        # Fast white balance
        image = auto_white_balance_fast(image)
        
        # Detect pattern
        pattern_type = detect_pattern_type(filename)
        detected_type = {
            "ac_pattern": "무도금화이트(0.12/0.15)",
            "other": "기타색상(no_overlay)"
        }.get(pattern_type, "기타색상(no_overlay)")
        
        logger.info(f"Pattern: {pattern_type}")
        
        # Enhanced cubic details
        image = enhance_cubic_details_simple(image)
        
        # Basic enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.05)
        
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
        
        # Final sharpening
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.6)
        
        # STEP 3: BACKGROUND COMPOSITE (if transparent)
        if has_transparency and 'original_transparent' in locals():
            logger.info(f"🖼️ STEP 3: Natural background compositing: {background_color}")
            
            # Apply all enhancements to transparent version
            enhanced_transparent = original_transparent.copy()
            
            # Resize transparent version
            enhanced_transparent = resize_to_target_dimensions(enhanced_transparent, 1200, 1560)
            
            # Apply enhancements to RGBA
            if enhanced_transparent.mode == 'RGBA':
                # Multi-threshold hole detection
                enhanced_transparent = ensure_ring_holes_transparent_multi_threshold(enhanced_transparent)
                
                # Split channels
                r, g, b, a = enhanced_transparent.split()
                rgb_image = Image.merge('RGB', (r, g, b))
                
                # Apply same enhancements
                rgb_image = auto_white_balance_fast(rgb_image)
                rgb_image = enhance_cubic_details_simple(rgb_image)
                brightness = ImageEnhance.Brightness(rgb_image)
                rgb_image = brightness.enhance(1.08)
                contrast = ImageEnhance.Contrast(rgb_image)
                rgb_image = contrast.enhance(1.05)
                sharpness = ImageEnhance.Sharpness(rgb_image)
                rgb_image = sharpness.enhance(1.6)
                
                # Pattern-specific enhancement
                if pattern_type == "ac_pattern":
                    # 12% white overlay
                    white_overlay = 0.12
                    img_array = np.array(rgb_image, dtype=np.float32)
                    img_array = img_array * (1 - white_overlay) + 255 * white_overlay
                    rgb_image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
                
                # Merge back with alpha
                r2, g2, b2 = rgb_image.split()
                enhanced_transparent = Image.merge('RGBA', (r2, g2, b2, a))
            
            # Natural composite with perfect edge blending
            image = composite_with_natural_blend(enhanced_transparent, background_color)
            
            # Final touch after compositing
            sharpness = ImageEnhance.Sharpness(image)
            image = sharpness.enhance(1.10)
        
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
            if metrics["brightness"] < 235:
                # Apply 15% white overlay as correction
                white_overlay = 0.15
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
                "edge_processing": "Natural multi-stage edge blending V11.0",
                "composite_method": "Advanced natural blending with edge color correction",
                "background_removal_method": "Multi-threshold approach",
                "threshold_configs": "5 levels from 240/50 to 200/90",
                "edge_quality_scoring": "Automatic best result selection",
                "natural_edge_features": [
                    "Multi-kernel feathering (3,5,7,9)",
                    "Dark edge pixel removal",
                    "Progressive alpha blending",
                    "Edge color spill correction",
                    "Anti-aliasing with sub-pixel accuracy",
                    "Bilateral filtering for smoothness"
                ],
                "hole_detection": "Multi-threshold (10-90 step 10)",
                "hole_shape_check": "Aspect ratio 0.5-2.0",
                "resize_method": "Aspect ratio aware with center crop",
                "processing_order": "1.Multi-threshold BG Removal → 2.Enhancement → 3.Natural Composite",
                "quality": "95",
                "expected_input": "2000x2600 (±30px tolerance)",
                "output_size": "1200x1560",
                "safety_features": "Wedding ring preservation with natural edges"
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
