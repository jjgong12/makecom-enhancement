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
# VERSION: V10.1-Consistent-Holes  
################################

VERSION = "V10.1-Consistent-Holes"

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

def create_background(size, color="#D8D8D8", style="gradient"):
    """Create natural gray background for jewelry - V10.1 BALANCED GRAY"""
    width, height = size
    
    if style == "gradient":
        # Create radial gradient background
        background = Image.new('RGB', size, color)
        bg_array = np.array(background, dtype=np.float32)
        
        # Create radial gradient
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width / 2, height / 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max(width, height)
        
        # Subtle gradient for natural look
        gradient = 1 - (distance * 0.06)  # 6% darkening at edges
        gradient = np.clip(gradient, 0.94, 1.0)
        
        # Apply gradient
        bg_array *= gradient[:, :, np.newaxis]
        
        return Image.fromarray(bg_array.astype(np.uint8))
    else:
        # Simple solid color
        return Image.new('RGB', size, color)

def remove_background_with_replicate(image: Image.Image) -> Image.Image:
    """Remove background using Replicate API - V10.1 CONSERVATIVE"""
    if not USE_REPLICATE or not REPLICATE_CLIENT:
        logger.warning("Replicate not available for background removal")
        return image
    
    try:
        logger.info("🔷 Removing background with Replicate (V10.1 conservative)")
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        # Use rembg model with CONSERVATIVE settings
        output = REPLICATE_CLIENT.run(
            "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
            input={
                "image": img_data_url,
                "model": "u2net",
                "alpha_matting": True,
                "alpha_matting_foreground_threshold": 240,  # Conservative
                "alpha_matting_background_threshold": 50,    # Conservative
                "alpha_matting_erode_size": 8               # Smaller
            }
        )
        
        if output:
            if isinstance(output, str):
                response = requests.get(output)
                result_image = Image.open(BytesIO(response.content))
            else:
                result_image = Image.open(BytesIO(base64.b64decode(output)))
            
            # MINIMAL hole processing
            if result_image.mode == 'RGBA':
                result_image = ensure_ring_holes_transparent_safe(result_image)
            
            logger.info("✅ Background removal successful")
            return result_image
        else:
            logger.warning("No output from background removal")
            return image
            
    except Exception as e:
        logger.error(f"Background removal error: {str(e)}")
        return image

def ensure_ring_holes_transparent_safe(image: Image.Image) -> Image.Image:
    """IMPROVED ring hole detection - V10.1 CONSISTENT"""
    if image.mode != 'RGBA':
        return image
    
    logger.info("🔍 Improved hole detection started")
    
    # Get alpha channel
    r, g, b, a = image.split()
    alpha_array = np.array(a, dtype=np.uint8)
    
    h, w = alpha_array.shape
    
    # Find semi-transparent areas that might be holes
    # Use higher threshold to catch partial transparency
    semi_transparent = (alpha_array < 100)  # Increased from 10
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(semi_transparent.astype(np.uint8))
    
    # Process each potential hole
    for label in range(1, num_labels):
        hole_mask = (labels == label)
        hole_size = np.sum(hole_mask)
        
        # Check if it's a reasonable hole size (not too big)
        if hole_size < (h * w * 0.1):  # Less than 10% of image
            # Get hole position
            coords = np.where(hole_mask)
            if len(coords[0]) > 0:
                center_y = int(np.mean(coords[0]))
                center_x = int(np.mean(coords[1]))
                
                # Check if hole is roughly in center area (typical ring hole position)
                if (0.3 * h < center_y < 0.7 * h) and (0.3 * w < center_x < 0.7 * w):
                    # Make this hole fully transparent
                    alpha_array[hole_mask] = 0
                    logger.info(f"Made hole at ({center_x}, {center_y}) fully transparent")
    
    # No edge processing - keep sharp edges
    
    logger.info("✅ Improved hole detection complete")
    
    # Create new image with corrected alpha
    a_new = Image.fromarray(alpha_array)
    return Image.merge('RGBA', (r, g, b, a_new))

def add_natural_edge_feathering(image: Image.Image) -> Image.Image:
    """NO EDGE PROCESSING - V10.1 REMOVED"""
    # Just return the image without any edge processing
    return image

def composite_with_light_gray_background(image, background_color="#D8D8D8"):
    """Natural composite WITHOUT shadow - V10.1 NO EDGE PROCESSING"""
    if image.mode == 'RGBA':
        # NO edge feathering - keep sharp edges
        # image = add_natural_edge_feathering(image)  # REMOVED in V10.1
        
        # Create background
        background = create_background(image.size, background_color, style="gradient")
        
        # Simple alpha blending
        r, g, b, a = image.split()
        
        # Convert to arrays for blending
        fg_array = np.array(image.convert('RGB'), dtype=np.float32)
        bg_array = np.array(background, dtype=np.float32)
        alpha_array = np.array(a, dtype=np.float32) / 255.0
        
        # Simple alpha blending
        for i in range(3):
            bg_array[:,:,i] = fg_array[:,:,i] * alpha_array + bg_array[:,:,i] * (1 - alpha_array)
        
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
    """Enhanced cubic details with gentle sharpening - V10.1 GENTLE"""
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
    """Apply center spotlight - V10.1 REDUCED"""
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
    """Enhanced wedding ring processing with gentle cubic detail - V10.1 GENTLE"""
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
    """Optimized enhancement - 12% white overlay for ac_ (1차) - V10.1 REDUCED"""
    
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
    """Resize image to exact target dimensions (for 2000x2600 input)"""
    width, height = image.size
    
    # Check if input is expected 2000x2600 or similar ratio
    expected_ratio = 2000 / 2600  # 0.769
    actual_ratio = width / height
    
    if abs(actual_ratio - expected_ratio) < 0.01:
        # Perfect ratio match - direct resize
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    else:
        # Different ratio - resize with aspect ratio
        logger.warning(f"Unexpected ratio: {width}x{height} ({actual_ratio:.3f})")
        
        # Calculate scale to fit
        scale_w = target_width / width
        scale_h = target_height / height
        scale = min(scale_w, scale_h)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center on white background if needed
        if new_width != target_width or new_height != target_height:
            background = Image.new('RGB', (target_width, target_height), (255, 255, 255))
            left = (target_width - new_width) // 2
            top = (target_height - new_height) // 2
            background.paste(resized, (left, top))
            return background
        
        return resized

def process_enhancement(job):
    """Main enhancement processing - V10.1 CONSISTENT VERSION"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    
    try:
        # Fast extraction
        filename = find_filename_fast(job)
        file_number = extract_file_number(filename) if filename else None
        image_data = find_input_data_fast(job)
        
        # Fixed gray background - BALANCED V10.1
        background_color = '#D8D8D8'  # Balanced gray
        
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
            logger.info("📸 STEP 1: PNG detected - removing background with V10.1 conservative settings")
            image = remove_background_with_replicate(image)
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
        
        # Gentle basic enhancement - V10.1
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)  # Reduced from 1.12
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.05)  # Reduced from 1.06
        
        # Apply pattern-specific enhancement
        image = apply_enhancement_optimized(image, pattern_type)
        
        # RESIZE
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
        
        # Gentle final sharpening - V10.1
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.6)  # Reduced from 1.8
        
        # STEP 3: BACKGROUND COMPOSITE (if transparent)
        if has_transparency and 'original_transparent' in locals():
            logger.info(f"🖼️ STEP 3: Natural background compositing (NO SHADOW): {background_color}")
            # Apply all enhancements to transparent version
            enhanced_transparent = original_transparent.copy()
            
            # Resize transparent version
            enhanced_transparent = enhanced_transparent.resize((1200, 1560), Image.Resampling.LANCZOS)
            
            # Apply enhancements to RGBA
            if enhanced_transparent.mode == 'RGBA':
                # NO additional hole processing - trust rembg
                
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
                    # 12% white overlay - V10.1
                    white_overlay = 0.12  # Reduced
                    img_array = np.array(rgb_image, dtype=np.float32)
                    img_array = img_array * (1 - white_overlay) + 255 * white_overlay
                    rgb_image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
                
                # Merge back with alpha
                r2, g2, b2 = rgb_image.split()
                enhanced_transparent = Image.merge('RGBA', (r2, g2, b2, a))
            
            # Natural composite with balanced gray background
            image = composite_with_light_gray_background(enhanced_transparent, background_color)
            
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
                # Apply 15% white overlay as correction - V10.1
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
                "spotlight_increased": "2.0-2.5%",
                "cubic_enhancement": "Gentle (120% unsharp)",
                "swinir_applied": swinir_applied,
                "swinir_timing": "AFTER resize",
                "png_support": True,
                "has_transparency": has_transparency,
                "background_composite": has_transparency,
                "background_removal": needs_background_removal,
                "background_color": background_color,
                "background_style": "Balanced gray gradient (#D8D8D8)",
                "gradient_edge_darkening": "6%",
                "shadow": "REMOVED - No shadow",
                "edge_processing": "REMOVED - Sharp edges only",
                "composite_method": "Simple alpha blending",
                "rembg_settings": "Conservative (240/50/8)",
                "ring_hole_detection": "Improved consistent detection",
                "hole_detection_details": "Detect semi-transparent areas (<100), center area focus",
                "processing_order": "1.Background Removal → 2.Gentle Enhancement → 3.Natural Composite",
                "quality": "95",
                "expected_input": "2000x2600",
                "output_size": "1200x1560",
                "safety_features": "Ring preservation priority"
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
