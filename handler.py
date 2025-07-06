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
import string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "V153-Advanced-Detail-Enhancement"

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
        for i in range(10):  # Reduced from 20
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
    """FAST base64 decode - Optimized for Make.com"""
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
    """Detect pattern type"""
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

def enhance_cubic_details_advanced(image: Image.Image) -> Image.Image:
    """Advanced cubic enhancement without AI models"""
    
    # 1. Multi-scale unsharp mask
    # Large details (for overall structure)
    large_detail = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    # Medium details (for medium-sized cubics)
    medium_detail = large_detail.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
    
    # Fine details (for tiny cubics)
    fine_detail = medium_detail.filter(ImageFilter.UnsharpMask(radius=0.5, percent=100, threshold=1))
    
    # 2. Edge enhancement for sparkle
    edges = fine_detail.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    # 3. Blend edge enhancement
    enhanced = Image.blend(fine_detail, edges, 0.3)
    
    # 4. Local contrast boost
    contrast = ImageEnhance.Contrast(enhanced)
    enhanced = contrast.enhance(1.1)
    
    # 5. Detail filter for micro-contrast
    enhanced = enhanced.filter(ImageFilter.DETAIL)
    
    return enhanced

def enhance_jewelry_details(image: Image.Image, pattern_type: str) -> Image.Image:
    """Jewelry-specific detail enhancement using OpenCV"""
    try:
        img_array = np.array(image)
        
        # 1. CLAHE for local contrast
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 2. Selective sharpening for bright areas (cubics)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Find bright areas (potential cubics/diamonds)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Dilate mask slightly to include edges
        kernel = np.ones((3,3), np.uint8)
        bright_mask = cv2.dilate(bright_mask, kernel, iterations=1)
        
        # Create sharpening kernel
        sharpen_kernel = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
        
        # Apply sharpening
        sharpened = cv2.filter2D(img_array, -1, sharpen_kernel)
        
        # Blend based on mask
        mask_3d = np.stack([bright_mask/255]*3, axis=2).astype(np.float32)
        result = img_array * (1 - mask_3d * 0.5) + sharpened * (mask_3d * 0.5)
        
        # 3. Additional edge enhancement for ac_bc and b_only patterns
        if pattern_type in ["ac_bc", "b_only"]:
            # Laplacian edge detection
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            
            # Add subtle edge enhancement
            edge_enhanced = cv2.addWeighted(result.astype(np.uint8), 0.9, 
                                          cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB), 0.1, 0)
            result = edge_enhanced
        
        return Image.fromarray(result.astype(np.uint8))
        
    except Exception as e:
        logger.warning(f"OpenCV enhancement failed: {e}, using PIL fallback")
        # Fallback to PIL-only enhancement
        return enhance_cubic_details_advanced(image)

def enhance_cubic_details_fast_quality(image: Image.Image, pattern_type: str) -> Image.Image:
    """Fast but high-quality detail enhancement - Main enhancement function"""
    
    # 1. Initial sharpening
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.3)
    
    # 2. Apply advanced multi-scale enhancement
    image = enhance_cubic_details_advanced(image)
    
    # 3. Apply jewelry-specific enhancement (includes OpenCV processing)
    image = enhance_jewelry_details(image, pattern_type)
    
    # 4. Pattern-specific fine-tuning
    if pattern_type in ["ac_bc", "b_only"]:
        # Extra enhancement for white/unplated patterns
        # These need more contrast to show cubic details
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.05)
        
        # Additional micro-sharpening
        image = image.filter(ImageFilter.UnsharpMask(radius=0.3, percent=80, threshold=1))
    
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

def apply_center_spotlight(image: Image.Image, intensity: float = 0.03) -> Image.Image:
    """Apply center spotlight - Reduced intensity"""
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
    """Fast wedding ring enhancement with cubic detail focus"""
    # Reduced spotlight
    image = apply_center_spotlight(image, 0.02)
    
    # Enhanced sharpness for cubic details
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.6)  # Increased for small cubic details
    
    # Slight contrast
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.03)
    
    # Multi-scale unsharp mask for various cubic sizes
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=3))
    
    return image

def apply_enhancement_optimized(image: Image.Image, pattern_type: str) -> Image.Image:
    """Optimized enhancement - Modified white overlay (10% primary, 3% additional)"""
    
    # Apply white overlay ONLY to ac_bc pattern (10% primary)
    if pattern_type == "ac_bc":
        # Unplated white - 10% white overlay
        white_overlay = 0.10
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Reduced brightness
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.02)  # Reduced from 1.05
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.96)
        
    elif pattern_type in ["a_only", "b_only"]:
        # a_ and b_ patterns - NO white overlay
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.06)  # Reduced from 1.10
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.96)
        
        # Enhanced sharpness
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.5)  # Increased from 1.4
        
    else:
        # Standard enhancement - NO white overlay
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.04)  # Reduced from 1.08
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.02)
    
    # Reduced center spotlight
    if pattern_type in ["a_only", "b_only"]:
        image = apply_center_spotlight(image, 0.02)
    else:
        image = apply_center_spotlight(image, 0.03)
    
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

def resize_to_width_1200(image: Image.Image) -> Image.Image:
    """Resize image to width 1200px maintaining aspect ratio"""
    width, height = image.size
    if width == 1200:
        return image
    
    target_width = 1200
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

def process_enhancement(job):
    """Main enhancement processing - ADVANCED DETAIL ENHANCEMENT"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    
    try:
        # Fast extraction
        filename = find_filename_fast(job)
        file_number = extract_file_number(filename) if filename else None
        image_data = find_input_data_fast(job)
        
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
        
        # Fast white balance
        image = auto_white_balance_fast(image)
        
        # Detect pattern
        pattern_type = detect_pattern_type(filename)
        detected_type = {
            "ac_bc": "무도금화이트(0.10+0.03)",
            "a_only": "a_패턴(no_overlay+spotlight2%)",
            "b_only": "b_패턴(no_overlay+spotlight2%)",
            "other": "기타색상(no_overlay)"
        }.get(pattern_type, "기타색상(no_overlay)")
        
        logger.info(f"Pattern: {pattern_type}")
        
        # Apply advanced cubic detail enhancement (replaces SwinIR)
        logger.info("Applying advanced detail enhancement")
        image = enhance_cubic_details_fast_quality(image, pattern_type)
        
        # Basic enhancement (reduced brightness)
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)  # Reduced from 1.12
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.02)
        
        # Apply optimized enhancement
        image = apply_enhancement_optimized(image, pattern_type)
        
        # Final sharpening (increased for cubic details)
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.5)  # Increased for cubic clarity
        
        # Resize to 1200px
        image = resize_to_width_1200(image)
        
        # Save to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=False, quality=95)  # Increased quality
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
        
        # Fast quality check (only for ac_bc)
        if pattern_type == "ac_bc":
            metrics = calculate_quality_metrics_fast(image)
            if metrics["brightness"] < 240:  # Simple check
                # Apply 3% additional white overlay
                white_overlay = 0.03  # Total 13%
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
                "white_overlay": "10% primary + 3% additional for ac_bc, 0% for others",
                "brightness_reduced": True,
                "sharpness_increased": "1.5-1.6 + multi-scale",
                "spotlight_reduced": "2-3%",
                "detail_enhancement": "Advanced multi-scale + OpenCV CLAHE",
                "enhancements_applied": [
                    "Multi-scale unsharp mask (3 levels)",
                    "OpenCV CLAHE for local contrast",
                    "Selective sharpening for bright areas",
                    "Edge enhancement for sparkle",
                    "Pattern-specific optimization"
                ],
                "processing_order": "White Balance → Advanced Detail Enhancement → Pattern Enhancement → Final Sharpening"
            }
        }
        
        logger.info("✅ Enhancement completed")
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
