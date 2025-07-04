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

VERSION = "V142-RealESRGAN"

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

def find_input_data_comprehensive(data, depth=0):
    """Find input data - ULTRA ENHANCED for Make.com"""
    if depth > 5:
        return None
    
    # If already string, return it
    if isinstance(data, str):
        if len(data) > 50 and not data.startswith('http'):
            return data
    
    if isinstance(data, dict):
        # Extended image keys
        image_keys = [
            'enhanced_image', 'image', 'image_data', 'base64_image',
            'imageBase64', 'image_base64', 'base64', 'img', 'photo',
            'picture', 'file', 'content', 'data', 'b64', 'base64_data',
            'image_content', 'file_content', 'thumbnail', 'raw_image'
        ]
        
        # 1. Direct key check
        for key in image_keys:
            if key in data:
                value = data[key]
                if isinstance(value, str) and len(value) > 50:
                    logger.info(f"Found base64 at key: {key}")
                    return value
                elif isinstance(value, dict):
                    result = find_input_data_comprehensive(value, depth + 1)
                    if result:
                        return result
        
        # 2. Check nested structures
        input_keys = ['input', 'inputs', 'data', 'payload', 'body', 'request', 
                     'params', 'arguments', 'job', 'event']
        for input_key in input_keys:
            if input_key in data:
                if isinstance(data[input_key], str) and len(data[input_key]) > 50:
                    logger.info(f"Found image data at {input_key}")
                    return data[input_key]
                elif isinstance(data[input_key], dict):
                    result = find_input_data_comprehensive(data[input_key], depth + 1)
                    if result:
                        return result
        
        # 3. Numeric keys (Make.com)
        for i in range(20):
            key = str(i)
            if key in data:
                if isinstance(data[key], str) and len(data[key]) > 50:
                    logger.info(f"Found at numeric key: {key}")
                    return data[key]
                elif isinstance(data[key], dict):
                    result = find_input_data_comprehensive(data[key], depth + 1)
                    if result:
                        return result
        
        # 4. Deep value search
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 50:
                if not value.startswith('http'):
                    logger.info(f"Found potential base64 at key: {key}")
                    return value
            elif isinstance(value, (dict, list)):
                result = find_input_data_comprehensive(value, depth + 1)
                if result:
                    return result
    
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, str) and len(item) > 50:
                logger.info(f"Found in list at index: {i}")
                return item
            elif isinstance(item, dict):
                result = find_input_data_comprehensive(item, depth + 1)
                if result:
                    return result
    
    return None

def find_filename_comprehensive(data, depth=0):
    """Find filename recursively"""
    if depth > 5:
        return None
    
    if isinstance(data, dict):
        filename_keys = [
            'filename', 'file_name', 'fileName', 'name', 'originalName', 
            'original_name', 'image_name', 'imageName', 'file', 'title'
        ]
        
        for key in filename_keys:
            if key in data and isinstance(data[key], str):
                value = data[key]
                if any(p in value.lower() for p in ['ac_', 'bc_', 'a_', 'b_', 'c_']):
                    logger.info(f"Found filename: {value}")
                    return value
                elif '.' in value and len(value) < 100:
                    return value
        
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

def decode_base64_ultra_safe(base64_str: str) -> bytes:
    """ULTRA SAFE base64 decode - V142 FINAL with Make.com compatibility"""
    try:
        if not base64_str:
            raise ValueError("Empty base64 string")
            
        if isinstance(base64_str, bytes):
            return base64_str
            
        base64_str = str(base64_str)
        
        # Step 1: Remove ALL possible prefixes
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[-1]
        elif 'data:' in base64_str:
            parts = base64_str.split(',')
            if len(parts) > 1:
                base64_str = parts[-1]
            else:
                base64_str = base64_str.split('base64')[-1].lstrip(',')
        
        # Step 2: ULTRA clean - remove everything except valid base64
        base64_str = base64_str.strip()
        base64_str = ''.join(base64_str.split())
        
        # Step 3: Handle common encoding issues
        replacements = [
            ('%2B', '+'), ('%2F', '/'), ('%3D', '='),
            ('%2b', '+'), ('%2f', '/'), ('%3d', '='),
            (' ', ''), ('\n', ''), ('\r', ''), ('\t', ''),
            ('\\n', ''), ('\\r', ''), ('\\t', ''),
            ('"', ''), ("'", ''), ('\\', ''),  # Added backslash removal
            ('&quot;', ''), ('&apos;', ''), ('&amp;', ''),  # HTML entities
            ('\u0020', ''), ('\u000A', ''), ('\u000D', '')  # Unicode spaces
        ]
        
        for old, new in replacements:
            base64_str = base64_str.replace(old, new)
        
        # Step 4: Keep ONLY valid base64 characters
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        base64_str = ''.join(c for c in base64_str if c in valid_chars)
        
        # Step 5: Check minimum length
        if len(base64_str) < 50:
            raise ValueError(f"Base64 too short: {len(base64_str)} chars")
        
        logger.info(f"Cleaned base64 length: {len(base64_str)}, first 50 chars: {base64_str[:50]}")
        
        # Step 6: Create strategies (MAKE.COM OPTIMIZED ORDER)
        strategies = []
        
        # Remove all padding first
        no_pad = base64_str.rstrip('=')
        
        # CRITICAL: Make.com sends without padding, so try this FIRST
        strategies.append(no_pad)  # Strategy 0: No padding (Make.com style)
        
        # Calculate correct padding
        padding_needed = (4 - len(no_pad) % 4) % 4
        correct_padded = no_pad + ('=' * padding_needed)
        strategies.append(correct_padded)  # Strategy 1: Correct padding
        
        # Original string (if different)
        if base64_str != no_pad and base64_str != correct_padded:
            strategies.append(base64_str)  # Strategy 2: Original
        
        # All possible padding combinations
        for i in range(4):
            padded = no_pad + ('=' * i)
            if padded not in strategies:
                strategies.append(padded)  # Strategy 3-6: All padding variants
        
        # Step 7: Try each strategy with multiple methods
        last_error = None
        for i, test_str in enumerate(strategies):
            if not test_str:
                continue
            
            # Method 1: Standard base64 with validation
            try:
                decoded = base64.b64decode(test_str, validate=True)
                # Verify it's valid image data
                img = Image.open(BytesIO(decoded))
                img.verify()  # Quick verify
                logger.info(f"✅ SUCCESS: Decoded with strategy {i} (standard, validated)")
                return decoded
            except Exception as e:
                last_error = e
            
            # Method 2: Standard base64 without validation
            try:
                decoded = base64.b64decode(test_str, validate=False)
                # Verify it's valid image data
                img = Image.open(BytesIO(decoded))
                img.verify()
                logger.info(f"✅ SUCCESS: Decoded with strategy {i} (standard, no validation)")
                return decoded
            except Exception as e:
                last_error = e
            
            # Method 3: URL-safe base64
            try:
                url_safe = test_str.replace('+', '-').replace('/', '_')
                decoded = base64.urlsafe_b64decode(url_safe)
                # Verify it's valid image data
                img = Image.open(BytesIO(decoded))
                img.verify()
                logger.info(f"✅ SUCCESS: Decoded with strategy {i} (urlsafe)")
                return decoded
            except Exception as e:
                last_error = e
            
            # Method 4: Alternative base64 (handle weird cases)
            try:
                # Try with standard lib directly
                import binascii
                decoded = binascii.a2b_base64(test_str)
                img = Image.open(BytesIO(decoded))
                img.verify()
                logger.info(f"✅ SUCCESS: Decoded with strategy {i} (binascii)")
                return decoded
            except:
                pass
        
        # If all strategies fail, provide detailed error info
        logger.error(f"❌ CRITICAL: All decode attempts failed")
        logger.error(f"Original length: {len(base64_str)}")
        logger.error(f"No padding length: {len(no_pad)}")
        logger.error(f"Padding needed: {padding_needed}")
        logger.error(f"First 100 chars: {base64_str[:100]}")
        logger.error(f"Last 100 chars: {base64_str[-100:]}")
        logger.error(f"Last error: {last_error}")
        
        # Last resort: try to decode ignoring errors
        try:
            decoded = base64.b64decode(no_pad + '==', validate=False)
            img = Image.open(BytesIO(decoded))
            logger.warning("⚠️ FALLBACK: Decoded with forced padding")
            return decoded
        except:
            pass
        
        raise ValueError(f"Base64 decode failed after all attempts. Last error: {last_error}")
        
    except Exception as e:
        logger.error(f"Fatal base64 decode error: {str(e)}")
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

def detect_wedding_ring_fast(image: Image.Image) -> bool:
    """Always return True since all images are wedding rings"""
    return True

def apply_replicate_enhancement(image: Image.Image, is_wedding_ring: bool, pattern_type: str) -> Image.Image:
    """Apply Replicate API enhancement"""
    if not USE_REPLICATE or not REPLICATE_CLIENT:
        logger.error("❌ Replicate not available")
        raise ValueError("Replicate API token not configured")
    
    try:
        # Check image size
        original_size = image.size
        width, height = original_size
        total_pixels = width * height
        MAX_PIXELS = 2000000
        
        need_resize = False
        if total_pixels > MAX_PIXELS:
            resize_factor = (MAX_PIXELS / total_pixels) ** 0.5
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            
            logger.info(f"Resizing for Replicate: {width}x{height} -> {new_width}x{new_height}")
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            need_resize = True
        
        # Convert to base64 for Replicate
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        # Always apply wedding ring enhancement
        logger.info("🔷 Applying Replicate enhancement for wedding ring")
        
        # Changed to Real-ESRGAN (same as thumbnail) due to permission issues
        output = REPLICATE_CLIENT.run(
            "nightmareai/real-esrgan:350d32041630ffbe63c8352783a26d94126809164e54085352f8326e53999085",
            input={
                "image": img_data_url,
                "scale": 2,
                "face_enhance": False,
                "model": "RealESRGAN_x4plus"
            }
        )
        
        if output:
            # Convert output back to PIL Image
            if isinstance(output, str):
                response = requests.get(output)
                enhanced_image = Image.open(BytesIO(response.content))
            else:
                if hasattr(output, 'read'):
                    enhanced_image = Image.open(output)
                else:
                    enhanced_image = Image.open(BytesIO(base64.b64decode(output)))
            
            # Resize back if needed
            if need_resize:
                logger.info(f"Resizing back to original: {original_size}")
                enhanced_image = enhanced_image.resize(original_size, Image.Resampling.LANCZOS)
            
            logger.info("✅ Replicate enhancement successful")
            return enhanced_image
        else:
            raise ValueError("Replicate enhancement failed - no output")
            
    except Exception as e:
        logger.error(f"❌ Replicate enhancement error: {str(e)}")
        raise

def auto_white_balance(image: Image.Image) -> Image.Image:
    """Apply automatic white balance correction"""
    img_array = np.array(image, dtype=np.float32)
    
    gray_mask = (
        (np.abs(img_array[:,:,0] - img_array[:,:,1]) < 10) & 
        (np.abs(img_array[:,:,1] - img_array[:,:,2]) < 10) &
        (img_array[:,:,0] > 200)
    )
    
    if np.sum(gray_mask) > 100:
        r_avg = np.mean(img_array[gray_mask, 0])
        g_avg = np.mean(img_array[gray_mask, 1])
        b_avg = np.mean(img_array[gray_mask, 2])
        
        gray_avg = (r_avg + g_avg + b_avg) / 3
        r_factor = gray_avg / r_avg if r_avg > 0 else 1
        g_factor = gray_avg / g_avg if g_avg > 0 else 1
        b_factor = gray_avg / b_avg if b_avg > 0 else 1
        
        img_array[:,:,0] *= r_factor
        img_array[:,:,1] *= g_factor
        img_array[:,:,2] *= b_factor
    
    img_array = np.clip(img_array, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))

def apply_center_spotlight(image: Image.Image, intensity: float = 0.10) -> Image.Image:
    """Apply center spotlight effect - Reduced for wedding rings"""
    width, height = image.size
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    
    # Reduced spotlight mask
    spotlight_mask = 1 + intensity * np.exp(-distance**2 * 0.8)
    spotlight_mask = np.clip(spotlight_mask, 1.0, 1.0 + intensity)
    
    img_array = np.array(image, dtype=np.float32)
    for i in range(3):
        img_array[:, :, i] *= spotlight_mask
    img_array = np.clip(img_array, 0, 255)
    
    return Image.fromarray(img_array.astype(np.uint8))

def apply_wedding_ring_enhancement(image: Image.Image) -> Image.Image:
    """Enhanced wedding ring processing - WITHOUT metallic highlight"""
    # Metallic highlight removed - skip this step entirely
    
    # 1. Center spotlight for cubic
    image = apply_center_spotlight(image, 0.08)
    
    # 2. Enhanced sharpness for cubic details
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.4)  # Reduced from 1.5
    
    # 3. Contrast for depth
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.03)  # Reduced to 1.03
    
    # 4. Detail enhancement with stronger settings
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
    
    return image

def apply_enhancement_v142(image: Image.Image, pattern_type: str, is_wedding_ring: bool) -> Image.Image:
    """V142 Enhancement with reduced brightness for wedding rings"""
    
    # Apply 3% white overlay to ALL patterns (as requested)
    white_overlay = 0.03
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array * (1 - white_overlay) + 255 * white_overlay
    img_array = np.clip(img_array, 0, 255)
    image = Image.fromarray(img_array.astype(np.uint8))
    
    if pattern_type == "ac_bc":
        # Unplated white - additional processing
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.02)  # Increased from 0.98
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.96)
        
        # Additional white overlay for ac_bc (total 18%)
        white_overlay = 0.15
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
    elif pattern_type in ["a_only", "b_only"]:
        # a_ and b_ patterns
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)  # Increased from 1.01
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.96)
        
        # Enhanced sharpness
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.4)  # Reduced from 1.5
        
    else:
        # Standard enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.05)  # Increased from 1.0
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.03)  # Reduced to 1.03
    
    # Apply reduced center spotlight
    image = apply_center_spotlight(image, 0.10)  # Reduced from 0.15
    
    # Wedding ring special enhancement (always applied)
    image = apply_wedding_ring_enhancement(image)
    
    return image

def resize_to_width_1200(image: Image.Image) -> Image.Image:
    """Resize image to width 1200px maintaining aspect ratio"""
    width, height = image.size
    target_width = 1200
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

def process_enhancement(job):
    """Main enhancement processing - Wedding Ring Optimized"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    logger.info(f"Replicate available: {USE_REPLICATE}")
    
    try:
        # Extract filename
        filename = find_filename_comprehensive(job)
        logger.info(f"Filename found: {filename}")
        file_number = extract_file_number(filename) if filename else None
        
        # Extract image data
        image_data = find_input_data_comprehensive(job)
        
        if not image_data:
            logger.error("Failed to find image data")
            return {
                "output": {
                    "error": "No image data found in input",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        logger.info(f"Found image data, length: {len(image_data)}")
        
        # Decode image with ULTRA SAFE method
        try:
            image_bytes = decode_base64_ultra_safe(image_data)
            image = Image.open(BytesIO(image_bytes))
            logger.info("✅ Successfully decoded and opened image")
        except Exception as e:
            logger.error(f"Failed to decode/open image: {str(e)}")
            return {
                "output": {
                    "error": f"Invalid base64 data: {str(e)}",
                    "status": "error",
                    "version": VERSION,
                    "decode_attempts": "all strategies failed"
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
        
        # Apply white balance correction
        image = auto_white_balance(image)
        
        # Detect pattern type
        pattern_type = detect_pattern_type(filename)
        detected_type = {
            "ac_bc": "무도금화이트(0.15+0.03)",
            "a_only": "a_패턴(0.03)",
            "b_only": "b_패턴(0.03)",
            "other": "기타색상(0.03)"
        }.get(pattern_type, "기타색상(0.03)")
        
        logger.info(f"Pattern type: {pattern_type}, Detected type: {detected_type}")
        
        # Always wedding ring
        is_wedding_ring = True
        logger.info(f"Wedding ring: Always True")
        
        # Apply Replicate enhancement if available
        replicate_applied = False
        if USE_REPLICATE:
            try:
                image = apply_replicate_enhancement(image, is_wedding_ring, pattern_type)
                replicate_applied = True
            except Exception as e:
                logger.error(f"Replicate enhancement failed: {str(e)}")
                # Continue with basic enhancement instead of returning error
                logger.warning("Continuing with basic enhancement only")
        
        # Basic enhancement (increased brightness and contrast)
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.1)  # Increased to 1.1 as requested
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.03)  # Reduced to 1.03
        
        # Apply V142 enhancement
        image = apply_enhancement_v142(image, pattern_type, is_wedding_ring)
        
        # Final sharpening
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.3)  # Reduced from 1.4
        
        # Resize to 1200px width
        image = resize_to_width_1200(image)
        logger.info(f"Resized to: {image.size}")
        
        # Save to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=True, quality=95)
        buffered.seek(0)
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # CRITICAL: Remove padding for Make.com
        enhanced_base64_no_padding = enhanced_base64.rstrip('=')
        logger.info(f"Output base64 length: {len(enhanced_base64_no_padding)} (padding removed)")
        
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
                "is_wedding_ring": True,
                "filename": filename,
                "enhanced_filename": enhanced_filename,
                "file_number": file_number,
                "original_size": list(original_size),
                "final_size": list(image.size),
                "version": VERSION,
                "status": "success",
                "white_overlay_applied": "3% base + pattern specific",
                "center_spotlight": "10% reduced",
                "wedding_ring_enhancement": "cubic_focus_only_no_metallic",
                "replicate_applied": replicate_applied,
                "base64_decode_method": "ultra_safe_v142_final",
                "make_com_compatible": True
            }
        }
        
        logger.info("✅ Enhancement completed successfully")
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
