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

VERSION = "V116-28PercentWhiteOverlay-APattternBrightness-Resize1200"

# Global cache to prevent duplicate processing
PROCESSED_IMAGES = {}

def find_input_data(data):
    """Find input data - optimized for speed"""
    
    # Fast return if already a string
    if isinstance(data, str):
        return data
    
    # Direct key access without logging
    if isinstance(data, dict):
        # Priority 1: Direct image keys
        image_keys = ['enhanced_image', 'image', 'image_data', 'base64_image', 
                     'imageBase64', 'image_base64', 'base64']
        
        for key in image_keys:
            if key in data and isinstance(data[key], str) and len(data[key]) > 100:
                return data[key]
        
        # Priority 2: Check 'input' key
        if 'input' in data:
            if isinstance(data['input'], str):
                return data['input']
            elif isinstance(data['input'], dict):
                # Check image keys in input
                for key in image_keys:
                    if key in data['input'] and isinstance(data['input'][key], str):
                        return data['input'][key]
        
        # Priority 3: Check numeric keys (Make.com) - limit to single digits
        for i in range(10):  # Only check 0-9
            key = str(i)
            if key in data:
                result = find_input_data(data[key])
                if result:
                    return result
        
        # Priority 4: Specific paths only
        # Direct path checking without loop
        if 'job' in data and isinstance(data['job'], dict) and 'input' in data['job']:
            result = find_input_data(data['job']['input'])
            if result:
                return result
        
        # Make.com specific path
        if '4' in data and isinstance(data['4'], dict):
            if 'data' in data['4'] and isinstance(data['4']['data'], dict):
                if 'output' in data['4']['data'] and isinstance(data['4']['data']['output'], dict):
                    if 'output' in data['4']['data']['output'] and isinstance(data['4']['data']['output']['output'], dict):
                        if 'enhanced_image' in data['4']['data']['output']['output']:
                            return data['4']['data']['output']['output']['enhanced_image']
    
    # Limited recursive search
    def quick_search(obj, depth=0):
        if depth > 3:  # Limit depth to 3
            return None
            
        if isinstance(obj, str) and len(obj) > 100:
            return obj
            
        if isinstance(obj, dict):
            # Only check specific keys
            for key in ['enhanced_image', 'image', 'image_data', 'input', 'data', 'output']:
                if key in obj:
                    if isinstance(obj[key], str) and len(obj[key]) > 100:
                        return obj[key]
                    else:
                        result = quick_search(obj[key], depth + 1)
                        if result:
                            return result
        
        return None
    
    result = quick_search(data)
    
    if not result:
        logger.error("No image data found!")
        
    return result

def find_filename_enhanced(data, depth=0):
    """Enhanced filename extraction for Make.com - checks EVERYWHERE"""
    if depth > 10:  # Increased depth limit
        return None
    
    found_filenames = []
    
    def extract_filenames(obj, current_depth):
        """Recursively extract all potential filenames"""
        if current_depth > 10:
            return
            
        if isinstance(obj, dict):
            # Extended list of filename keys
            filename_keys = [
                'filename', 'file_name', 'fileName', 'name', 'file',
                'originalName', 'original_name', 'originalFileName', 'original_file_name',
                'image_name', 'imageName', 'imageFileName', 'image_file_name',
                'ring_filename', 'ringFilename', 'product_name', 'productName',
                'title', 'label', 'id', 'identifier', 'reference'
            ]
            
            # Check all keys
            for key, value in obj.items():
                # Check if key itself looks like a filename pattern
                if isinstance(value, str) and any(pattern in value.lower() for pattern in ['ac_', 'bc_', 'a_', 'b_', 'c_']):
                    if len(value) < 100:  # Reasonable filename length
                        found_filenames.append(value)
                        logger.info(f"Found potential filename in value: {value}")
                
                # Check known filename keys
                if key.lower() in [k.lower() for k in filename_keys]:
                    if isinstance(value, str) and value and len(value) < 100:
                        found_filenames.append(value)
                        logger.info(f"Found filename at key '{key}': {value}")
                
                # Recursive search
                if isinstance(value, dict):
                    extract_filenames(value, current_depth + 1)
                elif isinstance(value, list):
                    for item in value:
                        extract_filenames(item, current_depth + 1)
                elif isinstance(value, str):
                    # Check if the string itself might contain JSON
                    if value.startswith('{') and value.endswith('}'):
                        try:
                            parsed = json.loads(value)
                            extract_filenames(parsed, current_depth + 1)
                        except:
                            pass
        
        elif isinstance(obj, list):
            for item in obj:
                extract_filenames(item, current_depth)
    
    # Start extraction
    extract_filenames(data, 0)
    
    # Filter and prioritize filenames
    valid_filenames = []
    for fname in found_filenames:
        # Check if it looks like our ring filename pattern
        if any(pattern in fname.lower() for pattern in ['ac_', 'bc_', 'a_', 'b_', 'c_']):
            valid_filenames.append(fname)
    
    if valid_filenames:
        # Return the most likely filename (first one with our pattern)
        filename = valid_filenames[0]
        logger.info(f"Selected filename: {filename} from {len(valid_filenames)} candidates")
        return filename
    
    # Log all keys at root level for debugging
    if depth == 0 and isinstance(data, dict):
        logger.info(f"Root level keys: {list(data.keys())}")
        # Also log some values to see structure
        for key in list(data.keys())[:5]:
            logger.info(f"Key '{key}' type: {type(data[key])}")
    
    return None

def decode_base64_safe(base64_str: str) -> bytes:
    """Safely decode base64 with automatic padding correction"""
    if not isinstance(base64_str, str):
        raise ValueError(f"Expected string, got {type(base64_str)}")
    
    # Remove data URL prefix if present
    if base64_str.startswith('data:'):
        base64_str = base64_str.split(',')[1]
    
    # Clean whitespace and newlines
    base64_str = base64_str.strip().replace('\n', '').replace('\r', '').replace(' ', '')
    
    # Add padding if needed
    padding = 4 - len(base64_str) % 4
    if padding != 4:
        base64_str += '=' * padding
    
    try:
        return base64.b64decode(base64_str)
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        # Try without padding
        try:
            return base64.b64decode(base64_str.rstrip('='))
        except:
            raise e

def detect_pattern_type(filename: str) -> str:
    """Detect pattern type: 'ac_bc' for unplated white, 'a_only' for a_ pattern, 'other' for rest"""
    if not filename:
        logger.warning("No filename found, defaulting to standard enhancement")
        return "other"
    
    # Convert to lowercase for case-insensitive check
    filename_lower = filename.lower()
    logger.info(f"Checking filename pattern: {filename_lower}")
    
    import re
    
    # Check for ac_ or bc_ patterns (unplated white)
    pattern_ac_bc = re.search(r'(ac_|bc_)', filename_lower)
    
    # Check for a_ pattern (NOT ac_)
    pattern_a_only = re.search(r'(?<!a)(?<!b)a_', filename_lower)
    
    if pattern_ac_bc:
        logger.info("Pattern detected: ac_ or bc_ (unplated white)")
        return "ac_bc"
    elif pattern_a_only:
        logger.info("Pattern detected: a_ only")
        return "a_only"
    else:
        logger.info("Pattern detected: other")
        return "other"

def apply_color_enhancement_simple(image: Image.Image, pattern_type: str, filename: str) -> Image.Image:
    """Enhanced with different settings based on pattern type"""
    
    logger.info(f"Applying enhancement - Filename: {filename}, Pattern type: {pattern_type}")
    
    if pattern_type == "ac_bc":
        # V116: 28% white overlay for ac_ and bc_ patterns
        logger.info("Applying unplated white enhancement (28% white overlay)")
        
        # First brightness adjustment
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.04)
        
        # Color adjustment
        color = ImageEnhance.Color(image)
        image = color.enhance(0.92)
        
        # Contrast
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.0)
        
        # Apply 28% white overlay
        img_array = np.array(image)
        img_array = img_array * 0.72 + 255 * 0.28  # 28% white overlay
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Final brightness
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.00)
        
    elif pattern_type == "a_only":
        # V116: Enhanced brightness and center focus for a_ pattern
        logger.info("Applying a_ pattern enhancement (increased brightness + center focus)")
        
        # Increased brightness for a_ pattern
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)  # Higher brightness
        
        # Color adjustment
        color = ImageEnhance.Color(image)
        image = color.enhance(0.94)  # Slightly more color
        
        # Contrast
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.02)  # Slight contrast boost
        
        # Apply subtle center focus for a_ pattern
        width, height = image.size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)
        
        # Slightly stronger center focus for a_ pattern
        focus_mask = 1 + 0.06 * np.exp(-distance**2 * 0.7)
        focus_mask = np.clip(focus_mask, 1.0, 1.06)
        
        img_array = np.array(image)
        for i in range(3):
            img_array[:, :, i] = np.clip(img_array[:, :, i] * focus_mask, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Final brightness boost
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.02)
        
    else:
        # Standard enhancement for other patterns
        logger.info("Standard enhancement (no white overlay)")
        
        # Standard brightness
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.04)
        
        # Color adjustment
        color = ImageEnhance.Color(image)
        image = color.enhance(0.92)
        
        # Contrast
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.0)
        
        # Final brightness
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.00)
    
    return image

def apply_center_focus(image: Image.Image) -> Image.Image:
    """Apply subtle center brightening to focus on ring"""
    width, height = image.size
    
    # Create radial gradient
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Distance from center
    distance = np.sqrt(X**2 + Y**2)
    
    # Create center focus mask (brighter in center, normal at edges)
    focus_mask = 1 + 0.04 * np.exp(-distance**2 * 0.8)
    focus_mask = np.clip(focus_mask, 1.0, 1.04)
    
    # Apply focus
    img_array = np.array(image)
    for i in range(3):
        img_array[:, :, i] = np.clip(img_array[:, :, i] * focus_mask, 0, 255)
    
    return Image.fromarray(img_array.astype(np.uint8))

def resize_to_width_1200(image: Image.Image) -> Image.Image:
    """Resize image to width 1200px while maintaining aspect ratio"""
    original_width, original_height = image.size
    logger.info(f"Original size: {original_width}x{original_height}")
    
    # Calculate new height maintaining aspect ratio
    target_width = 1200
    aspect_ratio = original_height / original_width
    target_height = int(target_width * aspect_ratio)
    
    # Resize using high quality resampling
    resized_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    logger.info(f"Resized to: {target_width}x{target_height}")
    
    return resized_image

def calculate_image_hash(image: Image.Image) -> str:
    """Calculate a simple hash to detect duplicate images"""
    # Resize to small size for fast comparison
    small = image.resize((8, 8), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())
    avg = sum(sum(pixel) for pixel in pixels) / len(pixels) / 3
    
    # Create binary hash
    hash_str = ""
    for pixel in pixels:
        if sum(pixel) / 3 > avg:
            hash_str += "1"
        else:
            hash_str += "0"
    
    return hash_str

def process_enhancement(job):
    """Enhancement processing - V116 with pattern-based enhancement and resize to 1200px width"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    logger.info(f"Input data type: {type(job)}")
    
    try:
        # Use enhanced filename detection
        filename = find_filename_enhanced(job)
        if filename:
            logger.info(f"Successfully extracted filename: {filename}")
        else:
            logger.warning("Could not extract filename from input - will use default enhancement")
            # Continue processing even without filename
        
        # Find image data
        image_data = find_input_data(job)
        
        if not image_data:
            return {
                "output": {
                    "error": "No image data found",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Ensure we have a string
        if not isinstance(image_data, str):
            return {
                "output": {
                    "error": f"Image data must be a string, got {type(image_data)}",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Check for duplicate processing
        image_preview = image_data[:100]
        current_time = time.time()
        
        # Clean old entries
        global PROCESSED_IMAGES
        PROCESSED_IMAGES = {k: v for k, v in PROCESSED_IMAGES.items() 
                          if current_time - v < 60}
        
        # Check if recently processed
        if image_preview in PROCESSED_IMAGES:
            logger.warning("Duplicate image detected, skipping processing")
            return {
                "output": {
                    "error": "Duplicate processing detected",
                    "status": "duplicate",
                    "version": VERSION
                }
            }
        
        # Mark as processed
        PROCESSED_IMAGES[image_preview] = current_time
        
        # Decode image with safe handling
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
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            else:
                image = image.convert('RGB')
        
        logger.info(f"Image loaded: {image.size}")
        
        # Store original size before any processing
        original_size = image.size
        
        # Detect pattern type
        pattern_type = detect_pattern_type(filename)
        
        # Set detected type for output
        if pattern_type == "ac_bc":
            detected_type = "무도금화이트"
        elif pattern_type == "a_only":
            detected_type = "a_패턴"
        else:
            detected_type = "기타색상"
        
        logger.info(f"Final detection - Type: {detected_type}, Filename: {filename}")
        
        # Basic enhancement - V116
        # 1. Brightness
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)  # Reduced from 1.10
        
        # 2. Contrast
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.05)
        
        # 3. Color
        color = ImageEnhance.Color(image)
        image = color.enhance(1.03)
        
        # 4. REMOVED apply_background_whitening() - causes gradient background
        
        # 5. Apply pattern-specific enhancement
        image = apply_color_enhancement_simple(image, pattern_type, filename)
        
        # 6. Apply center focus (already applied for a_ pattern in color enhancement)
        if pattern_type != "a_only":
            image = apply_center_focus(image)
        
        # 7. Light sharpening
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.2)
        
        # 8. Resize to 1200px width
        image = resize_to_width_1200(image)
        
        # Save to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=True, quality=95)
        buffered.seek(0)  # Reset buffer position
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        enhanced_base64_no_padding = enhanced_base64.rstrip('=')
        
        logger.info("Enhancement completed successfully")
        
        return {
            "output": {
                "enhanced_image": enhanced_base64_no_padding,
                "enhanced_image_with_prefix": f"data:image/png;base64,{enhanced_base64_no_padding}",
                "detected_type": detected_type,
                "pattern_type": pattern_type,
                "filename": filename,
                "original_size": list(original_size),
                "final_size": list(image.size),
                "version": VERSION,
                "status": "success"
            }
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        error_trace = traceback.format_exc()
        logger.error(error_trace)
        
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": VERSION,
                "traceback": error_trace
            }
        }

def handler(event):
    """RunPod handler function"""
    return process_enhancement(event)

# RunPod handler
runpod.serverless.start({"handler": handler})
