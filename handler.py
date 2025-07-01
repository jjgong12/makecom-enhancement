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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "V122-StrongerWeddingFocus-BrighterOverall"

# Global cache to prevent duplicate processing - DISABLED for testing
PROCESSED_IMAGES = {}

def extract_file_number(filename: str) -> str:
    """Extract number (001, 002, etc.) from filename"""
    if not filename:
        return None
    
    # Match patterns like 001, 002, etc. in the filename
    # This will find 3-digit numbers
    match = re.search(r'(\d{3})', filename)
    if match:
        number = match.group(1)
        logger.info(f"Extracted number: {number} from filename: {filename}")
        return number
    
    # Try to find 2-digit numbers if 3-digit not found
    match = re.search(r'(\d{2})', filename)
    if match:
        number = match.group(1).zfill(3)  # Pad to 3 digits
        logger.info(f"Extracted and padded number: {number} from filename: {filename}")
        return number
    
    logger.warning(f"No number found in filename: {filename}")
    return None

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

def detect_wedding_ring(image: Image.Image) -> bool:
    """Detect if image contains wedding rings based on ring characteristics"""
    # Convert to grayscale for analysis
    gray = image.convert('L')
    gray_array = np.array(gray)
    
    # Look for circular shapes (typical of rings)
    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray_array, (5, 5), 0)
    
    # Find edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detect circles using Hough transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=300
    )
    
    # If we find circular shapes, it's likely a ring
    if circles is not None and len(circles[0]) > 0:
        logger.info(f"Detected {len(circles[0])} circular shapes - likely wedding rings")
        return True
    
    # Additional check: look for metallic/bright regions in center
    height, width = gray_array.shape
    center_region = gray_array[height//3:2*height//3, width//3:2*width//3]
    
    # Check if center has bright metallic areas
    bright_pixels = np.sum(center_region > 200)
    total_pixels = center_region.size
    bright_ratio = bright_pixels / total_pixels
    
    if bright_ratio > 0.1:  # More than 10% bright pixels in center
        logger.info(f"Detected bright metallic areas ({bright_ratio:.2%}) - likely wedding rings")
        return True
    
    return False

def apply_wedding_ring_focus(image: Image.Image) -> Image.Image:
    """Apply enhanced focus and sharpness for wedding rings - V122 stronger focus"""
    logger.info("Applying wedding ring focus enhancement")
    
    # 1. Stronger center focus - V122 enhanced for better emphasis
    width, height = image.size
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    
    # V122: Stronger focus for wedding rings
    focus_mask = 1 + 0.035 * np.exp(-distance**2 * 1.5)  # Increased from 0.025 to 0.035
    focus_mask = np.clip(focus_mask, 1.0, 1.035)
    
    img_array = np.array(image)
    for i in range(3):
        img_array[:, :, i] = np.clip(img_array[:, :, i] * focus_mask, 0, 255)
    image = Image.fromarray(img_array.astype(np.uint8))
    
    # 2. Enhanced sharpness for ring details
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.3)  # Increased from 1.25
    
    # 3. Slightly enhanced contrast for definition
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.03)  # Increased from 1.02
    
    return image

def apply_color_enhancement_simple(image: Image.Image, pattern_type: str, filename: str, is_wedding_ring: bool) -> Image.Image:
    """Enhanced with different settings based on pattern type - V122 with stronger brightness and wedding focus"""
    
    logger.info(f"Applying enhancement - Filename: {filename}, Pattern type: {pattern_type}, Wedding ring: {is_wedding_ring}")
    
    if pattern_type == "ac_bc":
        # V122: Enhanced brightness and wedding ring specific white overlay
        logger.info("Applying unplated white enhancement")
        
        # Stronger brightness adjustment
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.03)  # Increased from 1.02
        
        # Color adjustment
        color = ImageEnhance.Color(image)
        image = color.enhance(0.96)
        
        # Minimal contrast
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.0)
        
        # Apply white overlay - more for wedding rings
        if is_wedding_ring:
            logger.info("Wedding ring detected - applying 18% white overlay")
            white_overlay_percent = 0.18  # Increased from 15% for wedding rings
        else:
            white_overlay_percent = 0.15  # Standard 15% for non-wedding rings
            
        img_array = np.array(image)
        img_array = img_array * (1 - white_overlay_percent) + 255 * white_overlay_percent
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # V122: Apply wedding ring focus for ac_ patterns too
        if is_wedding_ring:
            image = apply_wedding_ring_focus(image)
        
    elif pattern_type == "a_only":
        # V122: Enhanced brightness for a_ pattern
        logger.info("Applying a_ pattern enhancement (enhanced brightness + center focus)")
        
        # Enhanced brightness for a_ pattern
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.04)  # Increased from 1.03
        
        # Color adjustment
        color = ImageEnhance.Color(image)
        image = color.enhance(0.98)
        
        # Minimal contrast
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.02)  # Increased from 1.01
        
        # Apply enhanced center focus for a_ pattern
        width, height = image.size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)
        
        # Enhanced center focus
        focus_mask = 1 + 0.03 * np.exp(-distance**2 * 1.0)  # Increased from 0.025
        focus_mask = np.clip(focus_mask, 1.0, 1.03)
        
        img_array = np.array(image)
        for i in range(3):
            img_array[:, :, i] = np.clip(img_array[:, :, i] * focus_mask, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Apply wedding ring focus if detected
        if is_wedding_ring:
            image = apply_wedding_ring_focus(image)
        
    else:
        # Standard enhancement for other patterns - enhanced brightness
        logger.info("Standard enhancement (enhanced brightness)")
        
        # Enhanced natural brightness
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.03)  # Increased from 1.02
        
        # Color adjustment
        color = ImageEnhance.Color(image)
        image = color.enhance(0.98)
        
        # Slight contrast adjustment
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.02)  # Increased from 1.01
        
        # Apply wedding ring focus if detected
        if is_wedding_ring:
            image = apply_wedding_ring_focus(image)
    
    return image

def apply_center_focus(image: Image.Image) -> Image.Image:
    """Apply enhanced center brightening to focus on ring"""
    width, height = image.size
    
    # Create radial gradient
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Distance from center
    distance = np.sqrt(X**2 + Y**2)
    
    # Create enhanced center focus mask
    focus_mask = 1 + 0.025 * np.exp(-distance**2 * 1.2)  # Increased from 0.02
    focus_mask = np.clip(focus_mask, 1.0, 1.025)
    
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
    """Enhancement processing - V122 with stronger wedding focus and brightness"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    logger.info(f"Input data type: {type(job)}")
    
    try:
        # Use enhanced filename detection
        filename = find_filename_enhanced(job)
        file_number = None
        
        if filename:
            logger.info(f"Successfully extracted filename: {filename}")
            # Extract file number from filename
            file_number = extract_file_number(filename)
            if file_number:
                logger.info(f"Extracted file number: {file_number}")
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
        
        # Detect if wedding ring
        is_wedding_ring = detect_wedding_ring(image)
        
        # Basic enhancement - V122 with stronger settings
        # 1. Stronger brightness
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.03)  # Increased from 1.02
        
        # 2. Enhanced contrast
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.03)  # Increased from 1.02
        
        # 3. Natural color
        color = ImageEnhance.Color(image)
        image = color.enhance(1.02)  # Increased from 1.01
        
        # 4. Apply pattern-specific enhancement (includes wedding ring focus)
        image = apply_color_enhancement_simple(image, pattern_type, filename, is_wedding_ring)
        
        # 5. Apply center focus (only if not already applied and not wedding ring)
        if pattern_type != "a_only" and not is_wedding_ring:
            image = apply_center_focus(image)
        
        # 6. Enhanced sharpening
        if not is_wedding_ring:
            sharpness = ImageEnhance.Sharpness(image)
            image = sharpness.enhance(1.15)  # Increased from 1.10
        
        # 7. Resize to 1200px width
        image = resize_to_width_1200(image)
        
        # Save to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=True, quality=95)
        buffered.seek(0)  # Reset buffer position
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        enhanced_base64_no_padding = enhanced_base64.rstrip('=')
        
        logger.info("Enhancement completed successfully")
        
        # Build enhanced filename with number preservation
        enhanced_filename = filename
        if filename and file_number:
            # Create enhanced filename that preserves the number
            # For example: "ac_gold_ring_001.jpg" -> "ac_gold_ring_001_enhanced.jpg"
            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            extension = filename.rsplit('.', 1)[1] if '.' in filename else 'jpg'
            enhanced_filename = f"{base_name}_enhanced.{extension}"
        
        return {
            "output": {
                "enhanced_image": enhanced_base64_no_padding,
                "enhanced_image_with_prefix": f"data:image/png;base64,{enhanced_base64_no_padding}",
                "detected_type": detected_type,
                "pattern_type": pattern_type,
                "is_wedding_ring": is_wedding_ring,
                "filename": filename,
                "enhanced_filename": enhanced_filename,
                "file_number": file_number,  # Add file number to output
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
