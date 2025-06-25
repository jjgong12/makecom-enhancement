import runpod
import base64
import requests
from io import BytesIO
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np
import cv2
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import json
import traceback
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "Enhancement_V62_Brighter"

def create_session():
    """Create a session with retry strategy"""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def find_input_data(data):
    """Recursively find input data from various possible locations"""
    logger.info("Searching for input data...")
    
    # Log the structure (limited to prevent huge logs)
    logger.info(f"Input structure: {json.dumps(data, indent=2)[:1000]}...")
    
    # Direct access attempts
    if isinstance(data, dict):
        # Check top level
        if 'input' in data:
            return data['input']
        
        # Common RunPod structures
        common_paths = [
            ['job', 'input'],
            ['data', 'input'],
            ['payload', 'input'],
            ['body', 'input'],
            ['request', 'input']
        ]
        
        for path in common_paths:
            current = data
            for key in path:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    break
            else:
                logger.info(f"Found input at path: {'.'.join(path)}")
                return current
    
    # Recursive search function
    def recursive_search(obj, target_keys=None):
        if target_keys is None:
            target_keys = ['input', 'url', 'image_url', 'imageUrl', 'image_base64', 
                          'imageBase64', 'image', 'enhanced_image', 'base64_image']
        
        if isinstance(obj, dict):
            # Check for target keys
            for key in target_keys:
                if key in obj:
                    value = obj[key]
                    if key == 'input':
                        return value
                    else:
                        # Return as dict to maintain structure
                        return {key: value}
            
            # Recursive search in values
            for value in obj.values():
                result = recursive_search(value, target_keys)
                if result:
                    return result
                    
        elif isinstance(obj, list):
            for item in obj:
                result = recursive_search(item, target_keys)
                if result:
                    return result
        
        return None
    
    # Try recursive search
    result = recursive_search(data)
    if result:
        logger.info(f"Found input via recursive search: {type(result)}")
        return result
    
    # Last resort - check if the data itself is the input
    if isinstance(data, str) and len(data) > 100:
        logger.info("Using raw data as input")
        return data
    
    logger.warning("No input data found")
    return None

def validate_base64(data):
    """Validate and clean base64 string"""
    try:
        # Remove data URL prefix if present
        if isinstance(data, str) and 'base64,' in data:
            data = data.split('base64,')[1]
        
        # Remove whitespace
        if isinstance(data, str):
            data = data.strip()
        
        # Try decoding
        base64.b64decode(data)
        return True, data
    except Exception as e:
        logger.error(f"Base64 validation error: {str(e)}")
        return False, None

def decode_base64_safe(base64_str):
    """Decode base64 with automatic padding correction"""
    try:
        # Clean the string
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]
        
        # Fix padding if needed
        padding = 4 - len(base64_str) % 4
        if padding != 4:
            base64_str += '=' * padding
        
        return base64.b64decode(base64_str)
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise

def download_image_from_url(url):
    """Download image from URL"""
    try:
        session = create_session()
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        logger.error(f"Failed to download image: {str(e)}")
        raise

def apply_gamma_correction(image, gamma):
    """Apply gamma correction to brighten mid-tones"""
    try:
        # Convert to numpy array
        img_array = np.array(image).astype(float) / 255.0
        
        # Apply gamma correction
        corrected = np.power(img_array, gamma)
        
        # Convert back to 8-bit
        corrected = (corrected * 255).astype(np.uint8)
        
        return Image.fromarray(corrected)
    except Exception as e:
        logger.error(f"Gamma correction error: {str(e)}")
        return image

def detect_ring_color(image):
    """Detect ring color from image"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Get center region (where ring is likely to be)
        h, w = img_array.shape[:2]
        center_y, center_x = h // 2, w // 2
        region_size = min(h, w) // 3
        
        center_region = img_array[
            center_y - region_size:center_y + region_size,
            center_x - region_size:center_x + region_size
        ]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
        
        # Calculate average values
        avg_h = np.mean(hsv[:, :, 0])
        avg_s = np.mean(hsv[:, :, 1])
        avg_v = np.mean(hsv[:, :, 2])
        
        logger.info(f"Color analysis - H: {avg_h:.1f}, S: {avg_s:.1f}, V: {avg_v:.1f}")
        
        # Color detection logic
        if avg_v > 200 and avg_s < 30:  # Very bright and low saturation
            return 'white'
        elif avg_s < 40 and avg_v > 150:  # Low saturation, bright
            return 'white_gold'
        elif 15 <= avg_h <= 35 and avg_s > 30:  # Yellow hue with good saturation
            return 'yellow_gold'
        elif (avg_h < 15 or avg_h > 165) and avg_s > 20:  # Red/pink hue
            return 'rose_gold'
        else:
            return 'white'  # Default
            
    except Exception as e:
        logger.error(f"Color detection error: {str(e)}")
        return 'white'

def enhance_jewelry_image(image, metal_type='white'):
    """Apply enhancement specifically for jewelry photography"""
    try:
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Applying V62 enhancement for {metal_type}")
        
        # V62 보정값 - 더 밝고 하얗게
        # 1. 밝기 35% 증가 (V61: 25% → V62: 35%)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.35)
        
        # 2. 감마 보정 0.7 (V61: 0.8 → V62: 0.7) - 더 밝게
        image = apply_gamma_correction(image, 0.7)
        
        # 3. 채도 30% 감소 (V61: -20% → V62: -30%) - 더 무채색에 가깝게
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(0.7)
        
        # 4. 대비 살짝 증가로 선명도 유지
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)
        
        # 5. 샤프니스 증가
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        # 6. Convert to LAB for advanced adjustments
        img_array = np.array(image)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Increase L channel (lightness)
        lab[:, :, 0] = lab[:, :, 0] * 1.08
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
        
        # Adjust color based on metal type
        if metal_type == 'yellow_gold':
            # Warm up slightly
            lab[:, :, 2] = lab[:, :, 2] * 1.05  # More yellow
        elif metal_type == 'rose_gold':
            # Add pink tone
            lab[:, :, 1] = lab[:, :, 1] * 1.05  # More red
        elif metal_type == 'white_gold':
            # Cool down slightly
            lab[:, :, 2] = lab[:, :, 2] * 0.95  # Less yellow
        
        # Convert back to RGB
        lab = lab.astype(np.uint8)
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        image = Image.fromarray(img_array)
        
        # 7. Final sharpening
        image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=50, threshold=3))
        
        return image
        
    except Exception as e:
        logger.error(f"Enhancement error: {str(e)}")
        return image

def process_enhancement(job):
    """Process enhancement request"""
    logger.info(f"=== {VERSION} Started ===")
    logger.info(f"Received job: {json.dumps(job, indent=2)[:500]}...")
    
    start_time = time.time()
    
    try:
        # Extract input using the job parameter correctly
        job_input = job.get('input', {})
        
        # Find image data
        input_data = find_input_data(job_input)
        
        # If not found in job_input, try the whole job dict
        if not input_data:
            input_data = find_input_data(job)
        
        if not input_data:
            error_msg = "No image data provided in any expected field"
            logger.error(error_msg)
            return {
                "output": {
                    "error": error_msg,
                    "status": "error",
                    "available_fields": list(job_input.keys()) if isinstance(job_input, dict) else []
                }
            }
        
        # Extract image data from various possible formats
        image_data = None
        
        if isinstance(input_data, dict):
            # Try various keys
            for key in ['image', 'image_base64', 'imageBase64', 'base64_image', 
                       'url', 'image_url', 'imageUrl', 'enhanced_image']:
                if key in input_data:
                    image_data = input_data[key]
                    break
        elif isinstance(input_data, str):
            image_data = input_data
        
        if not image_data:
            return {
                "output": {
                    "error": "Could not extract image data from input",
                    "status": "error"
                }
            }
        
        # Process based on data type
        if isinstance(image_data, str) and image_data.startswith('http'):
            # URL input
            logger.info(f"Processing URL: {image_data[:100]}...")
            image = download_image_from_url(image_data)
        else:
            # Base64 input
            logger.info("Processing base64 image...")
            
            # Validate base64
            is_valid, clean_base64 = validate_base64(image_data)
            if not is_valid:
                return {
                    "output": {
                        "error": "Invalid base64 image data",
                        "status": "error"
                    }
                }
            
            # Decode image
            image_bytes = decode_base64_safe(clean_base64)
            image = Image.open(BytesIO(image_bytes))
        
        logger.info(f"Original image: {image.mode} {image.size}")
        
        # Detect ring color
        metal_type = detect_ring_color(image)
        logger.info(f"Detected metal type: {metal_type}")
        
        # Apply enhancements
        logger.info("Applying V62 enhancements (brighter settings)...")
        enhanced_image = enhance_jewelry_image(image, metal_type)
        
        # Save to base64
        logger.info("Encoding result...")
        buffered = BytesIO()
        
        # Use PNG for quality
        enhanced_image.save(buffered, format="PNG", optimize=True)
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        enhanced_base64_no_padding = enhanced_base64.rstrip('=')
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info(f"Enhancement completed in {processing_time:.2f}s")
        logger.info(f"Output size: {len(enhanced_base64_no_padding)} chars (no padding)")
        
        return {
            "output": {
                "enhanced_image": enhanced_base64_no_padding,
                "status": "success",
                "message": "Enhancement completed - V62 brighter settings",
                "processing_time": f"{processing_time:.2f}s",
                "detected_metal": metal_type,
                "original_size": list(image.size),
                "settings": {
                    "brightness": "135%",
                    "gamma": "0.7",
                    "saturation": "70%",
                    "contrast": "105%",
                    "sharpness": "110%",
                    "lab_lightness": "108%"
                }
            }
        }
        
    except Exception as e:
        error_msg = f"Enhancement failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            "output": {
                "error": error_msg,
                "status": "error",
                "traceback": traceback.format_exc()
            }
        }

# RunPod handler
logger.info(f"Starting RunPod {VERSION}...")
runpod.serverless.start({"handler": process_enhancement})
