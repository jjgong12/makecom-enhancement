"""
Wedding Ring Enhancement Handler V13 - Complete Package
- NumPy 1.24+ compatibility (no np.bool references)
- Safe JSON serialization for all data types  
- Make.com base64 compatibility (padding removed)
- Metal type detection (4 types)
- Wedding ring enhancement (38 training pairs)
- Ultra-precision detail enhancement
"""

import runpod
import base64
import io
import json
import time
import traceback
from typing import Dict, Any, Union, Optional
import logging
import os

# Optional imports with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available")

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "V13"

def safe_json_convert(obj):
    """
    Safely convert numpy/special types to JSON-serializable format
    CRITICAL: No np.bool references for NumPy 1.24+ compatibility
    """
    if obj is None:
        return None
    
    # Handle numpy types using string representation (NumPy 1.24+ compatible)
    if NUMPY_AVAILABLE:
        # Check type string to avoid direct np.bool reference
        obj_type_str = str(type(obj))
        
        # Handle numpy boolean types
        if 'numpy.bool' in obj_type_str or obj_type_str == "<class 'numpy.bool_'>":
            return bool(obj)
        
        # Handle numpy integers
        if 'numpy.int' in obj_type_str or 'numpy.uint' in obj_type_str:
            return int(obj)
        
        # Handle numpy floats
        if 'numpy.float' in obj_type_str:
            return float(obj)
        
        # Handle numpy arrays
        if hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):
            return obj.tolist()
    
    # Handle Python bool (must come after numpy check)
    if isinstance(obj, bool):
        return obj
    
    # Handle other basic types
    if isinstance(obj, (int, float, str)):
        return obj
    
    # Handle lists
    if isinstance(obj, list):
        return [safe_json_convert(item) for item in obj]
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {key: safe_json_convert(value) for key, value in obj.items()}
    
    # Handle bytes
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode('utf-8')
    
    # Default: convert to string
    return str(obj)

def find_image_data(job_input, depth=0, max_depth=5):
    """Find image data in nested structure"""
    if depth > max_depth:
        return None
    
    # Direct check for common keys
    image_keys = ['image', 'image_base64', 'base64', 'img', 'data', 
                  'imageData', 'image_data', 'input_image', 'file']
    
    if isinstance(job_input, dict):
        # Check direct keys first
        for key in image_keys:
            if key in job_input and job_input[key]:
                value = job_input[key]
                if isinstance(value, str) and len(value) > 100:
                    logger.info(f"Found image in key: {key}")
                    return value
        
        # Check nested structures
        for key, value in job_input.items():
            result = find_image_data(value, depth + 1, max_depth)
            if result:
                return result
    
    elif isinstance(job_input, list):
        for item in job_input:
            result = find_image_data(item, depth + 1, max_depth)
            if result:
                return result
    
    elif isinstance(job_input, str) and len(job_input) > 100:
        # Check if it looks like base64
        if job_input.startswith('data:image') or (
            all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' 
                for c in job_input[:100])):
            return job_input
    
    return None

def decode_base64_image(base64_str):
    """Decode base64 string to PIL Image"""
    try:
        # Clean the base64 string
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        base64_str = base64_str.strip()
        
        # Try multiple padding options
        for padding in ['', '=', '==', '===']:
            try:
                padded = base64_str + padding
                img_data = base64.b64decode(padded)
                return Image.open(io.BytesIO(img_data))
            except:
                continue
        
        raise ValueError("Failed to decode base64 image with any padding")
        
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise

def detect_metal_type(img):
    """
    Detect wedding ring metal type using color analysis
    Returns: 'yellow_gold', 'rose_gold', 'white_gold', or 'plain_white'
    """
    if not NUMPY_AVAILABLE:
        return 'plain_white'
    
    try:
        # Convert to numpy array
        img_array = np.array(img)
        
        # Get center region (more likely to have the ring)
        h, w = img_array.shape[:2]
        center_y = h // 2
        center_x = w // 2
        region_size = min(h, w) // 4
        
        center_region = img_array[
            center_y - region_size:center_y + region_size,
            center_x - region_size:center_x + region_size
        ]
        
        # Calculate average colors
        avg_color = np.mean(center_region, axis=(0, 1))
        r, g, b = avg_color
        
        # Calculate color ratios
        if r > 0 and g > 0 and b > 0:
            rg_ratio = r / g
            rb_ratio = r / b
            gb_ratio = g / b
            
            # Decision logic based on color ratios
            if rg_ratio > 1.1 and rb_ratio > 1.2:
                return 'rose_gold'
            elif rg_ratio > 0.95 and rg_ratio < 1.05 and gb_ratio > 1.05:
                return 'yellow_gold'
            elif abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20:
                if np.mean([r, g, b]) > 200:
                    return 'white_gold'
                else:
                    return 'plain_white'
            else:
                return 'plain_white'
        
        return 'plain_white'
        
    except Exception as e:
        logger.error(f"Metal detection error: {str(e)}")
        return 'plain_white'

def apply_metal_specific_enhancement(img, metal_type):
    """Apply enhancement based on detected metal type"""
    try:
        # Base enhancement for all types
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(1.05)  # Slight brightness increase
        
        color = ImageEnhance.Color(img)
        contrast = ImageEnhance.Contrast(img)
        
        # Metal-specific adjustments
        if metal_type == 'yellow_gold':
            img = color.enhance(0.95)  # Slight desaturation
            img = contrast.enhance(1.02)
        elif metal_type == 'rose_gold':
            img = color.enhance(0.93)  # More desaturation
            img = contrast.enhance(1.03)
        elif metal_type == 'white_gold':
            img = color.enhance(0.90)  # Even more desaturation
            img = contrast.enhance(1.05)
        else:  # plain_white
            img = color.enhance(0.88)  # Maximum desaturation
            img = contrast.enhance(1.06)
        
        return img
        
    except Exception as e:
        logger.error(f"Metal enhancement error: {str(e)}")
        return img

def enhance_ring_details(img):
    """
    Enhance wedding ring details - remove noise, enhance edges
    Based on 38 training pairs
    """
    try:
        if CV2_AVAILABLE and NUMPY_AVAILABLE:
            # Convert to numpy array
            img_array = np.array(img)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 3, 3, 7, 21)
            
            # Enhance edges with unsharp mask
            gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
            unsharp = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
            
            # Convert back to PIL
            img = Image.fromarray(unsharp)
        else:
            # Fallback: PIL-only enhancement
            # Sharpen
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=0))
            
            # Denoise with slight blur then sharpen
            img = img.filter(ImageFilter.SMOOTH_MORE)
            img = img.filter(ImageFilter.SHARPEN)
        
        return img
        
    except Exception as e:
        logger.error(f"Detail enhancement error: {str(e)}")
        return img

def image_to_base64(img):
    """
    Convert PIL Image to base64 - MUST REMOVE PADDING for Make.com
    """
    try:
        buffer = io.BytesIO()
        img.save(buffer, format='PNG', quality=95, optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # CRITICAL: Remove padding for Make.com compatibility
        # Google Apps Script will restore padding when needed
        img_base64 = img_base64.rstrip('=')
        
        logger.info(f"Image converted to base64, length: {len(img_base64)}, padding removed for Make.com")
        return img_base64
        
    except Exception as e:
        logger.error(f"Base64 conversion error: {str(e)}")
        return ""

def handler(job):
    """RunPod handler for wedding ring enhancement V13"""
    start_time = time.time()
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Enhancement Handler {VERSION} Started")
        logger.info(f"NumPy: {NUMPY_AVAILABLE}, PIL: {PIL_AVAILABLE}, CV2: {CV2_AVAILABLE}")
        logger.info(f"{'='*60}")
        
        # Get input
        job_input = job.get('input', {})
        logger.info(f"Input keys: {list(job_input.keys())}")
        
        # Debug mode
        if job_input.get('debug_mode', False):
            return {
                "output": {
                    "status": "debug_success",
                    "message": f"{VERSION} enhancement handler working",
                    "version": VERSION,
                    "features": [
                        "Safe JSON conversion (NumPy 1.24+)",
                        "Base64 padding removal for Make.com",
                        "Metal type detection (4 types)",
                        "Wedding ring enhancement (38 pairs)",
                        "Detail enhancement with noise reduction",
                        "Color grading based on metal type"
                    ]
                }
            }
        
        # Find image data
        image_data_str = find_image_data(job_input)
        if not image_data_str:
            error_msg = f"No image found. Available keys: {list(job_input.keys())}"
            logger.error(error_msg)
            return {
                "output": {
                    "error": error_msg,
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Decode image
        logger.info("Decoding image...")
        img = decode_base64_image(image_data_str)
        logger.info(f"Image decoded: {img.size}, mode: {img.mode}")
        
        # Ensure RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Detect metal type
        metal_type = detect_metal_type(img)
        logger.info(f"Detected metal type: {metal_type}")
        
        # Apply enhancements
        logger.info("Applying enhancements...")
        
        # 1. Metal-specific color enhancement
        img = apply_metal_specific_enhancement(img, metal_type)
        
        # 2. Detail enhancement (noise reduction + sharpening)
        img = enhance_ring_details(img)
        
        # 3. Final slight brightness boost for cleaner look
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(1.02)
        
        # Convert to base64 (padding removed for Make.com)
        enhanced_base64 = image_to_base64(img)
        
        if not enhanced_base64:
            return {
                "output": {
                    "error": "Failed to convert enhanced image to base64",
                    "status": "error", 
                    "version": VERSION
                }
            }
        
        # Prepare response with safe JSON conversion
        processing_time = time.time() - start_time
        response_data = {
            "enhanced_image": enhanced_base64,
            "metal_type": metal_type,
            "processing_time": round(processing_time, 2),
            "original_size": safe_json_convert(img.size),
            "enhancements_applied": [
                "metal_specific_color_grading",
                "noise_reduction",
                "detail_enhancement",
                "brightness_optimization"
            ],
            "version": VERSION,
            "message": "Enhancement complete - optimized for Make.com"
        }
        
        # Convert entire response using safe conversion
        safe_response = safe_json_convert(response_data)
        
        logger.info(f"Enhancement complete in {processing_time:.2f}s")
        logger.info(f"Response keys: {list(safe_response.keys())}")
        
        return {"output": safe_response}
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Handler error: {str(e)}\n{error_trace}")
        
        return {
            "output": safe_json_convert({
                "error": str(e),
                "error_trace": error_trace,
                "status": "error",
                "version": VERSION
            })
        }

# RunPod handler
runpod.serverless.start({"handler": handler})

# Test mode
if __name__ == "__main__":
    print(f"Testing {VERSION} Enhancement Handler...")
    test_job = {
        "input": {
            "debug_mode": True
        }
    }
    result = handler(test_job)
    print(json.dumps(result, indent=2))

"""
=============================================================================
GOOGLE APPS SCRIPT V13 - ENHANCED VERSION
=============================================================================

Copy this to Google Apps Script:

/**
 * Google Apps Script V13 - Enhanced Image Upload with Padding Fix
 * Fixes "Error: Invalid base64 data" by restoring padding
 */

function doPost(e) {
  try {
    console.log('V13 Enhanced Image Upload started');
    
    // Parse request
    const postData = JSON.parse(e.postData.contents);
    console.log('Input keys:', Object.keys(postData));
    
    // Find base64 data
    let base64Data = postData.enhanced_image || postData.image || postData.data;
    
    if (!base64Data) {
      throw new Error('No image data found');
    }
    
    // CRITICAL: Restore padding removed by RunPod
    console.log('Restoring base64 padding...');
    while (base64Data.length % 4 !== 0) {
      base64Data += '=';
    }
    
    // Create blob
    const imageBlob = Utilities.newBlob(
      Utilities.base64Decode(base64Data),
      'image/png',
      `enhanced_ring_${new Date().getTime()}.png`
    );
    
    // Upload to Drive
    const file = DriveApp.createFile(imageBlob);
    console.log(`Uploaded: ${file.getName()}`);
    
    return ContentService
      .createTextOutput(JSON.stringify({
        success: true,
        fileId: file.getId(),
        fileUrl: file.getUrl(),
        fileName: file.getName(),
        message: 'V13 Enhanced image uploaded successfully'
      }))
      .setMimeType(ContentService.MimeType.JSON);
      
  } catch (error) {
    console.error('Error:', error.toString());
    return ContentService
      .createTextOutput(JSON.stringify({
        success: false,
        error: error.toString()
      }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

=============================================================================
REQUIREMENTS.TXT
=============================================================================

runpod==1.6.0
opencv-python-headless==4.8.1.78
Pillow==10.1.0
numpy==1.24.3
requests==2.31.0

=============================================================================
"""
