import runpod
import base64
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import io
import os
import traceback
import time

# Version info
VERSION = "v27-enhancement"

class WeddingRingEnhancerV27:
    """v27 Wedding Ring Enhancement - Soft White without Overexposure"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Soft White Enhancement")
    
    def apply_simple_enhancement(self, image):
        """Soft white enhancement without overexposure - v27"""
        try:
            # 1. Brightness - reduced to prevent overexposure
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.18)  # Reduced from 1.25
            
            # 2. Contrast - moderate
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.10)  # Reduced from 1.12
            
            # 3. Color saturation - very subtle for white look
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.95)  # More desaturated for purer white
            
            # 4. Soft white background
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # Soft white background (not pure white to avoid harsh contrast)
            background_color = (248, 246, 243)  # Soft warm white
            
            # Create smooth gradient overlay instead of uniform
            y_gradient = np.linspace(0.7, 1.0, h).reshape((h, 1))
            x_gradient = np.linspace(1.0, 1.0, w).reshape((1, w))
            gradient_mask = y_gradient * x_gradient
            
            # Apply gradient overlay for natural lighting
            for i in range(3):
                overlay = background_color[i] * gradient_mask
                img_np[:, :, i] = img_np[:, :, i] * 0.88 + overlay * 0.12
            
            # Soft brightness adjustment (no harsh boost)
            img_np = np.clip(img_np * 1.02, 0, 255)
            
            # Gentle gamma correction
            gamma = 0.95  # Less aggressive than before
            img_np = np.power(img_np / 255.0, gamma) * 255
            img_np = np.clip(img_np, 0, 255)
            
            # Prevent overexposure in bright areas
            bright_mask = np.max(img_np, axis=2) > 240
            if np.any(bright_mask):
                for i in range(3):
                    img_np[bright_mask, i] = img_np[bright_mask, i] * 0.97
            
            return Image.fromarray(img_np.astype(np.uint8))
            
        except Exception as e:
            print(f"[{VERSION}] Enhancement error: {e}")
            return image

def handler(job):
    """RunPod handler function - V27 SOFT WHITE"""
    print(f"[{VERSION}] ====== Handler Started ======")
    
    try:
        job_input = job.get("input", {})
        print(f"[{VERSION}] Input type: {type(job_input)}")
        print(f"[{VERSION}] Input keys: {list(job_input.keys()) if isinstance(job_input, dict) else 'Not a dict'}")
        
        # Find base64 image - with 'image_base64' as priority
        base64_image = None
        
        # Direct access attempts
        if isinstance(job_input, dict):
            # Priority order including 'image_base64'
            for key in ['image_base64', 'image', 'base64', 'data', 'input', 'file', 'imageData']:
                if key in job_input:
                    value = job_input[key]
                    if isinstance(value, str) and len(value) > 100:
                        base64_image = value
                        print(f"[{VERSION}] Found image in key: {key}")
                        break
        
        # If still not found, check nested structure
        if not base64_image and isinstance(job_input, dict):
            for key, value in job_input.items():
                if isinstance(value, dict):
                    for sub_key in ['image_base64', 'image', 'base64', 'data']:
                        if sub_key in value and isinstance(value[sub_key], str) and len(value[sub_key]) > 100:
                            base64_image = value[sub_key]
                            print(f"[{VERSION}] Found image in nested: {key}.{sub_key}")
                            break
                if base64_image:
                    break
        
        # Last resort - if input is string
        if not base64_image and isinstance(job_input, str) and len(job_input) > 100:
            base64_image = job_input
            print(f"[{VERSION}] Input was direct base64 string")
        
        if not base64_image:
            return {
                "output": {
                    "enhanced_image": None,
                    "error": "No image data found in input",
                    "success": False,
                    "version": VERSION,
                    "debug_info": {
                        "input_keys": list(job_input.keys()) if isinstance(job_input, dict) else [],
                        "first_key": list(job_input.keys())[0] if isinstance(job_input, dict) and job_input else None
                    }
                }
            }
        
        # Process the image
        print(f"[{VERSION}] Base64 length: {len(base64_image)}")
        
        # Handle data URL format
        if ',' in base64_image and base64_image.startswith('data:'):
            base64_image = base64_image.split(',')[1]
            print(f"[{VERSION}] Removed data URL prefix")
        
        # Remove any whitespace
        base64_image = base64_image.strip()
        
        # Add padding if needed for decoding
        padding = 4 - len(base64_image) % 4
        if padding != 4:
            base64_image += '=' * padding
        
        # Decode base64 to image
        try:
            img_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(img_data))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            print(f"[{VERSION}] Image decoded successfully: {image.size}")
        except Exception as e:
            return {
                "output": {
                    "enhanced_image": None,
                    "error": f"Failed to decode base64: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Apply enhancement
        enhancer = WeddingRingEnhancerV27()
        enhanced_image = enhancer.apply_simple_enhancement(image)
        print(f"[{VERSION}] Enhancement applied with soft white settings")
        
        # Convert back to base64
        buffer = io.BytesIO()
        enhanced_image.save(buffer, format='PNG', quality=95)
        buffer.seek(0)
        
        # Encode to base64
        enhanced_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # CRITICAL: Remove padding for Make.com compatibility
        # Google Apps Script must add padding back:
        # while (base64Data.length % 4 !== 0) { base64Data += '='; }
        enhanced_base64 = enhanced_base64.rstrip('=')
        
        print(f"[{VERSION}] Enhanced base64 length: {len(enhanced_base64)}")
        
        # Return proper structure for Make.com
        # RunPod wraps this in {"data": {"output": ...}}
        # Make.com path: {{4.data.output.output.enhanced_image}}
        result = {
            "output": {
                "enhanced_image": enhanced_base64,
                "success": True,
                "version": VERSION,
                "original_size": list(image.size),
                "enhanced_size": list(enhanced_image.size),
                "processing_time": time.time() - job.get('start_time', time.time())
            }
        }
        
        print(f"[{VERSION}] ====== Success - Returning Result ======")
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"[{VERSION}] CRITICAL ERROR: {error_msg}")
        traceback.print_exc()
        
        return {
            "output": {
                "enhanced_image": None,
                "error": error_msg,
                "success": False,
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

# RunPod serverless start
if __name__ == "__main__":
    print("="*70)
    print(f"Wedding Ring Enhancement {VERSION}")
    print("V27 - Soft White Enhancement without Overexposure")
    print("IMPORTANT: Google Apps Script must add padding back!")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
