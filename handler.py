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
VERSION = "v25-enhancement"

class WeddingRingEnhancerV25:
    """v25 Wedding Ring Enhancement - Fixed Shadow Issues & Better Enhancement"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Fixed Shadow & Enhanced Color")
    
    def apply_simple_enhancement(self, image):
        """Simple but effective color enhancement - v25 improved"""
        try:
            # 1. Brightness - more increase for whiter look
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.15)  # Increased from 1.1
            
            # 2. Contrast - slightly more
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.08)  # Increased from 1.05
            
            # 3. Color saturation - keep subtle
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.03)  # Slightly increased from 1.02
            
            # 4. Background color adjustment WITHOUT shadow
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # Brighter background color
            background_color = (250, 248, 245)  # Even brighter beige
            
            # FIX: Create mask without edge darkening
            # Instead of blending edges, we'll brighten the whole image uniformly
            # This prevents the shadow effect at edges
            
            # Simple brightness overlay instead of edge blending
            brightness_overlay = np.full((h, w, 3), background_color, dtype=np.float32)
            
            # Very subtle uniform blending (10% only)
            for i in range(3):
                img_np[:, :, i] = img_np[:, :, i] * 0.9 + brightness_overlay[:, :, i] * 0.1
            
            # Additional overall brightness boost to match desired result
            img_np = np.clip(img_np * 1.02, 0, 255)
            
            return Image.fromarray(img_np.astype(np.uint8))
            
        except Exception as e:
            print(f"[{VERSION}] Enhancement error: {e}")
            return image

def handler(job):
    """RunPod handler function - V25 FIXED"""
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
        enhancer = WeddingRingEnhancerV25()
        enhanced_image = enhancer.apply_simple_enhancement(image)
        print(f"[{VERSION}] Enhancement applied")
        
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
    print("V25 - Fixed Shadow Issues & Better Enhancement")
    print("IMPORTANT: Google Apps Script must add padding back!")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
