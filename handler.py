import runpod
import base64
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import io
import os
import traceback
import time

# Version info
VERSION = "v31-enhancement"

class WeddingRingEnhancerV31:
    """v31 Wedding Ring Enhancement - Simple & Reliable"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Simple & Reliable Enhancement")
    
    def apply_simple_enhancement(self, image):
        """Simple color enhancement for bright white results"""
        try:
            # 1. Pre-sharpening for clarity
            image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=80, threshold=2))
            
            # 2. Brightness increase
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.18)
            
            # 3. Contrast for clarity
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.15)
            
            # 4. Slight saturation reduction for cleaner whites
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.95)
            
            # 5. Convert to numpy for background whitening
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # 6. Simple white background blend
            # Create white overlay
            white_color = (250, 250, 250)
            
            # Create simple edge mask
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_dilated = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
            
            # Create mask
            mask = np.ones((h, w), dtype=np.float32)
            mask[edges_dilated > 0] = 0
            mask = cv2.GaussianBlur(mask, (51, 51), 25)
            
            # Apply white overlay
            for i in range(3):
                img_np[:, :, i] = img_np[:, :, i] * (1 - mask * 0.15) + white_color[i] * mask * 0.15
            
            # 7. Simple gamma correction for brightness
            gamma = 0.9
            img_np = np.power(img_np / 255.0, gamma) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # 8. Final sharpness
            img_pil = Image.fromarray(img_np)
            enhancer = ImageEnhance.Sharpness(img_pil)
            img_pil = enhancer.enhance(1.2)
            
            return img_pil
            
        except Exception as e:
            print(f"[{VERSION}] Enhancement error: {e}")
            traceback.print_exc()
            return image

def handler(job):
    """RunPod handler function - V31 FIXED"""
    print(f"[{VERSION}] ====== Handler Started ======")
    
    try:
        job_input = job.get("input", {})
        print(f"[{VERSION}] Input type: {type(job_input)}")
        print(f"[{VERSION}] Input keys: {list(job_input.keys()) if isinstance(job_input, dict) else 'Not a dict'}")
        
        # Find base64 image
        base64_image = None
        
        # Direct access attempts - CRITICAL FIX: Added 'image_base64'
        if isinstance(job_input, dict):
            for key in ['image_base64', 'image', 'base64', 'data', 'input', 'file', 'imageData']:
                if key in job_input:
                    value = job_input[key]
                    if isinstance(value, str) and len(value) > 100:
                        base64_image = value
                        print(f"[{VERSION}] Found image in key: {key}")
                        break
        
        # Check nested structure
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
        
        # Direct string input
        if not base64_image and isinstance(job_input, str) and len(job_input) > 100:
            base64_image = job_input
            print(f"[{VERSION}] Input was direct base64 string")
        
        if not base64_image:
            print(f"[{VERSION}] ERROR: No base64 image found!")
            return {
                "output": {
                    "enhanced_image": None,
                    "error": "No image data found in input",
                    "success": False,
                    "version": VERSION,
                    "debug_info": {
                        "input_keys": list(job_input.keys()) if isinstance(job_input, dict) else [],
                        "input_sample": str(job_input)[:200] if job_input else "Empty"
                    }
                }
            }
        
        # Process the image
        print(f"[{VERSION}] Base64 length: {len(base64_image)}")
        
        # Handle data URL format
        if ',' in base64_image and base64_image.startswith('data:'):
            base64_image = base64_image.split(',')[1]
            print(f"[{VERSION}] Removed data URL prefix")
        
        # Clean base64
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
            print(f"[{VERSION}] DECODE ERROR: {e}")
            return {
                "output": {
                    "enhanced_image": None,
                    "error": f"Failed to decode base64: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Apply enhancement
        enhancer = WeddingRingEnhancerV31()
        enhanced_image = enhancer.apply_simple_enhancement(image)
        print(f"[{VERSION}] Enhancement applied")
        
        # Convert back to base64
        buffer = io.BytesIO()
        enhanced_image.save(buffer, format='PNG', quality=95)
        buffer.seek(0)
        
        # Encode to base64
        enhanced_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # CRITICAL: Remove padding for Make.com
        enhanced_base64 = enhanced_base64.rstrip('=')
        
        print(f"[{VERSION}] Enhanced base64 length: {len(enhanced_base64)}")
        
        # Create output structure that Make.com expects
        # RunPod wraps this in {"data": {"output": ...}}
        # So Make.com path becomes: {{4.data.output.output.enhanced_image}}
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
        print(f"[{VERSION}] Output keys: {list(result['output'].keys())}")
        print(f"[{VERSION}] Enhanced image exists: {'enhanced_image' in result['output'] and result['output']['enhanced_image'] is not None}")
        
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
    print("V31 - Simple & Reliable (Based on V24)")
    print("Features:")
    print("- Simple brightness (1.18)")
    print("- Moderate contrast (1.15)")
    print("- Subtle white background")
    print("- Reliable processing")
    print("IMPORTANT: Google Apps Script must add padding back!")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
