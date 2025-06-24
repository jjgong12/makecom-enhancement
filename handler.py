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
VERSION = "v30-enhancement"

class WeddingRingEnhancerV30:
    """v30 Wedding Ring Enhancement - Clear & Sharp like Image 3"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Clear & Sharp Enhancement")
    
    def apply_simple_enhancement(self, image):
        """Clear and sharp enhancement prioritizing clarity - v30"""
        try:
            # 1. FIRST: Strong sharpening to maintain detail
            # This is KEY - apply sharpening BEFORE brightness
            image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=100, threshold=1))
            
            # 2. Moderate brightness - not too much
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.15)  # Reduced from 1.28 to prevent washout
            
            # 3. Strong contrast for clarity
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.22)  # Increased for more definition
            
            # 4. Slight saturation reduction
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.94)  # Keep some color for natural look
            
            # 5. Convert to numpy for advanced processing
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # 6. Apply moderate CLAHE for balanced brightness
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(img_lab)
            
            # Moderate CLAHE to avoid over-brightening
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l_channel = clahe.apply(l_channel)
            
            # Slight L channel boost only
            l_channel = cv2.add(l_channel, 8)  # Much less than v29's 15
            l_channel = np.clip(l_channel, 0, 255)
            
            img_lab = cv2.merge([l_channel, a_channel, b_channel])
            img_np = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            
            # 7. Clean white background with less overlay
            background_color = (251, 250, 249)  # Slightly off-white for natural look
            
            # Edge detection for precise masking
            edges = cv2.Canny(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), 50, 150)
            edges_dilated = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=3)
            
            # Create mask avoiding product
            mask = np.ones((h, w), dtype=np.float32)
            mask[edges_dilated > 0] = 0
            mask = cv2.GaussianBlur(mask, (31, 31), 15)
            
            # Apply white overlay more subtly
            for i in range(3):
                img_np[:, :, i] = img_np[:, :, i] * (1 - mask * 0.12) + background_color[i] * mask * 0.12
            
            # 8. Gentle gamma correction
            gamma = 0.92  # Less aggressive than v29
            img_np = np.power(img_np / 255.0, gamma) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # 9. Detail enhancement using high-pass filter
            # This makes the image crisp like image 3
            img_float = img_np.astype(np.float32)
            
            # High-pass filter for detail extraction
            blurred = cv2.GaussianBlur(img_float, (0, 0), 3)
            detail = img_float - blurred
            
            # Add detail back with strength
            img_float = img_float + detail * 0.5
            img_np = np.clip(img_float, 0, 255).astype(np.uint8)
            
            # 10. Final sharpening pass
            img_pil = Image.fromarray(img_np)
            enhancer = ImageEnhance.Sharpness(img_pil)
            img_pil = enhancer.enhance(1.4)
            
            # 11. Micro-contrast enhancement for that "crisp" look
            img_np = np.array(img_pil)
            
            # Local contrast enhancement
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]]) / 9.0
            
            sharpened = cv2.filter2D(img_np, -1, kernel)
            img_np = cv2.addWeighted(img_np, 0.8, sharpened, 0.2, 0)
            
            return Image.fromarray(img_np.astype(np.uint8))
            
        except Exception as e:
            print(f"[{VERSION}] Enhancement error: {e}")
            traceback.print_exc()
            return image

def handler(job):
    """RunPod handler function - V30 CLEAR & SHARP"""
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
            print(f"[{VERSION}] Added {padding} padding for decoding")
        
        # Decode base64 to image
        try:
            img_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(img_data))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            print(f"[{VERSION}] Image decoded successfully: {image.size}")
        except Exception as e:
            print(f"[{VERSION}] Decode error: {e}")
            traceback.print_exc()
            return {
                "output": {
                    "enhanced_image": None,
                    "error": f"Failed to decode base64: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Apply enhancement
        try:
            enhancer = WeddingRingEnhancerV30()
            enhanced_image = enhancer.apply_simple_enhancement(image)
            print(f"[{VERSION}] Enhancement applied with clear & sharp settings")
        except Exception as e:
            print(f"[{VERSION}] Enhancement error: {e}")
            traceback.print_exc()
            return {
                "output": {
                    "enhanced_image": None,
                    "error": f"Enhancement failed: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Convert back to base64
        try:
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
        except Exception as e:
            print(f"[{VERSION}] Encoding error: {e}")
            traceback.print_exc()
            return {
                "output": {
                    "enhanced_image": None,
                    "error": f"Failed to encode image: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
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
        print(f"[{VERSION}] Result structure check - has 'output' key: {'output' in result}")
        print(f"[{VERSION}] Result['output'] keys: {list(result['output'].keys())}")
        
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
    print("V30 - Clear & Sharp Enhancement (Like Image 3)")
    print("Features:")
    print("- Sharpening first priority")
    print("- Moderate brightness (1.15)")
    print("- Strong contrast (1.22)")
    print("- High-pass detail enhancement")
    print("- Micro-contrast for crisp look")
    print("IMPORTANT: Google Apps Script must add padding back!")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
