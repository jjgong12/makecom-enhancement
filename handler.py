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
VERSION = "v29-enhancement"

class WeddingRingEnhancerV29:
    """v29 Wedding Ring Enhancement - Pure White Enhancement"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Pure White Enhancement")
    
    def apply_simple_enhancement(self, image):
        """Pure white enhancement for clean, bright results - v29"""
        try:
            # 1. Pre-sharpening for clarity
            image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=60, threshold=2))
            
            # 2. Stronger brightness for pure white look
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.28)  # Further increased from 1.22
            
            # 3. Higher contrast for crisp details
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.18)  # Increased from 1.15
            
            # 4. Reduced saturation for cleaner whites
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.92)  # More desaturated for pure white
            
            # 5. Convert to numpy for advanced processing
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # 6. Apply strong CLAHE for even brightness
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(img_lab)
            
            # Stronger CLAHE settings
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l_channel = clahe.apply(l_channel)
            
            # Boost L channel for more brightness
            l_channel = cv2.add(l_channel, 15)  # Add brightness to L channel
            l_channel = np.clip(l_channel, 0, 255)
            
            img_lab = cv2.merge([l_channel, a_channel, b_channel])
            img_np = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            
            # 7. Pure white background
            background_color = (253, 252, 251)  # Almost pure white
            
            # Create strong white overlay
            white_overlay = np.full((h, w, 3), background_color, dtype=np.float32)
            
            # Edge detection for masking
            edges = cv2.Canny(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), 40, 120)
            edges_dilated = cv2.dilate(edges, np.ones((7,7), np.uint8), iterations=2)
            
            # Create mask avoiding product edges
            mask = np.ones((h, w), dtype=np.float32)
            mask[edges_dilated > 0] = 0
            mask = cv2.GaussianBlur(mask, (51, 51), 25)
            
            # Apply white overlay more aggressively
            for i in range(3):
                img_np[:, :, i] = img_np[:, :, i] * (1 - mask * 0.25) + white_overlay[:, :, i] * mask * 0.25
            
            # 8. Gamma correction for overall brightness
            gamma = 0.85  # Lower gamma = brighter image
            img_np = np.power(img_np / 255.0, gamma) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # 9. Additional brightness push
            # Create brightness gradient (darker at top, brighter at bottom)
            brightness_gradient = np.linspace(0.95, 1.05, h).reshape((h, 1))
            brightness_mask = np.repeat(brightness_gradient, w, axis=1)
            
            for i in range(3):
                img_np[:, :, i] = np.clip(img_np[:, :, i] * brightness_mask * 1.03, 0, 255)
            
            # 10. Final color balance for pure white
            # Reduce any color casts
            img_np = img_np.astype(np.float32)
            
            # Calculate average color
            avg_r = np.mean(img_np[:, :, 0])
            avg_g = np.mean(img_np[:, :, 1])
            avg_b = np.mean(img_np[:, :, 2])
            avg_gray = (avg_r + avg_g + avg_b) / 3
            
            # Balance colors toward neutral
            img_np[:, :, 0] *= (avg_gray / avg_r) * 0.95 + 0.05
            img_np[:, :, 1] *= (avg_gray / avg_g) * 0.95 + 0.05
            img_np[:, :, 2] *= (avg_gray / avg_b) * 0.95 + 0.05
            
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # Final sharpening
            img_pil = Image.fromarray(img_np)
            enhancer = ImageEnhance.Sharpness(img_pil)
            img_pil = enhancer.enhance(1.3)
            
            return img_pil
            
        except Exception as e:
            print(f"[{VERSION}] Enhancement error: {e}")
            return image

def handler(job):
    """RunPod handler function - V29 PURE WHITE"""
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
        enhancer = WeddingRingEnhancerV29()
        enhanced_image = enhancer.apply_simple_enhancement(image)
        print(f"[{VERSION}] Enhancement applied with pure white settings")
        
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
    print("V29 - Pure White Enhancement")
    print("Features:")
    print("- Stronger brightness (1.28)")
    print("- Higher contrast (1.18)")
    print("- Reduced saturation (0.92)")
    print("- L channel boost (+15)")
    print("- Aggressive white overlay (25%)")
    print("- Color balance to neutral")
    print("IMPORTANT: Google Apps Script must add padding back!")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
