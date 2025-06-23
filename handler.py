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
VERSION = "v22-enhancement"

class WeddingRingEnhancerV22:
    """v22 Wedding Ring Enhancement - Simple Color Enhancement"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Simple Enhancement")
    
    def apply_simple_enhancement(self, image):
        """간단한 색감 보정만 적용"""
        try:
            # 1. 밝기 살짝 증가
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)
            
            # 2. 대비 약간 증가
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)
            
            # 3. 색상 채도 미세 조정
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.02)
            
            # 4. 배경색 부드럽게 조정
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # 전체적으로 밝은 톤 적용
            background_color = (245, 243, 240)  # 부드러운 베이지
            
            # 가장자리만 살짝 블렌딩
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(mask, (50, 50), (w-50, h-50), 1.0, -1)
            mask = cv2.GaussianBlur(mask, (101, 101), 50)
            
            # 배경색 블렌딩
            for i in range(3):
                img_np[:, :, i] = img_np[:, :, i] * mask + background_color[i] * (1 - mask) * 0.3
            
            return Image.fromarray(img_np.astype(np.uint8))
            
        except Exception as e:
            print(f"[{VERSION}] Enhancement error: {e}")
            return image

def handler(job):
    """RunPod handler function - FIXED OUTPUT STRUCTURE"""
    print(f"[{VERSION}] ====== Handler Started ======")
    
    try:
        job_input = job.get("input", {})
        print(f"[{VERSION}] Input type: {type(job_input)}")
        print(f"[{VERSION}] Input keys: {list(job_input.keys()) if isinstance(job_input, dict) else 'Not a dict'}")
        
        # Find base64 image - try multiple possible locations
        base64_image = None
        
        # Direct access attempts
        if isinstance(job_input, dict):
            # Try common keys
            for key in ['image', 'base64', 'data', 'input', 'file', 'imageData']:
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
                    for sub_key in ['image', 'base64', 'data']:
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
            error_result = {
                "output": {
                    "enhanced_image": None,
                    "error": "No image data found in input",
                    "success": False,
                    "version": VERSION,
                    "debug_info": {
                        "input_type": str(type(job_input)),
                        "input_keys": list(job_input.keys()) if isinstance(job_input, dict) else [],
                        "input_sample": str(job_input)[:200] if job_input else "Empty"
                    }
                }
            }
            print(f"[{VERSION}] ERROR: No image found, returning: {error_result}")
            return error_result
        
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
            error_result = {
                "output": {
                    "enhanced_image": None,
                    "error": f"Failed to decode base64: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
            print(f"[{VERSION}] ERROR decoding: {e}")
            return error_result
        
        # Apply enhancement
        enhancer = WeddingRingEnhancerV22()
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
        # So Make.com path will be: {{4.data.output.output.enhanced_image}}
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
        print(f"[{VERSION}] Output structure: {list(result.keys())}")
        print(f"[{VERSION}] Output.output keys: {list(result['output'].keys())}")
        
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
    print("Enhanced Handler with Fixed Output Structure")
    print("Make.com path: {{4.data.output.output.enhanced_image}}")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
