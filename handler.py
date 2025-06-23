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
VERSION = "v19-enhancement"

class WeddingRingEnhancerV19:
    """v19 Wedding Ring Enhancement - Simple Color Enhancement with Input Fix"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Simple Enhancement with correct input handling")
    
    def apply_simple_enhancement(self, image):
        """Simple color enhancement only - based on v16/v18"""
        try:
            # 1. Brightness slight increase
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)
            
            # 2. Contrast slight increase
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)
            
            # 3. Color saturation fine tuning
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.02)
            
            # 4. Soft beige background blending
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # Beige background color
            background_color = (245, 243, 240)
            
            # Create edge blending mask
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(mask, (50, 50), (w-50, h-50), 1.0, -1)
            mask = cv2.GaussianBlur(mask, (101, 101), 50)
            
            # Blend background color (30% only)
            for i in range(3):
                img_np[:, :, i] = img_np[:, :, i] * mask + background_color[i] * (1 - mask) * 0.3
            
            return Image.fromarray(img_np.astype(np.uint8))
            
        except Exception as e:
            print(f"[{VERSION}] Enhancement error: {e}")
            return image

# Global instance
enhancer_instance = None

def get_enhancer():
    """Singleton enhancer instance"""
    global enhancer_instance
    if enhancer_instance is None:
        enhancer_instance = WeddingRingEnhancerV19()
    return enhancer_instance

def find_base64_in_dict(data, depth=0, max_depth=10):
    """Find base64 image in nested dictionary - from v16"""
    if depth > max_depth:
        return None
    
    if isinstance(data, str) and len(data) > 100:
        return data
    
    if isinstance(data, dict):
        for key in ['image', 'base64', 'data', 'input', 'file']:
            if key in data and isinstance(data[key], str) and len(data[key]) > 100:
                return data[key]
        
        for value in data.values():
            result = find_base64_in_dict(value, depth + 1, max_depth)
            if result:
                return result
    
    elif isinstance(data, list):
        for item in data:
            result = find_base64_in_dict(item, depth + 1, max_depth)
            if result:
                return result
    
    return None

def decode_base64_image(base64_str):
    """Decode base64 string to PIL Image"""
    try:
        # Handle Data URL format
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        # Remove whitespace
        base64_str = base64_str.strip()
        
        # Add padding if needed
        padding = 4 - len(base64_str) % 4
        if padding != 4:
            base64_str += '=' * padding
        
        # Decode
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
        
    except Exception as e:
        print(f"[{VERSION}] Error decoding base64: {e}")
        raise

def encode_image_to_base64(image, format='PNG'):
    """Encode image to base64 (Make.com compatible)"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=95 if format == 'JPEG' else None)
        buffer.seek(0)
        
        # Base64 encoding - MUST remove padding for Make.com
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # CRITICAL: Remove padding for Make.com compatibility
        base64_str = base64_str.rstrip('=')
        
        return base64_str
        
    except Exception as e:
        print(f"[{VERSION}] Error encoding image: {e}")
        raise

def handler(job):
    """RunPod handler function"""
    try:
        start_time = time.time()
        job_input = job["input"]
        
        print(f"[{VERSION}] Processing started")
        print(f"[{VERSION}] Input keys: {list(job_input.keys())}")
        
        # Find base64 image in nested structure
        base64_image = find_base64_in_dict(job_input)
        if not base64_image:
            return {
                "output": {
                    "error": "No image data found in input",
                    "version": VERSION,
                    "success": False
                }
            }
        
        # Decode image
        image = decode_base64_image(base64_image)
        print(f"[{VERSION}] Image decoded: {image.size}")
        
        # Apply simple color enhancement
        enhancer = get_enhancer()
        enhanced = enhancer.apply_simple_enhancement(image)
        
        # Encode result
        enhanced_base64 = encode_image_to_base64(enhanced)
        
        # Processing time
        processing_time = time.time() - start_time
        print(f"[{VERSION}] Processing completed in {processing_time:.2f}s")
        
        # Return structure for Make.com
        # Make.com path: {{4.data.output.output.enhanced_image}}
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "success": True,
                "version": VERSION,
                "processing_time": round(processing_time, 2)
            }
        }
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"[{VERSION}] {error_msg}")
        traceback.print_exc()
        
        return {
            "output": {
                "error": error_msg,
                "success": False,
                "version": VERSION
            }
        }

# RunPod start
if __name__ == "__main__":
    print("="*70)
    print(f"Wedding Ring Enhancement {VERSION}")
    print("Simple Enhancement Handler (a_file)")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
