import runpod
import base64
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import io
import os
import time
import traceback

# Version
VERSION = "v35-enhancement"

def find_base64_in_dict(data, depth=0, max_depth=10):
    """Find base64 image in nested dictionary"""
    if depth > max_depth:
        return None
    
    if isinstance(data, str) and len(data) > 100:
        return data
    
    if isinstance(data, dict):
        for key in ['image_base64', 'image', 'base64', 'data', 'input', 'file', 'imageData']:
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
        # Handle data URL format
        if ',' in base64_str and base64_str.startswith('data:'):
            base64_str = base64_str.split(',')[1]
        
        # Clean base64
        base64_str = base64_str.strip()
        
        # Add padding for decoding
        padding = 4 - len(base64_str) % 4
        if padding != 4:
            base64_str += '=' * padding
        
        # Decode
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
        
    except Exception as e:
        print(f"[{VERSION}] Error decoding base64: {e}")
        raise

def encode_image_to_base64(image, format='JPEG'):
    """Encode image to base64 (Make.com compatible)"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=95)
        buffer.seek(0)
        
        # Base64 encode
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # CRITICAL: Remove padding for Make.com
        base64_str = base64_str.rstrip('=')
        
        return base64_str
        
    except Exception as e:
        print(f"[{VERSION}] Error encoding image: {e}")
        raise

def enhance_wedding_ring(image):
    """v35 Enhancement - Much brighter for white gold/plain white"""
    try:
        print(f"[{VERSION}] Starting v35 enhancement - extra bright white gold")
        
        # 1. Strong brightness boost for white metals
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.20)  # Increased from 1.12
        
        # 2. Contrast adjustment
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.10)  # Slightly increased
        
        # 3. Reduce saturation for whiter appearance
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(0.92)  # More desaturated for white metals
        
        # 4. Convert to numpy for advanced processing
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        # 5. Apply bright white background
        white_color = (254, 254, 254)  # Almost pure white
        
        # Create edge mask
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        
        # Create background mask
        mask = np.ones((h, w), dtype=np.float32)
        mask[edges_dilated > 0] = 0
        mask = cv2.GaussianBlur(mask, (31, 31), 15)
        
        # Apply white background more strongly
        for i in range(3):
            img_np[:, :, i] = img_np[:, :, i] * (1 - mask * 0.18) + white_color[i] * mask * 0.18
        
        # 6. Gamma correction for extra brightness
        gamma = 0.88  # Lower gamma = brighter
        img_np = np.power(img_np / 255.0, gamma) * 255
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        # 7. Additional brightness boost for white metals
        # Detect bright areas (likely white gold/plain white)
        bright_mask = gray > 180
        bright_mask = cv2.GaussianBlur(bright_mask.astype(np.float32), (15, 15), 7)
        
        # Brighten white metal areas
        for i in range(3):
            img_np[:, :, i] = np.clip(img_np[:, :, i] * (1 + bright_mask * 0.08), 0, 255)
        
        # 8. Final overall brightness
        img_np = np.clip(img_np * 1.05, 0, 255).astype(np.uint8)
        
        # 9. Light sharpening
        image = Image.fromarray(img_np)
        image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=30, threshold=3))
        
        print(f"[{VERSION}] Enhancement complete - extra bright for white metals")
        return image
        
    except Exception as e:
        print(f"[{VERSION}] Error in enhancement: {e}")
        traceback.print_exc()
        return image

def handler(job):
    """RunPod handler - v35 Enhancement"""
    print(f"[{VERSION}] ====== Enhancement Handler Started ======")
    print(f"[{VERSION}] Extra bright processing for white metals")
    
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        print(f"[{VERSION}] Input type: {type(job_input)}")
        print(f"[{VERSION}] Input keys: {list(job_input.keys()) if isinstance(job_input, dict) else 'Not a dict'}")
        
        # Find base64 image
        base64_image = find_base64_in_dict(job_input)
        
        if not base64_image:
            # Try direct string
            if isinstance(job_input, str) and len(job_input) > 100:
                base64_image = job_input
            else:
                return {
                    "output": {
                        "enhanced_image": None,
                        "error": "No image data found",
                        "success": False,
                        "version": VERSION,
                        "debug_info": {
                            "input_keys": list(job_input.keys()) if isinstance(job_input, dict) else [],
                            "input_length": len(str(job_input))
                        }
                    }
                }
        
        print(f"[{VERSION}] Base64 image found, length: {len(base64_image)}")
        
        # Decode image
        try:
            image = decode_base64_image(base64_image)
            print(f"[{VERSION}] Image decoded: {image.size}")
        except Exception as e:
            return {
                "output": {
                    "enhanced_image": None,
                    "error": f"Failed to decode image: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Check image size and resize if too large
        max_dimension = 4000
        if image.width > max_dimension or image.height > max_dimension:
            print(f"[{VERSION}] Image too large ({image.size}), resizing...")
            ratio = max_dimension / max(image.width, image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"[{VERSION}] Resized to: {image.size}")
        
        # Apply enhancement
        try:
            enhanced_image = enhance_wedding_ring(image)
            print(f"[{VERSION}] Enhancement applied successfully")
        except Exception as e:
            print(f"[{VERSION}] Error during enhancement: {e}")
            traceback.print_exc()
            enhanced_image = image
        
        # Encode result
        try:
            enhanced_base64 = encode_image_to_base64(enhanced_image, format='JPEG')
            print(f"[{VERSION}] Enhanced image encoded, length: {len(enhanced_base64)}")
        except Exception as e:
            return {
                "output": {
                    "enhanced_image": None,
                    "error": f"Failed to encode enhanced image: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return proper structure for Make.com
        result = {
            "output": {
                "enhanced_image": enhanced_base64,
                "success": True,
                "version": VERSION,
                "processing_time": round(processing_time, 2),
                "original_size": list(image.size),
                "enhancements_applied": [
                    "brightness_boost_1.20",
                    "contrast_1.10",
                    "desaturation_0.92",
                    "white_background_blend",
                    "gamma_correction_0.88",
                    "white_metal_brightening",
                    "final_brightness_1.05"
                ],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "warning": "Google Script must add padding: while (base64Data.length % 4 !== 0) { base64Data += '='; }"
            }
        }
        
        print(f"[{VERSION}] ====== Success - Returning Enhanced Image ======")
        print(f"[{VERSION}] Total processing time: {processing_time:.2f}s")
        
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
    print("V35 - Extra Bright for White Gold & Plain White")
    print("Features:")
    print("- 20% brightness boost (up from 12%)")
    print("- Stronger desaturation for white metals")
    print("- Additional white metal brightening")
    print("- Lower gamma for brighter appearance")
    print("- Max dimension limit: 4000px")
    print("CRITICAL: Padding is removed for Make.com")
    print("Google Apps Script MUST add padding back:")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
