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
VERSION = "v33-enhancement"

class WeddingRingEnhancerV33:
    """v33 Wedding Ring Enhancement - Brighter Enhancement"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Brighter Enhancement")
    
    def apply_simple_enhancement(self, image):
        """Slightly brighter color enhancement for whiter results"""
        try:
            # 1. Light sharpening first - like image 3
            image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=50, threshold=3))
            
            # 2. More brightness increase
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.12)  # Increased from 1.08
            
            # 3. Slightly more contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.08)  # Increased from 1.06
            
            # 4. Keep colors natural but cleaner
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.96)  # Slightly less saturation
            
            # 5. Convert to numpy for more white background
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # 6. More white background blend
            # Create whiter overlay
            white_color = (252, 252, 252)  # More white
            
            # Create simple edge mask
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 60, 150)
            edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
            
            # Create mask
            mask = np.ones((h, w), dtype=np.float32)
            mask[edges_dilated > 0] = 0
            mask = cv2.GaussianBlur(mask, (31, 31), 15)
            
            # Apply more white overlay
            for i in range(3):
                img_np[:, :, i] = img_np[:, :, i] * (1 - mask * 0.12) + white_color[i] * mask * 0.12
            
            # 7. More aggressive gamma correction for brightness
            gamma = 0.92  # More brightness
            img_np = np.power(img_np / 255.0, gamma) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # 8. Additional brightness push
            # Simple brightness adjustment
            img_np = np.clip(img_np * 1.03, 0, 255).astype(np.uint8)
            
            # 9. Final minimal sharpness
            img_pil = Image.fromarray(img_np)
            enhancer = ImageEnhance.Sharpness(img_pil)
            img_pil = enhancer.enhance(1.1)
            
            return img_pil
            
        except Exception as e:
            print(f"[{VERSION}] Enhancement error: {e}")
            traceback.print_exc()
            return image

def find_base64_in_dict(data, depth=0, max_depth=10):
    """중첩된 딕셔너리에서 base64 이미지 찾기"""
    if depth > max_depth:
        return None
    
    if isinstance(data, str) and len(data) > 100:
        return data
    
    if isinstance(data, dict):
        # 우선순위 키들 먼저 체크
        for key in ['image_base64', 'image', 'base64', 'data', 'input', 'file', 'imageData']:
            if key in data and isinstance(data[key], str) and len(data[key]) > 100:
                return data[key]
        
        # 모든 값 재귀적으로 체크
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
    """Base64 문자열을 PIL Image로 디코드"""
    try:
        # Data URL 형식 처리
        if ',' in base64_str and base64_str.startswith('data:'):
            base64_str = base64_str.split(',')[1]
        
        # 공백 제거
        base64_str = base64_str.strip()
        
        # Padding 추가 (디코딩용)
        padding = 4 - len(base64_str) % 4
        if padding != 4:
            base64_str += '=' * padding
        
        # 디코드
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
        # RGB로 변환
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
        
    except Exception as e:
        print(f"[{VERSION}] Error decoding base64: {e}")
        raise

def encode_image_to_base64(image, format='PNG'):
    """이미지를 base64로 인코딩 (Make.com 호환)"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=95 if format == 'JPEG' else None)
        buffer.seek(0)
        
        # Base64 인코딩
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # CRITICAL: Make.com을 위해 padding 제거
        base64_str = base64_str.rstrip('=')
        
        return base64_str
        
    except Exception as e:
        print(f"[{VERSION}] Error encoding image: {e}")
        raise

def handler(job):
    """RunPod handler function - V33 BRIGHTER ENHANCEMENT"""
    print(f"[{VERSION}] ====== Handler Started ======")
    print(f"[{VERSION}] Job start time recorded")
    
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        print(f"[{VERSION}] Input type: {type(job_input)}")
        print(f"[{VERSION}] Input keys: {list(job_input.keys()) if isinstance(job_input, dict) else 'Not a dict'}")
        
        # 입력이 딕셔너리가 아닌 경우 처리
        if not isinstance(job_input, dict):
            if isinstance(job_input, str) and len(job_input) > 100:
                # 입력이 직접 base64 문자열인 경우
                base64_image = job_input
                print(f"[{VERSION}] Direct base64 string input")
            else:
                print(f"[{VERSION}] Invalid input type: {type(job_input)}")
                return {
                    "output": {
                        "enhanced_image": None,
                        "error": "Invalid input type",
                        "success": False,
                        "version": VERSION
                    }
                }
        else:
            # Find base64 image using recursive search
            base64_image = find_base64_in_dict(job_input)
            
            if not base64_image:
                print(f"[{VERSION}] No base64 image found in input")
                return {
                    "output": {
                        "enhanced_image": None,
                        "error": "No image data found in input",
                        "success": False,
                        "version": VERSION,
                        "debug_info": {
                            "input_keys": list(job_input.keys()),
                            "input_sample": str(job_input)[:200]
                        }
                    }
                }
        
        print(f"[{VERSION}] Base64 image found, length: {len(base64_image)}")
        
        # Decode the image
        try:
            image = decode_base64_image(base64_image)
            print(f"[{VERSION}] Image decoded successfully: {image.size}")
        except Exception as e:
            print(f"[{VERSION}] Failed to decode image: {e}")
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
            enhancer = WeddingRingEnhancerV33()
            enhanced_image = enhancer.apply_simple_enhancement(image)
            print(f"[{VERSION}] Enhancement applied successfully")
        except Exception as e:
            print(f"[{VERSION}] Enhancement failed: {e}")
            return {
                "output": {
                    "enhanced_image": None,
                    "error": f"Enhancement failed: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Encode the result
        try:
            enhanced_base64 = encode_image_to_base64(enhanced_image, format='PNG')
            print(f"[{VERSION}] Enhanced image encoded, length: {len(enhanced_base64)}")
        except Exception as e:
            print(f"[{VERSION}] Failed to encode result: {e}")
            return {
                "output": {
                    "enhanced_image": None,
                    "error": f"Failed to encode result: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
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
                "processing_time": round(processing_time, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            }
        }
        
        print(f"[{VERSION}] ====== Success - Returning Result ======")
        print(f"[{VERSION}] Processing time: {processing_time:.2f}s")
        print(f"[{VERSION}] Output structure ready for Make.com")
        
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
    print("V33 - Brighter Enhancement")
    print("Features:")
    print("- More brightness (1.12)")
    print("- More contrast (1.08)")
    print("- Whiter background (252,252,252)")
    print("- Additional brightness push (1.03x)")
    print("- Gamma 0.92 for brighter results")
    print("- Recursive base64 search")
    print("- Full error handling")
    print("IMPORTANT: Google Apps Script must add padding back!")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
