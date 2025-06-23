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
VERSION = "v18-enhancement"

class WeddingRingEnhancerV18:
    """v18 Wedding Ring Enhancement - Simple Color Enhancement with Fixes"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Simple Enhancement with Make.com fixes")
    
    def apply_simple_enhancement(self, image):
        """간단한 색감 보정만 적용 - v16 기반"""
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
            
            # 4. 부드러운 베이지 배경색 블렌딩
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # 베이지 배경색
            background_color = (245, 243, 240)
            
            # 가장자리 블렌딩 마스크 생성
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(mask, (30, 30), (w-30, h-30), 1.0, -1)
            mask = cv2.GaussianBlur(mask, (61, 61), 30)
            
            # 배경색 블렌딩 (30% 정도만)
            for i in range(3):
                img_np[:, :, i] = img_np[:, :, i] * mask + background_color[i] * (1 - mask) * 0.3
            
            return Image.fromarray(img_np.astype(np.uint8))
            
        except Exception as e:
            print(f"[{VERSION}] Enhancement error: {e}")
            return image

def handler(job):
    """RunPod handler function - Simple enhancement only"""
    print(f"[{VERSION}] Handler started")
    job_input = job['input']
    
    try:
        # Get image data
        if 'image' not in job_input:
            raise ValueError("No 'image' field in input")
        
        image_data = job_input['image']
        
        # Decode base64 image
        if image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"[{VERSION}] Image loaded: {image.size}")
        
        # Create enhancer instance
        enhancer = WeddingRingEnhancerV18()
        
        # Apply simple enhancement
        enhanced_image = enhancer.apply_simple_enhancement(image)
        
        # Convert to base64
        buffered = io.BytesIO()
        enhanced_image.save(buffered, format="JPEG", quality=95)
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # IMPORTANT: Remove padding for Make.com
        enhanced_base64 = enhanced_base64.rstrip('=')
        
        # Return with proper structure for Make.com
        # Make.com expects: {{4.data.output.output.enhanced_image}}
        return {
            "output": {
                "enhanced_image": f"data:image/jpeg;base64,{enhanced_base64}",
                "status": "success",
                "version": VERSION,
                "processing_time": time.time() - job.get('start_time', time.time())
            }
        }
        
    except Exception as e:
        error_msg = f"Error in enhancement: {str(e)}\n{traceback.format_exc()}"
        print(f"[{VERSION}] {error_msg}")
        
        return {
            "output": {
                "error": error_msg,
                "status": "error",
                "version": VERSION
            }
        }

# RunPod serverless handler
runpod.serverless.start({"handler": handler})
