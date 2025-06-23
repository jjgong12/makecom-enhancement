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
VERSION = "v16-enhancement"

class WeddingRingEnhancerV16:
    """v16 Wedding Ring Enhancement - Simple Color Enhancement"""
    
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

# 전역 인스턴스
enhancer_instance = None

def get_enhancer():
    """싱글톤 enhancer 인스턴스"""
    global enhancer_instance
    if enhancer_instance is None:
        enhancer_instance = WeddingRingEnhancerV16()
    return enhancer_instance

def find_base64_in_dict(data, depth=0, max_depth=10):
    """중첩된 딕셔너리에서 base64 이미지 찾기"""
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
    """Base64 문자열을 PIL Image로 디코드"""
    try:
        # Data URL 형식 처리
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        # 공백 제거
        base64_str = base64_str.strip()
        
        # Padding 추가
        padding = 4 - len(base64_str) % 4
        if padding != 4:
            base64_str += '=' * padding
        
        # 디코드
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
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
        
        # Base64 인코딩 - padding 제거 안함!
        # Google Script는 padding이 있어야 함
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return base64_str
        
    except Exception as e:
        print(f"[{VERSION}] Error encoding image: {e}")
        raise

def handler(job):
    """RunPod 핸들러"""
    try:
        start_time = time.time()
        job_input = job["input"]
        
        print(f"[{VERSION}] Processing started")
        
        # Base64 이미지 찾기
        base64_image = find_base64_in_dict(job_input)
        if not base64_image:
            return {
                "output": {
                    "error": "No image data found",
                    "version": VERSION,
                    "success": False
                }
            }
        
        # 이미지 디코드
        image = decode_base64_image(base64_image)
        print(f"[{VERSION}] Image decoded: {image.size}")
        
        # 간단한 색감 보정
        enhancer = get_enhancer()
        enhanced = enhancer.apply_simple_enhancement(image)
        
        # 결과 인코딩
        enhanced_base64 = encode_image_to_base64(enhanced)
        
        # 처리 시간
        processing_time = time.time() - start_time
        print(f"[{VERSION}] Processing completed in {processing_time:.2f}s")
        
        # Return 구조
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

# RunPod 시작
if __name__ == "__main__":
    print("="*70)
    print(f"Wedding Ring Enhancement {VERSION}")
    print("Simple Enhancement Handler (a_파일)")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
