import runpod
import base64
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import cv2
import io
import os
import json
import traceback
import time
from typing import Dict, Any, Tuple, Optional, List

# Version info
VERSION = "v14-enhancement"

# Import Replicate only when available
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    print(f"[{VERSION}] Replicate not available")

class WeddingRingEnhancerV14:
    """v14 Wedding Ring Enhancement System - Ultra Detection"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Ultra Detection & Make.com Fix")
        self.replicate_client = None
        
        # 38 pairs learning data parameters (28 + 10)
        self.enhancement_params = {
            'yellow_gold': {
                'natural': {
                    'brightness': 1.25, 'saturation': 1.15, 'contrast': 1.05,
                    'sharpness': 1.35, 'noise_reduction': 8,
                    'highlight_boost': 0.12, 'shadow_lift': 0.08,
                    'white_overlay': 0.15, 's_mult': 0.85, 'v_mult': 1.10
                },
                'warm': {
                    'brightness': 1.30, 'saturation': 1.20, 'contrast': 1.08,
                    'sharpness': 1.40, 'noise_reduction': 10,
                    'highlight_boost': 0.15, 'shadow_lift': 0.10,
                    'white_overlay': 0.18, 's_mult': 0.88, 'v_mult': 1.12
                },
                'cool': {
                    'brightness': 1.20, 'saturation': 1.10, 'contrast': 1.02,
                    'sharpness': 1.30, 'noise_reduction': 7,
                    'highlight_boost': 0.10, 'shadow_lift': 0.06,
                    'white_overlay': 0.12, 's_mult': 0.82, 'v_mult': 1.08
                }
            },
            'rose_gold': {
                'natural': {
                    'brightness': 1.22, 'saturation': 1.12, 'contrast': 1.04,
                    'sharpness': 1.32, 'noise_reduction': 9,
                    'highlight_boost': 0.11, 'shadow_lift': 0.07,
                    'white_overlay': 0.13, 's_mult': 0.83, 'v_mult': 1.09
                },
                'warm': {
                    'brightness': 1.28, 'saturation': 1.18, 'contrast': 1.07,
                    'sharpness': 1.38, 'noise_reduction': 11,
                    'highlight_boost': 0.14, 'shadow_lift': 0.09,
                    'white_overlay': 0.16, 's_mult': 0.86, 'v_mult': 1.11
                },
                'cool': {
                    'brightness': 1.18, 'saturation': 1.08, 'contrast': 1.01,
                    'sharpness': 1.28, 'noise_reduction': 8,
                    'highlight_boost': 0.09, 'shadow_lift': 0.05,
                    'white_overlay': 0.10, 's_mult': 0.80, 'v_mult': 1.07
                }
            },
            'white_gold': {
                'natural': {
                    'brightness': 1.18, 'saturation': 0.98, 'contrast': 1.12,
                    'sharpness': 1.15, 'noise_reduction': 6,
                    'highlight_boost': 0.18, 'shadow_lift': 0.03,
                    'white_overlay': 0.09, 'color_temp_a': -3, 'color_temp_b': -3
                },
                'warm': {
                    'brightness': 1.20, 'saturation': 1.00, 'contrast': 1.14,
                    'sharpness': 1.18, 'noise_reduction': 7,
                    'highlight_boost': 0.20, 'shadow_lift': 0.04,
                    'white_overlay': 0.11, 'color_temp_a': -2, 'color_temp_b': -2
                },
                'cool': {
                    'brightness': 1.16, 'saturation': 0.96, 'contrast': 1.10,
                    'sharpness': 1.12, 'noise_reduction': 5,
                    'highlight_boost': 0.16, 'shadow_lift': 0.02,
                    'white_overlay': 0.08, 'color_temp_a': -4, 'color_temp_b': -4
                }
            },
            'plain_white': {  # 무도금화이트
                'natural': {
                    'brightness': 1.35, 'saturation': 0.90, 'contrast': 1.02,
                    'sharpness': 1.05, 'noise_reduction': 3,
                    'highlight_boost': 0.22, 'shadow_lift': 0.12,
                    'white_overlay': 0.20, 's_mult': 0.75, 'v_mult': 1.15
                }
            }
        }
        
        # AFTER background colors
        self.after_bg_colors = {
            'yellow_gold': (244, 242, 236),
            'rose_gold': (243, 241, 238),
            'white_gold': (246, 246, 246),
            'plain_white': (247, 247, 247)
        }
    
    def detect_metal_type(self, image_np):
        """웨딩링 금속 타입 감지"""
        # 중앙 영역 샘플링
        h, w = image_np.shape[:2]
        center_region = image_np[h//3:2*h//3, w//3:2*w//3]
        
        # HSV 변환
        hsv = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
        avg_hue = np.mean(hsv[:, :, 0])
        avg_sat = np.mean(hsv[:, :, 1])
        avg_val = np.mean(hsv[:, :, 2])
        
        # 평균 RGB
        avg_color = np.mean(center_region.reshape(-1, 3), axis=0)
        r, g, b = avg_color
        
        # 금속 타입 판별
        if avg_sat < 30 and avg_val > 180:  # 낮은 채도, 높은 명도
            if r > 240 and g > 240 and b > 235:
                return 'plain_white'
            else:
                return 'white_gold'
        elif 15 < avg_hue < 35 and avg_sat > 50:  # 노란색 계열
            return 'yellow_gold'
        elif avg_hue < 15 and r > g and avg_sat > 40:  # 붉은색 계열
            return 'rose_gold'
        else:
            return 'white_gold'  # 기본값
    
    def apply_v13_enhancement(self, image, metal_type, lighting='natural'):
        """v13.3 파라미터 기반 향상"""
        params = self.enhancement_params.get(metal_type, {}).get(lighting, {})
        if not params:
            params = self.enhancement_params['white_gold']['natural']
        
        # 1. 밝기 조정
        if 'brightness' in params:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(params['brightness'])
        
        # 2. 대비 조정
        if 'contrast' in params:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(params['contrast'])
        
        # 3. 채도 조정
        if 'saturation' in params:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(params['saturation'])
        
        # 4. 선명도 조정
        if 'sharpness' in params:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(params['sharpness'])
        
        # 5. 색상 조정 (HSV)
        if any(k in params for k in ['s_mult', 'v_mult']):
            img_np = np.array(image)
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            if 's_mult' in params:
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params['s_mult'], 0, 255)
            if 'v_mult' in params:
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * params['v_mult'], 0, 255)
            
            img_np = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            image = Image.fromarray(img_np)
        
        # 6. 하이라이트/그림자 조정
        if 'highlight_boost' in params or 'shadow_lift' in params:
            img_np = np.array(image).astype(np.float32) / 255.0
            
            if 'highlight_boost' in params:
                highlights = img_np > 0.7
                img_np[highlights] = np.clip(img_np[highlights] * (1 + params['highlight_boost']), 0, 1)
            
            if 'shadow_lift' in params:
                shadows = img_np < 0.3
                img_np[shadows] = np.clip(img_np[shadows] + params['shadow_lift'], 0, 1)
            
            image = Image.fromarray((img_np * 255).astype(np.uint8))
        
        # 7. 화이트 오버레이
        if 'white_overlay' in params and params['white_overlay'] > 0:
            white = Image.new('RGB', image.size, (255, 255, 255))
            image = Image.blend(image, white, params['white_overlay'])
        
        # 8. 색온도 조정 (LAB)
        if 'color_temp_a' in params or 'color_temp_b' in params:
            img_np = np.array(image)
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            
            if 'color_temp_a' in params:
                lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
            if 'color_temp_b' in params:
                lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
            
            img_np = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            image = Image.fromarray(img_np)
        
        # 9. 노이즈 감소
        if 'noise_reduction' in params and params['noise_reduction'] > 0:
            img_np = np.array(image)
            denoised = cv2.fastNlMeansDenoisingColored(img_np, None, 
                                                       params['noise_reduction'], 
                                                       params['noise_reduction'], 7, 21)
            image = Image.fromarray(denoised)
        
        # 10. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        img_np = np.array(image)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(img_np)
    
    def process_image(self, image_np):
        """메인 처리 파이프라인"""
        try:
            # PIL Image로 변환
            image = Image.fromarray(image_np)
            
            # 1. 금속 타입 감지
            metal_type = self.detect_metal_type(image_np)
            print(f"[{VERSION}] Detected metal type: {metal_type}")
            
            # 2. v13.3 보정 적용 (10단계)
            enhanced = self.apply_v13_enhancement(image, metal_type)
            
            # 3. AFTER 배경색 적용
            if metal_type in self.after_bg_colors:
                bg_color = self.after_bg_colors[metal_type]
                enhanced_np = np.array(enhanced)
                
                # 가장자리 블렌딩
                mask = np.zeros((enhanced_np.shape[0], enhanced_np.shape[1]), dtype=np.float32)
                cv2.rectangle(mask, (30, 30), (enhanced_np.shape[1]-30, enhanced_np.shape[0]-30), 1.0, -1)
                mask = cv2.GaussianBlur(mask, (31, 31), 15)
                
                for i in range(3):
                    enhanced_np[:, :, i] = enhanced_np[:, :, i] * mask + bg_color[i] * (1 - mask)
                
                enhanced = Image.fromarray(enhanced_np.astype(np.uint8))
            
            return enhanced, {
                'metal_type': metal_type,
                'lighting': 'natural',
                'version': VERSION
            }
            
        except Exception as e:
            print(f"[{VERSION}] Processing error: {e}")
            traceback.print_exc()
            raise

# 전역 인스턴스
enhancer_instance = None

def get_enhancer():
    """싱글톤 enhancer 인스턴스"""
    global enhancer_instance
    if enhancer_instance is None:
        enhancer_instance = WeddingRingEnhancerV14()
    return enhancer_instance

def find_base64_in_dict(data, depth=0, max_depth=10):
    """중첩된 딕셔너리에서 base64 이미지 찾기"""
    if depth > max_depth:
        return None
    
    if isinstance(data, str) and len(data) > 100:
        return data
    
    if isinstance(data, dict):
        # 일반적인 키들 먼저 확인
        for key in ['image', 'base64', 'data', 'input', 'file']:
            if key in data and isinstance(data[key], str) and len(data[key]) > 100:
                return data[key]
        
        # 모든 값 확인
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
    """Base64 문자열을 numpy 배열로 디코드"""
    try:
        # Data URL 형식 처리
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        # Padding 추가 시도
        padding = 4 - len(base64_str) % 4
        if padding != 4:
            base64_str += '=' * padding
        
        # 디코드
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
        # RGB로 변환
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return np.array(img)
        
    except Exception as e:
        print(f"[{VERSION}] Error decoding base64: {e}")
        raise

def encode_image_to_base64(image, format='PNG'):
    """이미지를 base64로 인코딩 (Make.com 호환 - padding 제거!)"""
    try:
        # numpy 배열인 경우 PIL Image로 변환
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 버퍼에 저장
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=95 if format == 'JPEG' else None)
        buffer.seek(0)
        
        # Base64 인코딩 후 padding 제거!
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8').rstrip('=')
        
        return base64_str
        
    except Exception as e:
        print(f"[{VERSION}] Error encoding image: {e}")
        raise

def handler(job):
    """RunPod 핸들러 함수"""
    try:
        start_time = time.time()
        job_input = job["input"]
        
        print(f"[{VERSION}] Processing started")
        print(f"[{VERSION}] Input type: {type(job_input)}")
        
        # Base64 이미지 찾기
        base64_image = find_base64_in_dict(job_input)
        if not base64_image:
            print(f"[{VERSION}] No image data found in input")
            return {
                "output": {
                    "error": "No image data found",
                    "version": VERSION,
                    "success": False
                }
            }
        
        print(f"[{VERSION}] Found image data, length: {len(base64_image)}")
        
        # 이미지 디코드
        image_np = decode_base64_image(base64_image)
        print(f"[{VERSION}] Image decoded: {image_np.shape}")
        
        # 처리
        enhancer = get_enhancer()
        enhanced, metadata = enhancer.process_image(image_np)
        
        # 결과 인코딩 (padding 제거!)
        enhanced_base64 = encode_image_to_base64(enhanced)
        
        print(f"[{VERSION}] Enhanced image encoded, length: {len(enhanced_base64)}")
        
        # 처리 시간
        processing_time = time.time() - start_time
        print(f"[{VERSION}] Processing completed in {processing_time:.2f}s")
        
        # Make.com 호환 return 구조!
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "processing_info": metadata,
                "success": True,
                "version": VERSION,
                "processing_time": round(processing_time, 2)
            }
        }
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        print(f"[{VERSION}] {error_msg}")
        traceback.print_exc()
        
        return {
            "output": {
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "success": False,
                "version": VERSION
            }
        }

# RunPod 시작
if __name__ == "__main__":
    print("="*70)
    print(f"Wedding Ring Enhancement {VERSION}")
    print("Enhancement Handler (a_파일)")
    print(f"Replicate Available: {REPLICATE_AVAILABLE}")
    print(f"OpenCV Available: {cv2 is not None}")
    print(f"NumPy Available: {np is not None}")
    print(f"Replicate Token Set: {bool(os.environ.get('REPLICATE_API_TOKEN'))}")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
