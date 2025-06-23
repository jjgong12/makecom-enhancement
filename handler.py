import runpod
import base64
import requests
from io import BytesIO
from PIL import Image, ImageEnhance
import numpy as np
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_image_data(job_input):
    """
    다양한 키에서 이미지 데이터 찾기
    """
    logger.info(f"Input keys: {list(job_input.keys())}")
    
    # 가능한 키들 체크
    possible_keys = ['image', 'image_base64', 'enhanced_image', 'base64', 'img', 'data', 'imageData', 'image_url']
    
    # 1. 직접 키 체크
    for key in possible_keys:
        if key in job_input and job_input[key]:
            logger.info(f"Found image in key: {key}")
            return job_input[key]
    
    # 2. 중첩된 구조 체크
    nested_paths = [
        ['input', 'image'],
        ['data', 'image'],
        ['body', 'image'],
        ['payload', 'image'],
        ['input', 'enhanced_image'],
        ['data', 'enhanced_image']
    ]
    
    for path in nested_paths:
        current = job_input
        try:
            for key in path:
                current = current.get(key, {})
            if current and isinstance(current, str):
                logger.info(f"Found image in nested path: {'.'.join(path)}")
                return current
        except:
            continue
    
    # 3. 모든 키 순회하며 base64 패턴 찾기
    for key, value in job_input.items():
        if isinstance(value, str) and len(value) > 1000:  # base64는 보통 길다
            if value.startswith('data:image') or looks_like_base64(value):
                logger.info(f"Found base64-like string in key: {key}")
                return value
        elif isinstance(value, dict):
            # 재귀적으로 찾기
            result = find_image_data(value)
            if result:
                return result
    
    return None

def looks_like_base64(s):
    """Base64 패턴인지 확인"""
    # Base64 패턴 체크 - 문자열 제대로 닫기
    base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
    return bool(base64_pattern.match(s[:100]))  # 처음 100자만 체크

def decode_image_data(image_data):
    """
    이미지 데이터 디코딩 (URL 또는 base64)
    """
    if image_data.startswith('http'):
        # URL인 경우
        logger.info(f"Fetching image from URL: {image_data}")
        response = requests.get(image_data)
        return response.content
    else:
        # Base64인 경우
        # data:image/png;base64, 접두사 제거
        if image_data.startswith('data:'):
            image_data = image_data.split(',', 1)[1]
        
        # 4가지 디코딩 시도
        for method in range(4):
            try:
                if method == 0:
                    # Direct decode
                    return base64.b64decode(image_data)
                elif method == 1:
                    # Add padding
                    padded = image_data + '=' * (4 - len(image_data) % 4)
                    return base64.b64decode(padded)
                elif method == 2:
                    # URL-safe decode
                    return base64.urlsafe_b64decode(image_data)
                elif method == 3:
                    # Force padding
                    padded = image_data + '==='
                    return base64.b64decode(padded)
            except Exception as e:
                logger.debug(f"Decode method {method} failed: {str(e)}")
                continue
        
        raise ValueError("Failed to decode base64 image")

def enhance_image(img):
    """
    매우 미세한 보정만 적용 - 3번에서 5번 이미지로의 변화처럼
    """
    try:
        # 1. 아주 약간의 밝기 증가 (1.05 = 5% 증가)
        brightness_enhancer = ImageEnhance.Brightness(img)
        img = brightness_enhancer.enhance(1.05)
        
        # 2. 아주 약간의 채도 감소로 더 깨끗한 느낌 (0.95 = 5% 감소)
        color_enhancer = ImageEnhance.Color(img)
        img = color_enhancer.enhance(0.95)
        
        # 3. 배경만 살짝 밝게 (선택적)
        img_array = np.array(img)
        
        # 밝은 픽셀(배경)만 더 밝게
        # 200 이상인 픽셀들을 5% 더 밝게
        mask = np.all(img_array > 200, axis=-1)
        if mask.any():
            for c in range(3):
                img_array[mask, c] = np.minimum(255, img_array[mask, c] * 1.05).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        
        logger.info("Simple brightness enhancement completed")
        return img
        
    except Exception as e:
        logger.error(f"Enhancement error: {str(e)}")
        return img

def handler(job):
    """
    RunPod handler for simple image enhancement
    """
    try:
        job_input = job['input']
        
        # 이미지 데이터 찾기
        image_data_str = find_image_data(job_input)
        if not image_data_str:
            # 디버깅을 위해 사용 가능한 키들 표시
            logger.error(f"No image found. Available keys: {list(job_input.keys())}")
            if job_input:
                # 첫 번째 키의 샘플 보여주기
                first_key = list(job_input.keys())[0]
                sample = str(job_input[first_key])[:200]
                logger.error(f"Sample data from '{first_key}': {sample}")
            raise ValueError("No image provided")
        
        # 이미지 디코딩
        image_data = decode_image_data(image_data_str)
        
        # 이미지 열기
        img = Image.open(BytesIO(image_data))
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        # 심플한 보정 적용
        enhanced_img = enhance_image(img)
        
        # 결과 인코딩 (padding 제거)
        buffered = BytesIO()
        enhanced_img.save(buffered, format="PNG", quality=95)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Padding 제거
        img_base64 = img_base64.rstrip('=')
        
        return {
            "output": {
                "enhanced_image": img_base64,
                "status": "success",
                "message": "Simple enhancement completed"
            }
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "failed"
            }
        }

runpod.serverless.start({"handler": handler})
