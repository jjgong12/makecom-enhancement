import runpod
import base64
import requests
from io import BytesIO
from PIL import Image, ImageEnhance
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enhance_image(img):
    """
    매우 미세한 보정만 적용 - 3번에서 5번 이미지로의 변화처럼
    """
    try:
        # numpy 배열로 변환
        img_array = np.array(img)
        
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
        
        # Base64 이미지 디코딩
        if 'image' in job_input:
            image_data = base64.b64decode(job_input['image'])
        else:
            # Make.com 경로 처리
            image_url = job_input.get('input', {}).get('enhanced_image', '')
            if not image_url:
                raise ValueError("No image provided")
            
            response = requests.get(image_url)
            image_data = response.content
        
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
