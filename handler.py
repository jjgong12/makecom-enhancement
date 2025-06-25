import json
import runpod
import base64
import requests
import time
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

def find_input_data(data):
    """재귀적으로 모든 가능한 경로에서 입력 데이터 찾기"""
    
    # 전체 구조 로깅 (디버깅용)
    print(f"전체 입력 데이터 구조: {json.dumps(data, indent=2)[:1000]}")
    
    # 직접 접근 시도
    if isinstance(data, dict):
        # 최상위 레벨 체크
        if 'input' in data:
            return data['input']
        
        # 일반적인 RunPod 구조들
        common_paths = [
            ['job', 'input'],
            ['data', 'input'],
            ['payload', 'input'],
            ['body', 'input'],
            ['request', 'input']
        ]
        
        for path in common_paths:
            current = data
            for key in path:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    break
            else:
                return current
    
    # 재귀적 탐색 - image_base64도 찾기
    def recursive_search(obj, target_keys=['input', 'url', 'image_url', 'imageUrl', 'image_base64', 'imageBase64']):
        if isinstance(obj, dict):
            for key in target_keys:
                if key in obj:
                    return obj[key] if key == 'input' else {key: obj[key]}
            
            for value in obj.values():
                result = recursive_search(value, target_keys)
                if result:
                    return result
        elif isinstance(obj, list):
            for item in obj:
                result = recursive_search(item, target_keys)
                if result:
                    return result
        
        return None
    
    result = recursive_search(data)
    print(f"재귀 탐색 결과: {result}")
    return result

def download_image_from_url(url):
    """URL에서 이미지를 다운로드하여 PIL Image 객체로 반환"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # 컨텐츠 타입 확인
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type and not url.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                print(f"경고: 이미지가 아닌 컨텐츠 타입: {content_type}")
            
            return Image.open(BytesIO(response.content))
            
        except Exception as e:
            print(f"다운로드 시도 {attempt + 1}/{max_retries} 실패: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise

def base64_to_image(base64_string):
    """Base64 문자열을 PIL Image로 변환"""
    # padding 복원
    padding = 4 - len(base64_string) % 4
    if padding != 4:
        base64_string += '=' * padding
    
    # data URL 형식 처리
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_data))

def enhance_image_brightness(image):
    """이미지의 전체적인 색감을 매우 밝고 하얗게 보정 (더 강한 버전)"""
    # RGB로 변환 (RGBA인 경우)
    if image.mode == 'RGBA':
        # 흰색 배경에 합성
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 1단계: 전체적으로 매우 밝게
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.5)  # 50% 밝게 (기존 30% -> 50%)
    
    # 2단계: 대비 감소로 부드럽게
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(0.85)  # 15% 대비 감소
    
    # 3단계: 채도 크게 감소 (더 하얗게)
    color = ImageEnhance.Color(image)
    image = color.enhance(0.6)  # 40% 채도 감소 (기존 20% -> 40%)
    
    # 4단계: 강력한 하이라이트 강화
    img_array = np.array(image)
    
    # RGB 각 채널별로 처리
    for i in range(3):
        channel = img_array[:, :, i].astype(np.float32) / 255.0
        
        # 감마 보정을 더 강하게 (밝은 톤 극대화)
        channel = np.power(channel, 0.6)  # 기존 0.8 -> 0.6
        
        # 추가로 밝은 영역을 더 밝게
        channel = np.where(channel > 0.5, 
                          channel + (1 - channel) * 0.3,  # 밝은 부분 30% 더 밝게
                          channel * 1.1)  # 어두운 부분도 10% 밝게
        
        # 클리핑
        channel = np.clip(channel, 0, 1)
        img_array[:, :, i] = (channel * 255).astype(np.uint8)
    
    # 5단계: 전체적으로 화이트 오버레이 효과
    white_overlay = np.ones_like(img_array) * 255
    alpha = 0.15  # 15% 흰색 오버레이
    img_array = (img_array * (1 - alpha) + white_overlay * alpha).astype(np.uint8)
    
    # PIL Image로 변환
    enhanced_image = Image.fromarray(img_array)
    
    # 6단계: 추가 밝기 조정
    brightness2 = ImageEnhance.Brightness(enhanced_image)
    enhanced_image = brightness2.enhance(1.1)  # 추가 10% 밝게
    
    # 7단계: 샤프니스는 약간만
    sharpness = ImageEnhance.Sharpness(enhanced_image)
    enhanced_image = sharpness.enhance(1.05)  # 5% 선명도 증가
    
    return enhanced_image

def image_to_base64(image, format='JPEG'):
    """PIL Image를 base64 문자열로 변환"""
    buffered = BytesIO()
    if format == 'JPEG' and image.mode == 'RGBA':
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[3])
        image = rgb_image
    
    image.save(buffered, format=format, quality=95)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Make.com을 위해 padding 제거
    img_base64_no_padding = img_base64.rstrip('=')
    
    print(f"Base64 길이 (padding 제거): {len(img_base64_no_padding)}")
    
    return img_base64_no_padding

def handler(event):
    """Enhancement 핸들러 함수"""
    try:
        print("Enhancement 이벤트 수신:", json.dumps(event, indent=2)[:500])
        
        # 입력 데이터 찾기
        input_data = find_input_data(event)
        
        if not input_data:
            raise ValueError("입력 데이터를 찾을 수 없습니다")
        
        # 이미지 소스 확인 (URL 또는 Base64)
        image = None
        
        # Base64 입력 처리
        if isinstance(input_data, dict):
            if 'image_base64' in input_data or 'imageBase64' in input_data:
                base64_str = input_data.get('image_base64') or input_data.get('imageBase64')
                print("Base64 이미지 입력 감지")
                image = base64_to_image(base64_str)
            elif 'url' in input_data or 'image_url' in input_data or 'imageUrl' in input_data:
                image_url = input_data.get('url') or input_data.get('image_url') or input_data.get('imageUrl')
                print(f"URL 입력 감지: {image_url}")
                image = download_image_from_url(image_url)
        elif isinstance(input_data, str):
            # 문자열인 경우 URL로 가정
            if input_data.startswith('http'):
                print(f"URL 문자열 입력: {input_data}")
                image = download_image_from_url(input_data)
            else:
                # Base64 문자열로 가정
                print("Base64 문자열 입력 감지")
                image = base64_to_image(input_data)
        
        if not image:
            raise ValueError(f"이미지를 로드할 수 없습니다. 입력: {input_data}")
        
        print(f"이미지 로드 완료: {image.size}")
        
        # 원본 크기 저장
        original_size = image.size
        
        # 이미지 향상 처리 (매우 밝고 하얗게)
        enhanced_image = enhance_image_brightness(image)
        print("이미지 향상 처리 완료 (강화된 버전)")
        
        # base64 변환 (padding 제거)
        enhanced_base64 = image_to_base64(enhanced_image)
        
        # 중첩된 output 구조로 반환
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "original_size": list(original_size),
                "enhanced_size": list(enhanced_image.size),
                "format": "base64_no_padding",
                "enhancement_applied": "strong_brightness_whitening_v56"
            }
        }
        
    except Exception as e:
        print(f"Enhancement 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "output": {
                "error": str(e),
                "status": "failed"
            }
        }

# RunPod 핸들러 등록
runpod.serverless.start({"handler": handler})
