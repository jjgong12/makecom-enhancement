import runpod
import base64
import requests
from io import BytesIO
from PIL import Image, ImageEnhance
import time

def brightness_correction(image):
    """원본 이미지의 전체적인 색감을 밝게 하얗게 보정"""
    enhancer = ImageEnhance.Brightness(image)
    brightened = enhancer.enhance(1.3)
    
    enhancer = ImageEnhance.Color(brightened)
    enhanced = enhancer.enhance(0.85)
    
    return enhanced

def enhance_with_replicate(image_path):
    """Replicate API를 사용하여 배경 제거 및 품질 향상"""
    try:
        # 이미지를 base64로 인코딩
        if isinstance(image_path, Image.Image):
            buffered = BytesIO()
            image_path.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
        else:
            with open(image_path, "rb") as img_file:
                img_str = base64.b64encode(img_file.read()).decode()
        
        image_uri = f"data:image/png;base64,{img_str}"
        
        # Replicate API 호출
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": "Bearer r8_CjO6q1W0Zh3mDnvtpuI6eBYuLLFqoP22D7cKs",
                "Content-Type": "application/json"
            },
            json={
                "version": "4067ee2a58f6c161d434a9c077cbc012dd2549b451aa0a3652cf0e7dc6c5da9f",
                "input": {
                    "image": image_uri,
                    "to_remove": ""
                }
            }
        )
        
        if response.status_code != 201:
            print(f"API 요청 실패: {response.status_code}")
            return None
            
        prediction = response.json()
        prediction_id = prediction['id']
        
        # 결과 대기
        max_attempts = 30
        for _ in range(max_attempts):
            time.sleep(1)
            
            response = requests.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers={"Authorization": "Bearer r8_CjO6q1W0Zh3mDnvtpuI6eBYuLLFqoP22D7cKs"}
            )
            
            result = response.json()
            
            if result['status'] == 'succeeded':
                output_url = result['output']
                
                img_response = requests.get(output_url)
                if img_response.status_code == 200:
                    return Image.open(BytesIO(img_response.content))
                else:
                    print(f"이미지 다운로드 실패: {img_response.status_code}")
                    return None
                    
            elif result['status'] == 'failed':
                print("처리 실패")
                return None
        
        print("시간 초과")
        return None
        
    except Exception as e:
        print(f"Replicate API 오류: {str(e)}")
        return None

def handler(job):
    """RunPod handler function"""
    try:
        job_input = job['input']
        
        # base64 이미지 가져오기
        base64_image = job_input.get('image')
        if not base64_image:
            return {"output": {"error": "No image provided", "success": False}}
        
        # base64 디코드
        try:
            # data:image/xxx;base64, 부분 제거
            if ',' in base64_image:
                base64_image = base64_image.split(',')[1]
            
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
            
            # RGBA로 변환
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
                
        except Exception as e:
            return {"output": {"error": f"Failed to decode image: {str(e)}", "success": False}}
        
        # 1. 먼저 전체적인 색감 보정
        brightened_image = brightness_correction(image)
        
        # 2. Replicate API로 배경 제거 및 품질 향상
        enhanced_image = enhance_with_replicate(brightened_image)
        
        if enhanced_image is None:
            # Replicate 실패 시 보정된 이미지만 반환
            enhanced_image = brightened_image
        
        # 결과를 base64로 인코딩
        output_buffer = BytesIO()
        enhanced_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        
        # Make.com용 - padding 제거
        enhanced_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        enhanced_base64 = enhanced_base64.rstrip('=')  # padding 제거
        
        return {
            "output": {
                "success": True,
                "enhanced_image": enhanced_base64,
                "message": "Image enhanced successfully"
            }
        }
        
    except Exception as e:
        return {
            "output": {
                "error": str(e),
                "success": False
            }
        }

runpod.serverless.start({"handler": handler})
