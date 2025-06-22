import cv2
import numpy as np
from PIL import Image
import base64
import io
import os
import replicate
import time
import requests


# v13.3 Wedding Ring Parameters - Based on 28 pairs of training data
WEDDING_RING_PARAMS = {
    'white_gold': {
        'natural': {'brightness': 1.18, 'contrast': 1.12, 'white_overlay': 0.09,
                   'sharpness': 1.15, 'color_temp_a': -3, 'color_temp_b': -3,
                   'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.01},
        'warm': {'brightness': 1.19, 'contrast': 1.13, 'white_overlay': 0.09,
                'sharpness': 1.14, 'color_temp_a': -3, 'color_temp_b': -3,
                'original_blend': 0.16, 'saturation': 1.03, 'gamma': 1.00},
        'cool': {'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.08,
                'sharpness': 1.16, 'color_temp_a': -3, 'color_temp_b': -3,
                'original_blend': 0.14, 'saturation': 1.01, 'gamma': 1.02}
    },
    'rose_gold': {
        'natural': {'brightness': 1.16, 'contrast': 1.10, 'white_overlay': 0.05,
                   'sharpness': 1.17, 'color_temp_a': 3, 'color_temp_b': 1,
                   'original_blend': 0.17, 'saturation': 1.05, 'gamma': 0.99},
        'warm': {'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.05,
                'sharpness': 1.16, 'color_temp_a': 3, 'color_temp_b': 1,
                'original_blend': 0.18, 'saturation': 1.06, 'gamma': 0.98},
        'cool': {'brightness': 1.15, 'contrast': 1.09, 'white_overlay': 0.04,
                'sharpness': 1.18, 'color_temp_a': 3, 'color_temp_b': 1,
                'original_blend': 0.16, 'saturation': 1.04, 'gamma': 1.00}
    },
    'champagne_gold': {
        'natural': {'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.12,
                   'sharpness': 1.16, 'color_temp_a': -4, 'color_temp_b': -4,
                   'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.00},
        'warm': {'brightness': 1.18, 'contrast': 1.12, 'white_overlay': 0.12,
                'sharpness': 1.15, 'color_temp_a': -4, 'color_temp_b': -4,
                'original_blend': 0.16, 'saturation': 1.03, 'gamma': 0.99},
        'cool': {'brightness': 1.16, 'contrast': 1.10, 'white_overlay': 0.11,
                'sharpness': 1.17, 'color_temp_a': -4, 'color_temp_b': -4,
                'original_blend': 0.14, 'saturation': 1.01, 'gamma': 1.01}
    },
    'yellow_gold': {
        'natural': {'brightness': 1.14, 'contrast': 1.08, 'white_overlay': 0.03,
                   'sharpness': 1.18, 'color_temp_a': 5, 'color_temp_b': 3,
                   'original_blend': 0.19, 'saturation': 1.08, 'gamma': 0.97},
        'warm': {'brightness': 1.15, 'contrast': 1.09, 'white_overlay': 0.03,
                'sharpness': 1.17, 'color_temp_a': 5, 'color_temp_b': 3,
                'original_blend': 0.20, 'saturation': 1.09, 'gamma': 0.96},
        'cool': {'brightness': 1.13, 'contrast': 1.07, 'white_overlay': 0.02,
                'sharpness': 1.19, 'color_temp_a': 5, 'color_temp_b': 3,
                'original_blend': 0.18, 'saturation': 1.07, 'gamma': 0.98}
    }
}

# Background colors from 28 AFTER images
AFTER_BG_COLORS = [
    (250, 249, 248), (251, 250, 249), (248, 247, 246), (249, 248, 247),
    (252, 251, 250), (250, 249, 248), (251, 250, 249), (249, 248, 247),
    (250, 249, 248), (248, 247, 246), (251, 250, 249), (252, 251, 250),
    (249, 248, 247), (250, 249, 248), (251, 250, 249), (248, 247, 246),
    (250, 249, 248), (249, 248, 247), (251, 250, 249), (252, 251, 250),
    (248, 247, 246), (250, 249, 248), (249, 248, 247), (251, 250, 249),
    (250, 249, 248), (252, 251, 250), (249, 248, 247), (251, 250, 249)
]

def detect_metal_type(img):
    """Detect metal type based on color analysis"""
    try:
        h, w = img.shape[:2]
        center_y, center_x = h // 2, w // 2
        
        # Sample center region
        sample_size = min(h, w) // 4
        center_region = img[center_y-sample_size:center_y+sample_size,
                          center_x-sample_size:center_x+sample_size]
        
        # Convert to HSV
        hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        
        # Calculate average hue and saturation
        avg_hue = np.mean(hsv[:, :, 0])
        avg_sat = np.mean(hsv[:, :, 1])
        avg_val = np.mean(hsv[:, :, 2])
        
        # Calculate RGB averages
        avg_b = np.mean(center_region[:, :, 0])
        avg_g = np.mean(center_region[:, :, 1])
        avg_r = np.mean(center_region[:, :, 2])
        
        # Enhanced metal detection logic
        if avg_sat < 30 and avg_val > 180:
            if avg_r > avg_b + 5:
                return 'champagne_gold'
            else:
                return 'white_gold'
        elif avg_r > avg_b + 10 and avg_sat > 30:
            return 'rose_gold'
        elif avg_r > avg_g + 5 and avg_g > avg_b + 5:
            return 'yellow_gold'
        else:
            return 'white_gold'
    except:
        return 'white_gold'

def detect_lighting(img):
    """Detect lighting condition"""
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Analyze color temperature
        avg_a = np.mean(a)
        avg_b = np.mean(b)
        
        if avg_b > 135:  # Warm tones
            return 'warm'
        elif avg_b < 125:  # Cool tones
            return 'cool'
        else:
            return 'natural'
    except:
        return 'natural'

def apply_v13_3_enhancement(img, metal_type, lighting):
    """Apply v13.3 enhancement based on 28 pairs of training data"""
    try:
        params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, WEDDING_RING_PARAMS['white_gold']['natural'])
        
        # Store original
        original = img.copy()
        result = img.astype(np.float32)
        
        # Step 1: Noise reduction
        result = cv2.bilateralFilter(result.astype(np.uint8), 5, 50, 50).astype(np.float32)
        
        # Step 2: Brightness adjustment
        result = result * params['brightness']
        
        # Step 3: Contrast adjustment
        mean = np.mean(result)
        result = (result - mean) * params['contrast'] + mean
        
        # Step 4: Sharpness enhancement
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * (params['sharpness'] - 1)
        kernel[1, 1] = kernel[1, 1] + 1
        sharpened = cv2.filter2D(result, -1, kernel)
        result = cv2.addWeighted(result, 1 - (params['sharpness'] - 1), sharpened, params['sharpness'] - 1, 0)
        
        # Step 5: Saturation adjustment
        hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= params['saturation']
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        
        # Step 6: White overlay (key for "white coating" effect)
        white_overlay = np.ones_like(result) * 255
        result = cv2.addWeighted(result, 1 - params['white_overlay'], white_overlay, params['white_overlay'], 0)
        
        # Step 7: Color temperature adjustment
        result[:, :, 2] = np.clip(result[:, :, 2] + params['color_temp_a'], 0, 255)  # Red channel
        result[:, :, 0] = np.clip(result[:, :, 0] + params['color_temp_b'], 0, 255)  # Blue channel
        
        # Step 8: CLAHE enhancement
        lab = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)
        
        # Step 9: Gamma correction
        result = np.power(result / 255.0, params['gamma']) * 255.0
        
        # Step 10: Blend with original
        result = cv2.addWeighted(original.astype(np.float32), params['original_blend'], 
                               result, 1 - params['original_blend'], 0)
        
        # Special handling for champagne gold to make it whiter
        if metal_type == 'champagne_gold':
            # Additional brightness and white overlay
            result = result * 1.30  # Increase brightness
            white_overlay = np.ones_like(result) * 255
            result = cv2.addWeighted(result, 0.85, white_overlay, 0.15, 0)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error in v13.3 enhancement: {e}")
        return img

def remove_padding_safe(base64_string):
    """Remove base64 padding safely for Make.com compatibility"""
    return base64_string.rstrip('=')

def handler(event):
    """Enhancement handler for wedding ring images"""
    try:
        # Extract image data
        input_data = event.get('input', {})
        image_data = input_data.get('image')
        
        if not image_data:
            return {"output": {"error": "No image provided in input", "status": "error"}}
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        
        # Detect metal type and lighting
        metal_type = detect_metal_type(img)
        lighting = detect_lighting(img)
        
        print(f"Detected: {metal_type} with {lighting} lighting")
        
        # Apply v13.3 enhancement
        enhanced = apply_v13_3_enhancement(img, metal_type, lighting)
        
        # Prepare for Replicate API
        client = replicate.Client(api_token=os.environ.get('REPLICATE_API_TOKEN'))
        
        # Convert to base64 for Replicate
        _, buffer = cv2.imencode('.png', enhanced)
        enhanced_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Use Replicate for background removal/refinement
        try:
            # Using Ideogram v2-turbo for fast processing
            output = client.run(
                "ideogram-ai/ideogram-v2-turbo",
                input={
                    "prompt": "wedding ring on clean white background, professional product photography, high quality, centered",
                    "image": f"data:image/png;base64,{enhanced_base64}",
                    "style": "photo",
                    "magic_prompt_option": "off",
                    "seed": 42
                }
            )
            
            # Download result
            if output and isinstance(output, list) and len(output) > 0:
                response = requests.get(output[0])
                if response.status_code == 200:
                    result_img = Image.open(io.BytesIO(response.content))
                    enhanced = cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Replicate processing failed, using direct enhancement: {e}")
        
        # Upscale 2x
        new_h, new_w = h * 2, w * 2
        upscaled = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create thumbnail (1000x1300)
        thumb_h, thumb_w = 1300, 1000
        
        # Find ring in the image
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (ring)
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            # Add padding
            pad = 50
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(enhanced.shape[1], x + w + pad)
            y2 = min(enhanced.shape[0], y + h + pad)
            
            # Crop ring region
            ring_crop = enhanced[y1:y2, x1:x2]
            
            # Resize to fit thumbnail maintaining aspect ratio
            crop_h, crop_w = ring_crop.shape[:2]
            scale = min(thumb_w/crop_w, thumb_h/crop_h) * 0.9
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            
            resized = cv2.resize(ring_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Create white background
            thumbnail = np.ones((thumb_h, thumb_w, 3), dtype=np.uint8) * 255
            
            # Center the ring
            y_offset = (thumb_h - new_h) // 4  # Place higher up
            x_offset = (thumb_w - new_w) // 2
            
            thumbnail[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        else:
            # Fallback: resize entire image
            thumbnail = cv2.resize(enhanced, (thumb_w, thumb_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', upscaled, [cv2.IMWRITE_JPEG_QUALITY, 95])
        main_base64 = remove_padding_safe(base64.b64encode(buffer).decode('utf-8'))
        
        _, buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 95])
        thumb_base64 = remove_padding_safe(base64.b64encode(buffer).decode('utf-8'))
        
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "status": "success",
                    "version": "v152"
                }
            }
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "error"
            }
        }
