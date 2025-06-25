import runpod
import base64
import requests
import time
import json
import numpy as np
from PIL import Image, ImageEnhance
import io
import cv2
from typing import Optional, Dict, Any, Union

def find_input_data(data: Dict[str, Any]) -> Optional[Union[str, Dict]]:
    """Find input data from various possible locations"""
    
    # Direct check for common keys
    if isinstance(data, dict):
        # Check for image_base64 first (most common)
        if 'image_base64' in data:
            return data['image_base64']
        if 'imageBase64' in data:
            return data['imageBase64']
        
        # Check for image/base64 keys
        if 'image' in data:
            return data['image']
        if 'base64' in data:
            return data['base64']
        
        # Check for URL keys
        if 'url' in data:
            return data['url']
        if 'image_url' in data:
            return data['image_url']
        if 'imageUrl' in data:
            return data['imageUrl']
            
        # Check input sub-object
        if 'input' in data:
            result = find_input_data(data['input'])
            if result:
                return result
                
        # Check job structure
        if 'job' in data and isinstance(data['job'], dict):
            if 'input' in data['job']:
                result = find_input_data(data['job']['input'])
                if result:
                    return result
                    
        # Check numbered keys (Make.com structure)
        for i in range(10):
            key = str(i)
            if key in data:
                result = find_input_data(data[key])
                if result:
                    return result
                    
        # Deep search in nested structures
        for key, value in data.items():
            if isinstance(value, dict):
                # Skip 'output' to avoid circular references
                if key != 'output':
                    result = find_input_data(value)
                    if result:
                        return result
                        
    return None

def download_image_from_url(url: str) -> Image.Image:
    """Download image from URL with retries"""
    headers_list = [
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
        {'User-Agent': 'Python-Requests/2.31.0'},
        {}
    ]
    
    for headers in headers_list:
        try:
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"Failed with headers {headers}: {e}")
            continue
            
    raise ValueError(f"Failed to download image from URL: {url}")

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image with padding fix"""
    # Remove data URL prefix if present
    if 'base64,' in base64_str:
        base64_str = base64_str.split('base64,')[1]
    
    # Fix padding if needed
    padding = 4 - len(base64_str) % 4
    if padding != 4:
        base64_str += '=' * padding
    
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert('RGB')

def detect_jewelry_color(image: Image.Image) -> str:
    """Detect jewelry color type with improved logic"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Get center region
    h, w = img_array.shape[:2]
    center_y, center_x = h // 2, w // 2
    region_size = min(h, w) // 3
    
    center_region = img_array[
        center_y - region_size:center_y + region_size,
        center_x - region_size:center_x + region_size
    ]
    
    # Convert to HSV
    hsv = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
    
    # Define color ranges (무도금화이트 우선 감지)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 50, 100]), np.array([35, 255, 255]))
    rose_mask = cv2.inRange(hsv, np.array([0, 30, 100]), np.array([15, 255, 255]))
    
    # Count pixels
    white_pixels = cv2.countNonZero(white_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    rose_pixels = cv2.countNonZero(rose_mask)
    total_pixels = center_region.shape[0] * center_region.shape[1]
    
    # Priority detection - 무도금화이트 first
    if white_pixels > total_pixels * 0.5:
        return "white_plain"
    
    # Check average brightness in center
    gray = cv2.cvtColor(center_region, cv2.COLOR_RGB2GRAY)
    avg_brightness = np.mean(gray)
    
    if avg_brightness > 200 and white_pixels > total_pixels * 0.3:
        return "white_plain"
    
    # Other colors
    if yellow_pixels > rose_pixels and yellow_pixels > total_pixels * 0.1:
        # Double check it's really yellow
        avg_b, avg_g, avg_r = np.mean(center_region, axis=(0, 1))
        if avg_r > avg_b and avg_g > avg_b:
            return "yellow_gold"
    
    if rose_pixels > yellow_pixels and rose_pixels > total_pixels * 0.1:
        return "rose_gold"
    
    return "white_gold"

def apply_color_enhancement_v60(image: Image.Image, jewelry_color: str) -> Image.Image:
    """Apply V60 color enhancement with updated values"""
    # Convert to numpy array
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # V60 specific adjustments
    brightness = 1.45  # 45% brightness increase
    contrast = 1.05
    gamma = 0.6
    saturation_factor = 0.65  # 35% saturation decrease
    additional_brightness = 0.02  # 2% additional brightness
    
    # Apply brightness
    img_array = img_array * brightness
    
    # Apply contrast
    img_array = (img_array - 0.5) * contrast + 0.5
    
    # Apply gamma correction
    img_array = np.power(np.clip(img_array, 0, 1), gamma)
    
    # Apply saturation adjustment
    # Convert to HSV for saturation control
    img_array_uint8 = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_array_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1] = hsv[:,:,1] * saturation_factor  # Reduce saturation
    hsv = np.clip(hsv, 0, 255)
    img_array_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    img_array = img_array_uint8.astype(np.float32) / 255.0
    
    # Apply additional brightness
    img_array = img_array + additional_brightness
    
    # Apply background brightening using LAB color space
    lab = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:,:,0] = lab[:,:,0] * 1.1  # Increase L channel (lightness)
    lab = np.clip(lab, 0, 255)
    img_array = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    
    # Color-specific adjustments
    if jewelry_color == "yellow_gold":
        # Subtle warm enhancement
        img_array[:,:,0] *= 1.02  # Red
        img_array[:,:,1] *= 1.01  # Green
    elif jewelry_color == "rose_gold":
        # Subtle rose enhancement
        img_array[:,:,0] *= 1.03  # Red
        img_array[:,:,1] *= 0.98  # Slight green reduction
    elif jewelry_color == "white_plain":
        # Extra brightness for white
        img_array = img_array * 1.05
        # Cool tone
        img_array[:,:,2] *= 1.02  # Slight blue enhancement
    
    # Final clipping
    img_array = np.clip(img_array, 0, 1)
    
    # Convert back to PIL Image
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Main handler function for RunPod"""
    try:
        print(f"Enhancement V60 received event: {json.dumps(event, indent=2)[:500]}...")
        
        # Find input data
        input_data = find_input_data(event)
        
        if not input_data:
            # Try direct access for RunPod structure
            if 'input' in event and isinstance(event['input'], dict):
                input_data = event['input'].get('image') or event['input'].get('image_base64')
            
            if not input_data:
                print("Failed to find input data. Event structure:")
                print(json.dumps(event, indent=2)[:1000])
                return {
                    "output": {
                        "error": "No input image found",
                        "status": "failed",
                        "version": "v60"
                    }
                }
        
        print(f"Found input data type: {type(input_data).__name__}")
        
        # Load image based on input type
        image = None
        
        if isinstance(input_data, str):
            if input_data.startswith('http'):
                print(f"Loading from URL: {input_data[:100]}...")
                image = download_image_from_url(input_data)
            else:
                print("Loading from base64 string")
                image = base64_to_image(input_data)
        elif isinstance(input_data, dict):
            # Check for various possible keys
            for key in ['image_base64', 'imageBase64', 'image', 'base64', 'data']:
                if key in input_data and isinstance(input_data[key], str):
                    print(f"Loading from dict key: {key}")
                    if input_data[key].startswith('http'):
                        image = download_image_from_url(input_data[key])
                    else:
                        image = base64_to_image(input_data[key])
                    break
        
        if image is None:
            return {
                "output": {
                    "error": "Failed to load image from input",
                    "status": "failed",
                    "version": "v60"
                }
            }
        
        print(f"Image loaded successfully: {image.size}")
        
        # Detect jewelry color
        jewelry_color = detect_jewelry_color(image)
        print(f"Detected jewelry color: {jewelry_color}")
        
        # Apply V60 enhancement
        enhanced_image = apply_color_enhancement_v60(image, jewelry_color)
        print("Enhancement applied successfully")
        
        # Convert to base64
        buffered = io.BytesIO()
        enhanced_image.save(buffered, format="PNG", quality=95)
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        enhanced_base64_no_padding = enhanced_base64.rstrip('=')
        print(f"Base64 length (no padding): {len(enhanced_base64_no_padding)}")
        
        # Return with proper structure
        return {
            "output": {
                "enhanced_image": enhanced_base64_no_padding,
                "detected_color": jewelry_color,
                "original_size": list(image.size),
                "enhanced_size": list(enhanced_image.size),
                "version": "v60_complete",
                "status": "success"
            }
        }
        
    except Exception as e:
        print(f"Error in Enhancement V60: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": "v60",
                "traceback": traceback.format_exc()
            }
        }

# RunPod serverless handler
runpod.serverless.start({"handler": handler})
