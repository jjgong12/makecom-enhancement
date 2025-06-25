import runpod
import base64
from io import BytesIO
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import traceback

VERSION = "v53"

def find_input_data(event):
    """Find base64 data from ALL possible locations"""
    # All possible keys for base64 data
    possible_keys = [
        'image_base64', 'imageBase64', 'image', 'base64', 
        'input_image', 'inputImage', 'img', 'photo',
        'base64Image', 'base64_image', 'image_data', 'imageData',
        'data', 'file', 'upload', 'content'
    ]
    
    def check_value(value):
        """Check if value is likely base64 image data"""
        if isinstance(value, str) and len(value) > 100:
            # Check if it's base64 or data URL
            if value.startswith('data:image'):
                return True
            # Check if it looks like base64
            if all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in value[:100]):
                return True
        return False
    
    def search_dict(d, path=""):
        """Recursively search dictionary for image data"""
        if not isinstance(d, dict):
            return None
            
        # Check all possible keys
        for key in possible_keys:
            if key in d:
                if check_value(d[key]):
                    print(f"Found image at: {path}{key}")
                    return d[key]
        
        # Check all keys (not just known ones)
        for key, value in d.items():
            if check_value(value):
                print(f"Found image at: {path}{key}")
                return value
            elif isinstance(value, dict):
                result = search_dict(value, f"{path}{key}.")
                if result:
                    return result
        
        return None
    
    # Search main event
    result = search_dict(event)
    if result:
        return result
    
    # If not found, dump structure for debugging
    print("No image data found - dumping structure")
    print(f"Event type: {type(event)}")
    print(f"Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not dict'}")
    if isinstance(event, dict):
        for key, value in event.items():
            if isinstance(value, dict):
                print(f"  {key}: {list(value.keys())}")
            elif isinstance(value, str):
                print(f"  {key}: string (length: {len(value)})")
            else:
                print(f"  {key}: {type(value)}")
    
    return None

def enhance_professional_style(image):
    """Professional jewelry photography style enhancement"""
    # Convert to LAB for better color control
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Brighten the L channel significantly (professional white background)
    l_channel = cv2.add(l_channel, 40)
    l_channel = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_channel = clahe.apply(l_channel)
    
    # Merge back
    enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Convert to PIL for final adjustments
    pil_img = Image.fromarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))
    
    # Professional style adjustments
    # Brightness - make it really bright like studio photos
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.25)
    
    # Contrast - subtle increase
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.15)
    
    # Color - slightly desaturate for clean look
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(0.95)
    
    # Sharpness - enhance details
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.2)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def handler(job):
    """RunPod handler function"""
    print(f"=== Enhancement Handler {VERSION} Started ===")
    
    try:
        # Find base64 data
        img_data = find_input_data(job)
        if not img_data:
            print(f"ERROR: No image data found in job structure")
            print(f"Job type: {type(job)}")
            print(f"Job keys: {list(job.keys()) if isinstance(job, dict) else 'Not dict'}")
            
            # Deep structure dump
            if isinstance(job, dict):
                for key, value in job.items():
                    print(f"\n--- Key: {key} ---")
                    if isinstance(value, dict):
                        print(f"  Type: dict, Keys: {list(value.keys())}")
                        for k, v in value.items():
                            if isinstance(v, str):
                                print(f"    {k}: string (length: {len(v)}, starts: {v[:50]}...)")
                            elif isinstance(v, dict):
                                print(f"    {k}: dict with keys: {list(v.keys())}")
                            else:
                                print(f"    {k}: {type(v)}")
                    elif isinstance(value, str):
                        print(f"  Type: string (length: {len(value)}, starts: {value[:50]}...)")
                    else:
                        print(f"  Type: {type(value)}")
            
            return {
                "output": {
                    "error": "No image data found in any known location",
                    "success": False,
                    "version": VERSION,
                    "checked_structure": True
                }
            }
        
        print(f"Found image data, length: {len(img_data)}")
        
        # Clean base64 data
        if img_data.startswith('data:'):
            img_data = img_data.split(',')[1]
        
        # Add padding for decoding if needed
        padding = 4 - (len(img_data) % 4)
        if padding != 4:
            img_data += '=' * padding
        
        # Decode image
        try:
            img_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image")
                
        except Exception as e:
            print(f"Decode error: {str(e)}")
            return {
                "output": {
                    "error": f"Failed to decode image: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        print(f"Image decoded: {img.shape}")
        
        # Apply professional enhancement
        enhanced = enhance_professional_style(img)
        
        # Encode result
        _, buffer = cv2.imencode('.jpg', enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        enhanced_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Remove padding for Make.com
        enhanced_base64 = enhanced_base64.rstrip('=')
        
        # Create data URL
        result_url = f"data:image/jpeg;base64,{enhanced_base64}"
        
        return {
            "output": {
                "enhanced_image": result_url,
                "success": True,
                "version": VERSION,
                "message": "Professional enhancement applied successfully"
            }
        }
        
    except Exception as e:
        print(f"Handler error: {str(e)}")
        print(traceback.format_exc())
        return {
            "output": {
                "error": str(e),
                "success": False,
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

runpod.serverless.start({"handler": handler})
