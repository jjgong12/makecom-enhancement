import runpod
import base64
from io import BytesIO
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import traceback

VERSION = "v53"

def find_input_data(event):
    """Find base64 data from various possible locations"""
    # Priority order - image_base64 first!
    direct_paths = [
        'image_base64', 'image', 'base64', 'imageBase64',
        'input.image_base64', 'input.image', 'input.base64'
    ]
    
    # Check direct paths
    for path in direct_paths:
        keys = path.split('.')
        value = event
        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    value = None
                    break
            if value and isinstance(value, str) and len(value) > 100:
                print(f"Found image data at: {path}")
                return value
        except:
            continue
    
    # Check numbered patterns (Make.com structure)
    for i in range(10):
        try:
            if str(i) in event:
                node = event[str(i)]
                if isinstance(node, dict) and 'data' in node:
                    data = node['data']
                    if isinstance(data, dict) and 'output' in data:
                        output = data['output']
                        if isinstance(output, dict) and 'output' in output:
                            inner_output = output['output']
                            if isinstance(inner_output, dict) and 'enhanced_image' in inner_output:
                                value = inner_output['enhanced_image']
                                if value and isinstance(value, str) and len(value) > 100:
                                    print(f"Found at: {i}.data.output.output.enhanced_image")
                                    return value
        except:
            continue
    
    # Check job.input structure (RunPod specific)
    if 'input' in event and isinstance(event['input'], dict):
        result = find_input_data(event['input'])
        if result:
            return result
    
    print("No image data found - dumping structure")
    print(f"Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not dict'}")
    if 'input' in event:
        print(f"Input keys: {list(event['input'].keys()) if isinstance(event['input'], dict) else 'Not dict'}")
    
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
            print(f"Job structure: {job}")
            return {
                "output": {
                    "error": "No image data found in input",
                    "success": False,
                    "version": VERSION
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
