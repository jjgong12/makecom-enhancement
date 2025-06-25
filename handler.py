import runpod
import base64
from io import BytesIO
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import traceback

VERSION = "enhancement_v50"
print(f"{VERSION} starting...")

def find_input_data(event):
    """Find base64 data from various possible locations"""
    # Check direct paths first
    direct_paths = [
        'image_base64', 'image', 'base64', 'imageBase64',
        'input.image_base64', 'input.image', 'input.base64'
    ]
    
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
    
    # Check numbered patterns
    for i in range(10):
        try:
            # Pattern 1: {i}.data.output.output.enhanced_image
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
                                    print(f"Found image data at: {i}.data.output.output.enhanced_image")
                                    return value
        except:
            continue
    
    # Check input dict
    if 'input' in event and isinstance(event['input'], dict):
        return find_input_data(event['input'])
    
    print("No image data found in any known location")
    return None

def enhance_handler(event):
    print(f"=== {VERSION} Handler Started ===")
    print(f"Event keys: {list(event.keys())}")
    
    try:
        # Find base64 data
        img_data = find_input_data(event)
        if not img_data:
            print(f"Event structure: {event}")
            return {
                "output": {
                    "error": "No base64 image data found",
                    "status": "error",
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
        except Exception as e:
            print(f"Base64 decode error: {str(e)}")
            return {
                "output": {
                    "error": f"Base64 decode error: {str(e)}",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Open image
        img = Image.open(BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        print(f"Image opened successfully: {img.width}x{img.height}")
        
        # Simple enhancement
        # Step 1: Brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.18)
        
        # Step 2: Contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.12)
        
        # Step 3: Color
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.05)
        
        # Step 4: Background whitening using LAB
        img_np = np.array(img)
        
        # Convert to LAB
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Increase brightness
        l = cv2.add(l, 15)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to PIL
        enhanced_img = Image.fromarray(img_np)
        
        # Save as JPEG
        output_buffer = BytesIO()
        enhanced_img.save(output_buffer, format='JPEG', quality=95)
        enhanced_bytes = output_buffer.getvalue()
        
        # Encode to base64 - IMPORTANT: Remove padding for Make.com
        enhanced_base64 = base64.b64encode(enhanced_bytes).decode('utf-8').rstrip('=')
        
        print(f"{VERSION} completed successfully")
        
        return {
            "output": {
                "enhanced_image": f"data:image/jpeg;base64,{enhanced_base64}",
                "status": "success",
                "version": VERSION,
                "size": f"{img.width}x{img.height}"
            }
        }
        
    except Exception as e:
        print(f"Error in {VERSION}: {str(e)}")
        print(traceback.format_exc())
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

runpod.serverless.start({"handler": enhance_handler})
