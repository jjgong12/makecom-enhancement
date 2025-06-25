import runpod
import base64
from io import BytesIO
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import traceback

VERSION = "enhancement_v47"
print(f"{VERSION} starting...")

def find_input_data(event):
    """Find base64 data from various possible locations"""
    # Direct paths
    paths = [
        'image_base64', 'image', 'base64', 'imageBase64',
        'input.image_base64', 'input.image', 'input.base64',
        'input.imageBase64'
    ]
    
    for path in paths:
        try:
            keys = path.split('.')
            value = event
            for key in keys:
                value = value.get(key, {})
            if value and isinstance(value, str) and len(value) > 100:
                return value
        except:
            continue
    
    # Check numbered patterns
    for i in range(10):
        path = f"{i}.data.output.output.enhanced_image"
        try:
            keys = path.split('.')
            value = event
            for key in keys:
                value = value.get(str(key), {})
            if value and isinstance(value, str) and len(value) > 100:
                return value
        except:
            continue
    
    return None

def enhance_handler(event):
    print(f"=== {VERSION} Handler Started ===")
    print(f"Event structure: {list(event.keys())}")
    
    try:
        # Find base64 data
        img_data = find_input_data(event)
        if not img_data:
            return {
                "output": {
                    "error": "No base64 image data found",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Clean base64 data
        if img_data.startswith('data:'):
            img_data = img_data.split(',')[1]
        
        # Add padding for decoding
        padding = 4 - (len(img_data) % 4)
        if padding != 4:
            img_data += '=' * padding
        
        # Decode image
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Simple enhancement - same for all images
        # This makes the overall image brighter and cleaner
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.2)  # 20% brighter
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)  # 10% more contrast
        
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.05)  # 5% more vivid
        
        # Convert to numpy for background brightening
        img_np = np.array(img)
        
        # Make background whiter
        # Simple approach: brighten pixels that are already bright
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        bright_mask = gray > 180  # Bright areas (likely background)
        
        # Brighten background areas
        img_np[bright_mask] = np.minimum(img_np[bright_mask] * 1.1, 255).astype(np.uint8)
        
        # Convert back to PIL
        enhanced_img = Image.fromarray(img_np)
        
        # Save as JPEG
        output_buffer = BytesIO()
        enhanced_img.save(output_buffer, format='JPEG', quality=95)
        enhanced_bytes = output_buffer.getvalue()
        
        # Encode to base64 and remove padding
        enhanced_base64 = base64.b64encode(enhanced_bytes).decode('utf-8')
        enhanced_base64 = enhanced_base64.rstrip('=')
        
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
                "version": VERSION
            }
        }

runpod.serverless.start({"handler": enhance_handler})
