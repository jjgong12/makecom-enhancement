import json
import base64
import traceback
from io import BytesIO
from typing import Dict, Any, Optional
from PIL import Image, ImageEnhance
import time

VERSION = "45"

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for wedding ring image enhancement
    v45: Fixed infinite loop and timeout issues
    """
    start_time = time.time()
    print(f"Enhancement Handler v{VERSION} starting at {start_time}")
    
    try:
        # Simple direct check first
        input_base64 = None
        
        # Check most common locations first
        if 'input' in event and isinstance(event['input'], dict):
            if 'image_base64' in event['input']:
                input_base64 = event['input']['image_base64']
                print("Found in input.image_base64")
            elif 'image' in event['input']:
                input_base64 = event['input']['image']
                print("Found in input.image")
        
        # Direct check
        if not input_base64:
            if 'image_base64' in event:
                input_base64 = event['image_base64']
                print("Found in image_base64")
            elif 'image' in event:
                input_base64 = event['image']
                print("Found in image")
        
        # If still not found, do limited search
        if not input_base64:
            print("Doing limited search...")
            input_base64 = find_base64_limited(event)
        
        if not input_base64:
            print("ERROR: No base64 data found")
            print(f"Event keys: {list(event.keys())}")
            return {
                "output": {
                    "status": "error",
                    "error": "No base64 input data found",
                    "version": VERSION
                }
            }
        
        print(f"Processing image, base64 length: {len(input_base64) if isinstance(input_base64, str) else 'unknown'}")
        
        # Process the image
        result = process_image(input_base64)
        
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        
        return result
        
    except Exception as e:
        print(f"ERROR in handler: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {
            "output": {
                "status": "error",
                "error": f"Handler error: {str(e)}",
                "version": VERSION
            }
        }


def find_base64_limited(obj: Any, depth: int = 0, max_depth: int = 5) -> Optional[str]:
    """Limited depth search to avoid infinite loops"""
    if depth > max_depth:
        return None
    
    if isinstance(obj, str):
        if len(obj) > 1000 and not obj.startswith('{'):
            # Extract base64 from data URL if needed
            if obj.startswith('data:image'):
                return obj.split(',')[1] if ',' in obj else obj
            return obj
    elif isinstance(obj, dict):
        # Priority keys
        for key in ['image_base64', 'image', 'base64', 'enhanced_image']:
            if key in obj:
                val = obj[key]
                if isinstance(val, str) and len(val) > 1000:
                    if val.startswith('data:image'):
                        return val.split(',')[1] if ',' in val else val
                    return val
        
        # Check numbered keys
        for i in range(10):
            if str(i) in obj:
                result = find_base64_limited(obj[str(i)], depth + 1, max_depth)
                if result:
                    return result
    
    return None


def process_image(input_base64: str) -> Dict[str, Any]:
    """Process the wedding ring image"""
    try:
        # Clean base64
        if isinstance(input_base64, str):
            input_base64 = input_base64.strip()
            if 'base64,' in input_base64:
                input_base64 = input_base64.split('base64,')[1]
            
            # Remove whitespace
            input_base64 = ''.join(input_base64.split())
            
            # Add padding if needed
            padding = 4 - len(input_base64) % 4
            if padding != 4:
                input_base64 += '=' * padding
        
        # Decode
        image_data = base64.b64decode(input_base64)
        
        # Open image
        img = Image.open(BytesIO(image_data))
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Simple enhancement
        # Brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.18)
        
        # Contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.15)
        
        # Color
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.95)
        
        # Save
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=92)
        buffer.seek(0)
        
        # Encode without padding for Make.com
        enhanced_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8').rstrip('=')
        
        return {
            "output": {
                "enhanced_image": f"data:image/jpeg;base64,{enhanced_base64}",
                "status": "success",
                "version": VERSION,
                "message": "Image enhanced successfully"
            }
        }
        
    except Exception as e:
        print(f"ERROR in process_image: {str(e)}")
        return {
            "output": {
                "status": "error",
                "error": f"Processing error: {str(e)}",
                "version": VERSION
            }
        }


# For RunPod
if __name__ == "__main__":
    print(f"Enhancement Handler v{VERSION} loaded and ready")
