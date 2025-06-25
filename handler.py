import json
import base64
import traceback
from io import BytesIO
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ImageEnhance
import numpy as np

VERSION = "44"

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for wedding ring image enhancement
    v44: Fixed base64 handling and response structure
    """
    print(f"Enhancement Handler v{VERSION} starting...")
    print(f"Event type: {type(event)}")
    print(f"Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
    
    enhancement_handler = EnhancementHandlerV44()
    
    try:
        # Find base64 input using recursive search
        input_base64 = enhancement_handler.find_string_recursive(event)
        
        if not input_base64:
            print("ERROR: No base64 data found")
            print(f"Full event (first 1000 chars): {str(event)[:1000]}")
            return enhancement_handler.create_error_response("No base64 input data found")
        
        print(f"Found base64 data, length: {len(input_base64)}")
        
        # Process the image
        result = enhancement_handler.process_image(input_base64)
        
        print(f"Result keys: {list(result.get('output', {}).keys())}")
        return result
        
    except Exception as e:
        print(f"ERROR in handler: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return enhancement_handler.create_error_response(f"Handler error: {str(e)}")


class EnhancementHandlerV44:
    def __init__(self):
        self.version = VERSION
        self.priority_keys = [
            'image_base64', 'image', 'base64', 'image_data',
            'input_image', 'input', 'data', 'enhanced_image',
            'imageBase64', 'base64Image', 'img'
        ]
    
    def find_string_recursive(self, obj: Any, path: str = "", depth: int = 0, max_depth: int = 10) -> Optional[str]:
        """Recursively search for base64 string in nested structure"""
        if depth > max_depth:
            return None
        
        if isinstance(obj, str):
            # Check if it's a data URL
            if obj.startswith('data:image'):
                print(f"Found data URL at path: {path}")
                base64_part = obj.split(',')[1] if ',' in obj else obj
                return base64_part
            # Check if it's likely base64 (long string)
            elif len(obj) > 1000:
                # Check if it's not JSON
                if not obj.strip().startswith('{') and not obj.strip().startswith('['):
                    print(f"Found potential base64 at path: {path}, length: {len(obj)}")
                    return obj
        
        elif isinstance(obj, dict):
            # Check priority keys first
            for key in self.priority_keys:
                if key in obj:
                    result = self.find_string_recursive(obj[key], f"{path}.{key}", depth + 1, max_depth)
                    if result:
                        return result
            
            # Check numbered keys (0-9)
            for i in range(10):
                key = str(i)
                if key in obj:
                    result = self.find_string_recursive(obj[key], f"{path}.{key}", depth + 1, max_depth)
                    if result:
                        return result
            
            # Check all other keys
            for key, value in obj.items():
                if key not in self.priority_keys and not key.isdigit():
                    result = self.find_string_recursive(value, f"{path}.{key}", depth + 1, max_depth)
                    if result:
                        return result
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                result = self.find_string_recursive(item, f"{path}[{i}]", depth + 1, max_depth)
                if result:
                    return result
        
        return None
    
    def decode_base64_safe(self, base64_str: str) -> bytes:
        """Safely decode base64 with automatic padding fix"""
        try:
            # Clean the input
            base64_str = base64_str.strip()
            
            # Remove any whitespace or newlines
            base64_str = ''.join(base64_str.split())
            
            # Remove data URL prefix if present
            if 'base64,' in base64_str:
                base64_str = base64_str.split('base64,')[1]
            
            # Try to decode as is first
            try:
                return base64.b64decode(base64_str)
            except:
                pass
            
            # Add padding if needed
            padding = 4 - len(base64_str) % 4
            if padding != 4:
                base64_str += '=' * padding
                print(f"Added {padding} padding characters")
            
            return base64.b64decode(base64_str)
            
        except Exception as e:
            print(f"Base64 decode error: {str(e)}")
            print(f"Base64 string first 100 chars: {base64_str[:100]}")
            raise
    
    def encode_base64_no_padding(self, data: bytes) -> str:
        """Encode to base64 without padding for Make.com compatibility"""
        # CRITICAL: Remove padding for Make.com
        return base64.b64encode(data).decode('utf-8').rstrip('=')
    
    def process_image(self, input_base64: str) -> Dict[str, Any]:
        """Process the wedding ring image with simple enhancement"""
        try:
            # Decode base64
            print("Decoding base64...")
            image_data = self.decode_base64_safe(input_base64)
            print(f"Decoded image data length: {len(image_data)}")
            
            # Open image
            img = Image.open(BytesIO(image_data))
            print(f"Image opened: {img.size}, mode: {img.mode}")
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print("Converted to RGB")
            
            # Simple but effective enhancement (based on v31 success)
            # Brightness
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.18)  # 18% brighter
            
            # Contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.15)  # 15% more contrast
            
            # Slight saturation reduction for cleaner look
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(0.95)  # 5% less saturated
            
            print("Enhancement applied")
            
            # Save to buffer with high quality
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=92, optimize=True)
            buffer.seek(0)
            
            # Encode without padding - CRITICAL FOR MAKE.COM
            enhanced_base64 = self.encode_base64_no_padding(buffer.getvalue())
            print(f"Enhanced base64 length: {len(enhanced_base64)}")
            print(f"Last 3 chars of base64: '{enhanced_base64[-3:]}'")  # Should NOT be '='
            
            # Return with correct structure
            result = {
                "output": {
                    "enhanced_image": f"data:image/jpeg;base64,{enhanced_base64}",
                    "status": "success",
                    "version": self.version,
                    "original_size": f"{img.width}x{img.height}",
                    "message": "Image enhanced successfully"
                }
            }
            
            print("Returning success response")
            return result
            
        except Exception as e:
            print(f"ERROR in process_image: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return self.create_error_response(f"Processing error: {str(e)}")
    
    def create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response with correct structure"""
        return {
            "output": {
                "status": "error",
                "error": error_message,
                "version": self.version
            }
        }


# For RunPod
if __name__ == "__main__":
    print(f"Enhancement Handler v{VERSION} loaded successfully")
