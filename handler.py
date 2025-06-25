import json
import base64
import traceback
from io import BytesIO
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ImageEnhance
import numpy as np

VERSION = "43"

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for wedding ring image enhancement
    v43: Optimized for speed while maintaining quality
    """
    print(f"Enhancement Handler v{VERSION} starting...")
    
    enhancement_handler = EnhancementHandlerV43()
    
    try:
        # Find base64 input using recursive search
        input_base64 = enhancement_handler.find_string_recursive(event)
        
        if not input_base64:
            print("ERROR: No base64 data found")
            print(f"Event keys: {list(event.keys())}")
            print(f"Event structure (first 500 chars): {str(event)[:500]}")
            return enhancement_handler.create_error_response("No base64 input data found")
        
        # Process the image
        result = enhancement_handler.process_image(input_base64)
        return result
        
    except Exception as e:
        print(f"ERROR in handler: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return enhancement_handler.create_error_response(f"Handler error: {str(e)}")


class EnhancementHandlerV43:
    def __init__(self):
        self.version = VERSION
        self.priority_keys = [
            'image_base64', 'image', 'base64', 'image_data',
            'input_image', 'input', 'data', 'enhanced_image'
        ]
    
    def find_string_recursive(self, obj: Any, path: str = "", depth: int = 0, max_depth: int = 10) -> Optional[str]:
        """Recursively search for base64 string in nested structure"""
        if depth > max_depth:
            return None
        
        if isinstance(obj, str):
            # Check if it's a data URL
            if obj.startswith('data:image'):
                print(f"Found data URL at path: {path}")
                return obj.split(',')[1] if ',' in obj else obj
            # Check if it's likely base64 (long string)
            elif len(obj) > 1000 and not obj.startswith('{'):
                print(f"Found potential base64 at path: {path}")
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
        # Clean the input
        base64_str = base64_str.strip()
        
        # Remove data URL prefix if present
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]
        
        # Add padding if needed
        padding = 4 - len(base64_str) % 4
        if padding != 4:
            base64_str += '=' * padding
        
        return base64.b64decode(base64_str)
    
    def encode_base64_no_padding(self, data: bytes) -> str:
        """Encode to base64 without padding for Make.com compatibility"""
        return base64.b64encode(data).decode('utf-8').rstrip('=')
    
    def process_image(self, input_base64: str) -> Dict[str, Any]:
        """Process the wedding ring image with optimized enhancement"""
        try:
            # Decode base64
            image_data = self.decode_base64_safe(input_base64)
            
            # Open image
            img = Image.open(BytesIO(image_data))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
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
            
            # Optional: Add subtle background blend if needed
            # This is fast and doesn't affect performance much
            if self.should_apply_background_blend(img):
                img = self.apply_quick_background_blend(img)
            
            # Save to buffer
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=92, optimize=True)
            buffer.seek(0)
            
            # Encode without padding
            enhanced_base64 = self.encode_base64_no_padding(buffer.getvalue())
            
            # Return with correct structure
            return {
                "output": {
                    "enhanced_image": f"data:image/jpeg;base64,{enhanced_base64}",
                    "status": "success",
                    "version": self.version,
                    "message": "Image enhanced successfully (optimized)"
                }
            }
            
        except Exception as e:
            print(f"ERROR in process_image: {str(e)}")
            return self.create_error_response(f"Processing error: {str(e)}")
    
    def should_apply_background_blend(self, img: Image.Image) -> bool:
        """Quick check if background blend would be beneficial"""
        # Sample corners to check if background is too dark/gray
        width, height = img.size
        corners = [
            img.getpixel((10, 10)),
            img.getpixel((width-10, 10)),
            img.getpixel((10, height-10)),
            img.getpixel((width-10, height-10))
        ]
        
        avg_brightness = sum(sum(c) / 3 for c in corners) / 4
        return avg_brightness < 200  # If corners are dark, apply blend
    
    def apply_quick_background_blend(self, img: Image.Image) -> Image.Image:
        """Quick background blend without heavy processing"""
        # Create a subtle gradient overlay
        width, height = img.size
        
        # Create beige overlay
        overlay = Image.new('RGB', (width, height), (245, 243, 240))
        
        # Quick mask based on brightness
        img_array = np.array(img)
        gray = np.mean(img_array, axis=2)
        
        # Simple threshold mask
        mask = (gray < 200).astype(np.uint8) * 30  # 30/255 = ~12% blend
        mask = Image.fromarray(mask, 'L')
        
        # Apply blend
        img = Image.composite(overlay, img, mask)
        
        return img
    
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
