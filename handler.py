import json
import base64
import traceback
from io import BytesIO
from typing import Dict, Any, Optional
from PIL import Image, ImageEnhance

VERSION = "46"

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for wedding ring image enhancement
    v46: Direct path checking without recursion (based on v41 approach)
    """
    print(f"Enhancement Handler v{VERSION} starting...")
    
    enhancement_handler = EnhancementHandlerV46()
    
    try:
        # Find base64 input using direct checks only
        input_base64 = enhancement_handler.find_input_data(event)
        
        if not input_base64:
            print("ERROR: No base64 data found")
            print(f"Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
            print(f"Event structure (first 500 chars): {str(event)[:500]}")
            return enhancement_handler.create_error_response("No base64 input data found")
        
        print(f"Found base64 data, length: {len(input_base64)}")
        
        # Process the image
        result = enhancement_handler.process_image(input_base64)
        return result
        
    except Exception as e:
        print(f"ERROR in handler: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return enhancement_handler.create_error_response(f"Handler error: {str(e)}")


class EnhancementHandlerV46:
    def __init__(self):
        self.version = VERSION
    
    def find_input_data(self, event: Dict[str, Any]) -> Optional[str]:
        """Find base64 input using direct path checks only - NO RECURSION"""
        # Check direct keys first
        direct_keys = ['image_base64', 'image', 'base64', 'imageBase64', 'img', 'input_image']
        for key in direct_keys:
            if key in event and isinstance(event[key], str) and len(event[key]) > 100:
                print(f"Found base64 in direct key: {key}")
                return self.clean_base64(event[key])
        
        # Check input dict
        if 'input' in event and isinstance(event['input'], dict):
            for key in direct_keys:
                if key in event['input'] and isinstance(event['input'][key], str) and len(event['input'][key]) > 100:
                    print(f"Found base64 in input.{key}")
                    return self.clean_base64(event['input'][key])
        
        # Check numbered keys with common patterns
        for i in range(10):
            str_i = str(i)
            
            # Direct numbered key
            if str_i in event and isinstance(event[str_i], str) and len(event[str_i]) > 100:
                print(f"Found base64 in key: {str_i}")
                return self.clean_base64(event[str_i])
            
            # Pattern: {i}.data.output.output.enhanced_image
            if str_i in event and isinstance(event[str_i], dict):
                current = event[str_i]
                
                # Check common patterns
                patterns = [
                    ['data', 'output', 'output', 'enhanced_image'],
                    ['data', 'output', 'enhanced_image'],
                    ['output', 'enhanced_image'],
                    ['enhanced_image']
                ]
                
                for pattern in patterns:
                    temp = current
                    valid = True
                    
                    for part in pattern:
                        if isinstance(temp, dict) and part in temp:
                            temp = temp[part]
                        else:
                            valid = False
                            break
                    
                    if valid and isinstance(temp, str) and len(temp) > 100:
                        print(f"Found base64 at path: {str_i}.{'.'.join(pattern)}")
                        return self.clean_base64(temp)
        
        # Check 'data' key patterns
        if 'data' in event and isinstance(event['data'], dict):
            data = event['data']
            
            # Check direct keys in data
            for key in direct_keys:
                if key in data and isinstance(data[key], str) and len(data[key]) > 100:
                    print(f"Found base64 in data.{key}")
                    return self.clean_base64(data[key])
            
            # Check data.output patterns
            if 'output' in data and isinstance(data['output'], dict):
                output = data['output']
                
                # Check direct keys in output
                for key in direct_keys:
                    if key in output and isinstance(output[key], str) and len(output[key]) > 100:
                        print(f"Found base64 in data.output.{key}")
                        return self.clean_base64(output[key])
                
                # Check data.output.output pattern
                if 'output' in output and isinstance(output['output'], dict):
                    output2 = output['output']
                    for key in direct_keys + ['enhanced_image']:
                        if key in output2 and isinstance(output2[key], str) and len(output2[key]) > 100:
                            print(f"Found base64 in data.output.output.{key}")
                            return self.clean_base64(output2[key])
        
        print("No base64 found in any known location")
        return None
    
    def clean_base64(self, data: str) -> str:
        """Extract base64 from data URL if needed"""
        if data.startswith('data:image'):
            return data.split(',')[1] if ',' in data else data
        return data
    
    def decode_base64_safe(self, base64_str: str) -> bytes:
        """Safely decode base64 with automatic padding fix"""
        # Clean the input
        base64_str = base64_str.strip()
        
        # Remove any whitespace or newlines
        base64_str = ''.join(base64_str.split())
        
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
        """Process the wedding ring image with simple enhancement"""
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
            
            # Color
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(0.95)  # 5% less saturated
            
            # Save to buffer
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=92, optimize=True)
            buffer.seek(0)
            
            # Encode without padding for Make.com
            enhanced_base64 = self.encode_base64_no_padding(buffer.getvalue())
            
            # Return with correct structure
            return {
                "output": {
                    "enhanced_image": f"data:image/jpeg;base64,{enhanced_base64}",
                    "status": "success",
                    "version": self.version,
                    "message": "Image enhanced successfully"
                }
            }
            
        except Exception as e:
            print(f"ERROR in process_image: {str(e)}")
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
