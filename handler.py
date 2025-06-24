import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
import io
import json
import logging
import traceback
from typing import Dict, Any, Tuple, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancementHandler:
    def __init__(self):
        self.version = "v41-fixed-decode"
        logger.info(f"Initializing {self.version}")
        
    def find_input_data(self, event: Dict[str, Any]) -> Optional[str]:
        """Find image data from various possible locations"""
        # Most common paths first
        simple_paths = [
            ['input', 'enhanced_image'],
            ['input', 'image_base64'],
            ['input', 'image'],
            ['enhanced_image'],
            ['image_base64'],
            ['image']
        ]
        
        # Check simple paths
        for path in simple_paths:
            try:
                data = event
                for key in path:
                    if isinstance(data, dict) and key in data:
                        data = data[key]
                    else:
                        break
                else:
                    if isinstance(data, str) and len(data) > 100:
                        logger.info(f"Found data at path: {'.'.join(path)}")
                        return data
            except:
                continue
        
        # Check numbered paths (like 4.data.output.output.enhanced_image)
        if 'input' in event:
            for i in range(10):
                key = str(i)
                if key in event['input']:
                    try:
                        # Try nested path
                        if isinstance(event['input'][key], dict):
                            paths = [
                                ['data', 'output', 'output', 'enhanced_image'],
                                ['data', 'output', 'enhanced_image'],
                                ['output', 'enhanced_image'],
                                ['enhanced_image']
                            ]
                            
                            for path in paths:
                                data = event['input'][key]
                                for subkey in path:
                                    if isinstance(data, dict) and subkey in data:
                                        data = data[subkey]
                                    else:
                                        break
                                else:
                                    if isinstance(data, str) and len(data) > 100:
                                        logger.info(f"Found data at numbered path: {key}.{'.'.join(path)}")
                                        return data
                    except:
                        continue
        
        logger.error(f"No valid image data found in event")
        logger.error(f"Event structure: {json.dumps(event, indent=2)[:500]}...")
        return None
    
    def decode_base64_safe(self, base64_str: str) -> bytes:
        """Safely decode base64 with automatic padding correction"""
        try:
            # Remove data URL prefix if present
            if base64_str.startswith('data:'):
                base64_str = base64_str.split(',')[1]
            
            # Clean the string
            base64_str = base64_str.strip()
            
            # Fix padding if needed
            padding = 4 - len(base64_str) % 4
            if padding != 4:
                base64_str += '=' * padding
            
            return base64.b64decode(base64_str)
        except Exception as e:
            logger.error(f"Error decoding base64: {str(e)}")
            logger.error(f"Base64 preview: {base64_str[:100]}...")
            raise
    
    def encode_base64_no_padding(self, data: bytes) -> str:
        """Encode to base64 without padding for Make.com"""
        return base64.b64encode(data).decode('utf-8').rstrip('=')
        
    def detect_ring_regions(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect ring regions using simple but effective methods"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Use edge detection to find potential ring areas
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and circularity
        ring_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100 or area > (width * height * 0.5):  # Skip too small or too large
                continue
                
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.3:  # Reasonably circular
                x, y, w, h = cv2.boundingRect(contour)
                ring_candidates.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'circularity': circularity,
                    'center': (x + w//2, y + h//2)
                })
        
        # Sort by area (largest first)
        ring_candidates.sort(key=lambda x: x['area'], reverse=True)
        
        # Primary region is the largest suitable candidate
        primary_region = None
        if ring_candidates:
            # Take up to 2 largest candidates (for wedding ring pairs)
            primary_region = ring_candidates[0]
            if len(ring_candidates) > 1:
                # Check if second candidate is close enough to be a pair
                dist = np.sqrt((ring_candidates[0]['center'][0] - ring_candidates[1]['center'][0])**2 +
                              (ring_candidates[0]['center'][1] - ring_candidates[1]['center'][1])**2)
                if dist < max(width, height) * 0.3:  # Close enough to be a pair
                    # Combine bounding boxes
                    x1 = min(ring_candidates[0]['bbox'][0], ring_candidates[1]['bbox'][0])
                    y1 = min(ring_candidates[0]['bbox'][1], ring_candidates[1]['bbox'][1])
                    x2 = max(ring_candidates[0]['bbox'][0] + ring_candidates[0]['bbox'][2],
                            ring_candidates[1]['bbox'][0] + ring_candidates[1]['bbox'][2])
                    y2 = max(ring_candidates[0]['bbox'][1] + ring_candidates[0]['bbox'][3],
                            ring_candidates[1]['bbox'][1] + ring_candidates[1]['bbox'][3])
                    primary_region = {
                        'bbox': (x1, y1, x2-x1, y2-y1),
                        'center': ((x1+x2)//2, (y1+y2)//2)
                    }
        
        return {
            'candidates': ring_candidates,
            'primary_region': primary_region
        }
    
    def detect_ring_color_enhanced(self, image: np.ndarray, ring_regions: Dict[str, Any]) -> str:
        """Enhanced color detection focusing on ring regions"""
        if ring_regions['primary_region']:
            x, y, w, h = ring_regions['primary_region']['bbox']
            # Add some padding
            padding = int(min(w, h) * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2*padding)
            h = min(image.shape[0] - y, h + 2*padding)
            
            roi = image[y:y+h, x:x+w]
        else:
            # Use center region
            h, w = image.shape[:2]
            roi = image[h//4:3*h//4, w//4:3*w//4]
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        
        # Calculate color statistics
        avg_b, avg_g, avg_r = cv2.mean(roi)[:3]
        avg_h, avg_s, avg_v = cv2.mean(hsv)[:3]
        avg_l, avg_a, avg_b_lab = cv2.mean(lab)[:3]
        
        # Detect plain white first (highest priority)
        if avg_v > 220 and avg_s < 20:  # Very bright and low saturation
            return 'plain_white'
        
        # White gold detection
        if avg_s < 30 and avg_v > 180 and abs(avg_a - 128) < 5:
            return 'white_gold'
        
        # Yellow gold detection (stricter to avoid false positives)
        yellow_hue = (15 <= avg_h <= 35)
        warm_tone = avg_a > 130
        sufficient_saturation = avg_s > 30
        
        if yellow_hue and warm_tone and sufficient_saturation:
            return 'yellow_gold'
        
        # Rose gold detection
        rose_hue = (avg_h < 15 or avg_h > 165)
        pink_tone = avg_a > 135
        moderate_saturation = 20 < avg_s < 60
        
        if rose_hue and pink_tone and moderate_saturation:
            return 'rose_gold'
        
        # Default
        return 'plain_white'
    
    def apply_professional_enhancement(self, image: np.ndarray, detected_color: str, 
                                     ring_regions: Dict[str, Any]) -> np.ndarray:
        """Apply professional jewelry photography style enhancement"""
        enhanced = image.copy()
        
        # Convert to LAB for better shadow/highlight control
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # 1. Brighten the background more aggressively
        # Create a mask for the background (non-ring areas)
        background_mask = np.ones_like(l_channel, dtype=np.float32)
        
        if ring_regions['primary_region']:
            x, y, w, h = ring_regions['primary_region']['bbox']
            # Expand the ring region slightly
            padding = int(min(w, h) * 0.3)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2*padding)
            h = min(image.shape[0] - y, h + 2*padding)
            
            # Create gradient mask
            center_x = x + w//2
            center_y = y + h//2
            
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                    max_dist = np.sqrt((h/2)**2 + (w/2)**2)
                    if dist < max_dist:
                        background_mask[i, j] = dist / max_dist
        
        # Apply strong brightening to background
        l_float = l_channel.astype(np.float32)
        l_float = l_float + (255 - l_float) * background_mask * 0.5  # Push towards white
        l_channel = np.clip(l_float, 0, 255).astype(np.uint8)
        
        # 2. Enhance the L channel for overall brightness
        l_channel = cv2.add(l_channel, 30)  # Increase overall brightness
        
        # 3. Apply CLAHE for local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_channel = clahe.apply(l_channel)
        
        # Merge back
        enhanced = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 4. Create shallow depth of field effect
        if ring_regions['primary_region']:
            # Create focus mask
            mask = np.zeros(image.shape[:2], dtype=np.float32)
            x, y, w, h = ring_regions['primary_region']['bbox']
            cv2.ellipse(mask, ((x + w//2), (y + h//2)), (w//2, h//2), 0, 0, 360, 1, -1)
            mask = cv2.GaussianBlur(mask, (51, 51), 20)
            
            # Blur the entire image slightly
            blurred = cv2.GaussianBlur(enhanced, (7, 7), 2)
            
            # Combine sharp and blurred based on mask
            for c in range(3):
                enhanced[:,:,c] = enhanced[:,:,c] * mask + blurred[:,:,c] * (1 - mask)
        
        # 5. Apply color-specific adjustments
        if detected_color == 'plain_white':
            # Make whites even brighter and cooler
            enhanced = cv2.addWeighted(enhanced, 1.3, np.ones_like(enhanced) * 255, 0.1, -30)
        
        # 6. Final touches
        # Convert to PIL for final adjustments
        pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        
        # Selective sharpening on ring area only
        if ring_regions['primary_region']:
            x, y, w, h = ring_regions['primary_region']['bbox']
            ring_crop = pil_img.crop((x, y, x+w, y+h))
            ring_sharp = ring_crop.filter(ImageFilter.SHARPEN)
            sharpness = ImageEnhance.Sharpness(ring_sharp)
            ring_sharp = sharpness.enhance(1.5)
            pil_img.paste(ring_sharp, (x, y))
        
        # Professional white balance
        color_enhancer = ImageEnhance.Color(pil_img)
        pil_img = color_enhancer.enhance(0.95)  # Slight desaturation
        
        # Subtle vignette
        width, height = pil_img.size
        vignette = Image.new('L', (width, height), 255)
        for i in range(width):
            for j in range(height):
                dist = np.sqrt((i - width/2)**2 + (j - height/2)**2)
                max_dist = np.sqrt((width/2)**2 + (height/2)**2)
                brightness = 255 - int(30 * (dist / max_dist)**2)
                vignette.putpixel((i, j), brightness)
        
        # Apply vignette
        vignette = vignette.convert('RGB')
        pil_img = Image.blend(pil_img, vignette, 0.1)
        
        # Boost highlights
        brightness_enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = brightness_enhancer.enhance(1.05)
        
        # Final conversion
        final_enhanced = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        return final_enhanced

def handler(event):
    """RunPod handler function"""
    try:
        logger.info("="*50)
        logger.info(f"Handler started with event keys: {list(event.keys())}")
        
        handler_instance = EnhancementHandler()
        
        # Find image data with improved logic
        enhanced_image_data = handler_instance.find_input_data(event)
        
        if not enhanced_image_data:
            raise ValueError("No image data found. Please check the input structure.")
        
        logger.info(f"Found image data, type: {type(enhanced_image_data)}, length: {len(str(enhanced_image_data)[:100])}...")
        
        # If the data is not a string, try to extract it
        if not isinstance(enhanced_image_data, str):
            logger.error(f"Image data is not a string, it's: {type(enhanced_image_data)}")
            raise ValueError(f"Expected string data but got {type(enhanced_image_data)}")
        
        # Decode base64 image with padding fix
        logger.info("Decoding base64 image...")
        image_bytes = handler_instance.decode_base64_safe(enhanced_image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image from base64")
        
        logger.info(f"Successfully decoded image: {image.shape}")
        
        # Detect ring regions
        ring_regions = handler_instance.detect_ring_regions(image)
        logger.info(f"Detected {len(ring_regions['candidates'])} ring candidates")
        
        # Detect color
        detected_color = handler_instance.detect_ring_color_enhanced(image, ring_regions)
        logger.info(f"Detected color: {detected_color}")
        
        # Apply enhancement
        enhanced_image = handler_instance.apply_professional_enhancement(image, detected_color, ring_regions)
        
        processing_info = {
            'original_shape': image.shape,
            'detected_color': detected_color,
            'ring_regions_found': len(ring_regions['candidates']),
            'primary_ring_bbox': ring_regions['primary_region']['bbox'] if ring_regions['primary_region'] else None,
            'enhancement_version': 'v41_fixed_decode'
        }
        
        # Convert result to base64 without padding
        _, buffer = cv2.imencode('.jpg', enhanced_image, 
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
        result_base64 = handler_instance.encode_base64_no_padding(buffer.tobytes())
        result_data_url = f"data:image/jpeg;base64,{result_base64}"
        
        # Return in nested structure for Make.com
        result = {
            "output": {
                "enhanced_image": result_data_url,
                "detected_color": detected_color,
                "processing_info": processing_info,
                "success": True
            }
        }
        
        logger.info(f"Processing completed successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "success": False
            }
        }

# For RunPod
if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
