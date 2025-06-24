import os
import json
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import logging
import traceback
import base64
from io import BytesIO
import requests
from typing import Dict, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancementHandler:
    def __init__(self):
        """Initialize the Enhancement Handler"""
        logger.info("Enhancement Handler v38 initialized")
        
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
            
            if circularity > 0.4:  # Reasonably circular
                x, y, w, h = cv2.boundingRect(contour)
                ring_candidates.append({
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2),
                    'area': area,
                    'circularity': circularity
                })
        
        # Sort by area (larger rings first)
        ring_candidates.sort(key=lambda x: x['area'], reverse=True)
        
        # Return the most likely ring regions
        return {
            'candidates': ring_candidates[:5],  # Top 5 candidates
            'primary_region': ring_candidates[0] if ring_candidates else None
        }
    
    def detect_ring_color_enhanced(self, image: np.ndarray, ring_regions: Dict[str, Any]) -> str:
        """Enhanced color detection focusing on detected ring regions"""
        if ring_regions['primary_region']:
            # Focus on the primary ring region
            bbox = ring_regions['primary_region']['bbox']
            x, y, w, h = bbox
            
            # Expand region slightly
            margin = 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.shape[1] - x, w + 2*margin)
            h = min(image.shape[0] - y, h + 2*margin)
            
            roi = image[y:y+h, x:x+w]
        else:
            # Fallback to center region
            height, width = image.shape[:2]
            center_y, center_x = height // 2, width // 2
            roi_size = min(width, height) // 3
            roi = image[
                max(0, center_y - roi_size):min(height, center_y + roi_size),
                max(0, center_x - roi_size):min(width, center_x + roi_size)
            ]
        
        # Convert to RGB and HSV
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate color statistics
        avg_hue = np.mean(hsv_roi[:, :, 0])
        avg_sat = np.mean(hsv_roi[:, :, 1])
        avg_val = np.mean(hsv_roi[:, :, 2])
        
        avg_r = np.mean(rgb_roi[:, :, 0])
        avg_g = np.mean(rgb_roi[:, :, 1])
        avg_b = np.mean(rgb_roi[:, :, 2])
        
        # Enhanced color detection logic
        color = "white_gold"  # Default
        
        # Yellow gold - strict criteria
        if (20 <= avg_hue <= 35 and avg_sat > 40 and avg_val > 120 and
            avg_r > avg_b + 25 and avg_g > avg_b + 15):
            color = "yellow_gold"
        
        # Rose gold
        elif ((0 <= avg_hue <= 15 or 340 <= avg_hue <= 360) and 
              avg_sat > 25 and avg_r > avg_g + 10 and avg_r > avg_b + 15):
            color = "rose_gold"
        
        # Unplated white - very bright and desaturated
        elif avg_sat < 15 and avg_val > 200 and abs(avg_r - avg_g) < 10 and abs(avg_g - avg_b) < 10:
            color = "unplated_white"
        
        logger.info(f"Color detection - HSV: ({avg_hue:.1f}, {avg_sat:.1f}, {avg_val:.1f}), "
                   f"RGB: ({avg_r:.1f}, {avg_g:.1f}, {avg_b:.1f}) -> {color}")
        
        return color
    
    def create_focus_map(self, image: np.ndarray, ring_regions: Dict[str, Any]) -> np.ndarray:
        """Create a focus map for depth of field effect"""
        height, width = image.shape[:2]
        focus_map = np.ones((height, width), dtype=np.float32)
        
        if ring_regions['primary_region']:
            # Get ring center
            center = ring_regions['primary_region']['center']
            cx, cy = center
            
            # Create radial gradient from ring center
            y_indices, x_indices = np.ogrid[:height, :width]
            distances = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
            
            # Normalize distances
            max_dist = np.sqrt(width**2 + height**2) / 2
            distances = distances / max_dist
            
            # Create smooth falloff (keep center sharp, blur edges)
            focus_map = 1.0 - np.clip(distances * 1.5, 0, 1)
            focus_map = np.power(focus_map, 2)  # Make falloff more gradual
            
        return focus_map
    
    def apply_professional_enhancement(self, image: np.ndarray, color: str, ring_regions: Dict[str, Any]) -> np.ndarray:
        """Apply professional jewelry photography style enhancement"""
        enhanced = image.copy()
        
        # Step 1: Significantly brighten the image (especially background)
        # Create a mask for likely background areas
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        _, background_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        background_mask = cv2.dilate(background_mask, np.ones((5,5), np.uint8), iterations=2)
        
        # Brighten background more aggressively
        background_brightened = cv2.addWeighted(enhanced, 1.0, np.ones_like(enhanced) * 255, 0.4, 0)
        enhanced = np.where(background_mask[..., np.newaxis] > 0, background_brightened, enhanced)
        
        # Step 2: Overall brightness boost
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=40)
        
        # Step 3: Apply depth of field effect
        focus_map = self.create_focus_map(enhanced, ring_regions)
        
        # Create blurred version
        pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=3))
        blurred_np = cv2.cvtColor(np.array(blurred), cv2.COLOR_RGB2BGR)
        
        # Blend based on focus map
        focus_map_3ch = np.stack([focus_map] * 3, axis=-1)
        enhanced = (enhanced * focus_map_3ch + blurred_np * (1 - focus_map_3ch)).astype(np.uint8)
        
        # Step 4: Color-specific adjustments
        if color == "unplated_white":
            # Pure white appearance
            enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * 1.08, 0, 255)  # Boost blue
            enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] * 1.02, 0, 255)  # Slight green
            enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 0.98, 0, 255)  # Reduce red
            
        elif color == "white_gold":
            # Cool white tone
            enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * 1.05, 0, 255)
            enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 0.97, 0, 255)
            
        elif color == "yellow_gold":
            # Warm golden tone
            enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] * 1.03, 0, 255)
            enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 1.05, 0, 255)
            
        elif color == "rose_gold":
            # Pink tone
            enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 1.08, 0, 255)
            enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * 0.96, 0, 255)
        
        # Step 5: Soften shadows
        # Convert to LAB for better shadow control
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Lift shadows
        shadow_mask = l_channel < 100
        l_channel[shadow_mask] = np.clip(l_channel[shadow_mask] * 1.3, 0, 255)
        
        lab[:, :, 0] = l_channel
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Step 6: Selective sharpening on rings
        if ring_regions['primary_region']:
            # Create mask for ring area
            mask = np.zeros(enhanced.shape[:2], dtype=np.uint8)
            bbox = ring_regions['primary_region']['bbox']
            x, y, w, h = bbox
            
            # Expand slightly for sharpening
            margin = 30
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(enhanced.shape[1] - x, w + 2*margin)
            h = min(enhanced.shape[0] - y, h + 2*margin)
            
            mask[y:y+h, x:x+w] = 255
            
            # Apply strong sharpening to ring area
            kernel = np.array([[-1, -1, -1],
                               [-1, 9.5, -1],
                               [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Blend sharpened with original based on mask
            mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
            enhanced = (sharpened * mask_3ch + enhanced * (1 - mask_3ch)).astype(np.uint8)
        
        # Step 7: Add subtle vignetting
        height, width = enhanced.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Create vignette mask
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        vignette = 1 - (dist_from_center / max_dist) * 0.15  # Very subtle
        vignette = np.clip(vignette, 0.85, 1.0)
        
        # Apply vignette
        vignette_3ch = np.stack([vignette] * 3, axis=-1)
        enhanced = (enhanced * vignette_3ch).astype(np.uint8)
        
        # Step 8: Boost highlights on metal
        # Create highlight mask
        highlight_mask = cv2.inRange(enhanced, (200, 200, 200), (255, 255, 255))
        highlight_mask = cv2.GaussianBlur(highlight_mask, (5, 5), 0)
        
        # Boost highlights
        highlights_boosted = cv2.addWeighted(enhanced, 1.0, np.ones_like(enhanced) * 255, 0.2, 0)
        mask_3ch = np.stack([highlight_mask] * 3, axis=-1) / 255.0
        enhanced = (highlights_boosted * mask_3ch + enhanced * (1 - mask_3ch)).astype(np.uint8)
        
        # Step 9: Final adjustments using PIL
        pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        
        # Increase contrast
        contrast_enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = contrast_enhancer.enhance(1.15)
        
        # Slight brightness boost
        brightness_enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = brightness_enhancer.enhance(1.05)
        
        # Final conversion
        final_enhanced = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        return final_enhanced
    
    def process_image(self, input_path: str) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        """Main processing function"""
        logger.info("Starting enhancement processing...")
        
        # Load image
        if input_path.startswith('http'):
            response = requests.get(input_path)
            image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(input_path)
        
        if image is None:
            raise ValueError("Failed to load image")
        
        original_shape = image.shape
        logger.info(f"Original image shape: {original_shape}")
        
        # Detect ring regions
        ring_regions = self.detect_ring_regions(image)
        logger.info(f"Detected {len(ring_regions['candidates'])} ring candidates")
        
        # Detect color using ring regions
        detected_color = self.detect_ring_color_enhanced(image, ring_regions)
        
        # Apply professional enhancement
        enhanced = self.apply_professional_enhancement(image, detected_color, ring_regions)
        
        processing_info = {
            'original_shape': original_shape,
            'detected_color': detected_color,
            'ring_regions_found': len(ring_regions['candidates']),
            'primary_ring_bbox': ring_regions['primary_region']['bbox'] if ring_regions['primary_region'] else None,
            'enhancement_version': 'v38_professional'
        }
        
        return enhanced, detected_color, processing_info

def handler(event):
    """RunPod handler function"""
    logger.info("=== Enhancement Handler v38 Started ===")
    
    try:
        handler_instance = EnhancementHandler()
        
        # Get input from event
        input_data = event.get('input', {})
        
        # Handle Make.com webhook structure
        if 'data' in input_data and 'output' in input_data['data']:
            if 'output' in input_data['data']['output']:
                enhanced_image_data = input_data['data']['output']['output'].get('enhanced_image', '')
            else:
                enhanced_image_data = input_data['data']['output'].get('enhanced_image', '')
        else:
            enhanced_image_data = input_data.get('enhanced_image', '')
        
        if not enhanced_image_data:
            # Try alternative path for Make.com
            if '4' in input_data and 'data' in input_data['4']:
                enhanced_image_data = input_data['4']['data']['output']['output'].get('enhanced_image', '')
        
        if not enhanced_image_data:
            raise ValueError("No enhanced_image data found in input")
        
        # Remove data URL prefix if present
        if enhanced_image_data.startswith('data:'):
            enhanced_image_data = enhanced_image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(enhanced_image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image from base64")
        
        # Process image
        enhanced_image, detected_color, processing_info = handler_instance.process_image(image)
        
        # Convert result to base64
        _, buffer = cv2.imencode('.jpg', enhanced_image, 
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
        result_base64 = base64.b64encode(buffer).decode('utf-8')
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
        logger.info(f"Detected color: {detected_color}")
        logger.info(f"Ring regions found: {processing_info['ring_regions_found']}")
        
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

# For local testing
if __name__ == "__main__":
    # Test with base64 encoded image
    test_event = {
        "input": {
            "enhanced_image": "base64_encoded_image_data_here"
        }
    }
    
    result = handler(test_event)
    print(json.dumps(result, indent=2))
