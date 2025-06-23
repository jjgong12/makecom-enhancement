import os
import io
import json
import base64
import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import replicate
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time

VERSION = "v3"

def log_debug(message):
    """Debug logging with version info"""
    print(f"[{VERSION}] {message}")

def decode_base64_image(base64_string):
    """Decode base64 image with enhanced error handling"""
    try:
        log_debug("Starting base64 decode process")
        
        # Remove whitespace and newlines
        base64_string = base64_string.strip().replace('\n', '').replace('\r', '').replace(' ', '')
        
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
            log_debug("Removed data URL prefix")
        
        # Clean any non-base64 characters
        import re
        base64_string = re.sub(r'[^A-Za-z0-9+/=]', '', base64_string)
        
        # Try standard decode first (without adding padding)
        try:
            image_data = base64.b64decode(base64_string, validate=True)
            log_debug("Standard decode successful")
        except Exception as e:
            log_debug(f"Standard decode failed: {e}, trying with padding")
            # Add padding if needed
            missing_padding = len(base64_string) % 4
            if missing_padding:
                base64_string += '=' * (4 - missing_padding)
            image_data = base64.b64decode(base64_string, validate=True)
            log_debug("Decode with padding successful")
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image data")
            
        log_debug(f"Image decoded successfully: {image.shape}")
        return image
        
    except Exception as e:
        log_debug(f"Base64 decode error: {e}")
        raise

def enhance_image_to_target_style(image):
    """Apply Image 3 style color enhancement - bright, clean, white background"""
    try:
        log_debug("Applying Image 3 target style color enhancement")
        
        # Convert to RGB for PIL processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Step 1: Brightness boost (25% increase like Image 3)
        brightness_enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = brightness_enhancer.enhance(1.25)
        log_debug("Applied 25% brightness boost")
        
        # Step 2: Slight contrast increase for clarity
        contrast_enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = contrast_enhancer.enhance(1.1)
        log_debug("Applied contrast enhancement")
        
        # Step 3: Reduce saturation slightly for clean look
        color_enhancer = ImageEnhance.Color(enhanced)
        enhanced = color_enhancer.enhance(0.95)
        log_debug("Applied saturation reduction for clean look")
        
        # Convert back to numpy for further processing
        enhanced_array = np.array(enhanced)
        enhanced_bgr = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2BGR)
        
        # Step 4: Apply white overlay (15% like Image 3)
        white_overlay = np.full_like(enhanced_bgr, 255, dtype=np.uint8)
        enhanced_bgr = cv2.addWeighted(enhanced_bgr, 0.85, white_overlay, 0.15, 0)
        log_debug("Applied 15% white overlay")
        
        # Step 5: Replace background with Image 3 style background
        # Target background: RGB(252, 250, 248) - very bright cream white
        target_bg_bgr = (248, 250, 252)  # BGR format for OpenCV
        
        # Create background mask (detect darker areas that should be background)
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        
        # Multiple threshold approach for better background detection
        bg_mask = np.zeros_like(gray, dtype=np.uint8)
        
        # Detect very dark areas (likely background/shadows)
        dark_mask = gray < 50
        bg_mask[dark_mask] = 255
        
        # Detect edges to refine mask
        edges = cv2.Canny(gray, 30, 100)
        kernel = np.ones((5,5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(bg_mask, edges_dilated)
        
        # Apply background replacement
        enhanced_bgr[combined_mask > 0] = target_bg_bgr
        log_debug("Applied Image 3 style background (252, 250, 248)")
        
        # Step 6: Final brightness adjustment to match Image 3
        enhanced_bgr = np.clip(enhanced_bgr.astype(float) * 1.05, 0, 255).astype(np.uint8)
        log_debug("Applied final brightness adjustment")
        
        return enhanced_bgr
        
    except Exception as e:
        log_debug(f"Enhancement error: {e}")
        # Return original with minimal processing
        return image

def create_thumbnail(image, target_size=(1000, 1300)):
    """Create centered thumbnail matching original style"""
    try:
        log_debug(f"Creating thumbnail {target_size}")
        
        h, w = image.shape[:2]
        
        # Find the ring area (assume center region)
        center_x, center_y = w // 2, h // 2
        
        # Estimate ring size (use center 60% of image)
        ring_size = min(w, h) * 0.6
        
        # Calculate crop area centered on ring
        crop_size = int(ring_size * 1.2)  # Add some padding
        
        x1 = max(0, center_x - crop_size // 2)
        y1 = max(0, center_y - crop_size // 2)
        x2 = min(w, center_x + crop_size // 2)
        y2 = min(h, center_y + crop_size // 2)
        
        # Crop the image
        cropped = image[y1:y2, x1:x2]
        
        # Resize to target size
        thumbnail = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        log_debug(f"Thumbnail created: {thumbnail.shape}")
        return thumbnail
        
    except Exception as e:
        log_debug(f"Thumbnail creation error: {e}")
        # Fallback: simple resize
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

def image_to_base64(image):
    """Convert image to base64 string without padding (Make.com compatible)"""
    try:
        _, buffer = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        # CRITICAL: Remove padding for Make.com compatibility
        img_base64 = img_base64.rstrip('=')
        log_debug("Image converted to base64 (padding removed)")
        return img_base64
    except Exception as e:
        log_debug(f"Base64 conversion error: {e}")
        raise

def safe_replicate_call(model, input_data, timeout=20):
    """Safe Replicate API call with timeout protection"""
    def run_replicate():
        client = replicate.Client(api_token=os.environ.get("REPLICATE_API_TOKEN"))
        return client.run(model, input=input_data)
    
    try:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(run_replicate)
            return future.result(timeout=timeout)
    except TimeoutError:
        log_debug(f"Replicate call timed out after {timeout}s")
        return None
    except Exception as e:
        log_debug(f"Replicate call failed: {e}")
        return None

def handler(event):
    """Main handler function"""
    try:
        log_debug("=== Enhancement Handler v153 Started ===")
        log_debug("Applying Image 3 target color style")
        
        # Extract image data from event
        job_input = event.get("input", {})
        log_debug(f"Event keys: {list(event.keys())}")
        log_debug(f"Input keys: {list(job_input.keys())}")
        
        # Get base64 image data
        base64_image = None
        possible_keys = ["image", "image_base64", "base64_image", "data"]
        
        for key in possible_keys:
            if key in job_input and job_input[key]:
                base64_image = job_input[key]
                log_debug(f"Found image data in key: {key}")
                break
        
        if not base64_image:
            # Check nested structures
            for key, value in job_input.items():
                if isinstance(value, dict) and "image" in value:
                    base64_image = value["image"]
                    log_debug(f"Found nested image data in: {key}")
                    break
        
        if not base64_image:
            raise ValueError("No image data found in event")
        
        log_debug(f"Base64 string length: {len(base64_image)}")
        log_debug(f"Base64 string start: {base64_image[:100]}...")
        
        # Decode image
        image = decode_base64_image(base64_image)
        log_debug(f"Image decoded: {image.shape}")
        
        # Apply Image 3 style enhancement
        enhanced_image = enhance_image_to_target_style(image)
        log_debug("Image 3 style enhancement completed")
        
        # Create thumbnail
        thumbnail = create_thumbnail(enhanced_image)
        log_debug("Thumbnail created")
        
        # Convert to base64
        enhanced_base64 = image_to_base64(enhanced_image)
        thumbnail_base64 = image_to_base64(thumbnail)
        
        # Prepare response with nested output structure for Make.com
        response = {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "version": VERSION,
                    "style": "Image_3_target_style",
                    "background_rgb": "252_250_248",
                    "brightness_boost": "25_percent",
                    "white_overlay": "15_percent",
                    "timestamp": int(time.time())
                }
            }
        }
        
        log_debug("=== Enhancement Handler v153 Completed Successfully ===")
        return response
        
    except Exception as e:
        log_debug(f"Handler error: {e}")
        
        # Return error response with same structure
        return {
            "output": {
                "enhanced_image": "",
                "thumbnail": "",
                "processing_info": {
                    "version": VERSION,
                    "error": str(e),
                    "timestamp": int(time.time())
                }
            }
        }

# RunPod serverless handler
if __name__ == "__main__":
    log_debug("Starting RunPod serverless handler")
    runpod.serverless.start({"handler": handler})
