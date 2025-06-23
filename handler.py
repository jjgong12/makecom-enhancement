import os
import sys
import runpod
import base64
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import traceback
import time

print("[v2] Starting Wedding Ring Enhancement Handler")
print(f"[v2] Python version: {sys.version}")
print(f"[v2] OpenCV version: {cv2.__version__}")
print("="*70)

def remove_padding_safe(base64_string):
    """Remove padding from base64 string for Make.com compatibility"""
    return base64_string.rstrip('=')

def decode_base64_image(base64_string):
    """Decode base64 image with enhanced error handling"""
    try:
        # Clean the base64 string
        base64_string = base64_string.strip()
        
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        # Try direct decode first
        try:
            image_data = base64.b64decode(base64_string)
        except:
            # Add padding if needed
            missing_padding = len(base64_string) % 4
            if missing_padding:
                base64_string += '=' * (4 - missing_padding)
            image_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        return image
    except Exception as e:
        print(f"[v2] Error decoding image: {str(e)}")
        raise

def find_image_in_event(event):
    """Enhanced image finding with multiple fallback strategies"""
    print("[v2] Starting image search...")
    print(f"[v2] Event type: {type(event)}")
    print(f"[v2] Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
    
    # Strategy 1: Direct input
    input_data = event.get("input", {})
    print(f"[v2] Input type: {type(input_data)}")
    
    if isinstance(input_data, dict):
        print(f"[v2] Input keys: {list(input_data.keys())}")
        # Common keys
        for key in ['image', 'image_base64', 'base64', 'img', 'data']:
            if key in input_data and input_data[key]:
                print(f"[v2] Found image in input.{key}")
                return input_data[key]
    
    # Strategy 2: Direct event keys
    for key in ['image', 'image_base64', 'base64']:
        if key in event and event[key]:
            print(f"[v2] Found image in event.{key}")
            return event[key]
    
    # Strategy 3: String input
    if isinstance(input_data, str) and len(input_data) > 100:
        print("[v2] Input is string, assuming base64")
        return input_data
    
    # Strategy 4: Nested search
    if isinstance(input_data, dict):
        for key, value in input_data.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"[v2] Found potential image in input.{key}")
                return value
    
    print("[v2] No image found in event")
    return None

def detect_metal_type(image):
    """Detect metal type based on color analysis"""
    # Convert to LAB color space for better color analysis
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Get center region
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2
    roi_size = min(h, w) // 4
    roi = lab[center_y-roi_size:center_y+roi_size, 
              center_x-roi_size:center_x+roi_size]
    
    # Calculate average LAB values
    avg_lab = np.mean(roi.reshape(-1, 3), axis=0)
    l, a, b = avg_lab
    
    # Detect metal type based on LAB values
    if b > 15:  # Yellow tones
        return "yellow_gold"
    elif a > 5:  # Red/pink tones
        return "rose_gold"
    elif l > 180:  # Very bright
        return "plain_white"
    else:
        return "white_gold"

def enhance_with_backlight_effect(image):
    """Apply enhancement to mimic backlight effect (like photos 4,5,6)"""
    # Step 1: Brighten overall image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Increase brightness significantly
    l = cv2.add(l, 35)  # Stronger brightening for backlight effect
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Step 2: Add soft glow effect
    # Create a blurred version
    blurred = cv2.GaussianBlur(enhanced, (31, 31), 15)
    
    # Blend for soft glow
    enhanced = cv2.addWeighted(enhanced, 0.65, blurred, 0.35, 0)
    
    # Step 3: Enhance colors
    # Increase saturation slightly
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = hsv[:,:,1] * 1.2  # 20% saturation increase
    hsv[:,:,1][hsv[:,:,1] > 255] = 255
    hsv[:,:,2] = hsv[:,:,2] * 1.15  # 15% value increase
    hsv[:,:,2][hsv[:,:,2] > 255] = 255
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Step 4: Final brightness adjustment
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.15, beta=20)
    
    return enhanced

def apply_ring_enhancement(image, metal_type):
    """Apply ring-specific enhancements"""
    # Denoise
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Sharpen details
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Blend
    enhanced = cv2.addWeighted(denoised, 0.6, sharpened, 0.4, 0)
    
    # Metal-specific color correction
    if metal_type == "yellow_gold":
        # Enhance yellow tones
        enhanced[:,:,0] = np.clip(enhanced[:,:,0] * 0.92, 0, 255).astype(np.uint8)  # Reduce blue
        enhanced[:,:,2] = np.clip(enhanced[:,:,2] * 1.08, 0, 255).astype(np.uint8)  # Increase red
    elif metal_type == "rose_gold":
        # Enhance pink tones
        enhanced[:,:,2] = np.clip(enhanced[:,:,2] * 1.1, 0, 255).astype(np.uint8)  # Increase red
        enhanced[:,:,1] = np.clip(enhanced[:,:,1] * 1.02, 0, 255).astype(np.uint8)  # Slight green
    elif metal_type == "white_gold":
        # Cool white tones
        enhanced[:,:,0] = np.clip(enhanced[:,:,0] * 1.03, 0, 255).astype(np.uint8)  # Slight blue
    
    return enhanced

def create_thumbnail(image):
    """Create a thumbnail with ring detection and cropping"""
    target_w, target_h = 1000, 1300
    
    # Try to detect ring
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to find ring
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour (likely the ring)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Add padding
        pad = 60
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        
        # Crop ring region
        ring_crop = image[y1:y2, x1:x2]
        
        # Resize to fit thumbnail maintaining aspect ratio
        crop_h, crop_w = ring_crop.shape[:2]
        scale = min(target_w/crop_w, target_h/crop_h) * 0.85
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        
        resized = cv2.resize(ring_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create white background
        thumbnail = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        
        # Center the ring (slightly higher for better composition)
        y_offset = int((target_h - new_h) * 0.35)  # Top 35%
        x_offset = (target_w - new_w) // 2
        
        thumbnail[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    else:
        # Fallback: resize entire image
        h, w = image.shape[:2]
        scale = min(target_w/w, target_h/h) * 0.8
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create white background
        thumbnail = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        
        # Center the image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        thumbnail[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return thumbnail

def handler(event):
    """RunPod handler function for enhancement"""
    start_time = time.time()
    
    try:
        print("\n" + "="*70)
        print("[v2] Handler started - Complete Fix Version")
        print("[v2] Event received, processing...")
        
        # Find image in event
        base64_image = find_image_in_event(event)
        
        if not base64_image:
            print("[v2] ERROR: No image found")
            return {
                "output": {
                    "error": "No image provided",
                    "status": "error",
                    "version": "v2"
                }
            }
        
        print(f"[v2] Image found, length: {len(base64_image)}")
        
        # Decode image
        image = decode_base64_image(base64_image)
        print(f"[v2] Image decoded: {image.shape}")
        
        # Detect metal type
        metal_type = detect_metal_type(image)
        print(f"[v2] Metal type detected: {metal_type}")
        
        # Apply backlight effect enhancement
        enhanced = enhance_with_backlight_effect(image)
        
        # Apply ring-specific enhancements
        enhanced = apply_ring_enhancement(enhanced, metal_type)
        
        # Create thumbnail
        thumbnail = create_thumbnail(enhanced)
        
        # Convert to base64
        # Main image
        _, buffer = cv2.imencode('.jpg', enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        main_base64 = remove_padding_safe(base64.b64encode(buffer).decode('utf-8'))
        
        # Thumbnail
        _, buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 95])
        thumb_base64 = remove_padding_safe(base64.b64encode(buffer).decode('utf-8'))
        
        processing_time = time.time() - start_time
        
        print(f"[v2] Processing complete in {processing_time:.2f}s")
        print("[v2] Returning results...")
        
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "status": "success",
                    "version": "v2",
                    "processing_time": f"{processing_time:.2f}s"
                }
            }
        }
        
    except Exception as e:
        print(f"[v2] ERROR in handler: {str(e)}")
        print(traceback.format_exc())
        
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": "v2",
                "traceback": traceback.format_exc()
            }
        }

# RunPod serverless entrypoint
if __name__ == "__main__":
    print("[v2] Starting RunPod serverless...")
    runpod.serverless.start({"handler": handler})
