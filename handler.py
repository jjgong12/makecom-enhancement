import runpod
import base64
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import io
import os
import time
import traceback

# Version
VERSION = "v37-enhancement"

def find_base64_in_dict(data, depth=0, max_depth=10):
    """Find base64 image in nested dictionary"""
    if depth > max_depth:
        return None
    
    if isinstance(data, str) and len(data) > 100:
        return data
    
    if isinstance(data, dict):
        for key in ['image_base64', 'image', 'base64', 'data', 'input', 'file', 'imageData']:
            if key in data and isinstance(data[key], str) and len(data[key]) > 100:
                return data[key]
        
        for value in data.values():
            result = find_base64_in_dict(value, depth + 1, max_depth)
            if result:
                return result
    
    elif isinstance(data, list):
        for item in data:
            result = find_base64_in_dict(item, depth + 1, max_depth)
            if result:
                return result
    
    return None

def decode_base64_image(base64_str):
    """Decode base64 string to PIL Image"""
    try:
        # Handle data URL format
        if ',' in base64_str and base64_str.startswith('data:'):
            base64_str = base64_str.split(',')[1]
        
        # Clean base64
        base64_str = base64_str.strip()
        
        # Add padding for decoding
        padding = 4 - len(base64_str) % 4
        if padding != 4:
            base64_str += '=' * padding
        
        # Decode
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
        
    except Exception as e:
        print(f"[{VERSION}] Error decoding base64: {e}")
        raise

def encode_image_to_base64(image, format='JPEG'):
    """Encode image to base64 (Make.com compatible)"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=95)
        buffer.seek(0)
        
        # Base64 encode
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # CRITICAL: Remove padding for Make.com
        base64_str = base64_str.rstrip('=')
        
        return base64_str
        
    except Exception as e:
        print(f"[{VERSION}] Error encoding image: {e}")
        raise

def detect_ring_regions(image):
    """Detect potential ring regions using edge detection and circular features"""
    try:
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours that might be rings
        ring_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Skip too small or too large contours
            if area < 100 or area > (w * h * 0.5):
                continue
            
            # Check circularity
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.3:  # Rings are somewhat circular
                    x, y, cw, ch = cv2.boundingRect(contour)
                    aspect_ratio = float(cw) / ch if ch > 0 else 0
                    
                    # Rings typically have aspect ratio close to 1
                    if 0.5 < aspect_ratio < 2.0:
                        ring_candidates.append({
                            'bbox': (x, y, x+cw, y+ch),
                            'center': (x + cw//2, y + ch//2),
                            'area': area,
                            'circularity': circularity
                        })
        
        # Sort by area (larger rings first)
        ring_candidates.sort(key=lambda x: x['area'], reverse=True)
        
        # If we found candidates, return the top ones
        if ring_candidates:
            print(f"[{VERSION}] Found {len(ring_candidates)} potential ring regions")
            return ring_candidates[:3]  # Return top 3 candidates
        
        # If no rings found, return center region as fallback
        print(f"[{VERSION}] No rings detected, using center region")
        cx, cy = w//2, h//2
        size = min(w, h) // 3
        return [{
            'bbox': (cx-size//2, cy-size//2, cx+size//2, cy+size//2),
            'center': (cx, cy),
            'area': size * size,
            'circularity': 0
        }]
        
    except Exception as e:
        print(f"[{VERSION}] Error in ring detection: {e}")
        # Return center region as fallback
        h, w = image.size[1], image.size[0]
        cx, cy = w//2, h//2
        size = min(w, h) // 3
        return [{
            'bbox': (cx-size//2, cy-size//2, cx+size//2, cy+size//2),
            'center': (cx, cy),
            'area': size * size,
            'circularity': 0
        }]

def detect_metal_color_v37(image, ring_regions=None):
    """v37: Much stricter color detection - yellow gold only for TRUE yellow"""
    try:
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        # If ring regions provided, analyze those
        if ring_regions and len(ring_regions) > 0:
            color_votes = []
            
            for region in ring_regions:
                x1, y1, x2, y2 = region['bbox']
                # Ensure bounds are valid
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    roi = img_np[y1:y2, x1:x2]
                    
                    # Calculate color statistics for this region
                    r_mean = np.mean(roi[:, :, 0])
                    g_mean = np.mean(roi[:, :, 1])
                    b_mean = np.mean(roi[:, :, 2])
                    
                    brightness = (r_mean + g_mean + b_mean) / 3
                    max_channel = max(r_mean, g_mean, b_mean)
                    min_channel = min(r_mean, g_mean, b_mean)
                    saturation = max_channel - min_channel
                    
                    # Calculate yellowness (how much yellow vs other colors)
                    yellowness = (r_mean + g_mean) / 2 - b_mean
                    
                    print(f"[{VERSION}] Region analysis - R:{r_mean:.1f} G:{g_mean:.1f} B:{b_mean:.1f}")
                    print(f"[{VERSION}] Brightness:{brightness:.1f} Saturation:{saturation:.1f} Yellowness:{yellowness:.1f}")
                    
                    # Determine color for this region
                    # 1. Rose Gold - clear pinkish tone
                    if r_mean - b_mean > 25 and r_mean > g_mean > b_mean and r_mean - g_mean > 10:
                        color_votes.append("rose_gold")
                    # 2. Yellow Gold - ONLY if clearly yellow
                    elif yellowness > 40 and g_mean > 180 and saturation > 30:
                        color_votes.append("yellow_gold")
                    # 3. Plain White - very bright and no color
                    elif brightness > 235 and saturation < 8:
                        color_votes.append("plain_white")
                    # 4. White Gold - bright metallic
                    elif brightness > 190 and saturation < 25:
                        color_votes.append("white_gold")
                    # 5. Default to plain white if uncertain
                    else:
                        color_votes.append("plain_white")
            
            # Vote on final color
            if color_votes:
                from collections import Counter
                color_count = Counter(color_votes)
                detected_color = color_count.most_common(1)[0][0]
                print(f"[{VERSION}] Color votes: {dict(color_count)}")
                print(f"[{VERSION}] Final detection: {detected_color}")
                return detected_color
        
        # Fallback: analyze center region
        center_y, center_x = h//2, w//2
        crop_size = min(h, w) // 3
        
        y1 = max(0, center_y - crop_size)
        y2 = min(h, center_y + crop_size)
        x1 = max(0, center_x - crop_size)
        x2 = min(w, center_x + crop_size)
        
        center_region = img_np[y1:y2, x1:x2]
        
        # Calculate color statistics
        r_mean = np.mean(center_region[:, :, 0])
        g_mean = np.mean(center_region[:, :, 1])
        b_mean = np.mean(center_region[:, :, 2])
        
        brightness = (r_mean + g_mean + b_mean) / 3
        max_channel = max(r_mean, g_mean, b_mean)
        min_channel = min(r_mean, g_mean, b_mean)
        saturation = max_channel - min_channel
        yellowness = (r_mean + g_mean) / 2 - b_mean
        
        print(f"[{VERSION}] Center region - R:{r_mean:.1f} G:{g_mean:.1f} B:{b_mean:.1f}")
        print(f"[{VERSION}] Yellowness:{yellowness:.1f}")
        
        # Very strict detection
        if r_mean - b_mean > 25 and r_mean > g_mean > b_mean and r_mean - g_mean > 10:
            return "rose_gold"
        elif yellowness > 40 and g_mean > 180 and saturation > 30:
            return "yellow_gold"
        elif brightness > 235 and saturation < 8:
            return "plain_white"
        elif brightness > 190 and saturation < 25:
            return "white_gold"
        else:
            return "plain_white"  # Default to plain white instead of yellow gold
            
    except Exception as e:
        print(f"[{VERSION}] Error in metal detection: {e}")
        return "plain_white"  # Safer default

def enhance_wedding_ring_v37(image, metal_type=None):
    """v37 Enhancement - Color-aware brightness enhancement"""
    try:
        print(f"[{VERSION}] Enhancing {metal_type} ring")
        
        # Metal-specific enhancement parameters
        if metal_type == "yellow_gold":
            # True yellow gold - warm enhancement
            brightness_factor = 1.14
            contrast_factor = 1.10
            saturation_factor = 1.05  # Keep yellow tones
            gamma = 0.93
            background_blend = 0.12
        elif metal_type == "plain_white":
            # Extra bright for plain white
            brightness_factor = 1.25
            contrast_factor = 1.08
            saturation_factor = 0.88  # More desaturated
            gamma = 0.83  # Brighter
            background_blend = 0.22
        elif metal_type == "rose_gold":
            # Warm enhancement for rose gold
            brightness_factor = 1.16
            contrast_factor = 1.10
            saturation_factor = 1.03  # Keep warm tones
            gamma = 0.91
            background_blend = 0.14
        elif metal_type == "white_gold":
            # Cool enhancement for white gold
            brightness_factor = 1.20
            contrast_factor = 1.10
            saturation_factor = 0.92
            gamma = 0.86
            background_blend = 0.20
        else:
            # Default to plain white settings
            brightness_factor = 1.25
            contrast_factor = 1.08
            saturation_factor = 0.88
            gamma = 0.83
            background_blend = 0.22
        
        # 1. Apply brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        
        # 2. Apply contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        
        # 3. Apply saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation_factor)
        
        # 4. Convert to numpy for advanced processing
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        # 5. Apply white background blend
        white_color = (254, 254, 254)
        
        # Create edge mask
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        
        # Create background mask
        mask = np.ones((h, w), dtype=np.float32)
        mask[edges_dilated > 0] = 0
        mask = cv2.GaussianBlur(mask, (31, 31), 15)
        
        # Apply white background
        for i in range(3):
            img_np[:, :, i] = img_np[:, :, i] * (1 - mask * background_blend) + white_color[i] * mask * background_blend
        
        # 6. Gamma correction
        img_np = np.power(img_np / 255.0, gamma) * 255
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        # 7. Light sharpening
        image = Image.fromarray(img_np)
        image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=30, threshold=3))
        
        print(f"[{VERSION}] Enhancement complete for {metal_type}")
        return image
        
    except Exception as e:
        print(f"[{VERSION}] Error in enhancement: {e}")
        traceback.print_exc()
        return image

def handler(job):
    """RunPod handler - v37 Ring Detection + Stricter Color"""
    print(f"[{VERSION}] ====== Enhancement Handler Started ======")
    print(f"[{VERSION}] Ring detection + stricter color classification")
    
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        print(f"[{VERSION}] Input type: {type(job_input)}")
        print(f"[{VERSION}] Input keys: {list(job_input.keys()) if isinstance(job_input, dict) else 'Not a dict'}")
        
        # Find base64 image
        base64_image = find_base64_in_dict(job_input)
        
        if not base64_image:
            # Try direct string
            if isinstance(job_input, str) and len(job_input) > 100:
                base64_image = job_input
            else:
                return {
                    "output": {
                        "enhanced_image": None,
                        "error": "No image data found",
                        "success": False,
                        "version": VERSION,
                        "debug_info": {
                            "input_keys": list(job_input.keys()) if isinstance(job_input, dict) else [],
                            "input_length": len(str(job_input))
                        }
                    }
                }
        
        print(f"[{VERSION}] Base64 image found, length: {len(base64_image)}")
        
        # Decode image
        try:
            image = decode_base64_image(base64_image)
            print(f"[{VERSION}] Image decoded: {image.size}")
        except Exception as e:
            return {
                "output": {
                    "enhanced_image": None,
                    "error": f"Failed to decode image: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Check image size and resize if too large
        max_dimension = 4000
        if image.width > max_dimension or image.height > max_dimension:
            print(f"[{VERSION}] Image too large ({image.size}), resizing...")
            ratio = max_dimension / max(image.width, image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"[{VERSION}] Resized to: {image.size}")
        
        # Detect ring regions first
        try:
            ring_regions = detect_ring_regions(image)
            print(f"[{VERSION}] Ring detection complete")
        except Exception as e:
            print(f"[{VERSION}] Ring detection failed: {e}")
            ring_regions = None
        
        # Apply color-aware enhancement with ring regions
        try:
            metal_type = detect_metal_color_v37(image, ring_regions)
            enhanced_image = enhance_wedding_ring_v37(image, metal_type)
            print(f"[{VERSION}] Enhancement applied successfully")
        except Exception as e:
            print(f"[{VERSION}] Error during enhancement: {e}")
            traceback.print_exc()
            enhanced_image = image
            metal_type = "unknown"
        
        # Encode result
        try:
            enhanced_base64 = encode_image_to_base64(enhanced_image, format='JPEG')
            print(f"[{VERSION}] Enhanced image encoded, length: {len(enhanced_base64)}")
        except Exception as e:
            return {
                "output": {
                    "enhanced_image": None,
                    "error": f"Failed to encode enhanced image: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return proper structure for Make.com
        result = {
            "output": {
                "enhanced_image": enhanced_base64,
                "success": True,
                "version": VERSION,
                "processing_time": round(processing_time, 2),
                "original_size": list(image.size),
                "detected_metal": metal_type,
                "ring_regions_found": len(ring_regions) if ring_regions else 0,
                "enhancements_applied": {
                    "yellow_gold": "brightness_1.14_contrast_1.10_sat_1.05",
                    "plain_white": "brightness_1.25_contrast_1.08_desat_0.88",
                    "rose_gold": "brightness_1.16_contrast_1.10_sat_1.03",
                    "white_gold": "brightness_1.20_contrast_1.10_desat_0.92"
                }.get(metal_type, "default"),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "warning": "Google Script must add padding: while (base64Data.length % 4 !== 0) { base64Data += '='; }"
            }
        }
        
        print(f"[{VERSION}] ====== Success - Returning Enhanced Image ======")
        print(f"[{VERSION}] Detected metal: {metal_type}")
        print(f"[{VERSION}] Total processing time: {processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"[{VERSION}] CRITICAL ERROR: {error_msg}")
        traceback.print_exc()
        
        return {
            "output": {
                "enhanced_image": None,
                "error": error_msg,
                "success": False,
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

# RunPod serverless start
if __name__ == "__main__":
    print("="*70)
    print(f"Wedding Ring Enhancement {VERSION}")
    print("V37 - Ring Detection + Stricter Color Classification")
    print("Features:")
    print("- Ring region detection using edge detection")
    print("- Much stricter yellow gold detection (only true yellow)")
    print("- Default to plain white instead of yellow gold")
    print("- Analyze multiple ring regions and vote")
    print("- Plain white: Extra bright (25% boost)")
    print("- Max dimension limit: 4000px")
    print("CRITICAL: Padding is removed for Make.com")
    print("Google Apps Script MUST add padding back:")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
