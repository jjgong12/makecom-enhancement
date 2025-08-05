import runpod
import os
import json
import time
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import cv2
import logging
import re
import requests
import string
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor
import functools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################
# ENHANCEMENT HANDLER - 1200x1560
# VERSION: Enhancement-V3-Simplified-Korean
################################

VERSION = "Enhancement-V3-Simplified-Korean"

# Global rembg session with U2Net
REMBG_SESSION = None

# Korean font
KOREAN_FONT_PATH = "/tmp/NotoSansKR-Regular.ttf"

def init_rembg_session():
    """Initialize rembg session with U2Net for faster processing"""
    global REMBG_SESSION
    if REMBG_SESSION is None:
        try:
            from rembg import new_session
            REMBG_SESSION = new_session('u2net')
            logger.info("✅ U2Net session initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize rembg: {e}")
            REMBG_SESSION = None
    return REMBG_SESSION

# Initialize on module load
init_rembg_session()

def download_korean_font():
    """Simple Korean font download"""
    if os.path.exists(KOREAN_FONT_PATH):
        return KOREAN_FONT_PATH
    
    try:
        url = 'https://github.com/google/fonts/raw/main/ofl/notosanskr/NotoSansKR-Regular.ttf'
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            with open(KOREAN_FONT_PATH, 'wb') as f:
                f.write(response.content)
            logger.info(f"✅ Korean font downloaded")
            return KOREAN_FONT_PATH
    except:
        pass
    
    return None

def get_font(size):
    """Get font - simplified"""
    download_korean_font()
    
    if os.path.exists(KOREAN_FONT_PATH):
        try:
            return ImageFont.truetype(KOREAN_FONT_PATH, size)
        except:
            pass
    
    # Fallback to system fonts
    for font_path in [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    ]:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                pass
    
    return ImageFont.load_default()

def get_text_size(draw, text, font):
    """Get text size"""
    try:
        # Try new method first
        bbox = draw.textbbox((0, 0), text, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except AttributeError:
        # Old method
        return draw.textsize(text, font=font)

def wrap_text(text, font, max_width, draw):
    """Simple text wrapping"""
    lines = []
    paragraphs = text.split('\n')
    
    for paragraph in paragraphs:
        if not paragraph:
            lines.append('')
            continue
        
        words = paragraph.split()
        if not words:
            words = list(paragraph)  # For Korean
        
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word]) if current_line else word
            width, _ = get_text_size(draw, test_line, font)
            
            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
    
    return lines

def create_md_talk_section(text_content=None, width=1200):
    """Create MD TALK section - simplified"""
    logger.info("🔤 Creating MD TALK section")
    
    fixed_width = 1200
    fixed_height = 400
    
    left_margin = 100
    right_margin = 100
    top_margin = 60
    content_width = fixed_width - left_margin - right_margin
    
    # Get fonts
    title_font = get_font(48)
    body_font = get_font(28)
    
    # Create white background
    section_img = Image.new('RGB', (fixed_width, fixed_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Title
    title = "MD TALK"
    title_width, title_height = get_text_size(draw, title, title_font)
    title_x = (fixed_width - title_width) // 2
    draw.text((title_x, top_margin), title, font=title_font, fill=(40, 40, 40))
    
    # Text content
    if text_content and text_content.strip():
        text = text_content.replace('MD TALK', '').replace('MD Talk', '').strip()
    else:
        text = """이 제품은 일상에서도 부담없이 착용할 수 있는 편안한 디자인으로 매일의 스타일링에 포인트를 더해줍니다. 특별한 날은 물론 평범한 일상까지 모든 순간을 빛나게 만들어주는 당신만의 특별한 주얼리입니다."""
    
    # Wrap text
    wrapped_lines = wrap_text(text, body_font, content_width, draw)
    
    # Draw text
    line_height = 45
    title_bottom_margin = 60
    y_pos = top_margin + title_height + title_bottom_margin
    
    for line in wrapped_lines:
        if line:
            line_width, _ = get_text_size(draw, line, body_font)
            line_x = (fixed_width - line_width) // 2
            draw.text((line_x, y_pos), line, font=body_font, fill=(80, 80, 80))
            y_pos += line_height
            
            if y_pos > fixed_height - 50:
                break
    
    return section_img

def create_design_point_section(text_content=None, width=1200):
    """Create DESIGN POINT section - simplified"""
    logger.info("🔤 Creating DESIGN POINT section")
    
    fixed_width = 1200
    fixed_height = 350
    
    left_margin = 100
    right_margin = 100
    top_margin = 60
    content_width = fixed_width - left_margin - right_margin
    
    # Get fonts
    title_font = get_font(48)
    body_font = get_font(24)
    
    # Create white background
    section_img = Image.new('RGB', (fixed_width, fixed_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Title
    title = "DESIGN POINT"
    title_width, title_height = get_text_size(draw, title, title_font)
    title_x = (fixed_width - title_width) // 2
    draw.text((title_x, top_margin), title, font=title_font, fill=(40, 40, 40))
    
    # Text content
    if text_content and text_content.strip():
        text = text_content.replace('DESIGN POINT', '').replace('Design Point', '').strip()
    else:
        text = """남성 단품은 무광 텍스처와 유광 라인의 조화가 견고한 감성을 전하고 여자 단품은 파베 세팅과 섬세한 밀그레인의 디테일 화려하면서도 고급스러운 반짝임을 표현합니다"""
    
    # Wrap text
    wrapped_lines = wrap_text(text, body_font, content_width, draw)
    
    # Draw text
    line_height = 40
    title_bottom_margin = 70
    y_pos = top_margin + title_height + title_bottom_margin
    
    for line in wrapped_lines:
        if line:
            line_width, _ = get_text_size(draw, line, body_font)
            line_x = (fixed_width - line_width) // 2
            draw.text((line_x, y_pos), line, font=body_font, fill=(80, 80, 80))
            y_pos += line_height
            
            if y_pos > fixed_height - 50:
                break
    
    return section_img

def find_special_mode(data, path=""):
    """Find special mode in nested data structures"""
    if isinstance(data, str):
        if data in ['both_text_sections', 'md_talk', 'design_point']:
            return data
        return None
    
    if isinstance(data, dict):
        if 'special_mode' in data and data['special_mode']:
            return data['special_mode']
        
        for key, value in data.items():
            if key in ['enhanced_image', 'image', 'base64', 'image_base64'] and isinstance(value, str) and len(value) > 1000:
                continue
                
            if isinstance(value, dict):
                result = find_special_mode(value, f"{path}.{key}")
                if result:
                    return result
            elif isinstance(value, str) and value in ['both_text_sections', 'md_talk', 'design_point']:
                return value
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    result = find_special_mode(item, f"{path}.{key}[{i}]")
                    if result:
                        return result
    
    return None

def find_text_content(data, content_type):
    """Find text content for MD TALK or DESIGN POINT"""
    if isinstance(data, dict):
        if content_type == 'md_talk':
            keys = ['md_talk_content', 'md_talk', 'md_talk_text', 'text_content', 'claude_text']
        elif content_type == 'design_point':
            keys = ['design_point_content', 'design_point', 'design_point_text', 'text_content', 'claude_text']
        else:
            keys = ['text_content', 'claude_text', 'text']
        
        for key in keys:
            if key in data and isinstance(data[key], str) and data[key].strip():
                return data[key]
        
        for key, value in data.items():
            if key in ['enhanced_image', 'image', 'base64', 'image_base64'] and isinstance(value, str) and len(value) > 1000:
                continue
                
            if isinstance(value, dict):
                result = find_text_content(value, content_type)
                if result:
                    return result
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        result = find_text_content(item, content_type)
                        if result:
                            return result
    
    return None

def fast_ring_detection_phase1(image: Image.Image, max_candidates=20):
    """PHASE 1: Fast Ring Detection"""
    try:
        logger.info("🎯 PHASE 1: Fast Ring Detection Started")
        start_time = time.time()
        
        if image.mode != 'RGB':
            image_rgb = image.convert('RGB')
        else:
            image_rgb = image
            
        img_array = np.array(image_rgb)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        h, w = gray.shape
        
        min_radius = int(min(h, w) * 0.05)
        max_radius = int(min(h, w) * 0.4)
        
        circles = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT,
            dp=2,
            minDist=min_radius * 2,
            param1=100,
            param2=50,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        ring_candidates = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for i, (x, y, r) in enumerate(circles[0]):
                if r < min_radius or r > max_radius:
                    continue
                    
                y1 = max(0, y - r - 10)
                y2 = min(h, y + r + 10)
                x1 = max(0, x - r - 10)
                x2 = min(w, x + r + 10)
                
                region = gray[y1:y2, x1:x2]
                
                center_mask = np.zeros_like(region)
                cv2.circle(center_mask, (x - x1, y - y1), int(r * 0.5), 255, -1)
                
                center_brightness = np.mean(region[center_mask > 0]) if np.any(center_mask > 0) else 0
                edge_brightness = np.mean(region[center_mask == 0]) if np.any(center_mask == 0) else 0
                
                brightness_diff = abs(center_brightness - edge_brightness)
                score = brightness_diff / 255.0
                
                ring_candidates.append({
                    'id': i,
                    'center': (int(x), int(y)),
                    'radius': int(r),
                    'score': float(score),
                    'bbox': (x1, y1, x2, y2),
                    'inner_radius': max(1, int(r * 0.3)),
                    'type': 'circle'
                })
        
        if len(ring_candidates) < 3:
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours[:50]:
                area = cv2.contourArea(contour)
                if area < 500 or area > (h * w * 0.5):
                    continue
                
                if len(contour) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(contour)
                        center, (width, height), angle = ellipse
                        
                        if 0.7 < width/height < 1.3:
                            radius = int((width + height) / 4)
                            if min_radius < radius < max_radius:
                                x, y = int(center[0]), int(center[1])
                                ring_candidates.append({
                                    'id': len(ring_candidates),
                                    'center': (x, y),
                                    'radius': radius,
                                    'score': 0.5,
                                    'bbox': (max(0, x-radius-10), 
                                           max(0, y-radius-10),
                                           min(w, x+radius+10),
                                           min(h, y+radius+10)),
                                    'inner_radius': max(1, int(radius * 0.3)),
                                    'type': 'ellipse'
                                })
                    except:
                        continue
        
        ring_candidates.sort(key=lambda x: x['score'], reverse=True)
        ring_candidates = ring_candidates[:max_candidates]
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Phase 1 complete in {elapsed:.2f}s")
        
        return {
            'candidates': ring_candidates,
            'image_size': (w, h),
            'detection_time': elapsed,
            'method': 'fast_detection',
            'total_candidates': len(ring_candidates)
        }
        
    except Exception as e:
        logger.error(f"Fast ring detection failed: {e}")
        return {
            'candidates': [],
            'error': str(e),
            'image_size': image.size,
            'detection_time': 0,
            'method': 'fast_detection'
        }

def precise_ring_removal_phase2(image: Image.Image, detection_result: dict):
    """PHASE 2: Precise Background Removal"""
    try:
        from rembg import remove
        
        logger.info("✨ PHASE 2: Precise Ring Removal Started")
        start_time = time.time()
        
        candidates = detection_result.get('candidates', [])
        if not candidates:
            logger.warning("No ring candidates found, applying general removal")
            return u2net_original_optimized_removal(image)
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        global REMBG_SESSION
        if REMBG_SESSION is None:
            REMBG_SESSION = init_rembg_session()
        
        r, g, b, a = image.split()
        alpha_array = np.array(a, dtype=np.uint8)
        rgb_array = np.array(image.convert('RGB'))
        
        processed_rings = []
        
        for i, candidate in enumerate(candidates[:10]):
            x1, y1, x2, y2 = candidate['bbox']
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(image.width, x2 + margin)
            y2 = min(image.height, y2 + margin)
            
            ring_region = image.crop((x1, y1, x2, y2))
            
            buffered = BytesIO()
            ring_region.save(buffered, format="PNG")
            buffered.seek(0)
            
            output = remove(
                buffered.getvalue(),
                session=REMBG_SESSION,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=50,
                alpha_matting_erode_size=10,
                only_mask=False,
                post_process_mask=True
            )
            
            processed_region = Image.open(BytesIO(output))
            if processed_region.mode != 'RGBA':
                processed_region = processed_region.convert('RGBA')
            
            _, _, _, region_alpha = processed_region.split()
            region_alpha_array = np.array(region_alpha)
            
            cx, cy = candidate['center']
            radius = candidate['radius']
            inner_radius = candidate['inner_radius']
            
            local_cx = cx - x1
            local_cy = cy - y1
            
            hole_mask = np.zeros_like(region_alpha_array)
            cv2.circle(hole_mask, (local_cx, local_cy), inner_radius, 255, -1)
            
            region_gray = cv2.cvtColor(np.array(ring_region.convert('RGB')), cv2.COLOR_RGB2GRAY)
            center_brightness = np.mean(
                region_gray[max(0, local_cy-10):min(region_gray.shape[0], local_cy+10),
                           max(0, local_cx-10):min(region_gray.shape[1], local_cx+10)]
            )
            
            if center_brightness > 230:
                region_alpha_array[hole_mask > 0] = 0
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                dilated = cv2.dilate(hole_mask, kernel, iterations=2)
                transition = (dilated > 0) & (hole_mask == 0)
                region_alpha_array[transition] = region_alpha_array[transition] // 2
            
            region_alpha_array = cv2.bilateralFilter(region_alpha_array, 9, 75, 75)
            
            alpha_float = region_alpha_array.astype(np.float32) / 255.0
            k = 150
            threshold = 0.5
            alpha_float = 1 / (1 + np.exp(-k * (alpha_float - threshold)))
            region_alpha_array = (alpha_float * 255).astype(np.uint8)
            
            processed_rings.append({
                'bbox': (x1, y1, x2, y2),
                'alpha': region_alpha_array,
                'center': candidate['center'],
                'radius': candidate['radius'],
                'has_hole': center_brightness > 230
            })
            
            alpha_array[y1:y2, x1:x2] = region_alpha_array
        
        processed_mask = np.zeros_like(alpha_array)
        for ring in processed_rings:
            x1, y1, x2, y2 = ring['bbox']
            processed_mask[y1:y2, x1:x2] = 255
        
        if np.any(processed_mask == 0):
            gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
            h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
            
            is_background = (
                ((gray > 240) | (gray < 20)) |
                ((s < 30) & (v > 200)) |
                ((s < 20) & (v < 50))
            )
            
            unprocessed = processed_mask == 0
            alpha_array[unprocessed & is_background] = 0
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha_binary = (alpha_array > 128).astype(np.uint8)
        alpha_cleaned = cv2.morphologyEx(alpha_binary, cv2.MORPH_OPEN, kernel)
        alpha_cleaned = cv2.morphologyEx(alpha_cleaned, cv2.MORPH_CLOSE, kernel)
        
        alpha_array = cv2.GaussianBlur(alpha_array, (3, 3), 0.5)
        alpha_array[alpha_cleaned == 0] = 0
        
        a_new = Image.fromarray(alpha_array)
        result = Image.merge('RGBA', (r, g, b, a_new))
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Phase 2 complete in {elapsed:.2f}s")
        
        return {
            'image': result,
            'processed_rings': len(processed_rings),
            'processing_time': elapsed,
            'rings_with_holes': sum(1 for r in processed_rings if r['has_hole']),
            'method': 'precise_focused_removal'
        }
        
    except Exception as e:
        logger.error(f"Precise removal failed: {e}")
        return {
            'image': u2net_original_optimized_removal(image),
            'error': str(e),
            'method': 'fallback_general_removal'
        }

def combined_two_phase_processing(image: Image.Image):
    """Combined 2-phase processing"""
    logger.info("🚀 Starting 2-Phase Ring Processing")
    total_start = time.time()
    
    detection_result = fast_ring_detection_phase1(image, max_candidates=15)
    removal_result = precise_ring_removal_phase2(image, detection_result)
    
    total_elapsed = time.time() - total_start
    
    if isinstance(removal_result, dict) and 'image' in removal_result:
        result_image = removal_result['image']
    else:
        result_image = removal_result
    
    logger.info(f"✨ Total processing time: {total_elapsed:.2f}s")
    
    return result_image

def u2net_original_optimized_removal(image: Image.Image) -> Image.Image:
    """Original optimized removal method"""
    try:
        from rembg import remove
        
        global REMBG_SESSION
        if REMBG_SESSION is None:
            REMBG_SESSION = init_rembg_session()
            if REMBG_SESSION is None:
                return image
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        contrast = ImageEnhance.Contrast(image)
        image_enhanced = contrast.enhance(1.3)
        
        buffered = BytesIO()
        image_enhanced.save(buffered, format="PNG", compress_level=1)
        buffered.seek(0)
        img_data = buffered.getvalue()
        
        output = remove(
            img_data,
            session=REMBG_SESSION,
            alpha_matting=True,
            alpha_matting_foreground_threshold=270,
            alpha_matting_background_threshold=0,
            alpha_matting_erode_size=0,
            only_mask=False,
            post_process_mask=True
        )
        
        result_image = Image.open(BytesIO(output))
        
        if result_image.mode != 'RGBA':
            result_image = result_image.convert('RGBA')
        
        return result_image
        
    except Exception as e:
        logger.error(f"U2Net removal failed: {e}")
        if image.mode != 'RGBA':
            return image.convert('RGBA')
        return image

def u2net_optimized_removal(image: Image.Image) -> Image.Image:
    """2-Phase optimized removal"""
    try:
        result = combined_two_phase_processing(image)
        
        if result and result.mode == 'RGBA':
            return result
        else:
            return u2net_original_optimized_removal(image)
            
    except Exception as e:
        logger.error(f"2-Phase removal failed: {e}")
        return u2net_original_optimized_removal(image)

def image_to_base64(image, keep_transparency=True):
    """Convert to base64 WITH padding"""
    buffered = BytesIO()
    
    if image.mode != 'RGBA' and keep_transparency:
        image = image.convert('RGBA')
    
    if image.mode == 'RGBA':
        image.save(buffered, format='PNG', compress_level=3, optimize=True)
    else:
        image.save(buffered, format='PNG', optimize=True, compress_level=3)
    
    buffered.seek(0)
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return base64_str

def resize_image_proportional(image, target_width=1200, target_height=1560):
    """Resize image proportionally"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    original_width, original_height = image.size
    
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    scale = min(scale_x, scale_y)
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    result = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))
    
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    
    result.paste(resized, (paste_x, paste_y), resized)
    
    if result.mode != 'RGBA':
        result = result.convert('RGBA')
    
    return result

def process_special_mode(job):
    """Process special modes - MD TALK and DESIGN POINT"""
    special_mode = job.get('special_mode', '')
    
    if not special_mode:
        special_mode = find_special_mode(job)
    
    logger.info(f"🔤 Processing special mode: '{special_mode}'")
    
    if not special_mode or special_mode not in ['both_text_sections', 'md_talk', 'design_point']:
        md_talk_content = find_text_content(job, 'md_talk')
        design_point_content = find_text_content(job, 'design_point')
        
        if md_talk_content and design_point_content:
            special_mode = 'both_text_sections'
        elif md_talk_content:
            special_mode = 'md_talk'
        elif design_point_content:
            special_mode = 'design_point'
        else:
            return {
                "output": {
                    "error": f"Invalid or missing special mode",
                    "status": "error",
                    "version": VERSION
                }
            }
    
    if special_mode == 'both_text_sections':
        md_talk_text = find_text_content(job, 'md_talk')
        design_point_text = find_text_content(job, 'design_point')
        
        if not md_talk_text:
            md_talk_text = """각도에 따라 달라지는 빛의 결들이 두 사람의 특별한 순간순간을 더 찬란하게 만들며 360도 새겨진 패턴으로 매일 새로운 반짝임을 보여줍니다 :)"""
        
        if not design_point_text:
            design_point_text = """입체적인 컷팅 위로 섬세하게 빛나는 패턴이 고급스러움을 완성하며 각진 텍스처가 심플하면서 유니크한 매력을 더해줍니다."""
        
        md_section = create_md_talk_section(md_talk_text)
        design_section = create_design_point_section(design_point_text)
        
        md_base64 = image_to_base64(md_section, keep_transparency=False)
        design_base64 = image_to_base64(design_section, keep_transparency=False)
        
        return {
            "output": {
                "images": [
                    {
                        "enhanced_image": md_base64,
                        "enhanced_image_with_prefix": f"data:image/png;base64,{md_base64}",
                        "section_type": "md_talk",
                        "filename": "ac_wedding_004.png",
                        "file_number": "004",
                        "final_size": list(md_section.size),
                        "format": "PNG"
                    },
                    {
                        "enhanced_image": design_base64,
                        "enhanced_image_with_prefix": f"data:image/png;base64,{design_base64}",
                        "section_type": "design_point", 
                        "filename": "ac_wedding_008.png",
                        "file_number": "008",
                        "final_size": list(design_section.size),
                        "format": "PNG"
                    }
                ],
                "total_images": 2,
                "special_mode": special_mode,
                "sections_included": ["MD_TALK", "DESIGN_POINT"],
                "version": VERSION,
                "status": "success",
                "base64_padding": "INCLUDED"
            }
        }
    
    elif special_mode == 'md_talk':
        text_content = find_text_content(job, 'md_talk')
        
        if not text_content:
            text_content = """이 제품은 일상에서도 부담없이 착용할 수 있는 편안한 디자인으로 매일의 스타일링에 포인트를 더해줍니다. 특별한 날은 물론 평범한 일상까지 모든 순간을 빛나게 만들어주는 당신만의 특별한 주얼리입니다."""
        
        section_image = create_md_talk_section(text_content)
        section_base64 = image_to_base64(section_image, keep_transparency=False)
        
        return {
            "output": {
                "enhanced_image": section_base64,
                "enhanced_image_with_prefix": f"data:image/png;base64,{section_base64}",
                "section_type": "md_talk",
                "filename": "ac_wedding_004.png",
                "file_number": "004",
                "final_size": list(section_image.size),
                "version": VERSION,
                "status": "success",
                "format": "PNG",
                "special_mode": special_mode,
                "base64_padding": "INCLUDED"
            }
        }
    
    elif special_mode == 'design_point':
        text_content = find_text_content(job, 'design_point')
        
        if not text_content:
            text_content = """남성 단품은 무광 텍스처와 유광 라인의 조화가 견고한 감성을 전하고 여자 단품은 파베 세팅과 섬세한 밀그레인의 디테일로 화려하면서도 고급스러운 반짝임을 표현합니다."""
        
        section_image = create_design_point_section(text_content)
        section_base64 = image_to_base64(section_image, keep_transparency=False)
        
        return {
            "output": {
                "enhanced_image": section_base64,
                "enhanced_image_with_prefix": f"data:image/png;base64,{section_base64}",
                "section_type": "design_point",
                "filename": "ac_wedding_008.png",
                "file_number": "008",
                "final_size": list(section_image.size),
                "version": VERSION,
                "status": "success",
                "format": "PNG",
                "special_mode": special_mode,
                "base64_padding": "INCLUDED"
            }
        }

def extract_file_number(filename: str) -> str:
    """Extract number from filename"""
    if not filename:
        return None
    
    match = re.search(r'(\d{3})', filename)
    if match:
        return match.group(1)
    
    match = re.search(r'(\d{2})', filename)
    if match:
        return match.group(1).zfill(3)
    
    return None

def find_input_data_fast(data, depth=0, max_depth=10):
    """Find input data"""
    if depth > max_depth:
        return None
        
    if isinstance(data, str) and len(data) > 50:
        sample = data[:100].strip()
        if all(c in string.ascii_letters + string.digits + '+/=' for c in sample):
            return data
    
    if isinstance(data, dict):
        priority_keys = ['enhanced_image', 'image', 'image_base64', 'base64', 'img', 
                        'input_image', 'original_image', 'base64_image', 'imageData']
        
        for key in priority_keys:
            if key in data and isinstance(data[key], str) and len(data[key]) > 50:
                return data[key]
        
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 1000:
                sample = value[:100].strip()
                if all(c in string.ascii_letters + string.digits + '+/=' for c in sample):
                    return value
            elif isinstance(value, (dict, list)):
                result = find_input_data_fast(value, depth + 1, max_depth)
                if result:
                    return result
    
    elif isinstance(data, list):
        for item in data:
            result = find_input_data_fast(item, depth + 1, max_depth)
            if result:
                return result
    
    return None

def find_filename_fast(data, depth=0, max_depth=10):
    """Find filename"""
    if depth > max_depth:
        return None
        
    if isinstance(data, dict):
        for key in ['filename', 'file_name', 'name', 'fileName', 'file', 'fname']:
            if key in data and isinstance(data[key], str) and data[key].strip():
                return data[key]
        
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                result = find_filename_fast(value, depth + 1, max_depth)
                if result:
                    return result
    
    elif isinstance(data, list):
        for item in data:
            result = find_filename_fast(item, depth + 1, max_depth)
            if result:
                return result
    
    return None

def decode_base64_fast(base64_str: str) -> bytes:
    """Fast base64 decode"""
    try:
        if not base64_str or len(base64_str) < 50:
            raise ValueError("Invalid base64 string")
        
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[-1]
        
        base64_str = ''.join(base64_str.split())
        
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        base64_str = ''.join(c for c in base64_str if c in valid_chars)
        
        try:
            decoded = base64.b64decode(base64_str, validate=True)
            return decoded
        except:
            no_pad = base64_str.rstrip('=')
            padding_needed = (4 - len(no_pad) % 4) % 4
            padded = no_pad + ('=' * padding_needed)
            decoded = base64.b64decode(padded, validate=True)
            return decoded
            
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise ValueError(f"Invalid base64 data: {str(e)}")

def handler(event):
    """Enhancement handler - Simplified Korean"""
    try:
        logger.info(f"=== {VERSION} Started ===")
        
        special_mode = find_special_mode(event)
        
        if special_mode and special_mode in ['both_text_sections', 'md_talk', 'design_point']:
            return process_special_mode(event)
        
        filename = find_filename_fast(event)
        image_data_str = find_input_data_fast(event)
        
        if not image_data_str:
            md_talk_content = find_text_content(event, 'md_talk')
            design_point_content = find_text_content(event, 'design_point')
            
            if md_talk_content and design_point_content:
                event['special_mode'] = 'both_text_sections'
                return process_special_mode(event)
            elif md_talk_content:
                event['special_mode'] = 'md_talk'
                return process_special_mode(event)
            elif design_point_content:
                event['special_mode'] = 'design_point'
                return process_special_mode(event)
            
            raise ValueError("No input image data or text content found")
        
        start_time = time.time()
        image_bytes = decode_base64_fast(image_data_str)
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        decode_time = time.time() - start_time
        
        start_time = time.time()
        image = u2net_optimized_removal(image)
        removal_time = time.time() - start_time
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        start_time = time.time()
        image = resize_image_proportional(image, 1200, 1560)
        resize_time = time.time() - start_time
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        start_time = time.time()
        enhanced_base64 = image_to_base64(image, keep_transparency=True)
        encode_time = time.time() - start_time
        
        total_time = decode_time + removal_time + resize_time + encode_time
        
        output_filename = filename or "enhanced_image.png"
        file_number = extract_file_number(output_filename)
        
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "enhanced_image_with_prefix": f"data:image/png;base64,{enhanced_base64}",
                "filename": output_filename,
                "file_number": file_number,
                "final_size": list(image.size),
                "version": VERSION,
                "status": "success",
                "format": "PNG",
                "mode": "RGBA",
                "has_transparency": True,
                "transparency_preserved": True,
                "background_removed": True,
                "ring_holes_applied": True,
                "base64_padding": "INCLUDED",
                "compression": "level_3",
                "processing_times": {
                    "decode": f"{decode_time:.2f}s",
                    "two_phase_processing": f"{removal_time:.2f}s",
                    "resize": f"{resize_time:.2f}s",
                    "encode": f"{encode_time:.2f}s",
                    "total": f"{total_time:.2f}s"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

# RunPod handler
runpod.serverless.start({"handler": handler})
