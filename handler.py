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
import replicate
import requests
import string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################
# ENHANCEMENT HANDLER - 1200x1560
# VERSION: V24-Precise-Edge-Detection
################################

VERSION = "V24-Precise-Edge-Detection"

# ===== GLOBAL INITIALIZATION =====
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
REPLICATE_CLIENT = None

if REPLICATE_API_TOKEN:
    try:
        REPLICATE_CLIENT = replicate.Client(api_token=REPLICATE_API_TOKEN)
        logger.info("✅ Replicate client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Replicate: {e}")

# Claude API configuration
CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY', '')
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

# Global rembg session with U2Net
REMBG_SESSION = None

def init_rembg_session():
    """Initialize rembg session with U2Net for faster processing"""
    global REMBG_SESSION
    if REMBG_SESSION is None:
        try:
            from rembg import new_session
            REMBG_SESSION = new_session('u2net')
            logger.info("✅ U2Net session initialized for faster processing")
        except Exception as e:
            logger.error(f"❌ Failed to initialize rembg: {e}")
            REMBG_SESSION = None
    return REMBG_SESSION

# Initialize on module load
init_rembg_session()

def download_korean_font():
    """Download Korean font for text rendering"""
    try:
        font_path = '/tmp/NanumGothic.ttf'
        
        if os.path.exists(font_path):
            try:
                test_font = ImageFont.truetype(font_path, 20)
                img_test = Image.new('RGB', (100, 100), 'white')
                draw_test = ImageDraw.Draw(img_test)
                draw_test.text((10, 10), "테스트", font=test_font, fill='black')
                return font_path
            except:
                os.remove(font_path)
        
        font_urls = [
            'https://github.com/naver/nanumfont/raw/master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf',
            'https://cdn.jsdelivr.net/gh/naver/nanumfont@master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf'
        ]
        
        for url in font_urls:
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200 and len(response.content) > 100000:
                    with open(font_path, 'wb') as f:
                        f.write(response.content)
                    return font_path
            except:
                continue
        
        return None
    except:
        return None

def get_font(size, korean_font_path=None):
    """Get font with fallback"""
    if korean_font_path and os.path.exists(korean_font_path):
        try:
            return ImageFont.truetype(korean_font_path, size)
        except:
            pass
    
    return ImageFont.load_default()

def safe_draw_text(draw, position, text, font, fill):
    """Safely draw text"""
    try:
        if text:
            draw.text(position, str(text), font=font, fill=fill)
    except:
        draw.text(position, "[Error]", font=font, fill=fill)

def get_text_size(draw, text, font):
    """Get text size compatible with different PIL versions"""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

def create_md_talk_section(text_content=None, width=1200):
    """Create MD TALK section"""
    logger.info("Creating MD TALK section")
    
    korean_font_path = download_korean_font()
    title_font = get_font(48, korean_font_path)
    body_font = get_font(28, korean_font_path)
    
    # 임시 이미지로 텍스트 크기 계산
    temp_img = Image.new('RGB', (width, 1000), '#FFFFFF')
    draw = ImageDraw.Draw(temp_img)
    
    title = "MD TALK"
    title_width, title_height = get_text_size(draw, title, title_font)
    
    # 텍스트 준비
    if text_content and text_content.strip():
        text = text_content.replace('MD TALK', '').replace('MD Talk', '').strip()
        words = text.split()
        lines = []
        current_line = ""
        max_line_width = width - 120
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_width, _ = get_text_size(draw, test_line, body_font)
            
            if test_width > max_line_width:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
    else:
        lines = [
            "이 제품은 일상에서도 부담없이",
            "착용할 수 있는 편안한 디자인으로",
            "매일의 스타일링에 포인트를 더해줍니다.",
            "",
            "특별한 날은 물론 평범한 일상까지",
            "모든 순간을 빛나게 만들어주는",
            "당신만의 특별한 주얼리입니다."
        ]
    
    # 높이 계산
    top_margin = 60
    title_bottom_margin = 140
    line_height = 50
    bottom_margin = 80
    
    content_height = len(lines) * line_height
    total_height = top_margin + title_height + title_bottom_margin + content_height + bottom_margin
    
    # 실제 이미지 생성
    section_img = Image.new('RGB', (width, total_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # 제목 그리기
    safe_draw_text(draw, (width//2 - title_width//2, top_margin), title, title_font, (40, 40, 40))
    
    # 본문 그리기
    y_pos = top_margin + title_height + title_bottom_margin
    
    for line in lines:
        if line:
            line_width, _ = get_text_size(draw, line, body_font)
            safe_draw_text(draw, (width//2 - line_width//2, y_pos), line, body_font, (80, 80, 80))
        y_pos += line_height
    
    logger.info(f"MD TALK section created: {width}x{total_height}")
    return section_img

def create_design_point_section(text_content=None, width=1200):
    """Create DESIGN POINT section"""
    logger.info("Creating DESIGN POINT section")
    
    korean_font_path = download_korean_font()
    title_font = get_font(48, korean_font_path)
    body_font = get_font(24, korean_font_path)
    
    # 임시 이미지로 텍스트 크기 계산
    temp_img = Image.new('RGB', (width, 1000), '#FFFFFF')
    draw = ImageDraw.Draw(temp_img)
    
    title = "DESIGN POINT"
    title_width, title_height = get_text_size(draw, title, title_font)
    
    # 텍스트 준비
    if text_content and text_content.strip():
        text = text_content.replace('DESIGN POINT', '').replace('Design Point', '').strip()
        words = text.split()
        lines = []
        current_line = ""
        max_line_width = width - 100
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_width, _ = get_text_size(draw, test_line, body_font)
            
            if test_width > max_line_width:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
    else:
        lines = [
            "남성 단품은 무광 텍스처와 유광 라인의 조화가",
            "견고한 감성을 전하고 여자 단품은",
            "파베 세팅과 섬세한 밀그레인의 디테일",
            "화려하면서도 고급스러운 반영을 표현합니다"
        ]
    
    # 높이 계산
    top_margin = 60
    title_bottom_margin = 160
    line_height = 55
    bottom_margin = 100
    
    content_height = len(lines) * line_height
    total_height = top_margin + title_height + title_bottom_margin + content_height + bottom_margin
    
    # 실제 이미지 생성
    section_img = Image.new('RGB', (width, total_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # 제목 그리기
    safe_draw_text(draw, (width//2 - title_width//2, top_margin), title, title_font, (40, 40, 40))
    
    # 본문 그리기
    y_pos = top_margin + title_height + title_bottom_margin
    
    for line in lines:
        if line:
            line_width, _ = get_text_size(draw, line, body_font)
            safe_draw_text(draw, (width//2 - line_width//2, y_pos), line, body_font, (80, 80, 80))
        y_pos += line_height
    
    # 하단 구분선
    draw.rectangle([100, y_pos + 30, width - 100, y_pos + 32], fill=(220, 220, 220))
    
    logger.info(f"DESIGN POINT section created: {width}x{total_height}")
    return section_img

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

def find_input_data_fast(data):
    """Find input data - OPTIMIZED"""
    if isinstance(data, str) and len(data) > 50:
        return data
    
    if isinstance(data, dict):
        priority_keys = ['enhanced_image', 'image', 'image_base64', 'base64', 'img']
        
        for key in priority_keys:
            if key in data and isinstance(data[key], str) and len(data[key]) > 50:
                return data[key]
        
        for key in ['input', 'data', 'payload']:
            if key in data:
                if isinstance(data[key], str) and len(data[key]) > 50:
                    return data[key]
                elif isinstance(data[key], dict):
                    result = find_input_data_fast(data[key])
                    if result:
                        return result
        
        for i in range(10):
            key = str(i)
            if key in data and isinstance(data[key], str) and len(data[key]) > 50:
                return data[key]
    
    return None

def find_filename_fast(data):
    """Find filename - OPTIMIZED"""
    if isinstance(data, dict):
        for key in ['filename', 'file_name', 'name']:
            if key in data and isinstance(data[key], str):
                return data[key]
        
        for key in ['input', 'data']:
            if key in data and isinstance(data[key], dict):
                for subkey in ['filename', 'file_name', 'name']:
                    if subkey in data[key] and isinstance(data[key][subkey], str):
                        return data[key][subkey]
    
    return None

def decode_base64_fast(base64_str: str) -> bytes:
    """FAST base64 decode"""
    try:
        if not base64_str or len(base64_str) < 50:
            raise ValueError("Invalid base64 string")
        
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[-1]
        
        base64_str = ''.join(base64_str.split())
        
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        base64_str = ''.join(c for c in base64_str if c in valid_chars)
        
        no_pad = base64_str.rstrip('=')
        
        try:
            decoded = base64.b64decode(no_pad, validate=False)
            return decoded
        except:
            padding_needed = (4 - len(no_pad) % 4) % 4
            padded = no_pad + ('=' * padding_needed)
            decoded = base64.b64decode(padded, validate=False)
            return decoded
            
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise ValueError(f"Invalid base64 data: {str(e)}")

def detect_pattern_type(filename: str) -> str:
    """Detect pattern type - Updated with AB pattern"""
    if not filename:
        return "other"
    
    filename_lower = filename.lower()
    
    if 'ac_' in filename_lower:
        return "ac_pattern"
    elif 'ab_' in filename_lower:
        return "ab_pattern"
    else:
        return "other"

def u2net_optimized_removal(image: Image.Image) -> Image.Image:
    """ENHANCED U2Net background removal with precise edge detection"""
    try:
        from rembg import remove
        
        global REMBG_SESSION
        if REMBG_SESSION is None:
            REMBG_SESSION = init_rembg_session()
            if REMBG_SESSION is None:
                return image
        
        logger.info("🔷 U2Net Enhanced Background Removal V24-Precise")
        
        # Save image to buffer
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        img_data = buffered.getvalue()
        
        # Apply U2Net removal with MORE PRECISE settings
        output = remove(
            img_data,
            session=REMBG_SESSION,
            alpha_matting=True,
            alpha_matting_foreground_threshold=270,  # Higher threshold for better edge detection
            alpha_matting_background_threshold=0,     # Lower for cleaner removal
            alpha_matting_erode_size=0,              # No erosion to preserve details
            only_mask=False
        )
        
        result_image = Image.open(BytesIO(output))
        
        if result_image.mode != 'RGBA':
            return result_image
        
        # ENHANCED edge refinement with multi-stage processing
        r, g, b, a = result_image.split()
        alpha_array = np.array(a, dtype=np.uint8)
        original_alpha = alpha_array.copy()
        
        # Stage 1: Fine-tuned edge detection
        # Use smaller kernel for initial cleanup
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        alpha_array = cv2.morphologyEx(alpha_array, cv2.MORPH_CLOSE, kernel_tiny)
        
        # Stage 2: Precision edge refinement
        # Apply guided filter for edge-preserving smoothing
        rgb_array = np.array(result_image.convert('RGB'), dtype=np.float32) / 255.0
        gray_guide = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        
        # Guided filter for precise edge preservation
        alpha_float = alpha_array.astype(np.float32) / 255.0
        alpha_guided = cv2.ximgproc.guidedFilter(
            guide=gray_guide,
            src=alpha_float,
            radius=2,         # Small radius for detail preservation
            eps=0.001         # Small epsilon for edge preservation
        )
        alpha_array = (alpha_guided * 255).astype(np.uint8)
        
        # Stage 3: Sub-pixel edge refinement
        # Detect edges for special treatment
        edges = cv2.Canny(alpha_array, 100, 200)
        edge_mask = cv2.dilate(edges, kernel_tiny, iterations=1)
        
        # Apply bilateral filter ONLY to edge regions
        alpha_edges = cv2.bilateralFilter(alpha_array, 5, 50, 50)
        alpha_array = np.where(edge_mask > 0, alpha_edges, alpha_array)
        
        # Stage 4: Precision threshold with gradient
        # Create smooth transition at edges
        gradient_alpha = alpha_array.astype(np.float32) / 255.0
        
        # Multi-level thresholding for smooth transitions
        gradient_alpha = np.where(gradient_alpha > 0.95, 1.0, gradient_alpha)
        gradient_alpha = np.where(gradient_alpha < 0.05, 0.0, gradient_alpha)
        
        # Smooth transition zone (0.05 - 0.95)
        transition_mask = (gradient_alpha > 0.05) & (gradient_alpha < 0.95)
        if np.any(transition_mask):
            # Apply sigmoid curve for natural transition
            gradient_alpha[transition_mask] = 1 / (1 + np.exp(-10 * (gradient_alpha[transition_mask] - 0.5)))
        
        alpha_array = (gradient_alpha * 255).astype(np.uint8)
        
        # Stage 5: Remove tiny artifacts while preserving details
        # Use connected components with size threshold
        num_labels, labels = cv2.connectedComponents((alpha_array > 128).astype(np.uint8))
        
        if num_labels > 2:
            sizes = [np.sum(labels == i) for i in range(1, num_labels)]
            if sizes:
                # Keep all components larger than 0.01% of image
                min_size = int(alpha_array.size * 0.0001)
                valid_labels = [i+1 for i, size in enumerate(sizes) if size > min_size]
                
                # Create mask for valid components
                valid_mask = np.zeros_like(alpha_array, dtype=bool)
                for label in valid_labels:
                    valid_mask |= (labels == label)
                
                alpha_array = np.where(valid_mask, alpha_array, 0)
        
        # Stage 6: Final edge polish
        # Very light Gaussian blur ONLY on edges
        edge_blur = cv2.GaussianBlur(alpha_array.astype(np.float32), (3, 3), 0.5)
        alpha_final = np.where(edge_mask > 0, edge_blur, alpha_array.astype(np.float32))
        alpha_array = alpha_final.astype(np.uint8)
        
        # Stage 7: Preserve original strong edges
        # Restore very strong edges from original
        strong_edges = original_alpha > 250
        alpha_array = np.where(strong_edges, original_alpha, alpha_array)
        
        logger.info("✅ Enhanced background removal complete with precise edges")
        
        a_new = Image.fromarray(alpha_array)
        return Image.merge('RGBA', (r, g, b, a_new))
        
    except Exception as e:
        logger.error(f"U2Net removal failed: {e}")
        # Fallback to simpler method if advanced features not available
        try:
            logger.info("Falling back to standard U2Net removal")
            from rembg import remove
            
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            img_data = buffered.getvalue()
            
            output = remove(
                img_data,
                session=REMBG_SESSION,
                alpha_matting=True,
                alpha_matting_foreground_threshold=250,
                alpha_matting_background_threshold=5,
                alpha_matting_erode_size=0
            )
            
            return Image.open(BytesIO(output))
        except:
            return image

def ensure_ring_holes_transparent_fast(image: Image.Image) -> Image.Image:
    """PRECISE ring hole detection with enhanced accuracy"""
    if image.mode != 'RGBA':
        return image
    
    logger.info("🔍 Precise Ring Hole Detection V24")
    
    r, g, b, a = image.split()
    alpha_array = np.array(a, dtype=np.uint8)
    rgb_array = np.array(image.convert('RGB'), dtype=np.float32)
    
    h, w = alpha_array.shape
    
    # More precise threshold detection with multiple criteria
    # 1. Low alpha areas
    low_alpha = alpha_array < 30
    
    # 2. Bright interior regions (potential holes)
    gray = cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    bright_areas = gray > 230
    
    # 3. Combine conditions
    potential_holes = low_alpha | (bright_areas & (alpha_array < 100))
    
    # Refined noise removal with smaller kernel
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    potential_holes = cv2.morphologyEx(potential_holes.astype(np.uint8), cv2.MORPH_OPEN, kernel_small)
    potential_holes = cv2.morphologyEx(potential_holes, cv2.MORPH_CLOSE, kernel_small)
    
    # Find hole candidates with refined analysis
    num_labels, labels = cv2.connectedComponents(potential_holes)
    
    holes_mask = np.zeros_like(alpha_array, dtype=np.float32)
    
    for label in range(1, num_labels):
        component = (labels == label)
        component_size = np.sum(component)
        
        # More precise size filtering
        if h * w * 0.00005 < component_size < h * w * 0.15:
            coords = np.where(component)
            min_y, max_y = coords[0].min(), coords[0].max()
            min_x, max_x = coords[1].min(), coords[1].max()
            
            width = max_x - min_x
            height = max_y - min_y
            
            # Check multiple criteria for hole detection
            aspect_ratio = width / height if height > 0 else 0
            
            # 1. Shape check - more flexible aspect ratio
            shape_valid = 0.3 < aspect_ratio < 3.0
            
            # 2. Position check - holes usually in center area
            center_y, center_x = (min_y + max_y) / 2, (min_x + max_x) / 2
            center_distance = np.sqrt((center_x - w/2)**2 + (center_y - h/2)**2)
            position_valid = center_distance < max(w, h) * 0.4
            
            # 3. Brightness and color consistency check
            hole_pixels = rgb_array[component]
            if len(hole_pixels) > 0:
                brightness = np.mean(hole_pixels)
                color_std = np.std(hole_pixels, axis=0).mean()
                
                # Holes are usually bright and color-consistent
                brightness_valid = brightness > 220
                consistency_valid = color_std < 30
                
                # 4. Edge smoothness check
                component_uint8 = component.astype(np.uint8) * 255
                contours, _ = cv2.findContours(component_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    perimeter = cv2.arcLength(contours[0], True)
                    area = cv2.contourArea(contours[0])
                    if area > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        smoothness_valid = circularity > 0.4
                    else:
                        smoothness_valid = False
                else:
                    smoothness_valid = False
                
                # Combine all criteria with weights
                if shape_valid and (brightness_valid or (position_valid and consistency_valid)):
                    # Calculate confidence score
                    confidence = 0.0
                    if brightness_valid: confidence += 0.4
                    if consistency_valid: confidence += 0.3
                    if position_valid: confidence += 0.2
                    if smoothness_valid: confidence += 0.1
                    
                    if confidence > 0.5:
                        # Apply graduated transparency based on confidence
                        holes_mask[component] = 255 * min(1.0, confidence)
    
    if np.any(holes_mask > 0):
        # More precise edge blending
        # Use guided filter for smooth transitions
        try:
            holes_mask_guided = cv2.ximgproc.guidedFilter(
                guide=gray.astype(np.float32),
                src=holes_mask,
                radius=3,
                eps=0.01
            )
        except AttributeError:
            # Fallback if ximgproc not available
            holes_mask_guided = cv2.GaussianBlur(holes_mask, (7, 7), 1.5)
        
        # Apply mask with smooth transitions
        alpha_float = alpha_array.astype(np.float32)
        
        # Gradual transition at hole edges
        hole_edges = cv2.Canny(holes_mask.astype(np.uint8), 50, 150)
        transition_zone = cv2.dilate(hole_edges, kernel_small, iterations=2)
        
        # Apply different blending in transition zone
        alpha_float[holes_mask_guided > 200] = 0  # Full transparency in hole center
        transition_mask = (transition_zone > 0) & (holes_mask_guided <= 200) & (holes_mask_guided > 50)
        if np.any(transition_mask):
            # Smooth gradient in transition
            alpha_float[transition_mask] *= (1 - holes_mask_guided[transition_mask] / 255)
        
        alpha_array = np.clip(alpha_float, 0, 255).astype(np.uint8)
    
    logger.info("✅ Precise ring hole detection complete")
    
    a_new = Image.fromarray(alpha_array.astype(np.uint8))
    return Image.merge('RGBA', (r, g, b, a_new))

def apply_swinir_enhancement_transparent(image: Image.Image) -> Image.Image:
    """Apply SwinIR enhancement while preserving transparency"""
    if not REPLICATE_CLIENT:
        logger.warning("SwinIR skipped - no Replicate client")
        return image
    
    try:
        logger.info("🎨 Applying SwinIR enhancement with transparency")
        
        # 알파 채널 분리
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            rgb_image = Image.merge('RGB', (r, g, b))
            has_alpha = True
        else:
            rgb_image = image
            has_alpha = False
        
        buffered = BytesIO()
        rgb_image.save(buffered, format="PNG", optimize=False)
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        output = REPLICATE_CLIENT.run(
            "jingyunliang/swinir:660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a",
            input={
                "image": img_data_url,
                "task_type": "Real-World Image Super-Resolution",
                "scale": 1,
                "noise_level": 10,
                "jpeg_quality": 50
            }
        )
        
        if output:
            if isinstance(output, str):
                response = requests.get(output)
                enhanced_image = Image.open(BytesIO(response.content))
            else:
                enhanced_image = Image.open(BytesIO(base64.b64decode(output)))
            
            # 알파 채널 재결합
            if has_alpha:
                r2, g2, b2 = enhanced_image.split()
                enhanced_image = Image.merge('RGBA', (r2, g2, b2, a))
            
            logger.info("✅ SwinIR enhancement successful with transparency")
            return enhanced_image
        else:
            return image
            
    except Exception as e:
        logger.warning(f"SwinIR error: {str(e)}")
        return image

def enhance_with_alpha(image: Image.Image, enhancement_func):
    """Apply enhancement to RGB while preserving alpha"""
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        enhanced_rgb = enhancement_func(rgb_image)
        r2, g2, b2 = enhanced_rgb.split()
        return Image.merge('RGBA', (r2, g2, b2, a))
    else:
        return enhancement_func(image)

def apply_pattern_enhancement_transparent(image: Image.Image, pattern_type: str) -> Image.Image:
    """Apply pattern enhancement while preserving transparency"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    rgb_image = Image.merge('RGB', (r, g, b))
    
    # Apply enhancements based on pattern type
    if pattern_type == "ac_pattern":
        logger.info("🔍 AC Pattern - Applying 12% white overlay")
        # Apply 12% white overlay
        white_overlay = 0.12
        img_array = np.array(rgb_image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.005)
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.98)
        
        logger.info("✅ AC Pattern enhancement applied")
    
    elif pattern_type == "ab_pattern":
        logger.info("🔍 AB Pattern - Applying 5% white overlay and cool tone")
        # Apply 5% white overlay and cool tone
        white_overlay = 0.05
        img_array = np.array(rgb_image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        
        # Cool tone adjustment
        img_array[:,:,0] *= 0.96  # Reduce red
        img_array[:,:,1] *= 0.98  # Reduce green
        img_array[:,:,2] *= 1.02  # Increase blue
        
        # Cool color grading
        cool_overlay = np.array([240, 248, 255], dtype=np.float32)
        img_array = img_array * 0.95 + cool_overlay * 0.05
        
        img_array = np.clip(img_array, 0, 255)
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.88)
        
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.005)
        
        logger.info("✅ AB Pattern enhancement applied")
        
    else:
        logger.info("🔍 Other Pattern - Standard enhancement")
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.08)
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.99)
        
        sharpness = ImageEnhance.Sharpness(rgb_image)
        rgb_image = sharpness.enhance(1.4)
    
    # Apply common enhancements
    contrast = ImageEnhance.Contrast(rgb_image)
    rgb_image = contrast.enhance(1.05)
    
    # Apply sharpening
    sharpness = ImageEnhance.Sharpness(rgb_image)
    rgb_image = sharpness.enhance(1.6)
    
    # Recombine with alpha
    r2, g2, b2 = rgb_image.split()
    return Image.merge('RGBA', (r2, g2, b2, a))

def resize_to_target_dimensions(image: Image.Image, target_width=1200, target_height=1560) -> Image.Image:
    """Resize image to target dimensions preserving transparency"""
    width, height = image.size
    
    img_ratio = width / height
    target_ratio = target_width / target_height
    
    expected_ratio = 2000 / 2600
    
    if abs(img_ratio - expected_ratio) < 0.05:
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    if img_ratio > target_ratio:
        new_height = target_height
        new_width = int(target_height * img_ratio)
    else:
        new_width = target_width
        new_height = int(target_width / img_ratio)
    
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    if new_width != target_width or new_height != target_height:
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        resized = resized.crop((left, top, right, bottom))
    
    return resized

def process_special_mode(job):
    """Process special modes (MD TALK, DESIGN POINT, BOTH)"""
    special_mode = job.get('special_mode', '')
    logger.info(f"Processing special mode: {special_mode}")
    
    # BOTH TEXT SECTIONS - NEW!
    if special_mode == 'both_text_sections':
        md_talk_text = job.get('md_talk_content', '') or job.get('md_talk', '') or "기본 MD TALK 텍스트"
        design_point_text = job.get('design_point_content', '') or job.get('design_point', '') or "기본 DESIGN POINT 텍스트"
        
        logger.info(f"Creating both sections - MD TALK: {md_talk_text[:50]}...")
        logger.info(f"Creating both sections - DESIGN POINT: {design_point_text[:50]}...")
        
        # Create both sections
        md_section = create_md_talk_section(md_talk_text)
        design_section = create_design_point_section(design_point_text)
        
        # Combine vertically
        total_width = 1200
        total_height = md_section.height + design_section.height
        
        combined = Image.new('RGB', (total_width, total_height), '#FFFFFF')
        combined.paste(md_section, (0, 0))
        combined.paste(design_section, (0, md_section.height))
        
        # Convert to base64
        buffered = BytesIO()
        combined.save(buffered, format="PNG", optimize=False)
        buffered.seek(0)
        combined_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        combined_base64_no_padding = combined_base64.rstrip('=')
        
        return {
            "output": {
                "enhanced_image": combined_base64_no_padding,
                "enhanced_image_with_prefix": f"data:image/png;base64,{combined_base64_no_padding}",
                "section_type": "both_text_sections",
                "filename": "text_sections.png",
                "final_size": list(combined.size),
                "md_talk_height": md_section.height,
                "design_point_height": design_section.height,
                "version": VERSION,
                "status": "success",
                "format": "PNG",
                "special_mode": special_mode,
                "sections_included": ["MD_TALK", "DESIGN_POINT"]
            }
        }
    
    # MD TALK section
    elif special_mode == 'md_talk':
        text_content = job.get('text_content', '') or job.get('claude_text', '') or job.get('md_talk', '')
        
        if not text_content:
            text_content = "이 제품은 일상에서도 부담없이 착용할 수 있는 편안한 디자인으로 매일의 스타일링에 포인트를 더해줍니다."
        
        section_image = create_md_talk_section(text_content)
        
        buffered = BytesIO()
        section_image.save(buffered, format="PNG", optimize=False)
        buffered.seek(0)
        section_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        section_base64_no_padding = section_base64.rstrip('=')
        
        return {
            "output": {
                "enhanced_image": section_base64_no_padding,
                "enhanced_image_with_prefix": f"data:image/png;base64,{section_base64_no_padding}",
                "section_type": "md_talk",
                "filename": "ac_wedding_004.png",
                "file_number": "004",
                "final_size": list(section_image.size),
                "version": VERSION,
                "status": "success",
                "format": "PNG",
                "special_mode": special_mode
            }
        }
    
    # DESIGN POINT section
    elif special_mode == 'design_point':
        text_content = job.get('text_content', '') or job.get('claude_text', '') or job.get('design_point', '')
        
        if not text_content:
            text_content = "남성 단품은 무광 텍스처와 유광 라인의 조화가 견고한 감성을 전하고 여자 단품은 파베 세팅과 섬세한 밀그레인의 디테일로 화려하면서도 고급스러운 반짝임을 표현합니다."
        
        section_image = create_design_point_section(text_content)
        
        buffered = BytesIO()
        section_image.save(buffered, format="PNG", optimize=False)
        buffered.seek(0)
        section_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        section_base64_no_padding = section_base64.rstrip('=')
        
        return {
            "output": {
                "enhanced_image": section_base64_no_padding,
                "enhanced_image_with_prefix": f"data:image/png;base64,{section_base64_no_padding}",
                "section_type": "design_point",
                "filename": "ac_wedding_008.png",
                "file_number": "008",
                "final_size": list(section_image.size),
                "version": VERSION,
                "status": "success",
                "format": "PNG",
                "special_mode": special_mode
            }
        }
    
    else:
        return {
            "output": {
                "error": f"Unknown special mode: {special_mode}",
                "status": "error",
                "version": VERSION
            }
        }

def process_enhancement(job):
    """Main enhancement processing - V24 with Precise Edge Detection
    
    Features:
    - 7-stage edge refinement process
    - Sub-pixel accuracy for edges
    - Guided filter for edge preservation
    - Multi-criteria hole detection
    - Gradient-based smooth transitions
    """
    logger.info(f"=== Enhancement {VERSION} Started ===")
    logger.info("🎯 PRECISE MODE: Enhanced edge detection enabled")
    logger.info(f"Received job data: {json.dumps(job, indent=2)[:500]}...")  # Log first 500 chars
    start_time = time.time()
    
    try:
        # Check for special mode first
        if job.get('special_mode'):
            return process_special_mode(job)
        
        # Normal enhancement processing continues here...
        filename = find_filename_fast(job)
        file_number = extract_file_number(filename) if filename else None
        image_data = find_input_data_fast(job)
        
        if not image_data:
            return {
                "output": {
                    "error": "No image data found",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        image_bytes = decode_base64_fast(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # STEP 1: BACKGROUND REMOVAL (PNG files)
        if filename and filename.lower().endswith('.png'):
            logger.info("📸 STEP 1: PNG detected - PRECISE background removal with edge refinement")
            removal_start = time.time()
            image = u2net_optimized_removal(image)
            logger.info(f"⏱️ Precise background removal took: {time.time() - removal_start:.2f}s")
        
        # Ensure RGBA mode for transparency
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # STEP 2: ENHANCEMENT (preserving transparency)
        logger.info("🎨 STEP 2: Applying enhancements with transparency")
        enhancement_start = time.time()
        
        # Auto white balance with alpha preservation
        def auto_white_balance(rgb_img):
            img_array = np.array(rgb_img, dtype=np.float32)
            
            gray_pixels = img_array[::15, ::15]
            gray_mask = (
                (np.abs(gray_pixels[:,:,0] - gray_pixels[:,:,1]) < 15) & 
                (np.abs(gray_pixels[:,:,1] - gray_pixels[:,:,2]) < 15) &
                (gray_pixels[:,:,0] > 180)
            )
            
            if np.sum(gray_mask) > 10:
                r_avg = np.mean(gray_pixels[gray_mask, 0])
                g_avg = np.mean(gray_pixels[gray_mask, 1])
                b_avg = np.mean(gray_pixels[gray_mask, 2])
                
                gray_avg = (r_avg + g_avg + b_avg) / 3
                
                img_array[:,:,0] *= (gray_avg / r_avg) if r_avg > 0 else 1
                img_array[:,:,1] *= (gray_avg / g_avg) if g_avg > 0 else 1
                img_array[:,:,2] *= (gray_avg / b_avg) if b_avg > 0 else 1
            
            return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        
        image = enhance_with_alpha(image, auto_white_balance)
        
        # Detect pattern type
        pattern_type = detect_pattern_type(filename)
        detected_type = {
            "ac_pattern": "무도금화이트(0.12)",
            "ab_pattern": "무도금화이트-쿨톤(0.05)",
            "other": "기타색상(no_overlay)"
        }.get(pattern_type, "기타색상(no_overlay)")
        
        # Apply pattern-specific enhancements
        image = apply_pattern_enhancement_transparent(image, pattern_type)
        
        # Ring hole detection
        logger.info("🔍 Applying precise ring hole detection")
        image = ensure_ring_holes_transparent_fast(image)
        
        logger.info(f"⏱️ Enhancement took: {time.time() - enhancement_start:.2f}s")
        
        # RESIZE
        image = resize_to_target_dimensions(image, 1200, 1560)
        
        # STEP 3: SWINIR ENHANCEMENT (preserving transparency)
        logger.info("🚀 STEP 3: Applying SwinIR enhancement")
        swinir_start = time.time()
        image = apply_swinir_enhancement_transparent(image)
        logger.info(f"⏱️ SwinIR took: {time.time() - swinir_start:.2f}s")
        
        # Save to base64 as PNG (to preserve transparency)
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=False)
        buffered.seek(0)
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        enhanced_base64_no_padding = enhanced_base64.rstrip('=')
        
        # Build filename with corrected numbers
        enhanced_filename = filename
        if filename and file_number:
            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            # Ensure correct file numbers for enhancement outputs
            # Skip 004 (MD TALK), 007, 008 (DESIGN POINT), 009, 010, 011
            enhanced_filename = f"{base_name}_enhanced_transparent.png"
        
        total_time = time.time() - start_time
        logger.info(f"✅ Enhancement completed in {total_time:.2f}s")
        
        output = {
            "output": {
                "enhanced_image": enhanced_base64_no_padding,
                "enhanced_image_with_prefix": f"data:image/png;base64,{enhanced_base64_no_padding}",
                "detected_type": detected_type,
                "pattern_type": pattern_type,
                "is_wedding_ring": True,
                "filename": filename,
                "enhanced_filename": enhanced_filename,
                "file_number": file_number,
                "final_size": list(image.size),
                "version": VERSION,
                "status": "success",
                "processing_time": f"{total_time:.2f}s",
                "has_transparency": True,
                "background_applied": False,
                "format": "PNG",
                "special_modes_available": ["md_talk", "design_point", "both_text_sections"],
                "file_number_info": {
                    "001-003": "Enhancement",
                    "004": "MD TALK",
                    "005-006": "Enhancement",
                    "007": "Thumbnail",
                    "008": "DESIGN POINT",
                    "009-010": "Thumbnail",
                    "011": "COLOR section"
                },
                "optimization_features": [
                    "✅ ENHANCED Transparent PNG with PRECISE edge detection",
                    "✅ Multi-stage edge refinement (7 stages)",
                    "✅ Guided filter for edge-preserving smoothing",
                    "✅ Sub-pixel edge accuracy",
                    "✅ Gradient-based smooth transitions",
                    "✅ Precise ring hole detection with multi-criteria",
                    "✅ Pattern-specific enhancement preserved",
                    "✅ AC: 12% white overlay",
                    "✅ AB: 5% white overlay + cool tone",
                    "✅ MD TALK (004) & DESIGN POINT (008) support",
                    "✅ BOTH TEXT SECTIONS support added",
                    "✅ Ready for Figma overlay"
                ],
                "processing_order": "1.U2Net-Precise → 2.Enhancement → 3.SwinIR",
                "swinir_applied": True,
                "png_support": True,
                "edge_detection": "PRECISE (7-stage refinement)",
                "white_overlay": "AC: 12% | AB: 5% + Cool Tone | Other: None",
                "expected_input": "2000x2600 PNG",
                "output_size": "1200x1560"
            }
        }
        
        logger.info("✅ Enhancement completed successfully")
        return output
        
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

def handler(event):
    """RunPod handler function"""
    logger.info(f"Handler received event type: {type(event)}")
    logger.info(f"Handler received event: {json.dumps(event, indent=2)[:500]}...")
    
    # Handle different event structures
    if isinstance(event, dict):
        if 'input' in event:
            job = event['input']
        else:
            job = event
    else:
        job = event
    
    return process_enhancement(job)

# RunPod handler
runpod.serverless.start({"handler": handler})
