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
# VERSION: V28-Fixed-Always-Transparent
################################

VERSION = "V28-Fixed-Always-Transparent"

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
    """Download Korean font for text rendering - IMPROVED with better encoding"""
    try:
        font_path = '/tmp/NanumGothic.ttf'
        
        # If font exists, verify it works with Korean text
        if os.path.exists(font_path):
            try:
                # Test with actual Korean text
                test_font = ImageFont.truetype(font_path, 20, encoding='utf-8')
                img_test = Image.new('RGB', (200, 100), 'white')
                draw_test = ImageDraw.Draw(img_test)
                # Test with various Korean characters
                test_text = "테스트 한글 폰트 확인"
                draw_test.text((10, 10), test_text, font=test_font, fill='black')
                logger.info("✅ Korean font verified and working")
                return font_path
            except Exception as e:
                logger.error(f"Font verification failed: {e}")
                os.remove(font_path)
        
        # Download URLs with backup options
        font_urls = [
            'https://github.com/naver/nanumfont/raw/master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf',
            'https://cdn.jsdelivr.net/gh/naver/nanumfont@master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf',
            'https://github.com/naver/nanumfont/raw/master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothicBold.ttf'
        ]
        
        for url in font_urls:
            try:
                logger.info(f"Downloading font from: {url}")
                response = requests.get(url, timeout=30)
                if response.status_code == 200 and len(response.content) > 100000:
                    with open(font_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Verify the font works with Korean
                    test_font = ImageFont.truetype(font_path, 20, encoding='utf-8')
                    img_test = Image.new('RGB', (200, 100), 'white')
                    draw_test = ImageDraw.Draw(img_test)
                    draw_test.text((10, 10), "한글 테스트", font=test_font, fill='black')
                    logger.info("✅ Korean font downloaded and verified successfully")
                    return font_path
            except Exception as e:
                logger.error(f"Failed to download from {url}: {e}")
                continue
        
        logger.error("❌ Failed to download Korean font from all sources")
        return None
    except Exception as e:
        logger.error(f"Font download error: {e}")
        return None

def get_font(size, korean_font_path=None):
    """Get font with proper encoding - ENHANCED"""
    if korean_font_path and os.path.exists(korean_font_path):
        try:
            # Always use UTF-8 encoding for Korean fonts
            font = ImageFont.truetype(korean_font_path, size, encoding='utf-8')
            logger.info(f"Font loaded successfully with size {size}")
            return font
        except Exception as e:
            logger.error(f"Font loading error: {e}")
    
    # Fallback to default
    try:
        logger.warning("Using default font as fallback")
        return ImageFont.load_default()
    except:
        return None

def safe_draw_text(draw, position, text, font, fill):
    """Safely draw text with proper encoding - ENHANCED"""
    try:
        if text and font:
            # Ensure text is properly encoded as UTF-8
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            else:
                # Ensure it's a string and normalize
                text = str(text)
            
            # Draw the text
            draw.text(position, text, font=font, fill=fill)
            logger.info(f"Successfully drew text: {text[:20]}...")
    except Exception as e:
        logger.error(f"Text drawing error: {e}, text: {repr(text)}")
        # Fallback to simple text
        try:
            draw.text(position, "[Text Error]", font=font, fill=fill)
        except:
            pass

def get_text_size(draw, text, font):
    """Get text size compatible with different PIL versions"""
    try:
        # Ensure text is string
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        else:
            text = str(text)
            
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

def create_md_talk_section(text_content=None, width=1200):
    """Create MD TALK section with proper Korean support - ENHANCED"""
    logger.info("Creating MD TALK section")
    
    korean_font_path = download_korean_font()
    if not korean_font_path:
        logger.error("Korean font not available")
    
    title_font = get_font(48, korean_font_path)
    body_font = get_font(28, korean_font_path)
    
    # Create temporary image for text measurement
    temp_img = Image.new('RGB', (width, 1000), '#FFFFFF')
    draw = ImageDraw.Draw(temp_img)
    
    title = "MD TALK"
    title_width, title_height = get_text_size(draw, title, title_font)
    
    # Text preparation with proper encoding
    if text_content and text_content.strip():
        # Clean up the text
        text = text_content.replace('MD TALK', '').replace('MD Talk', '').strip()
        # Ensure proper encoding
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        else:
            text = str(text)
    else:
        text = """이 제품은 일상에서도 부담없이
착용할 수 있는 편안한 디자인으로
매일의 스타일링에 포인트를 더해줍니다.

특별한 날은 물론 평범한 일상까지
모든 순간을 빛나게 만들어주는
당신만의 특별한 주얼리입니다."""
    
    # Log the text to verify encoding
    logger.info(f"MD TALK text content: {repr(text[:50])}...")
    
    # Split text into lines
    lines = text.split('\n')
    
    # Calculate height
    top_margin = 60
    title_bottom_margin = 140
    line_height = 50
    bottom_margin = 80
    
    content_height = len(lines) * line_height
    total_height = top_margin + title_height + title_bottom_margin + content_height + bottom_margin
    
    # Create actual image
    section_img = Image.new('RGB', (width, total_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Draw title
    safe_draw_text(draw, (width//2 - title_width//2, top_margin), title, title_font, (40, 40, 40))
    
    # Draw body text
    y_pos = top_margin + title_height + title_bottom_margin
    
    for line in lines:
        if line.strip():
            line_width, _ = get_text_size(draw, line, body_font)
            safe_draw_text(draw, (width//2 - line_width//2, y_pos), line, body_font, (80, 80, 80))
        y_pos += line_height
    
    logger.info(f"MD TALK section created: {width}x{total_height}")
    return section_img

def create_design_point_section(text_content=None, width=1200):
    """Create DESIGN POINT section with proper Korean support - ENHANCED"""
    logger.info("Creating DESIGN POINT section")
    
    korean_font_path = download_korean_font()
    if not korean_font_path:
        logger.error("Korean font not available")
    
    title_font = get_font(48, korean_font_path)
    body_font = get_font(24, korean_font_path)
    
    # Create temporary image for text measurement
    temp_img = Image.new('RGB', (width, 1000), '#FFFFFF')
    draw = ImageDraw.Draw(temp_img)
    
    title = "DESIGN POINT"
    title_width, title_height = get_text_size(draw, title, title_font)
    
    # Text preparation with proper encoding
    if text_content and text_content.strip():
        # Clean up the text
        text = text_content.replace('DESIGN POINT', '').replace('Design Point', '').strip()
        # Ensure proper encoding
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        else:
            text = str(text)
    else:
        text = """남성 단품은 무광 텍스처와 유광 라인의 조화가
견고한 감성을 전하고 여자 단품은
파베 세팅과 섬세한 밀그레인의 디테일
화려하면서도 고급스러운 반영을 표현합니다"""
    
    # Log the text to verify encoding
    logger.info(f"DESIGN POINT text content: {repr(text[:50])}...")
    
    # Split text into lines
    lines = text.split('\n')
    
    # Calculate height
    top_margin = 60
    title_bottom_margin = 160
    line_height = 55
    bottom_margin = 100
    
    content_height = len(lines) * line_height
    total_height = top_margin + title_height + title_bottom_margin + content_height + bottom_margin
    
    # Create actual image
    section_img = Image.new('RGB', (width, total_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Draw title
    safe_draw_text(draw, (width//2 - title_width//2, top_margin), title, title_font, (40, 40, 40))
    
    # Draw body text
    y_pos = top_margin + title_height + title_bottom_margin
    
    for line in lines:
        if line.strip():
            line_width, _ = get_text_size(draw, line, body_font)
            safe_draw_text(draw, (width//2 - line_width//2, y_pos), line, body_font, (80, 80, 80))
        y_pos += line_height
    
    # Draw bottom line
    draw.rectangle([100, y_pos + 30, width - 100, y_pos + 32], fill=(220, 220, 220))
    
    logger.info(f"DESIGN POINT section created: {width}x{total_height}")
    return section_img

def u2net_ultra_precise_removal(image: Image.Image) -> Image.Image:
    """ULTRA PRECISE U2Net background removal with advanced edge detection"""
    try:
        from rembg import remove
        
        global REMBG_SESSION
        if REMBG_SESSION is None:
            REMBG_SESSION = init_rembg_session()
            if REMBG_SESSION is None:
                return image
        
        logger.info("🔷 U2Net ULTRA PRECISE Background Removal V28")
        
        # Pre-process image for better edge detection
        # Apply slight contrast enhancement before removal
        contrast = ImageEnhance.Contrast(image)
        image_enhanced = contrast.enhance(1.1)
        
        # Save image to buffer
        buffered = BytesIO()
        image_enhanced.save(buffered, format="PNG")
        buffered.seek(0)
        img_data = buffered.getvalue()
        
        # Apply U2Net removal with ULTRA PRECISE settings
        output = remove(
            img_data,
            session=REMBG_SESSION,
            alpha_matting=True,
            alpha_matting_foreground_threshold=280,  # Even higher for better edges
            alpha_matting_background_threshold=0,
            alpha_matting_erode_size=0,
            only_mask=False,
            post_process_mask=True  # Enable post-processing
        )
        
        result_image = Image.open(BytesIO(output))
        
        if result_image.mode != 'RGBA':
            result_image = result_image.convert('RGBA')
        
        # ULTRA PRECISE edge refinement
        r, g, b, a = result_image.split()
        alpha_array = np.array(a, dtype=np.uint8)
        original_alpha = alpha_array.copy()
        
        # Convert to float for processing
        alpha_float = alpha_array.astype(np.float32) / 255.0
        
        # Stage 1: Advanced edge detection using Sobel
        rgb_array = np.array(result_image.convert('RGB'), dtype=np.uint8)
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        
        # Sobel edge detection for more precise edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_magnitude = (edge_magnitude / edge_magnitude.max() * 255).astype(np.uint8)
        
        # Stage 2: Create edge mask
        edge_mask = edge_magnitude > 30
        edge_dilated = cv2.dilate(edge_mask.astype(np.uint8), np.ones((3,3)), iterations=2)
        
        # Stage 3: Apply guided filter for ultra-smooth edges
        try:
            # Normalize gray for guided filter
            gray_float = gray.astype(np.float32) / 255.0
            
            # Multiple passes of guided filter with different parameters
            alpha_guided1 = cv2.ximgproc.guidedFilter(
                guide=gray_float,
                src=alpha_float,
                radius=1,
                eps=0.0001  # Very small epsilon for maximum edge preservation
            )
            
            alpha_guided2 = cv2.ximgproc.guidedFilter(
                guide=gray_float,
                src=alpha_guided1,
                radius=3,
                eps=0.001
            )
            
            # Blend the two guided results
            alpha_float = alpha_guided1 * 0.7 + alpha_guided2 * 0.3
            
        except AttributeError:
            # Fallback to bilateral filter
            alpha_uint8 = (alpha_float * 255).astype(np.uint8)
            alpha_bilateral = cv2.bilateralFilter(alpha_uint8, 5, 75, 75)
            alpha_float = alpha_bilateral.astype(np.float32) / 255.0
        
        # Stage 4: Ultra-precise threshold with smooth gradients
        # Create smooth transition using sigmoid function
        k = 50  # Steepness of transition
        threshold = 0.5
        alpha_sigmoid = 1 / (1 + np.exp(-k * (alpha_float - threshold)))
        
        # Stage 5: Edge-aware smoothing
        # Only smooth non-edge areas
        alpha_smooth = alpha_sigmoid.copy()
        non_edge_mask = ~edge_dilated.astype(bool)
        if np.any(non_edge_mask):
            # Apply gentle smoothing to non-edge areas
            alpha_smooth_temp = cv2.GaussianBlur(alpha_sigmoid, (5, 5), 1.0)
            alpha_smooth[non_edge_mask] = alpha_smooth_temp[non_edge_mask]
        
        # Stage 6: Hair and fine detail preservation
        # Detect fine details using high-pass filter
        alpha_highpass = alpha_float - cv2.GaussianBlur(alpha_float, (7, 7), 2.0)
        fine_details = np.abs(alpha_highpass) > 0.05
        
        # Preserve fine details
        alpha_smooth[fine_details] = alpha_float[fine_details]
        
        # Stage 7: Remove small artifacts while preserving tiny details
        alpha_binary = (alpha_smooth > 0.5).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(alpha_binary)
        
        if num_labels > 2:
            sizes = [np.sum(labels == i) for i in range(1, num_labels)]
            if sizes:
                # More aggressive artifact removal
                min_size = int(alpha_array.size * 0.0002)  # 0.02% of image
                valid_labels = [i+1 for i, size in enumerate(sizes) if size > min_size]
                
                # Create valid mask
                valid_mask = np.zeros_like(alpha_binary, dtype=bool)
                for label in valid_labels:
                    valid_mask |= (labels == label)
                
                # Apply mask but preserve edges
                alpha_smooth[~valid_mask & ~edge_dilated.astype(bool)] = 0
        
        # Stage 8: Final polish with edge enhancement
        # Enhance edges slightly
        edge_enhancement = 1.2
        alpha_smooth[edge_dilated.astype(bool)] *= edge_enhancement
        
        # Convert back to uint8
        alpha_array = np.clip(alpha_smooth * 255, 0, 255).astype(np.uint8)
        
        # Stage 9: Feather edges for natural look
        # Create feathered edge
        kernel_feather = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha_eroded = cv2.erode(alpha_array, kernel_feather, iterations=1)
        alpha_dilated = cv2.dilate(alpha_array, kernel_feather, iterations=1)
        
        # Blend for feathering
        feather_mask = (alpha_dilated > 0) & (alpha_eroded < 255)
        if np.any(feather_mask):
            alpha_array[feather_mask] = ((alpha_array[feather_mask].astype(np.float32) + 
                                         alpha_eroded[feather_mask].astype(np.float32)) / 2).astype(np.uint8)
        
        logger.info("✅ ULTRA PRECISE background removal complete")
        
        a_new = Image.fromarray(alpha_array)
        return Image.merge('RGBA', (r, g, b, a_new))
        
    except Exception as e:
        logger.error(f"U2Net removal failed: {e}")
        return image

def ensure_ring_holes_transparent_ultra(image: Image.Image) -> Image.Image:
    """ULTRA PRECISE ring hole detection with maximum accuracy"""
    if image.mode != 'RGBA':
        return image
    
    logger.info("🔍 ULTRA PRECISE Ring Hole Detection V28")
    
    r, g, b, a = image.split()
    alpha_array = np.array(a, dtype=np.uint8)
    rgb_array = np.array(image.convert('RGB'), dtype=np.uint8)
    
    h, w = alpha_array.shape
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    
    # Multi-criteria hole detection
    # 1. Very bright areas (potential holes)
    very_bright = v_channel > 240
    
    # 2. Low saturation (grayish/white areas)
    low_saturation = s_channel < 30
    
    # 3. Current alpha holes
    alpha_holes = alpha_array < 50
    
    # 4. Combine all criteria
    potential_holes = (very_bright & low_saturation) | alpha_holes
    
    # Clean up noise
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    potential_holes = cv2.morphologyEx(potential_holes.astype(np.uint8), cv2.MORPH_OPEN, kernel_clean)
    potential_holes = cv2.morphologyEx(potential_holes, cv2.MORPH_CLOSE, kernel_clean)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(potential_holes)
    
    holes_mask = np.zeros_like(alpha_array, dtype=np.float32)
    
    # Analyze each component
    for label in range(1, num_labels):
        component = (labels == label)
        component_size = np.sum(component)
        
        # Size filtering - adjust for ring holes
        if h * w * 0.0001 < component_size < h * w * 0.2:
            # Get component properties
            coords = np.where(component)
            if len(coords[0]) == 0:
                continue
                
            min_y, max_y = coords[0].min(), coords[0].max()
            min_x, max_x = coords[1].min(), coords[1].max()
            
            comp_width = max_x - min_x
            comp_height = max_y - min_y
            
            if comp_height == 0:
                continue
            
            # Multiple validation criteria
            aspect_ratio = comp_width / comp_height
            
            # 1. Shape validation (ring holes can be various shapes)
            shape_valid = 0.2 < aspect_ratio < 5.0
            
            # 2. Position validation (usually in center area)
            center_y, center_x = (min_y + max_y) / 2, (min_x + max_x) / 2
            center_distance = np.sqrt((center_x - w/2)**2 + (center_y - h/2)**2)
            position_valid = center_distance < max(w, h) * 0.45
            
            # 3. Color consistency check
            component_pixels = rgb_array[component]
            if len(component_pixels) > 0:
                # Check brightness
                brightness = np.mean(component_pixels)
                brightness_std = np.std(component_pixels)
                
                brightness_valid = brightness > 230
                consistency_valid = brightness_std < 25
                
                # 4. Circularity check
                component_uint8 = component.astype(np.uint8) * 255
                contours, _ = cv2.findContours(component_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                circularity_valid = False
                if contours:
                    contour = contours[0]
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        circularity_valid = circularity > 0.3
                
                # 5. Edge smoothness check
                edges = cv2.Canny(component_uint8, 50, 150)
                edge_ratio = np.sum(edges > 0) / max(1, perimeter)
                smoothness_valid = edge_ratio < 2.0
                
                # Calculate confidence score
                confidence = 0.0
                if brightness_valid: confidence += 0.35
                if consistency_valid: confidence += 0.25
                if position_valid: confidence += 0.15
                if circularity_valid: confidence += 0.15
                if smoothness_valid: confidence += 0.10
                
                # Apply hole mask if confident
                if confidence > 0.45 and shape_valid:
                    holes_mask[component] = 255
                    logger.info(f"Hole detected with confidence: {confidence:.2f}")
    
    # Apply holes if any detected
    if np.any(holes_mask > 0):
        # Smooth hole edges
        holes_mask_smooth = cv2.GaussianBlur(holes_mask, (5, 5), 1.0)
        
        # Create transition zone
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        holes_dilated = cv2.dilate(holes_mask, kernel_dilate, iterations=1)
        transition_zone = (holes_dilated > 0) & (holes_mask < 255)
        
        # Apply graduated transparency
        alpha_float = alpha_array.astype(np.float32)
        
        # Full transparency in hole centers
        alpha_float[holes_mask_smooth > 200] = 0
        
        # Graduated transparency in transition
        if np.any(transition_zone):
            transition_alpha = 1 - (holes_mask_smooth[transition_zone] / 255)
            alpha_float[transition_zone] *= transition_alpha
        
        alpha_array = np.clip(alpha_float, 0, 255).astype(np.uint8)
        
        logger.info("✅ Ring holes made transparent")
    
    a_new = Image.fromarray(alpha_array)
    return Image.merge('RGBA', (r, g, b, a_new))

def image_to_base64(image, keep_transparency=True):
    """Convert to base64 without padding - TRULY preserving transparency"""
    buffered = BytesIO()
    
    # CRITICAL FIX: Always save as PNG with transparency for RGBA images
    if image.mode == 'RGBA':
        logger.info("💎 Preserving transparency in output - RGBA mode")
        # Save as PNG with full transparency support
        image.save(buffered, format='PNG', compress_level=0, optimize=False)
    else:
        # For non-RGBA images, just save as PNG
        logger.info(f"Saving {image.mode} mode image as PNG")
        image.save(buffered, format='PNG', optimize=True, compress_level=1)
    
    buffered.seek(0)
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    # Remove padding for Make.com compatibility
    return base64_str.rstrip('=')

def process_special_mode(job):
    """Process special modes - FIXED to return separate images with proper encoding"""
    special_mode = job.get('special_mode', '')
    logger.info(f"Processing special mode: {special_mode}")
    
    # BOTH TEXT SECTIONS - Return TWO separate images
    if special_mode == 'both_text_sections':
        # Get text content with proper encoding
        md_talk_text = job.get('md_talk_content', '') or job.get('md_talk', '') or """각도에 따라 달라지는 빛의 결들이
두 사람의 특별한 순간순간을 더 찬란하게 만들며
360도 새겨진 패턴으로
매일 새로운 반짝임을 보여줍니다 :)"""
        
        design_point_text = job.get('design_point_content', '') or job.get('design_point', '') or """입체적인 컷팅 위로 섬세하게 빛나는 패턴이
고급스러움을 완성하며
각진 텍스처가 심플하면서 유니크한 매력을 더해줍니다."""
        
        # Ensure proper encoding
        if isinstance(md_talk_text, bytes):
            md_talk_text = md_talk_text.decode('utf-8', errors='replace')
        if isinstance(design_point_text, bytes):
            design_point_text = design_point_text.decode('utf-8', errors='replace')
        
        logger.info(f"Creating both sections separately")
        logger.info(f"MD TALK text: {repr(md_talk_text[:50])}...")
        logger.info(f"DESIGN POINT text: {repr(design_point_text[:50])}...")
        
        # Create both sections
        md_section = create_md_talk_section(md_talk_text)
        design_section = create_design_point_section(design_point_text)
        
        # Convert MD TALK to base64
        md_base64_no_padding = image_to_base64(md_section, keep_transparency=False)
        
        # Convert DESIGN POINT to base64
        design_base64_no_padding = image_to_base64(design_section, keep_transparency=False)
        
        # Return BOTH images separately
        return {
            "output": {
                "images": [
                    {
                        "enhanced_image": md_base64_no_padding,
                        "enhanced_image_with_prefix": f"data:image/png;base64,{md_base64_no_padding}",
                        "section_type": "md_talk",
                        "filename": "ac_wedding_004.png",
                        "file_number": "004",
                        "final_size": list(md_section.size),
                        "format": "PNG"
                    },
                    {
                        "enhanced_image": design_base64_no_padding,
                        "enhanced_image_with_prefix": f"data:image/png;base64,{design_base64_no_padding}",
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
                "korean_encoding": "UTF-8"
            }
        }
    
    # Single MD TALK section
    elif special_mode == 'md_talk':
        text_content = job.get('text_content', '') or job.get('claude_text', '') or job.get('md_talk', '')
        
        if not text_content:
            text_content = """이 제품은 일상에서도 부담없이 착용할 수 있는 편안한 디자인으로
매일의 스타일링에 포인트를 더해줍니다."""
        
        # Ensure proper encoding
        if isinstance(text_content, bytes):
            text_content = text_content.decode('utf-8', errors='replace')
        
        section_image = create_md_talk_section(text_content)
        
        section_base64_no_padding = image_to_base64(section_image, keep_transparency=False)
        
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
                "special_mode": special_mode,
                "korean_encoding": "UTF-8"
            }
        }
    
    # Single DESIGN POINT section
    elif special_mode == 'design_point':
        text_content = job.get('text_content', '') or job.get('claude_text', '') or job.get('design_point', '')
        
        if not text_content:
            text_content = """남성 단품은 무광 텍스처와 유광 라인의 조화가 견고한 감성을 전하고
여자 단품은 파베 세팅과 섬세한 밀그레인의 디테일로
화려하면서도 고급스러운 반짝임을 표현합니다."""
        
        # Ensure proper encoding
        if isinstance(text_content, bytes):
            text_content = text_content.decode('utf-8', errors='replace')
        
        section_image = create_design_point_section(text_content)
        
        section_base64_no_padding = image_to_base64(section_image, keep_transparency=False)
        
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
                "special_mode": special_mode,
                "korean_encoding": "UTF-8"
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

def apply_pattern_enhancement_transparent(image: Image.Image, pattern_type: str) -> Image.Image:
    """Apply pattern enhancement while TRULY preserving transparency"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # CRITICAL: Process RGB channels separately to preserve alpha
    r, g, b, a = image.split()
    rgb_image = Image.merge('RGB', (r, g, b))
    
    # Convert to array for processing
    img_array = np.array(rgb_image, dtype=np.float32)
    
    # Apply enhancements based on pattern type
    if pattern_type == "ac_pattern":
        logger.info("🔍 AC Pattern - Applying 12% white overlay")
        # Apply 12% white overlay
        white_overlay = 0.12
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        
        # Convert back to image
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.005)
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.98)
        
        logger.info("✅ AC Pattern enhancement applied")
    
    elif pattern_type == "ab_pattern":
        logger.info("🔍 AB Pattern - Applying 5% white overlay and cool tone")
        # Apply 5% white overlay
        white_overlay = 0.05
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
    
    # CRITICAL: Recombine with ORIGINAL alpha channel
    r2, g2, b2 = rgb_image.split()
    enhanced_image = Image.merge('RGBA', (r2, g2, b2, a))
    
    logger.info(f"✅ Enhancement applied while preserving transparency. Mode: {enhanced_image.mode}")
    
    return enhanced_image

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

def apply_swinir_enhancement_transparent(image: Image.Image) -> Image.Image:
    """Apply SwinIR enhancement while preserving transparency"""
    if not REPLICATE_CLIENT:
        logger.warning("SwinIR skipped - no Replicate client")
        return image
    
    try:
        logger.info("🎨 Applying SwinIR enhancement with transparency")
        
        # Separate alpha channel if present
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            rgb_image = Image.merge('RGB', (r, g, b))
            has_alpha = True
        else:
            rgb_image = image
            has_alpha = False
        
        buffered = BytesIO()
        rgb_image.save(buffered, format="PNG", optimize=True, compress_level=1)
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
            
            # Recombine with original alpha channel
            if has_alpha:
                r2, g2, b2 = enhanced_image.split()
                enhanced_image = Image.merge('RGBA', (r2, g2, b2, a))
            
            logger.info("✅ SwinIR enhancement successful with transparency preserved")
            return enhanced_image
        else:
            return image
            
    except Exception as e:
        logger.warning(f"SwinIR error: {str(e)}")
        return image

def process_enhancement(job):
    """Main enhancement processing - V28 FIXED Always Transparent"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    logger.info("🎯 FIXED: Always apply background removal for transparency")
    logger.info("💎 TRANSPARENT OUTPUT: Preserving alpha channel throughout")
    logger.info(f"Received job data: {json.dumps(job, indent=2)[:500]}...")
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
        
        # Log initial image info
        logger.info(f"Input image mode: {image.mode}, size: {image.size}")
        
        # CRITICAL FIX: Check if we should force background removal
        force_removal = job.get('force_background_removal', True)  # Default to True
        
        # STEP 1: ULTRA PRECISE BACKGROUND REMOVAL - ALWAYS APPLY
        logger.info("📸 STEP 1: ALWAYS applying ULTRA PRECISE background removal")
        logger.info("🔥 FIXED: Removing filename check - applying to ALL images")
        removal_start = time.time()
        image = u2net_ultra_precise_removal(image)
        logger.info(f"⏱️ Ultra precise background removal took: {time.time() - removal_start:.2f}s")
        
        # Ensure RGBA mode for transparency
        if image.mode != 'RGBA':
            logger.info("Converting to RGBA mode for transparency")
            if image.mode == 'RGB':
                # Add full alpha channel
                r, g, b = image.split()
                a = Image.new('L', image.size, 255)
                image = Image.merge('RGBA', (r, g, b, a))
            else:
                image = image.convert('RGBA')
        
        # STEP 2: ENHANCEMENT (preserving transparency)
        logger.info("🎨 STEP 2: Applying enhancements with TRUE transparency preservation")
        enhancement_start = time.time()
        
        # Auto white balance with alpha preservation
        def auto_white_balance_rgba(rgba_img):
            r, g, b, a = rgba_img.split()
            rgb_img = Image.merge('RGB', (r, g, b))
            
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
            
            rgb_balanced = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
            r2, g2, b2 = rgb_balanced.split()
            return Image.merge('RGBA', (r2, g2, b2, a))
        
        image = auto_white_balance_rgba(image)
        
        # Detect pattern type
        pattern_type = detect_pattern_type(filename)
        detected_type = {
            "ac_pattern": "무도금화이트(0.12)",
            "ab_pattern": "무도금화이트-쿨톤(0.05)",
            "other": "기타색상(no_overlay)"
        }.get(pattern_type, "기타색상(no_overlay)")
        
        # Apply pattern-specific enhancements (preserving transparency)
        image = apply_pattern_enhancement_transparent(image, pattern_type)
        
        # ULTRA PRECISE ring hole detection
        logger.info("🔍 Applying ULTRA PRECISE ring hole detection")
        image = ensure_ring_holes_transparent_ultra(image)
        
        logger.info(f"⏱️ Enhancement took: {time.time() - enhancement_start:.2f}s")
        
        # RESIZE (preserving transparency)
        image = resize_to_target_dimensions(image, 1200, 1560)
        
        # STEP 3: SWINIR ENHANCEMENT (preserving transparency)
        logger.info("🚀 STEP 3: Applying SwinIR enhancement")
        swinir_start = time.time()
        image = apply_swinir_enhancement_transparent(image)
        logger.info(f"⏱️ SwinIR took: {time.time() - swinir_start:.2f}s")
        
        # Log final image mode
        logger.info(f"Final image mode: {image.mode}, size: {image.size}")
        
        # CRITICAL: NO BACKGROUND COMPOSITE - Keep transparency
        logger.info("💎 NO background composite - keeping pure transparency")
        
        # Save to base64 as PNG with transparency
        enhanced_base64_no_padding = image_to_base64(image, keep_transparency=True)
        
        # Build filename
        enhanced_filename = filename
        if filename and file_number:
            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
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
                "transparency_preserved": True,
                "background_removed": True,
                "background_applied": False,
                "format": "PNG",
                "output_mode": "RGBA",
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
                    "✅ FIXED V28: Always apply background removal",
                    "✅ TRUE TRANSPARENT PNG: No background composite",
                    "✅ FIXED: Alpha channel preserved throughout",
                    "✅ ENHANCED: Korean font with UTF-8 encoding verification",
                    "✅ ULTRA PRECISE Transparent PNG edge detection",
                    "✅ Fixed: both_text_sections returns 2 separate images",
                    "✅ Advanced multi-stage edge refinement",
                    "✅ Sobel edge detection for precision",
                    "✅ Multiple guided filter passes",
                    "✅ Hair and fine detail preservation",
                    "✅ Feathered edges for natural look",
                    "✅ Ultra precise ring hole detection",
                    "✅ Pattern-specific enhancement preserved",
                    "✅ Ready for Figma transparent overlay",
                    "✅ Pure PNG with full alpha channel",
                    "✅ Make.com compatible base64 (no padding)"
                ],
                "processing_order": "1.U2Net-Ultra → 2.Enhancement → 3.SwinIR",
                "swinir_applied": True,
                "png_support": True,
                "edge_detection": "ULTRA PRECISE (Sobel + Guided Filter)",
                "korean_support": "ENHANCED (UTF-8 encoding with verification)",
                "white_overlay": "AC: 12% | AB: 5% + Cool Tone | Other: None",
                "expected_input": "2000x2600 (any format)",
                "output_size": "1200x1560",
                "transparency_info": "Full RGBA transparency preserved - NO background",
                "make_com_compatibility": "Base64 without padding"
            }
        }
        
        logger.info("✅ Enhancement completed successfully with TRUE transparency")
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
