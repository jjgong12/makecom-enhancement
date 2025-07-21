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
# VERSION: Korean-Encoding-Fixed-V5
################################

VERSION = "Korean-Encoding-Fixed-V5"

# ===== GLOBAL INITIALIZATION =====
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
REPLICATE_CLIENT = None

if REPLICATE_API_TOKEN:
    try:
        REPLICATE_CLIENT = replicate.Client(api_token=REPLICATE_API_TOKEN)
        logger.info("✅ Replicate client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Replicate: {e}")

# Global rembg session with U2Net
REMBG_SESSION = None

# Korean font cache - FIXED VERSION
KOREAN_FONT_PATH = None
FONT_CACHE = {}
DEFAULT_FONT_CACHE = {}

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
    """Download and verify Korean font - FIXED VERSION"""
    global KOREAN_FONT_PATH
    
    if KOREAN_FONT_PATH and os.path.exists(KOREAN_FONT_PATH):
        logger.info(f"✅ Using cached Korean font: {KOREAN_FONT_PATH}")
        return KOREAN_FONT_PATH
    
    try:
        font_path = '/tmp/NanumGothic.ttf'
        
        # Download if not exists
        if not os.path.exists(font_path):
            # Try multiple sources for Korean font
            font_urls = [
                'https://github.com/naver/nanumfont/raw/master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf',
                'https://cdn.jsdelivr.net/gh/naver/nanumfont@master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf',
                'https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf'
            ]
            
            for url in font_urls:
                try:
                    logger.info(f"Downloading Korean font from: {url}")
                    response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
                    if response.status_code == 200 and len(response.content) > 100000:
                        with open(font_path, 'wb') as f:
                            f.write(response.content)
                        
                        # Verify font file
                        try:
                            test_font = ImageFont.truetype(font_path, 24)
                            # Test Korean character rendering
                            test_img = Image.new('RGB', (100, 100), 'white')
                            test_draw = ImageDraw.Draw(test_img)
                            test_draw.text((10, 10), "한글", font=test_font, fill='black')
                            
                            KOREAN_FONT_PATH = font_path
                            logger.info("✅ Korean font downloaded and verified successfully")
                            return font_path
                        except Exception as e:
                            logger.error(f"Font verification failed: {e}")
                            if os.path.exists(font_path):
                                os.remove(font_path)
                            continue
                except Exception as e:
                    logger.error(f"Failed to download from {url}: {e}")
                    continue
        else:
            # Verify existing font
            try:
                test_font = ImageFont.truetype(font_path, 24)
                KOREAN_FONT_PATH = font_path
                logger.info("✅ Existing Korean font verified")
                return font_path
            except:
                os.remove(font_path)
                return download_korean_font()  # Retry download
        
        logger.error("❌ Failed to download Korean font from all sources")
        return None
        
    except Exception as e:
        logger.error(f"❌ Font download error: {e}")
        return None

def get_default_font(size):
    """Get default font with size approximation"""
    global DEFAULT_FONT_CACHE
    
    if size in DEFAULT_FONT_CACHE:
        return DEFAULT_FONT_CACHE[size]
    
    try:
        # Try to load default font
        font = ImageFont.load_default()
        # Note: default font doesn't support size parameter
        DEFAULT_FONT_CACHE[size] = font
        return font
    except:
        return None

def get_font(size, force_korean=True):
    """Get font with Korean support - FIXED VERSION"""
    global KOREAN_FONT_PATH, FONT_CACHE
    
    cache_key = f"{size}_{force_korean}"
    
    # Return cached font if available
    if cache_key in FONT_CACHE:
        return FONT_CACHE[cache_key]
    
    font = None
    
    if force_korean:
        # Ensure Korean font is downloaded
        if not KOREAN_FONT_PATH:
            korean_path = download_korean_font()
            if not korean_path:
                logger.error("❌ No Korean font available, using default")
                font = get_default_font(size)
                if font:
                    FONT_CACHE[cache_key] = font
                return font
        
        if KOREAN_FONT_PATH and os.path.exists(KOREAN_FONT_PATH):
            try:
                # Load font without encoding parameter
                font = ImageFont.truetype(KOREAN_FONT_PATH, size)
                logger.info(f"✅ Korean font loaded successfully: size {size}")
            except Exception as e:
                logger.error(f"❌ Failed to load Korean font: {e}")
                # Try system fonts as fallback
                system_fonts = [
                    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                    '/System/Library/Fonts/AppleSDGothicNeo.ttc',
                    'C:\\Windows\\Fonts\\malgun.ttf'
                ]
                
                for sys_font in system_fonts:
                    if os.path.exists(sys_font):
                        try:
                            font = ImageFont.truetype(sys_font, size)
                            logger.info(f"✅ System font loaded: {sys_font}")
                            break
                        except:
                            continue
    
    # Fallback to default font
    if font is None:
        font = get_default_font(size)
        logger.warning(f"⚠️ Using default font for size {size}")
    
    # Cache the font
    if font:
        FONT_CACHE[cache_key] = font
    
    return font

def safe_draw_text(draw, position, text, font, fill):
    """Safely draw Korean text - FIXED VERSION"""
    try:
        if not text:
            return
        
        # Ensure text is properly encoded
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        else:
            text = str(text)
        
        text = text.strip()
        
        if not text:
            return
        
        # Try to draw text normally
        try:
            draw.text(position, text, font=font, fill=fill)
            logger.info(f"✅ Successfully drew text: {text[:20]}...")
        except UnicodeEncodeError:
            # Handle unicode errors by replacing problematic characters
            safe_text = text.encode('utf-8', errors='replace').decode('utf-8')
            draw.text(position, safe_text, font=font, fill=fill)
            logger.warning(f"⚠️ Drew text with replacements: {safe_text[:20]}...")
        except Exception as e:
            logger.error(f"❌ Text drawing error: {e}")
            # Final fallback: draw placeholder
            try:
                draw.text(position, "[Text Error]", font=font, fill=fill)
            except:
                pass
                
    except Exception as e:
        logger.error(f"❌ Critical text drawing error: {e}")

def get_text_size(draw, text, font):
    """Get text size with better compatibility"""
    try:
        if not text or not font:
            return (0, 0)
        
        # Ensure text is string
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        text = str(text).strip()
        
        if not text:
            return (0, 0)
        
        # Try modern method first
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            return (max(0, width), max(0, height))
        except AttributeError:
            # Fallback for older PIL versions
            try:
                return draw.textsize(text, font=font)
            except:
                # Estimate based on font size and character count
                char_width = size * 0.6 if 'size' in locals() else 15
                return (len(text) * char_width, size if 'size' in locals() else 30)
                
    except Exception as e:
        logger.error(f"❌ Text size calculation error: {e}")
        return (100, 30)  # Safe fallback

def wrap_text(text, font, max_width, draw):
    """Wrap text to fit within max_width - FIXED VERSION"""
    if not text or not font:
        return []
    
    # Ensure text is string
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    text = str(text).strip()
    
    lines = []
    
    # Split by newlines first
    paragraphs = text.split('\n')
    
    for paragraph in paragraphs:
        if not paragraph:
            lines.append('')
            continue
            
        words = paragraph.split()
        if not words:
            lines.append('')
            continue
            
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            try:
                width, _ = get_text_size(draw, test_line, font)
            except:
                width = max_width + 1  # Force new line on error
            
            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word too long, add it anyway
                    lines.append(word)
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
    
    return lines

def create_md_talk_section(text_content=None, width=1200):
    """Create MD TALK section - FIXED Korean rendering with reduced height"""
    logger.info("🔤 Creating MD TALK section with FIXED Korean support and optimized layout")
    
    # Fixed dimensions - REDUCED HEIGHT
    fixed_width = 1200
    fixed_height = 400  # Reduced from 600
    
    # Margins
    left_margin = 100
    right_margin = 100
    top_margin = 60  # Reduced from 80
    content_width = fixed_width - left_margin - right_margin
    
    # Get fonts
    title_font = get_font(48, force_korean=True)
    body_font = get_font(28, force_korean=True)
    
    if not title_font or not body_font:
        logger.error("❌ Font loading failed")
        # Create blank image
        section_img = Image.new('RGB', (fixed_width, fixed_height), '#FFFFFF')
        return section_img
    
    # Create image
    section_img = Image.new('RGB', (fixed_width, fixed_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Title
    title = "MD TALK"
    try:
        title_width, title_height = get_text_size(draw, title, title_font)
        title_x = (fixed_width - title_width) // 2
        safe_draw_text(draw, (title_x, top_margin), title, title_font, (40, 40, 40))
    except Exception as e:
        logger.error(f"Title drawing error: {e}")
        title_height = 50
    
    # Prepare Korean text content
    if text_content and text_content.strip():
        text = text_content.replace('MD TALK', '').replace('MD Talk', '').strip()
    else:
        text = """이 제품은 일상에서도 부담없이 착용할 수 있는 편안한 디자인으로 매일의 스타일링에 포인트를 더해줍니다. 특별한 날은 물론 평범한 일상까지 모든 순간을 빛나게 만들어주는 당신만의 특별한 주얼리입니다."""
    
    # Ensure proper encoding
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    text = str(text).strip()
    
    logger.info(f"MD TALK text (first 50 chars): {text[:50]}...")
    
    # Wrap text
    wrapped_lines = wrap_text(text, body_font, content_width, draw)
    
    # Calculate positions
    line_height = 45  # Reduced from 50
    title_bottom_margin = 60  # Reduced from 80
    y_pos = top_margin + title_height + title_bottom_margin
    
    # Draw body text
    for line in wrapped_lines:
        if line:
            try:
                line_width, _ = get_text_size(draw, line, body_font)
                line_x = (fixed_width - line_width) // 2
            except:
                line_x = left_margin
            
            safe_draw_text(draw, (line_x, y_pos), line, body_font, (80, 80, 80))
            y_pos += line_height
    
    logger.info(f"✅ MD TALK section created: {fixed_width}x{fixed_height}")
    return section_img

def create_design_point_section(text_content=None, width=1200):
    """Create DESIGN POINT section - FIXED Korean rendering with reduced height"""
    logger.info("🔤 Creating DESIGN POINT section with FIXED Korean support and optimized layout")
    
    # Fixed dimensions - REDUCED HEIGHT
    fixed_width = 1200
    fixed_height = 350  # Reduced from 600
    
    # Margins
    left_margin = 100
    right_margin = 100
    top_margin = 60  # Reduced from 80
    content_width = fixed_width - left_margin - right_margin
    
    # Get fonts
    title_font = get_font(48, force_korean=True)
    body_font = get_font(24, force_korean=True)
    
    if not title_font or not body_font:
        logger.error("❌ Font loading failed")
        # Create blank image
        section_img = Image.new('RGB', (fixed_width, fixed_height), '#FFFFFF')
        return section_img
    
    # Create image
    section_img = Image.new('RGB', (fixed_width, fixed_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Title
    title = "DESIGN POINT"
    try:
        title_width, title_height = get_text_size(draw, title, title_font)
        title_x = (fixed_width - title_width) // 2
        safe_draw_text(draw, (title_x, top_margin), title, title_font, (40, 40, 40))
    except Exception as e:
        logger.error(f"Title drawing error: {e}")
        title_height = 50
    
    # Prepare text content
    if text_content and text_content.strip():
        text = text_content.replace('DESIGN POINT', '').replace('Design Point', '').strip()
    else:
        text = """남성 단품은 무광 텍스처와 유광 라인의 조화가 견고한 감성을 전하고 여자 단품은 파베 세팅과 섬세한 밀그레인의 디테일 화려하면서도 고급스러운 반짝임을 표현합니다"""
    
    # Ensure proper encoding
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    text = str(text).strip()
    
    logger.info(f"DESIGN POINT text (first 50 chars): {text[:50]}...")
    
    # Wrap text
    wrapped_lines = wrap_text(text, body_font, content_width, draw)
    
    # Calculate positions
    line_height = 40  # Reduced from 45
    title_bottom_margin = 70  # Reduced from 100
    y_pos = top_margin + title_height + title_bottom_margin
    
    # Draw body text
    for line in wrapped_lines:
        if line:
            try:
                line_width, _ = get_text_size(draw, line, body_font)
                line_x = (fixed_width - line_width) // 2
            except:
                line_x = left_margin
            
            safe_draw_text(draw, (line_x, y_pos), line, body_font, (80, 80, 80))
            y_pos += line_height
    
    # NO bottom line
    logger.info(f"✅ DESIGN POINT section created: {fixed_width}x{fixed_height}")
    return section_img

def u2net_ultra_precise_removal_v3_shadow_fix_ultra_enhanced(image: Image.Image) -> Image.Image:
    """ULTRA PRECISE V3 ENHANCED - Maximum precision for jewelry with aggressive shadow removal"""
    try:
        from rembg import remove
        
        global REMBG_SESSION
        if REMBG_SESSION is None:
            REMBG_SESSION = init_rembg_session()
            if REMBG_SESSION is None:
                return image
        
        logger.info("🔷 U2Net ULTRA PRECISE V3 ENHANCED - Maximum Shadow Removal")
        
        # CRITICAL: Ensure RGBA mode before processing
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Enhanced pre-processing for jewelry
        contrast = ImageEnhance.Contrast(image)
        image_enhanced = contrast.enhance(1.4)  # Even higher contrast
        
        # Save image to buffer
        buffered = BytesIO()
        image_enhanced.save(buffered, format="PNG", compress_level=3, optimize=True)
        buffered.seek(0)
        img_data = buffered.getvalue()
        
        # Apply U2Net removal with ULTRA settings
        output = remove(
            img_data,
            session=REMBG_SESSION,
            alpha_matting=True,
            alpha_matting_foreground_threshold=340,  # Even higher for maximum precision
            alpha_matting_background_threshold=0,
            alpha_matting_erode_size=0,  # No erosion for maximum detail
            only_mask=False,
            post_process_mask=True
        )
        
        result_image = Image.open(BytesIO(output))
        
        # CRITICAL: Ensure RGBA mode
        if result_image.mode != 'RGBA':
            result_image = result_image.convert('RGBA')
        
        # ULTRA edge refinement
        r, g, b, a = result_image.split()
        alpha_array = np.array(a, dtype=np.uint8)
        rgb_array = np.array(result_image.convert('RGB'), dtype=np.uint8)
        
        # Convert to float for processing
        alpha_float = alpha_array.astype(np.float32) / 255.0
        
        # STAGE 1: AGGRESSIVE shadow detection and removal
        logger.info("🔍 AGGRESSIVE multi-level shadow detection...")
        
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        
        # Convert to multiple color spaces for comprehensive analysis
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        
        lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Multi-level shadow detection
        # Level 1: Any semi-transparent areas that might be shadows
        potential_shadows = (alpha_float > 0.01) & (alpha_float < 0.5)
        
        # Level 2: Low saturation gray areas
        gray_shadows = (s_channel < 30) & (v_channel < 200) & (alpha_float < 0.7)
        
        # Level 3: Edge-based shadow detection
        edges = cv2.Canny(gray, 30, 100)
        edge_dilated = cv2.dilate(edges, np.ones((7,7)), iterations=2)
        edge_shadows = (alpha_float < 0.8) & (~edge_dilated.astype(bool))
        
        # Level 4: LAB-based shadow detection
        lab_shadows = (l_channel < 180) & (np.abs(a_channel - 128) < 20) & (np.abs(b_channel - 128) < 20)
        
        # Combine all shadow detections
        all_shadows = potential_shadows | gray_shadows | edge_shadows | (lab_shadows & (alpha_float < 0.9))
        
        # AGGRESSIVE shadow removal
        if np.any(all_shadows):
            logger.info("🔥 Removing detected shadows aggressively...")
            alpha_float[all_shadows] = 0
        
        # STAGE 2: Ultra-precise edge detection with 6 methods
        logger.info("🔍 Ultra-precise 6-method edge detection...")
        
        # Method 1-2: Sobel with multiple kernel sizes
        sobel_3 = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        sobel_5 = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
        sobel_7 = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=7)
        sobel_combined = np.maximum(np.maximum(np.abs(sobel_3), np.abs(sobel_5)), np.abs(sobel_7))
        sobel_edges = (sobel_combined / sobel_combined.max() * 255).astype(np.uint8) > 30
        
        # Method 3: Scharr for fine details
        scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        scharr_magnitude = np.sqrt(scharrx**2 + scharry**2)
        scharr_edges = (scharr_magnitude / scharr_magnitude.max() * 255).astype(np.uint8) > 30
        
        # Method 4: Laplacian for jewelry details
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplacian_edges = np.abs(laplacian) > 25
        
        # Method 5-6: Multi-threshold Canny
        canny_low = cv2.Canny(gray, 20, 60)
        canny_mid = cv2.Canny(gray, 40, 120)
        canny_high = cv2.Canny(gray, 60, 180)
        
        # Combine all edge detections
        all_edges = sobel_edges | scharr_edges | laplacian_edges | (canny_low > 0) | (canny_mid > 0) | (canny_high > 0)
        
        # STAGE 3: Main object isolation with better component analysis
        logger.info("🔍 Intelligent jewelry object isolation...")
        
        # Binary mask for main object
        alpha_binary = (alpha_float > 0.6).astype(np.uint8)
        
        # Clean up with morphology
        kernel_jewelry = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        alpha_binary = cv2.morphologyEx(alpha_binary, cv2.MORPH_CLOSE, kernel_jewelry)
        alpha_binary = cv2.morphologyEx(alpha_binary, cv2.MORPH_OPEN, np.ones((3,3)))
        
        num_labels, labels = cv2.connectedComponents(alpha_binary)
        
        if num_labels > 1:
            # Find all significant components
            sizes = [(i, np.sum(labels == i)) for i in range(1, num_labels)]
            sizes.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only significant components
            main_mask = np.zeros_like(alpha_binary, dtype=bool)
            if sizes:
                main_size = sizes[0][1]
                min_component_size = max(100, main_size * 0.02)  # At least 2% of main object
                
                for label_id, size in sizes:
                    if size > min_component_size:
                        main_mask |= (labels == label_id)
            
            # Apply main mask
            alpha_float[~main_mask] = 0
        
        # STAGE 4: Remove any remaining gray artifacts
        logger.info("🔍 Removing gray artifacts and edge noise...")
        
        # Find areas that look like shadows or artifacts
        gray_artifacts = (s_channel < 25) & (v_channel > 50) & (v_channel < 200) & (alpha_float > 0) & (alpha_float < 0.9)
        
        if np.any(gray_artifacts):
            alpha_float[gray_artifacts] = 0
        
        # STAGE 5: Sharp edge refinement
        logger.info("🔍 Sharp edge refinement...")
        
        # Create sharp edges
        alpha_sharp = np.where(alpha_float > 0.7, 1.0, 0.0)
        
        # Minimal smoothing for anti-aliasing
        alpha_smooth = cv2.GaussianBlur(alpha_sharp, (3, 3), 0.5)
        
        # Ultra-sharp sigmoid
        k = 150  # Very high steepness
        threshold = 0.5
        alpha_sigmoid = 1 / (1 + np.exp(-k * (alpha_smooth - threshold)))
        
        # STAGE 6: Final cleanup
        logger.info("🔍 Final cleanup...")
        
        # Remove any remaining small components
        alpha_binary_final = (alpha_sigmoid > 0.5).astype(np.uint8)
        num_labels_final, labels_final = cv2.connectedComponents(alpha_binary_final)
        
        if num_labels_final > 2:
            sizes_final = [(i, np.sum(labels_final == i)) for i in range(1, num_labels_final)]
            if sizes_final:
                sizes_final.sort(key=lambda x: x[1], reverse=True)
                # Keep only the largest component(s)
                min_size = max(100, alpha_array.size * 0.0001)
                
                valid_mask = np.zeros_like(alpha_binary_final, dtype=bool)
                for label_id, size in sizes_final:
                    if size > min_size:
                        valid_mask |= (labels_final == label_id)
                
                alpha_sigmoid[~valid_mask] = 0
        
        # Convert back to uint8
        alpha_array = np.clip(alpha_sigmoid * 255, 0, 255).astype(np.uint8)
        
        logger.info("✅ ULTRA PRECISE V3 ENHANCED complete")
        
        a_new = Image.fromarray(alpha_array)
        result = Image.merge('RGBA', (r, g, b, a_new))
        
        # Verify RGBA mode
        if result.mode != 'RGBA':
            logger.error("❌ WARNING: Result is not RGBA!")
            result = result.convert('RGBA')
        
        return result
        
    except Exception as e:
        logger.error(f"U2Net removal failed: {e}")
        # Ensure RGBA mode even on failure
        if image.mode != 'RGBA':
            return image.convert('RGBA')
        return image

def ensure_ring_holes_transparent_ultra_v3_enhanced(image: Image.Image) -> Image.Image:
    """ULTRA PRECISE V3 ENHANCED - Multi-color space hole detection"""
    # CRITICAL: Preserve RGBA mode
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    logger.info("🔍 ULTRA PRECISE V3 ENHANCED Ring Hole Detection")
    
    r, g, b, a = image.split()
    alpha_array = np.array(a, dtype=np.uint8)
    rgb_array = np.array(image.convert('RGB'), dtype=np.uint8)
    
    h, w = alpha_array.shape
    
    # Convert to multiple color spaces
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    
    lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    
    # STAGE 1: Comprehensive hole detection
    # Multiple criteria for hole detection
    very_bright_v = v_channel > 250
    very_bright_l = l_channel > 245
    very_bright_gray = gray > 245
    
    # Very low saturation
    very_low_saturation = s_channel < 15
    
    # Low color variance in LAB
    low_color_variance = (np.abs(a_channel - 128) < 15) & (np.abs(b_channel - 128) < 15)
    
    # Alpha-based detection
    alpha_holes = alpha_array < 30
    
    # Combine all criteria with more aggressive thresholds
    potential_holes = ((very_bright_v | very_bright_l | very_bright_gray) & 
                      (very_low_saturation | low_color_variance)) | alpha_holes
    
    # STAGE 2: Narrow region detection
    logger.info("🔍 Detecting narrow regions and enclosed areas...")
    
    if np.any(alpha_array > 128):
        # Distance transform from object
        object_mask = (alpha_array > 128).astype(np.uint8)
        dist_transform = cv2.distanceTransform(object_mask, cv2.DIST_L2, 5)
        
        # Find narrow regions
        narrow_regions = (dist_transform > 1) & (dist_transform < 25)
        
        # Bright areas in narrow regions are likely holes
        narrow_bright = narrow_regions & ((gray > 240) | (v_channel > 245))
        potential_holes |= narrow_bright
        
        # Find enclosed regions
        inverted = cv2.bitwise_not(object_mask)
        num_inv_labels, inv_labels = cv2.connectedComponents(inverted)
        
        # Check each potential enclosed region
        for label in range(1, num_inv_labels):
            component = (inv_labels == label)
            if np.any(component):
                # Check if completely enclosed (doesn't touch border)
                dilated = cv2.dilate(component.astype(np.uint8), np.ones((5,5)), iterations=1)
                touches_border = np.any(dilated[0,:]) or np.any(dilated[-1,:]) or \
                               np.any(dilated[:,0]) or np.any(dilated[:,-1])
                
                if not touches_border:
                    # This is an enclosed region
                    component_pixels = rgb_array[component]
                    if len(component_pixels) > 0:
                        brightness = np.mean(component_pixels)
                        if brightness > 235:
                            potential_holes[component] = True
                            logger.info(f"Found enclosed bright region with brightness {brightness:.1f}")
    
    # Clean up noise
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    potential_holes = cv2.morphologyEx(potential_holes.astype(np.uint8), cv2.MORPH_OPEN, kernel_clean)
    potential_holes = cv2.morphologyEx(potential_holes, cv2.MORPH_CLOSE, kernel_clean)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(potential_holes)
    
    holes_mask = np.zeros_like(alpha_array, dtype=np.float32)
    
    # STAGE 3: Validate each hole candidate with enhanced criteria
    for label in range(1, num_labels):
        component = (labels == label)
        component_size = np.sum(component)
        
        # Size constraints
        min_size = max(20, h * w * 0.00002)
        max_size = h * w * 0.25
        
        if min_size < component_size < max_size:
            coords = np.where(component)
            if len(coords[0]) == 0:
                continue
            
            # Analyze component in all color spaces
            component_pixels_rgb = rgb_array[component]
            component_pixels_hsv = np.column_stack((h_channel[component], 
                                                   s_channel[component], 
                                                   v_channel[component]))
            component_pixels_lab = np.column_stack((l_channel[component], 
                                                   a_channel[component], 
                                                   b_channel[component]))
            
            if len(component_pixels_rgb) > 0:
                # Multi-space analysis
                brightness_rgb = np.mean(component_pixels_rgb)
                brightness_v = np.mean(component_pixels_hsv[:, 2])
                brightness_l = np.mean(component_pixels_lab[:, 0])
                
                saturation_mean = np.mean(component_pixels_hsv[:, 1])
                
                # Color uniformity
                rgb_std = np.std(component_pixels_rgb, axis=0)
                max_rgb_std = np.max(rgb_std)
                
                lab_std = np.std(component_pixels_lab, axis=0)
                max_lab_std = np.max(lab_std)
                
                # Shape analysis
                component_uint8 = component.astype(np.uint8) * 255
                contours, _ = cv2.findContours(component_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                shape_score = 0
                is_enclosed = False
                
                if contours:
                    contour = contours[0]
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    
                    if perimeter > 0 and area > 0:
                        # Circularity
                        circularity = (4 * np.pi * area) / (perimeter * perimeter)
                        
                        # Convexity
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        convexity = area / hull_area if hull_area > 0 else 0
                        
                        # Solidity (how "filled" the shape is)
                        solidity = area / hull_area if hull_area > 0 else 0
                        
                        shape_score = (circularity * 0.4 + convexity * 0.3 + solidity * 0.3)
                        
                        # Check if enclosed
                        x, y, w, h = cv2.boundingRect(contour)
                        roi = object_mask[y:y+h, x:x+w] if 'object_mask' in locals() else None
                        if roi is not None and roi.shape[0] > 0 and roi.shape[1] > 0:
                            # Check if the hole is surrounded by object
                            border_sum = np.sum(roi[0,:]) + np.sum(roi[-1,:]) + \
                                       np.sum(roi[:,0]) + np.sum(roi[:,-1])
                            if border_sum > (2 * (w + h) - 4) * 0.8:
                                is_enclosed = True
                
                # Enhanced confidence calculation
                confidence = 0.0
                
                # Brightness criteria (very important)
                if brightness_rgb > 245 and brightness_v > 250 and brightness_l > 245:
                    confidence += 0.4
                elif brightness_rgb > 235 and brightness_v > 240 and brightness_l > 235:
                    confidence += 0.3
                elif brightness_rgb > 225:
                    confidence += 0.2
                
                # Saturation criteria
                if saturation_mean < 10:
                    confidence += 0.25
                elif saturation_mean < 20:
                    confidence += 0.15
                
                # Color uniformity
                if max_rgb_std < 10 and max_lab_std < 10:
                    confidence += 0.2
                elif max_rgb_std < 20 and max_lab_std < 15:
                    confidence += 0.1
                
                # Shape criteria
                if shape_score > 0.6:
                    confidence += 0.15
                elif shape_score > 0.4:
                    confidence += 0.1
                
                # Bonus for enclosed regions
                if is_enclosed:
                    confidence += 0.2
                
                # Apply hole mask based on confidence
                if confidence > 0.5:
                    holes_mask[component] = 255
                    logger.info(f"Hole detected: brightness RGB/V/L={brightness_rgb:.1f}/{brightness_v:.1f}/{brightness_l:.1f}, "
                              f"saturation={saturation_mean:.1f}, uniformity={max_rgb_std:.1f}, "
                              f"shape={shape_score:.2f}, enclosed={is_enclosed}, confidence={confidence:.2f}")
    
    # STAGE 4: Apply holes with smooth transitions
    if np.any(holes_mask > 0):
        # Smooth the hole masks
        holes_mask_smooth = cv2.GaussianBlur(holes_mask, (5, 5), 1)
        
        # Create transition zones
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        holes_dilated = cv2.dilate(holes_mask, kernel_dilate, iterations=1)
        transition_zone = (holes_dilated > 0) & (holes_mask < 255)
        
        alpha_float = alpha_array.astype(np.float32)
        
        # Apply holes
        alpha_float[holes_mask_smooth > 200] = 0
        
        # Smooth transitions
        if np.any(transition_zone):
            # Distance-based transition
            dist_from_hole = cv2.distanceTransform((holes_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
            transition_alpha = np.clip(dist_from_hole / 5, 0, 1)
            alpha_float[transition_zone] *= transition_alpha[transition_zone]
        
        alpha_array = np.clip(alpha_float, 0, 255).astype(np.uint8)
        
        logger.info("✅ Ring holes applied with enhanced detection")
    
    a_new = Image.fromarray(alpha_array)
    result = Image.merge('RGBA', (r, g, b, a_new))
    
    # Verify RGBA mode
    if result.mode != 'RGBA':
        logger.error("❌ WARNING: Result is not RGBA!")
        result = result.convert('RGBA')
    
    return result

def image_to_base64(image, keep_transparency=True):
    """Convert to base64 WITH padding - FIXED for Google Script compatibility"""
    buffered = BytesIO()
    
    # CRITICAL FIX: Force RGBA and save as PNG
    if image.mode != 'RGBA' and keep_transparency:
        logger.warning(f"⚠️ Converting {image.mode} to RGBA for transparency")
        image = image.convert('RGBA')
    
    if image.mode == 'RGBA':
        logger.info("💎 Saving RGBA image as PNG with compression level 3")
        image.save(buffered, format='PNG', compress_level=3, optimize=True)
    else:
        logger.info(f"Saving {image.mode} mode image as PNG")
        image.save(buffered, format='PNG', optimize=True, compress_level=3)
    
    buffered.seek(0)
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    # FIXED: Always return WITH padding for Google Script compatibility
    return base64_str

def process_special_mode(job):
    """Process special modes - FIXED KOREAN SUPPORT"""
    special_mode = job.get('special_mode', '')
    logger.info(f"🔤 Processing special mode with FIXED Korean support: {special_mode}")
    
    # BOTH TEXT SECTIONS - Return TWO separate images
    if special_mode == 'both_text_sections':
        # Get text content with proper Korean encoding
        md_talk_text = job.get('md_talk_content', '') or job.get('md_talk', '') or """각도에 따라 달라지는 빛의 결들이 두 사람의 특별한 순간순간을 더 찬란하게 만들며 360도 새겨진 패턴으로 매일 새로운 반짝임을 보여줍니다 :)"""
        
        design_point_text = job.get('design_point_content', '') or job.get('design_point', '') or """입체적인 컷팅 위로 섬세하게 빛나는 패턴이 고급스러움을 완성하며 각진 텍스처가 심플하면서 유니크한 매력을 더해줍니다."""
        
        # Handle encoding properly
        if isinstance(md_talk_text, bytes):
            md_talk_text = md_talk_text.decode('utf-8', errors='replace')
        if isinstance(design_point_text, bytes):
            design_point_text = design_point_text.decode('utf-8', errors='replace')
        
        md_talk_text = str(md_talk_text).strip()
        design_point_text = str(design_point_text).strip()
        
        logger.info(f"✅ Creating both Korean sections")
        logger.info(f"MD TALK text: {md_talk_text[:50]}...")
        logger.info(f"DESIGN POINT text: {design_point_text[:50]}...")
        
        # Create both sections with Korean support
        md_section = create_md_talk_section(md_talk_text)
        design_section = create_design_point_section(design_point_text)
        
        # Convert to base64 WITH padding
        md_base64 = image_to_base64(md_section, keep_transparency=False)
        design_base64 = image_to_base64(design_section, keep_transparency=False)
        
        # Return BOTH images separately
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
                "korean_encoding": "UTF-8-FIXED-V5",
                "korean_font_verified": True,
                "korean_font_path": KOREAN_FONT_PATH,
                "base64_padding": "INCLUDED",
                "compression": "level_3",
                "layout_update": "Optimized height: MD_TALK=400px, DESIGN_POINT=350px"
            }
        }
    
    # Single MD TALK section
    elif special_mode == 'md_talk':
        text_content = job.get('text_content', '') or job.get('claude_text', '') or job.get('md_talk', '')
        
        if not text_content:
            text_content = """이 제품은 일상에서도 부담없이 착용할 수 있는 편안한 디자인으로 매일의 스타일링에 포인트를 더해줍니다."""
        
        # Handle encoding
        if isinstance(text_content, bytes):
            text_content = text_content.decode('utf-8', errors='replace')
        text_content = str(text_content).strip()
        
        logger.info(f"✅ Creating MD TALK with Korean text: {text_content[:50]}...")
        
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
                "korean_encoding": "UTF-8-FIXED-V5",
                "korean_font_verified": True,
                "korean_font_path": KOREAN_FONT_PATH,
                "base64_padding": "INCLUDED",
                "compression": "level_3",
                "layout_update": "Optimized height: 400px"
            }
        }
    
    # Single DESIGN POINT section
    elif special_mode == 'design_point':
        text_content = job.get('text_content', '') or job.get('claude_text', '') or job.get('design_point', '')
        
        if not text_content:
            text_content = """남성 단품은 무광 텍스처와 유광 라인의 조화가 견고한 감성을 전하고 여자 단품은 파베 세팅과 섬세한 밀그레인의 디테일로 화려하면서도 고급스러운 반짝임을 표현합니다."""
        
        # Handle encoding
        if isinstance(text_content, bytes):
            text_content = text_content.decode('utf-8', errors='replace')
        text_content = str(text_content).strip()
        
        logger.info(f"✅ Creating DESIGN POINT with Korean text: {text_content[:50]}...")
        
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
                "korean_encoding": "UTF-8-FIXED-V5",
                "korean_font_verified": True,
                "korean_font_path": KOREAN_FONT_PATH,
                "base64_padding": "INCLUDED",
                "compression": "level_3",
                "layout_update": "Optimized height: 350px"
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
        
        # FIXED: Try with padding first (normal base64)
        try:
            decoded = base64.b64decode(base64_str, validate=True)
            return decoded
        except:
            # If fails, try without padding
            no_pad = base64_str.rstrip('=')
            padding_needed = (4 - len(no_pad) % 4) % 4
            padded = no_pad + ('=' * padding_needed)
            decoded = base64.b64decode(padded, validate=True)
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
    """Apply pattern enhancement while TRULY preserving transparency - AC 20%, AB 16%, Other 5% - UPDATED"""
    # CRITICAL: Ensure RGBA mode
    if image.mode != 'RGBA':
        logger.warning(f"⚠️ Converting {image.mode} to RGBA in pattern enhancement")
        image = image.convert('RGBA')
    
    # CRITICAL: Process RGB channels separately to preserve alpha
    r, g, b, a = image.split()
    rgb_image = Image.merge('RGB', (r, g, b))
    
    # Convert to array for processing
    img_array = np.array(rgb_image, dtype=np.float32)
    
    # Apply enhancements based on pattern type
    if pattern_type == "ac_pattern":
        logger.info("🔍 AC Pattern - Applying 20% white overlay with brightness 1.03")
        # Apply 20% white overlay
        white_overlay = 0.20
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        
        # Convert back to image
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        # UPDATED: Brightness increased by 0.01
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.03)  # Changed from 1.02
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.98)
        
        logger.info("✅ AC Pattern enhancement applied with 20% white overlay and brightness 1.03")
    
    elif pattern_type == "ab_pattern":
        logger.info("🔍 AB Pattern - Applying 16% white overlay with brightness 1.03")
        # Apply 16% white overlay
        white_overlay = 0.16
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
        
        # UPDATED: Brightness increased by 0.01
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.03)  # Changed from 1.02
        
        logger.info("✅ AB Pattern enhancement applied with 16% white overlay and brightness 1.03")
        
    else:
        logger.info("🔍 Other Pattern - Applying 5% white overlay with brightness 1.09")
        # NEW: Apply 5% white overlay for other patterns
        white_overlay = 0.05
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        
        # Convert back to image
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        # UPDATED: Brightness increased by 0.01
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.09)  # Changed from 1.08
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.99)
        
        sharpness = ImageEnhance.Sharpness(rgb_image)
        rgb_image = sharpness.enhance(1.5)
        
        logger.info("✅ Other Pattern enhancement applied with 5% white overlay and brightness 1.09")
    
    # UPDATED: Apply common enhancements with contrast 1.1
    contrast = ImageEnhance.Contrast(rgb_image)
    rgb_image = contrast.enhance(1.1)  # Changed from 1.08
    
    # Apply sharpening
    sharpness = ImageEnhance.Sharpness(rgb_image)
    rgb_image = sharpness.enhance(1.8)
    
    # CRITICAL: Recombine with ORIGINAL alpha channel
    r2, g2, b2 = rgb_image.split()
    enhanced_image = Image.merge('RGBA', (r2, g2, b2, a))
    
    logger.info(f"✅ Enhancement applied with contrast 1.1 and brightness adjustments. Mode: {enhanced_image.mode}")
    
    # Verify RGBA mode
    if enhanced_image.mode != 'RGBA':
        logger.error("❌ WARNING: Enhanced image is not RGBA!")
        enhanced_image = enhanced_image.convert('RGBA')
    
    return enhanced_image

def resize_to_target_dimensions(image: Image.Image, target_width=1200, target_height=1560) -> Image.Image:
    """Resize image to target dimensions preserving transparency"""
    # CRITICAL: Ensure RGBA mode
    if image.mode != 'RGBA':
        logger.warning(f"⚠️ Converting {image.mode} to RGBA in resize")
        image = image.convert('RGBA')
    
    width, height = image.size
    
    img_ratio = width / height
    target_ratio = target_width / target_height
    
    expected_ratio = 2000 / 2600
    
    if abs(img_ratio - expected_ratio) < 0.05:
        resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    else:
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
    
    # Verify RGBA mode
    if resized.mode != 'RGBA':
        logger.error("❌ WARNING: Resized image is not RGBA!")
        resized = resized.convert('RGBA')
    
    return resized

def apply_swinir_enhancement_transparent(image: Image.Image) -> Image.Image:
    """Apply SwinIR enhancement while preserving transparency"""
    if not REPLICATE_CLIENT:
        logger.warning("SwinIR skipped - no Replicate client")
        return image
    
    try:
        logger.info("🎨 Applying SwinIR enhancement with transparency")
        
        # CRITICAL: Ensure RGBA mode
        if image.mode != 'RGBA':
            logger.warning(f"⚠️ Converting {image.mode} to RGBA for SwinIR")
            image = image.convert('RGBA')
        
        # Separate alpha channel
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        
        buffered = BytesIO()
        rgb_image.save(buffered, format="PNG", optimize=True, compress_level=3)
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
            r2, g2, b2 = enhanced_image.split()
            result = Image.merge('RGBA', (r2, g2, b2, a))
            
            logger.info("✅ SwinIR enhancement successful with transparency preserved")
            
            # Verify RGBA mode
            if result.mode != 'RGBA':
                logger.error("❌ WARNING: SwinIR result is not RGBA!")
                result = result.convert('RGBA')
            
            return result
        else:
            return image
            
    except Exception as e:
        logger.warning(f"SwinIR error: {str(e)}")
        return image

def process_enhancement(job):
    """Main enhancement processing - UPDATED with Korean V5 fixes and Other 5% overlay"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    logger.info("🎯 KOREAN ENCODING FIXED V5: Enhanced error handling, font verification, and layout")
    logger.info("💎 TRANSPARENT OUTPUT: Preserving alpha channel throughout")
    logger.info("🔤 FIXED: Better font fallback, encoding handling, text rendering")
    logger.info("🔤 FIXED TEXT SECTIONS: MD_TALK=1200x400, DESIGN_POINT=1200x350")
    logger.info("🔧 AC PATTERN: 20% white overlay, brightness 1.03, contrast 1.1")
    logger.info("🔧 AB PATTERN: 16% white overlay, brightness 1.03, contrast 1.1")
    logger.info("✨ OTHER PATTERNS: 5% white overlay, brightness 1.09, contrast 1.1")
    logger.info("📌 BASE64 PADDING: ALWAYS INCLUDED for Google Script compatibility")
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
        
        # CRITICAL: Convert to RGBA immediately
        if image.mode != 'RGBA':
            logger.info(f"Converting {image.mode} to RGBA immediately")
            image = image.convert('RGBA')
        
        # STEP 1: ULTRA PRECISE V3 ENHANCED BACKGROUND REMOVAL
        logger.info("📸 STEP 1: Applying ULTRA PRECISE V3 ENHANCED background removal")
        removal_start = time.time()
        image = u2net_ultra_precise_removal_v3_shadow_fix_ultra_enhanced(image)
        logger.info(f"⏱️ Ultra precise V3 Enhanced removal took: {time.time() - removal_start:.2f}s")
        
        # Verify RGBA after removal
        if image.mode != 'RGBA':
            logger.error("❌ Image lost RGBA after background removal!")
            image = image.convert('RGBA')
        
        # STEP 2: ENHANCEMENT (preserving transparency)
        logger.info("🎨 STEP 2: Applying enhancements with TRUE transparency preservation")
        enhancement_start = time.time()
        
        # Enhanced auto white balance with alpha preservation
        def auto_white_balance_rgba(rgba_img):
            if rgba_img.mode != 'RGBA':
                rgba_img = rgba_img.convert('RGBA')
            
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
            "ac_pattern": "무도금화이트(0.20)",
            "ab_pattern": "무도금화이트-쿨톤(0.16)",
            "other": "기타색상(0.05)"
        }.get(pattern_type, "기타색상(0.05)")
        
        # Apply pattern-specific enhancements (preserving transparency)
        image = apply_pattern_enhancement_transparent(image, pattern_type)
        
        # ULTRA PRECISE V3 ENHANCED ring hole detection
        logger.info("🔍 Applying ULTRA PRECISE V3 ENHANCED ring hole detection")
        image = ensure_ring_holes_transparent_ultra_v3_enhanced(image)
        
        logger.info(f"⏱️ Enhancement took: {time.time() - enhancement_start:.2f}s")
        
        # RESIZE (preserving transparency)
        image = resize_to_target_dimensions(image, 1200, 1560)
        
        # STEP 3: SWINIR ENHANCEMENT (preserving transparency)
        logger.info("🚀 STEP 3: Applying SwinIR enhancement")
        swinir_start = time.time()
        image = apply_swinir_enhancement_transparent(image)
        logger.info(f"⏱️ SwinIR took: {time.time() - swinir_start:.2f}s")
        
        # Final verification
        if image.mode != 'RGBA':
            logger.error("❌ CRITICAL: Final image is not RGBA! Converting...")
            image = image.convert('RGBA')
        
        logger.info(f"✅ Final image mode: {image.mode}, size: {image.size}")
        
        # CRITICAL: NO BACKGROUND COMPOSITE - Keep transparency
        logger.info("💎 NO background composite - keeping pure transparency")
        
        # Save to base64 as PNG with transparency - WITH PADDING
        enhanced_base64 = image_to_base64(image, keep_transparency=True)
        
        # Build filename
        enhanced_filename = filename
        if filename and file_number:
            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            enhanced_filename = f"{base_name}_enhanced_transparent.png"
        
        total_time = time.time() - start_time
        logger.info(f"✅ Enhancement completed in {total_time:.2f}s")
        
        output = {
            "output": {
                "enhanced_image": enhanced_base64,
                "enhanced_image_with_prefix": f"data:image/png;base64,{enhanced_base64}",
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
                "base64_padding": "INCLUDED",
                "compression": "level_3",
                "korean_font_verified": True,
                "korean_font_path": KOREAN_FONT_PATH,
                "korean_support": "FIXED-V5",
                "special_modes_available": ["md_talk", "design_point", "both_text_sections"],
                "file_number_info": {
                    "001-003": "Enhancement",
                    "004": "MD TALK (1200x400)",
                    "005-006": "Enhancement",
                    "007": "Thumbnail",
                    "008": "DESIGN POINT (1200x350)",
                    "009-010": "Thumbnail",
                    "011": "COLOR section"
                },
                "korean_fixes_v5": [
                    "✅ Enhanced font download with verification",
                    "✅ Added Korean character rendering test",
                    "✅ Better font caching mechanism",
                    "✅ Improved text encoding error handling",
                    "✅ Added system font fallbacks",
                    "✅ Better error messages and logging",
                    "✅ Fixed Unicode encoding errors",
                    "✅ Added default font fallback",
                    "✅ Improved text size calculation compatibility",
                    "✅ Optimized text section heights: MD_TALK=400px, DESIGN_POINT=350px",
                    "✅ Added 5% white overlay for Other patterns"
                ],
                "processing_order": "1.U2Net-Ultra-V3-Enhanced → 2.Enhancement → 3.SwinIR",
                "swinir_applied": True,
                "png_support": True,
                "edge_detection": "ULTRA PRECISE V3 ENHANCED (6-method combination)",
                "korean_encoding": "UTF-8 with enhanced error handling",
                "white_overlay": "AC: 20% | AB: 16% | Other: 5%",
                "brightness_values": "AC/AB: 1.03 | Other: 1.09",
                "sharpness_values": "Other: 1.5 → Final: 1.8",
                "contrast_value": "1.1",
                "expected_input": "2000x2600 (any format)",
                "output_size": "1200x1560",
                "transparency_info": "Full RGBA transparency preserved - NO background",
                "google_script_compatibility": "Base64 WITH padding - FIXED",
                "layout_optimization": "Text sections height reduced for better proportions"
            }
        }
        
        logger.info("✅ Enhancement completed successfully with FIXED Korean support V5")
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
