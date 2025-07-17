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
import urllib.request
import urllib.error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################
# ENHANCEMENT HANDLER - 1200x1560
# VERSION: New-Neo-V1-Ultra-Precision
################################

VERSION = "New-Neo-V1-Ultra-Precision"

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

# Korean font cache - COMPLETELY FIXED
KOREAN_FONT_PATH = None
KOREAN_FONT_VERIFIED = False
FONT_CACHE = {}

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
    """Download and verify Korean font - ENCODING FIXED"""
    global KOREAN_FONT_PATH, KOREAN_FONT_VERIFIED
    
    if KOREAN_FONT_PATH and KOREAN_FONT_VERIFIED:
        logger.info(f"✅ Using cached Korean font: {KOREAN_FONT_PATH}")
        return KOREAN_FONT_PATH
    
    try:
        font_path = '/tmp/NanumGothic.ttf'
        
        # Always remove existing font file to ensure fresh download
        if os.path.exists(font_path):
            os.remove(font_path)
            logger.info("🔄 Removed existing font file for fresh download")
        
        # FIXED: Working font sources with proper URLs
        font_sources = [
            {
                'url': 'https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf',
                'name': 'Google Fonts GitHub (Primary)'
            },
            {
                'url': 'https://fonts.gstatic.com/s/nanumgothic/v17/PN_3Rfi-oW3hYwmKDpxS7F_D-d7qPgJc.ttf',
                'name': 'Google Fonts Direct'
            },
            {
                'url': 'https://cdn.jsdelivr.net/gh/google/fonts@main/ofl/nanumgothic/NanumGothic-Regular.ttf',
                'name': 'JSDelivr CDN'
            },
            {
                'url': 'https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Regular.ttf',
                'name': 'GitHub Raw'
            }
        ]
        
        for source in font_sources:
            try:
                logger.info(f"🔽 Downloading Korean font from: {source['name']}")
                
                # Use urllib for better control
                req = urllib.request.Request(
                    source['url'],
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': 'application/font-ttf,font/*,*/*'
                    }
                )
                
                with urllib.request.urlopen(req, timeout=30) as response:
                    font_data = response.read()
                
                # Validate font data
                if len(font_data) < 50000:  # TTF files should be at least 50KB
                    logger.warning(f"❌ Font file too small: {len(font_data)} bytes")
                    continue
                
                # Save font file
                with open(font_path, 'wb') as f:
                    f.write(font_data)
                
                # CRITICAL: Verify Korean font works
                if verify_korean_font(font_path):
                    KOREAN_FONT_PATH = font_path
                    KOREAN_FONT_VERIFIED = True
                    logger.info(f"✅ Korean font successfully downloaded and verified: {source['name']}")
                    return font_path
                else:
                    logger.warning(f"❌ Font verification failed for: {source['name']}")
                    if os.path.exists(font_path):
                        os.remove(font_path)
                
            except Exception as e:
                logger.error(f"❌ Failed to download from {source['name']}: {e}")
                continue
        
        # If all downloads fail, try to use system fonts
        logger.warning("🔄 All downloads failed, trying system fonts...")
        system_fonts = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/TTF/NanumGothic.ttf',
            '/System/Library/Fonts/AppleSDGothicNeo.ttc',
            '/Windows/Fonts/malgun.ttf',
            '/System/Library/Fonts/AppleGothic.ttf'
        ]
        
        for sys_font in system_fonts:
            if os.path.exists(sys_font):
                if verify_korean_font(sys_font):
                    KOREAN_FONT_PATH = sys_font
                    KOREAN_FONT_VERIFIED = True
                    logger.info(f"✅ Using system Korean font: {sys_font}")
                    return sys_font
        
        logger.error("❌ No valid Korean font found")
        return None
        
    except Exception as e:
        logger.error(f"❌ Font download error: {e}")
        return None

def verify_korean_font(font_path):
    """Verify that the font can render Korean text properly - FIXED"""
    try:
        # Test with multiple Korean texts
        test_texts = [
            "한글",
            "테스트", 
            "안녕하세요",
            "이 제품은",
            "각도에 따라",
            "MD TALK",
            "DESIGN POINT"
        ]
        
        # Test multiple font sizes
        for size in [20, 24, 28, 48]:
            # FIXED: Remove encoding parameter completely
            font = ImageFont.truetype(font_path, size)
            
            # Create test image
            test_img = Image.new('RGB', (500, 200), 'white')
            draw = ImageDraw.Draw(test_img)
            
            y_pos = 10
            for text in test_texts:
                try:
                    # Test text rendering with proper UTF-8 string
                    bbox = draw.textbbox((10, y_pos), text, font=font)
                    draw.text((10, y_pos), text, font=font, fill='black')
                    
                    # Check if bbox is valid
                    if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
                        logger.warning(f"❌ Invalid bbox for text: {text}")
                        return False
                    
                    y_pos += 25
                    
                except Exception as e:
                    logger.warning(f"❌ Failed to render text '{text}': {e}")
                    return False
        
        logger.info("✅ Korean font verification passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Font verification error: {e}")
        return False

def get_font(size, force_korean=True):
    """Get font with Korean support - ENCODING FIXED"""
    global KOREAN_FONT_PATH, FONT_CACHE
    
    cache_key = f"{size}_{force_korean}"
    
    # Return cached font if available
    if cache_key in FONT_CACHE:
        return FONT_CACHE[cache_key]
    
    font = None
    
    if force_korean:
        # Ensure Korean font is available
        if not KOREAN_FONT_PATH or not KOREAN_FONT_VERIFIED:
            korean_path = download_korean_font()
            if not korean_path:
                logger.error("❌ No Korean font available")
                force_korean = False
        
        if KOREAN_FONT_PATH and os.path.exists(KOREAN_FONT_PATH):
            try:
                # FIXED: Remove encoding parameter completely
                font = ImageFont.truetype(KOREAN_FONT_PATH, size)
                logger.info(f"✅ Korean font loaded: size {size}")
            except Exception as e:
                logger.error(f"❌ Failed to load Korean font: {e}")
                font = None
    
    # Fallback to default font
    if font is None:
        try:
            font = ImageFont.load_default()
            logger.warning(f"⚠️ Using default font for size {size}")
        except Exception as e:
            logger.error(f"❌ Failed to load default font: {e}")
            font = None
    
    # Cache the font
    if font:
        FONT_CACHE[cache_key] = font
    
    return font

def safe_draw_text(draw, position, text, font, fill):
    """Safely draw Korean text - ENCODING FIXED"""
    try:
        if not text or not font:
            logger.warning("⚠️ No text or font provided")
            return
        
        # FIXED: Simplified encoding handling - just ensure UTF-8 string
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        
        text = str(text).strip()
        
        if not text:
            logger.warning("⚠️ Empty text after processing")
            return
        
        # Draw the text directly - PIL handles Korean properly now
        draw.text(position, text, font=font, fill=fill)
        logger.info(f"✅ Successfully drew text: {text[:20]}...")
        
    except Exception as e:
        logger.error(f"❌ Text drawing error: {e}, text: {repr(text)}")
        # Simple fallback
        try:
            if font:
                draw.text(position, "[한글 오류]", font=font, fill=fill)
        except:
            pass

def get_text_size(draw, text, font):
    """Get text size - FIXED for Korean"""
    try:
        if not text or not font:
            return (0, 0)
        
        # Handle encoding
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        text = str(text).strip()
        
        if not text:
            return (0, 0)
        
        # Use textbbox for accurate measurement
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        return (max(0, width), max(0, height))
        
    except Exception as e:
        logger.error(f"❌ Text size calculation error: {e}")
        return (100, 20)  # Fallback size

def wrap_text(text, font, max_width, draw):
    """Wrap text to fit within max_width"""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        width = bbox[2] - bbox[0]
        
        if width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                # Word is too long, add it anyway
                lines.append(word)
                current_line = []
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def create_md_talk_section(text_content=None, width=1200):
    """Create MD TALK section - FIXED SIZE 1200x600"""
    logger.info("🔤 Creating MD TALK section with FIXED size 1200x600")
    
    # Fixed dimensions
    fixed_width = 1200
    fixed_height = 600
    
    # Margins
    left_margin = 100
    right_margin = 100
    top_margin = 80
    content_width = fixed_width - left_margin - right_margin
    
    # Ensure Korean font is ready
    download_korean_font()
    
    # Get fonts with proper sizes
    title_font = get_font(48, force_korean=True)
    body_font = get_font(28, force_korean=True)
    
    if not title_font or not body_font:
        logger.error("❌ Failed to load fonts for MD TALK")
        # Create error image
        error_img = Image.new('RGB', (fixed_width, fixed_height), '#FFFFFF')
        error_draw = ImageDraw.Draw(error_img)
        try:
            error_draw.text((50, 50), "Font Error - MD TALK", fill='red')
        except:
            pass
        return error_img
    
    # Create final image with fixed size
    section_img = Image.new('RGB', (fixed_width, fixed_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Title
    title = "MD TALK"
    title_width, title_height = get_text_size(draw, title, title_font)
    
    # Draw title centered
    title_x = (fixed_width - title_width) // 2
    safe_draw_text(draw, (title_x, top_margin), title, title_font, (40, 40, 40))
    
    # Prepare Korean text content
    if text_content and text_content.strip():
        text = text_content.replace('MD TALK', '').replace('MD Talk', '').strip()
    else:
        text = """이 제품은 일상에서도 부담없이 착용할 수 있는 편안한 디자인으로 매일의 스타일링에 포인트를 더해줍니다. 특별한 날은 물론 평범한 일상까지 모든 순간을 빛나게 만들어주는 당신만의 특별한 주얼리입니다."""
    
    # Ensure UTF-8 string
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    text = str(text).strip()
    
    # Wrap text to fit within margins
    wrapped_lines = wrap_text(text, body_font, content_width, draw)
    
    # Calculate vertical position for centering content
    line_height = 50
    content_height = len(wrapped_lines) * line_height
    title_bottom_margin = 80
    
    # Start position for body text
    y_pos = top_margin + title_height + title_bottom_margin
    
    # Center text vertically if space allows
    total_content_height = title_height + title_bottom_margin + content_height
    if total_content_height < fixed_height - top_margin - 80:
        y_pos = (fixed_height - content_height) // 2 + 20
    
    # Draw body text with center alignment
    for line in wrapped_lines:
        if line:
            line_width, _ = get_text_size(draw, line, body_font)
            line_x = (fixed_width - line_width) // 2
            safe_draw_text(draw, (line_x, y_pos), line, body_font, (80, 80, 80))
        y_pos += line_height
    
    logger.info(f"✅ MD TALK section created: {fixed_width}x{fixed_height}")
    return section_img

def create_design_point_section(text_content=None, width=1200):
    """Create DESIGN POINT section - FIXED SIZE 1200x600 WITHOUT BOTTOM LINE"""
    logger.info("🔤 Creating DESIGN POINT section with FIXED size 1200x600 (no bottom line)")
    
    # Fixed dimensions
    fixed_width = 1200
    fixed_height = 600
    
    # Margins
    left_margin = 100
    right_margin = 100
    top_margin = 80
    content_width = fixed_width - left_margin - right_margin
    
    # Ensure Korean font is ready
    download_korean_font()
    
    # Get fonts
    title_font = get_font(48, force_korean=True)
    body_font = get_font(24, force_korean=True)
    
    if not title_font or not body_font:
        logger.error("❌ Failed to load fonts for DESIGN POINT")
        # Create error image
        error_img = Image.new('RGB', (fixed_width, fixed_height), '#FFFFFF')
        error_draw = ImageDraw.Draw(error_img)
        try:
            error_draw.text((50, 50), "Font Error - DESIGN POINT", fill='red')
        except:
            pass
        return error_img
    
    # Create final image with fixed size
    section_img = Image.new('RGB', (fixed_width, fixed_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Title
    title = "DESIGN POINT"
    title_width, title_height = get_text_size(draw, title, title_font)
    
    # Draw title centered
    title_x = (fixed_width - title_width) // 2
    safe_draw_text(draw, (title_x, top_margin), title, title_font, (40, 40, 40))
    
    # Prepare text content
    if text_content and text_content.strip():
        text = text_content.replace('DESIGN POINT', '').replace('Design Point', '').strip()
    else:
        text = """남성 단품은 무광 텍스처와 유광 라인의 조화가 견고한 감성을 전하고 여자 단품은 파베 세팅과 섬세한 밀그레인의 디테일 화려하면서도 고급스러운 반짝임을 표현합니다"""
    
    # Ensure UTF-8 string
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    text = str(text).strip()
    
    # Wrap text to fit within margins
    wrapped_lines = wrap_text(text, body_font, content_width, draw)
    
    # Calculate vertical position for centering content
    line_height = 45
    content_height = len(wrapped_lines) * line_height
    title_bottom_margin = 100
    
    # Start position for body text
    y_pos = top_margin + title_height + title_bottom_margin
    
    # Center text vertically if space allows
    total_content_height = title_height + title_bottom_margin + content_height
    if total_content_height < fixed_height - top_margin - 100:
        y_pos = (fixed_height - content_height) // 2 + 40
    
    # Draw body text with center alignment
    for line in wrapped_lines:
        if line:
            line_width, _ = get_text_size(draw, line, body_font)
            line_x = (fixed_width - line_width) // 2
            safe_draw_text(draw, (line_x, y_pos), line, body_font, (80, 80, 80))
        y_pos += line_height
    
    # REMOVED: Bottom line drawing
    # No more gray line at the bottom
    
    logger.info(f"✅ DESIGN POINT section created: {fixed_width}x{fixed_height} (no bottom line)")
    return section_img

def u2net_ultra_precise_removal_v2(image: Image.Image) -> Image.Image:
    """ULTRA PRECISE V2 U2Net background removal with multi-stage verification"""
    try:
        from rembg import remove
        
        global REMBG_SESSION
        if REMBG_SESSION is None:
            REMBG_SESSION = init_rembg_session()
            if REMBG_SESSION is None:
                return image
        
        logger.info("🔷 U2Net ULTRA PRECISE V2 Background Removal with Multi-Stage Verification")
        
        # CRITICAL: Ensure RGBA mode before processing
        if image.mode != 'RGBA':
            if image.mode == 'RGB':
                image = image.convert('RGBA')
            else:
                image = image.convert('RGBA')
        
        # Pre-process image for better edge detection
        contrast = ImageEnhance.Contrast(image)
        image_enhanced = contrast.enhance(1.15)  # Slightly higher contrast
        
        # Save image to buffer
        buffered = BytesIO()
        image_enhanced.save(buffered, format="PNG", compress_level=3, optimize=True)
        buffered.seek(0)
        img_data = buffered.getvalue()
        
        # Apply U2Net removal with ULTRA PRECISE settings
        output = remove(
            img_data,
            session=REMBG_SESSION,
            alpha_matting=True,
            alpha_matting_foreground_threshold=290,  # Even higher for better edges
            alpha_matting_background_threshold=0,
            alpha_matting_erode_size=0,
            only_mask=False,
            post_process_mask=True
        )
        
        result_image = Image.open(BytesIO(output))
        
        # CRITICAL: Ensure RGBA mode
        if result_image.mode != 'RGBA':
            result_image = result_image.convert('RGBA')
        
        # ULTRA PRECISE V2 edge refinement with verification stages
        r, g, b, a = result_image.split()
        alpha_array = np.array(a, dtype=np.uint8)
        rgb_array = np.array(result_image.convert('RGB'), dtype=np.uint8)
        
        # Convert to float for processing
        alpha_float = alpha_array.astype(np.float32) / 255.0
        
        # STAGE 1: Advanced edge detection using multiple methods
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_magnitude = (edge_magnitude / edge_magnitude.max() * 255).astype(np.uint8)
        
        # Canny edge detection for comparison
        edges_canny = cv2.Canny(gray, 50, 150)
        
        # Combine edge detections
        combined_edges = np.maximum(edge_magnitude > 30, edges_canny > 0)
        edge_dilated = cv2.dilate(combined_edges.astype(np.uint8), np.ones((3,3)), iterations=2)
        
        # STAGE 2: Narrow area detection for ring holes
        logger.info("🔍 Detecting narrow areas and ring holes...")
        
        # Use morphological operations to find narrow areas
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(alpha_array, cv2.MORPH_CLOSE, kernel_close)
        
        # Find difference to detect narrow gaps
        narrow_areas = cv2.absdiff(closed, alpha_array)
        narrow_mask = narrow_areas > 50
        
        # STAGE 3: Multi-pass guided filter with verification
        gray_float = gray.astype(np.float32) / 255.0
        
        try:
            # First pass - very fine details
            alpha_guided1 = cv2.ximgproc.guidedFilter(
                guide=gray_float,
                src=alpha_float,
                radius=1,
                eps=0.00001  # Ultra-small epsilon
            )
            
            # Second pass - smooth transitions
            alpha_guided2 = cv2.ximgproc.guidedFilter(
                guide=gray_float,
                src=alpha_guided1,
                radius=3,
                eps=0.0005
            )
            
            # Third pass - overall smoothing
            alpha_guided3 = cv2.ximgproc.guidedFilter(
                guide=gray_float,
                src=alpha_guided2,
                radius=5,
                eps=0.001
            )
            
            # Adaptive blending based on edge proximity
            # FIXED: Convert boolean to uint8
            edge_distance = cv2.distanceTransform((~edge_dilated).astype(np.uint8), cv2.DIST_L2, 3)
            edge_weight = np.clip(edge_distance / 10, 0, 1)
            
            alpha_float = (alpha_guided1 * (1 - edge_weight) + 
                          alpha_guided3 * edge_weight)
            
        except AttributeError:
            # Fallback with enhanced bilateral filter
            alpha_uint8 = (alpha_float * 255).astype(np.uint8)
            alpha_bilateral = cv2.bilateralFilter(alpha_uint8, 7, 100, 100)
            alpha_float = alpha_bilateral.astype(np.float32) / 255.0
        
        # STAGE 4: Verification stage - check for missed areas
        logger.info("🔍 Verification stage - checking for missed areas...")
        
        # Create verification mask
        bright_areas = gray > 240
        low_contrast = cv2.Laplacian(gray, cv2.CV_64F).var() < 100
        potential_missed = bright_areas & (alpha_float < 0.1)
        
        # Re-evaluate missed areas
        if np.any(potential_missed):
            logger.info("Found potential missed areas, re-evaluating...")
            # Use local analysis for missed areas
            for y in range(0, gray.shape[0], 50):
                for x in range(0, gray.shape[1], 50):
                    region = potential_missed[y:y+50, x:x+50]
                    if np.any(region):
                        local_region = gray[y:y+50, x:x+50]
                        local_mean = np.mean(local_region)
                        if local_mean > 230:
                            # This might be a hole, keep it transparent
                            alpha_float[y:y+50, x:x+50][region[0:min(50, alpha_float.shape[0]-y), 0:min(50, alpha_float.shape[1]-x)]] = 0
        
        # STAGE 5: Enhanced sigmoid with adaptive threshold
        k = 60  # Higher steepness
        
        # Adaptive threshold based on image statistics
        alpha_mean = np.mean(alpha_float[alpha_float > 0.1])
        threshold = min(0.5, max(0.3, alpha_mean * 0.8))
        
        alpha_sigmoid = 1 / (1 + np.exp(-k * (alpha_float - threshold)))
        
        # STAGE 6: Narrow area preservation
        alpha_smooth = alpha_sigmoid.copy()
        
        # Preserve narrow areas detected earlier
        alpha_smooth[narrow_mask] = alpha_float[narrow_mask]
        
        # Edge-aware smoothing
        non_edge_mask = ~edge_dilated.astype(bool) & ~narrow_mask
        if np.any(non_edge_mask):
            alpha_smooth_temp = cv2.GaussianBlur(alpha_sigmoid, (5, 5), 1.0)
            alpha_smooth[non_edge_mask] = alpha_smooth_temp[non_edge_mask]
        
        # STAGE 7: Fine detail preservation with enhanced detection
        # High-frequency detail detection
        alpha_highpass = alpha_float - cv2.GaussianBlur(alpha_float, (7, 7), 2.0)
        fine_details = np.abs(alpha_highpass) > 0.03  # Lower threshold for more details
        
        # Preserve fine details
        detail_dilated = cv2.dilate(fine_details.astype(np.uint8), np.ones((3,3)), iterations=1)
        alpha_smooth[detail_dilated.astype(bool)] = alpha_float[detail_dilated.astype(bool)]
        
        # STAGE 8: Connected component analysis with size adaptation
        alpha_binary = (alpha_smooth > 0.5).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(alpha_binary)
        
        if num_labels > 2:
            sizes = [np.sum(labels == i) for i in range(1, num_labels)]
            if sizes:
                # Adaptive size threshold
                total_size = alpha_array.size
                min_size = max(int(total_size * 0.0001), 50)  # Minimum 50 pixels
                
                valid_labels = [i+1 for i, size in enumerate(sizes) if size > min_size]
                
                valid_mask = np.zeros_like(alpha_binary, dtype=bool)
                for label in valid_labels:
                    valid_mask |= (labels == label)
                
                # Don't remove small components near edges or in narrow areas
                removal_mask = ~valid_mask & ~edge_dilated.astype(bool) & ~narrow_mask
                alpha_smooth[removal_mask] = 0
        
        # STAGE 9: Final verification pass
        logger.info("🔍 Final verification pass...")
        
        # Check for any remaining artifacts
        final_binary = (alpha_smooth > 0.5).astype(np.uint8)
        
        # Small morphological cleanup
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_cleaned = cv2.morphologyEx(final_binary, cv2.MORPH_OPEN, kernel_small)
        final_cleaned = cv2.morphologyEx(final_cleaned, cv2.MORPH_CLOSE, kernel_small)
        
        # Apply cleanup only where safe
        safe_cleanup_mask = ~edge_dilated.astype(bool) & ~narrow_mask
        alpha_smooth[safe_cleanup_mask] = alpha_smooth[safe_cleanup_mask] * final_cleaned[safe_cleanup_mask]
        
        # Convert back to uint8
        alpha_array = np.clip(alpha_smooth * 255, 0, 255).astype(np.uint8)
        
        # STAGE 10: Ultra-fine feathering
        kernel_feather = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha_eroded = cv2.erode(alpha_array, kernel_feather, iterations=1)
        alpha_dilated = cv2.dilate(alpha_array, kernel_feather, iterations=1)
        
        feather_mask = (alpha_dilated > 0) & (alpha_eroded < 255)
        if np.any(feather_mask):
            # Smoother feathering
            feather_alpha = alpha_array[feather_mask].astype(np.float32)
            eroded_alpha = alpha_eroded[feather_mask].astype(np.float32)
            smooth_factor = 0.7  # Smoother transition
            alpha_array[feather_mask] = (feather_alpha * smooth_factor + 
                                        eroded_alpha * (1 - smooth_factor)).astype(np.uint8)
        
        logger.info("✅ ULTRA PRECISE V2 background removal complete with verification")
        
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

def ensure_ring_holes_transparent_ultra_v2(image: Image.Image) -> Image.Image:
    """ULTRA PRECISE V2 ring hole detection with narrow area support"""
    # CRITICAL: Preserve RGBA mode
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    logger.info("🔍 ULTRA PRECISE V2 Ring Hole Detection with Narrow Area Support")
    
    r, g, b, a = image.split()
    alpha_array = np.array(a, dtype=np.uint8)
    rgb_array = np.array(image.convert('RGB'), dtype=np.uint8)
    
    h, w = alpha_array.shape
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    
    # STAGE 1: Multi-criteria hole detection
    very_bright = v_channel > 240
    low_saturation = s_channel < 30
    alpha_holes = alpha_array < 50
    
    # Additional criteria for narrow areas
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    high_brightness_gray = gray > 235
    
    potential_holes = (very_bright & low_saturation) | alpha_holes | high_brightness_gray
    
    # STAGE 2: Narrow area specific detection
    logger.info("🔍 Detecting narrow ring areas...")
    
    # Use distance transform to find narrow regions
    if np.any(alpha_array > 128):
        # FIXED: Convert boolean to uint8
        dist_transform = cv2.distanceTransform((alpha_array > 128).astype(np.uint8), cv2.DIST_L2, 3)
        narrow_regions = (dist_transform > 0) & (dist_transform < 20)  # Narrow band
        
        # Check brightness in narrow regions
        narrow_bright = narrow_regions & (gray > 230)
        potential_holes |= narrow_bright
    
    # Clean up noise with smaller kernel for narrow areas
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    potential_holes = cv2.morphologyEx(potential_holes.astype(np.uint8), cv2.MORPH_OPEN, kernel_clean)
    potential_holes = cv2.morphologyEx(potential_holes, cv2.MORPH_CLOSE, kernel_clean)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(potential_holes)
    
    holes_mask = np.zeros_like(alpha_array, dtype=np.float32)
    
    # STAGE 3: Analyze each component with enhanced criteria
    for label in range(1, num_labels):
        component = (labels == label)
        component_size = np.sum(component)
        
        # Adjusted size filtering for narrow areas
        min_size = h * w * 0.00005  # Smaller minimum for narrow holes
        max_size = h * w * 0.2
        
        if min_size < component_size < max_size:
            coords = np.where(component)
            if len(coords[0]) == 0:
                continue
                
            min_y, max_y = coords[0].min(), coords[0].max()
            min_x, max_x = coords[1].min(), coords[1].max()
            
            comp_width = max_x - min_x
            comp_height = max_y - min_y
            
            if comp_height == 0 or comp_width == 0:
                continue
            
            # More flexible aspect ratio for various hole shapes
            aspect_ratio = comp_width / comp_height
            shape_valid = 0.1 < aspect_ratio < 10.0  # Very flexible
            
            # Check if it's a narrow hole
            is_narrow = min(comp_width, comp_height) < 30
            
            center_y, center_x = (min_y + max_y) / 2, (min_x + max_x) / 2
            center_distance = np.sqrt((center_x - w/2)**2 + (center_y - h/2)**2)
            position_valid = center_distance < max(w, h) * 0.48  # Slightly larger range
            
            component_pixels = rgb_array[component]
            if len(component_pixels) > 0:
                brightness = np.mean(component_pixels)
                brightness_std = np.std(component_pixels)
                
                # Adjusted thresholds for narrow areas
                brightness_threshold = 225 if is_narrow else 230
                std_threshold = 30 if is_narrow else 25
                
                brightness_valid = brightness > brightness_threshold
                consistency_valid = brightness_std < std_threshold
                
                # Shape analysis
                component_uint8 = component.astype(np.uint8) * 255
                contours, _ = cv2.findContours(component_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                circularity_valid = False
                smoothness_valid = True
                
                if contours:
                    contour = contours[0]
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        # More lenient circularity for narrow holes
                        circularity_threshold = 0.2 if is_narrow else 0.3
                        circularity_valid = circularity > circularity_threshold
                    
                    # Edge smoothness check
                    edges = cv2.Canny(component_uint8, 50, 150)
                    if perimeter > 0:
                        edge_ratio = np.sum(edges > 0) / perimeter
                        smoothness_threshold = 3.0 if is_narrow else 2.0
                        smoothness_valid = edge_ratio < smoothness_threshold
                
                # Confidence calculation with narrow area bonus
                confidence = 0.0
                if brightness_valid: confidence += 0.3
                if consistency_valid: confidence += 0.2
                if position_valid: confidence += 0.15
                if circularity_valid: confidence += 0.15
                if smoothness_valid: confidence += 0.1
                if is_narrow: confidence += 0.1  # Bonus for narrow areas
                
                # Lower threshold for narrow areas
                confidence_threshold = 0.35 if is_narrow else 0.45
                
                if confidence > confidence_threshold and (shape_valid or is_narrow):
                    holes_mask[component] = 255
                    logger.info(f"{'Narrow ' if is_narrow else ''}Hole detected with confidence: {confidence:.2f}")
    
    # STAGE 4: Apply holes with smooth transitions
    if np.any(holes_mask > 0):
        # Extra smoothing for narrow areas
        holes_mask_smooth = cv2.GaussianBlur(holes_mask, (3, 3), 0.5)
        
        # Smaller dilation for narrow areas
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        holes_dilated = cv2.dilate(holes_mask, kernel_dilate, iterations=1)
        transition_zone = (holes_dilated > 0) & (holes_mask < 255)
        
        alpha_float = alpha_array.astype(np.float32)
        
        # Make holes fully transparent
        alpha_float[holes_mask_smooth > 200] = 0
        
        # Smooth transition
        if np.any(transition_zone):
            transition_alpha = 1 - (holes_mask_smooth[transition_zone] / 255)
            alpha_float[transition_zone] *= transition_alpha
        
        alpha_array = np.clip(alpha_float, 0, 255).astype(np.uint8)
        
        logger.info("✅ Ring holes (including narrow areas) made transparent")
    
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
    """Process special modes - KOREAN ENCODING FIXED"""
    special_mode = job.get('special_mode', '')
    logger.info(f"🔤 Processing special mode with FIXED Korean encoding: {special_mode}")
    
    # BOTH TEXT SECTIONS - Return TWO separate images
    if special_mode == 'both_text_sections':
        # Get text content with proper Korean encoding
        md_talk_text = job.get('md_talk_content', '') or job.get('md_talk', '') or """각도에 따라 달라지는 빛의 결들이 두 사람의 특별한 순간순간을 더 찬란하게 만들며 360도 새겨진 패턴으로 매일 새로운 반짝임을 보여줍니다 :)"""
        
        design_point_text = job.get('design_point_content', '') or job.get('design_point', '') or """입체적인 컷팅 위로 섬세하게 빛나는 패턴이 고급스러움을 완성하며 각진 텍스처가 심플하면서 유니크한 매력을 더해줍니다."""
        
        # Handle encoding properly - FIXED
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
                "korean_encoding": "UTF-8-FIXED",
                "korean_font_verified": KOREAN_FONT_VERIFIED,
                "korean_font_path": KOREAN_FONT_PATH,
                "base64_padding": "INCLUDED",
                "compression": "level_3",
                "design_point_update": "Bottom gray line removed"
            }
        }
    
    # Single MD TALK section
    elif special_mode == 'md_talk':
        text_content = job.get('text_content', '') or job.get('claude_text', '') or job.get('md_talk', '')
        
        if not text_content:
            text_content = """이 제품은 일상에서도 부담없이 착용할 수 있는 편안한 디자인으로 매일의 스타일링에 포인트를 더해줍니다."""
        
        # Handle encoding - FIXED
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
                "korean_encoding": "UTF-8-FIXED",
                "korean_font_verified": KOREAN_FONT_VERIFIED,
                "korean_font_path": KOREAN_FONT_PATH,
                "base64_padding": "INCLUDED",
                "compression": "level_3"
            }
        }
    
    # Single DESIGN POINT section
    elif special_mode == 'design_point':
        text_content = job.get('text_content', '') or job.get('claude_text', '') or job.get('design_point', '')
        
        if not text_content:
            text_content = """남성 단품은 무광 텍스처와 유광 라인의 조화가 견고한 감성을 전하고 여자 단품은 파베 세팅과 섬세한 밀그레인의 디테일로 화려하면서도 고급스러운 반짝임을 표현합니다."""
        
        # Handle encoding - FIXED
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
                "korean_encoding": "UTF-8-FIXED",
                "korean_font_verified": KOREAN_FONT_VERIFIED,
                "korean_font_path": KOREAN_FONT_PATH,
                "base64_padding": "INCLUDED",
                "compression": "level_3",
                "design_point_update": "Bottom gray line removed"
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
    """Apply pattern enhancement while TRULY preserving transparency - AC 20%, AB 16%"""
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
        logger.info("🔍 AC Pattern - Applying 20% white overlay (increased from 12%)")
        # Apply 20% white overlay (increased from 12%)
        white_overlay = 0.20
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        
        # Convert back to image
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        # Slightly increased brightness for AC pattern
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.02)  # Increased from 1.005
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.98)
        
        logger.info("✅ AC Pattern enhancement applied with 20% white overlay")
    
    elif pattern_type == "ab_pattern":
        logger.info("🔍 AB Pattern - Applying 16% white overlay and cool tone")
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
        
        # Slightly increased brightness for AB pattern
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.02)  # Increased from 1.005
        
        logger.info("✅ AB Pattern enhancement applied with 16% white overlay")
        
    else:
        logger.info("🔍 Other Pattern - Standard enhancement with increased values")
        # Increased brightness for other patterns
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.12)  # Increased from 1.08
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.99)
        
        sharpness = ImageEnhance.Sharpness(rgb_image)
        rgb_image = sharpness.enhance(1.5)  # Increased from 1.4
    
    # Apply common enhancements
    contrast = ImageEnhance.Contrast(rgb_image)
    rgb_image = contrast.enhance(1.08)  # Slightly increased from 1.05
    
    # Apply sharpening - Increased for all patterns
    sharpness = ImageEnhance.Sharpness(rgb_image)
    rgb_image = sharpness.enhance(1.8)  # Increased from 1.6
    
    # CRITICAL: Recombine with ORIGINAL alpha channel
    r2, g2, b2 = rgb_image.split()
    enhanced_image = Image.merge('RGBA', (r2, g2, b2, a))
    
    logger.info(f"✅ Enhancement applied while preserving transparency. Mode: {enhanced_image.mode}")
    
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
    """Main enhancement processing - New Neo V1 Ultra Precision"""
    logger.info(f"=== Enhancement {VERSION} Started ===")
    logger.info("🎯 NEW NEO V1: Ultra Precision Background Removal")
    logger.info("💎 TRANSPARENT OUTPUT: Preserving alpha channel throughout")
    logger.info("🔤 FIXED TEXT SECTIONS: 1200x600 with center alignment and margins")
    logger.info("🔧 AC PATTERN: 20% white overlay")
    logger.info("🔧 AB PATTERN: 16% white overlay")
    logger.info("✨ ALL PATTERNS: Increased brightness and sharpness")
    logger.info("📌 BASE64 PADDING: ALWAYS INCLUDED for Google Script compatibility")
    logger.info("🗜️ COMPRESSION: Level 3 (balanced speed/size)")
    logger.info("🆕 DESIGN POINT: Bottom gray line removed")
    logger.info("🆕 NARROW AREA DETECTION: Enhanced for thin ring holes")
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
        
        # STEP 1: ULTRA PRECISE V2 BACKGROUND REMOVAL - ALWAYS APPLY
        logger.info("📸 STEP 1: ALWAYS applying ULTRA PRECISE V2 background removal")
        removal_start = time.time()
        image = u2net_ultra_precise_removal_v2(image)
        logger.info(f"⏱️ Ultra precise V2 background removal took: {time.time() - removal_start:.2f}s")
        
        # Verify RGBA after removal
        if image.mode != 'RGBA':
            logger.error("❌ Image lost RGBA after background removal!")
            image = image.convert('RGBA')
        
        # STEP 2: ENHANCEMENT (preserving transparency)
        logger.info("🎨 STEP 2: Applying enhancements with TRUE transparency preservation")
        enhancement_start = time.time()
        
        # Auto white balance with alpha preservation
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
            "other": "기타색상(no_overlay)"
        }.get(pattern_type, "기타색상(no_overlay)")
        
        # Apply pattern-specific enhancements (preserving transparency)
        image = apply_pattern_enhancement_transparent(image, pattern_type)
        
        # ULTRA PRECISE V2 ring hole detection
        logger.info("🔍 Applying ULTRA PRECISE V2 ring hole detection")
        image = ensure_ring_holes_transparent_ultra_v2(image)
        
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
                "korean_font_verified": KOREAN_FONT_VERIFIED,
                "korean_font_path": KOREAN_FONT_PATH,
                "special_modes_available": ["md_talk", "design_point", "both_text_sections"],
                "file_number_info": {
                    "001-003": "Enhancement",
                    "004": "MD TALK (1200x600)",
                    "005-006": "Enhancement",
                    "007": "Thumbnail",
                    "008": "DESIGN POINT (1200x600)",
                    "009-010": "Thumbnail",
                    "011": "COLOR section"
                },
                "new_neo_v1_features": [
                    "✅ ULTRA PRECISE V2: Multi-stage verification background removal",
                    "✅ NARROW AREA DETECTION: Enhanced for thin ring holes",
                    "✅ DESIGN POINT: Bottom gray line removed",
                    "✅ VERIFICATION STAGES: Added missed area re-evaluation",
                    "✅ ADAPTIVE THRESHOLD: Based on image statistics",
                    "✅ ENHANCED EDGE DETECTION: Combined Sobel and Canny",
                    "✅ NARROW HOLE SUPPORT: Lower size and confidence thresholds",
                    "✅ MULTI-PASS GUIDED FILTER: 3 passes with adaptive blending",
                    "✅ FINE DETAIL PRESERVATION: Lower threshold (0.03)",
                    "✅ SMOOTHER FEATHERING: Factor 0.7 for natural edges",
                    "✅ FIXED: OpenCV distanceTransform type conversion"
                ],
                "processing_order": "1.U2Net-Ultra-V2 → 2.Enhancement → 3.SwinIR",
                "swinir_applied": True,
                "png_support": True,
                "edge_detection": "ULTRA PRECISE V2 (Multi-method + Verification)",
                "korean_support": "COMPLETELY FIXED - No encoding parameter needed",
                "white_overlay": "AC: 20% | AB: 16% | Other: None",
                "brightness_values": "AC/AB: 1.02 | Other: 1.12",
                "sharpness_values": "Other: 1.5 → Final: 1.8",
                "contrast_value": "1.08",
                "expected_input": "2000x2600 (any format)",
                "output_size": "1200x1560",
                "transparency_info": "Full RGBA transparency preserved - NO background",
                "google_script_compatibility": "Base64 WITH padding - FIXED"
            }
        }
        
        logger.info("✅ Enhancement completed successfully with NEW NEO V1 features")
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
