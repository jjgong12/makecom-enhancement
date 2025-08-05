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
# VERSION: Enhancement-V3-TwoPhase-KoreanFixed
################################

VERSION = "Enhancement-V3-TwoPhase-KoreanFixed"

# Global rembg session with U2Net
REMBG_SESSION = None

# Korean font cache
KOREAN_FONT_PATH = None
FONT_CACHE = {}
DEFAULT_FONT_CACHE = {}

def init_rembg_session():
    """Initialize rembg session with U2Net for faster processing"""
    global REMBG_SESSION
    if REMBG_SESSION is None:
        try:
            from rembg import new_session
            # Use U2Net for faster processing
            REMBG_SESSION = new_session('u2net')
            logger.info("✅ U2Net session initialized")
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
        return KOREAN_FONT_PATH
    
    try:
        # Try multiple font options
        font_options = [
            {
                'path': '/tmp/NotoSansKR-Regular.ttf',
                'urls': [
                    'https://github.com/google/fonts/raw/main/ofl/notosanskr/NotoSansKR-Regular.ttf',
                    'https://cdn.jsdelivr.net/gh/google/fonts@main/ofl/notosanskr/NotoSansKR-Regular.ttf'
                ]
            },
            {
                'path': '/tmp/NanumGothic.ttf',
                'urls': [
                    'https://github.com/naver/nanumfont/raw/master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf',
                    'https://cdn.jsdelivr.net/gh/naver/nanumfont@master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf'
                ]
            },
            {
                'path': '/tmp/MalgunGothic.ttf',
                'urls': [
                    'https://github.com/codejamninja/fonts/raw/master/MalgunGothic.ttf'
                ]
            }
        ]
        
        for font_option in font_options:
            font_path = font_option['path']
            
            # Check if already exists
            if os.path.exists(font_path):
                try:
                    # Test font with Korean text
                    test_font = ImageFont.truetype(font_path, 24)
                    test_img = Image.new('RGB', (100, 50), 'white')
                    test_draw = ImageDraw.Draw(test_img)
                    test_draw.text((10, 10), "한글", font=test_font, fill='black')
                    
                    KOREAN_FONT_PATH = font_path
                    logger.info(f"✅ Korean font loaded from cache: {font_path}")
                    return font_path
                except Exception as e:
                    logger.error(f"Cached font verification failed: {e}")
                    os.remove(font_path)
                    continue
            
            # Try downloading
            for url in font_option['urls']:
                try:
                    logger.info(f"Downloading font from: {url}")
                    response = requests.get(url, timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
                    
                    if response.status_code == 200 and len(response.content) > 100000:
                        with open(font_path, 'wb') as f:
                            f.write(response.content)
                        
                        # Verify font with Korean text
                        test_font = ImageFont.truetype(font_path, 24)
                        test_img = Image.new('RGB', (100, 50), 'white')
                        test_draw = ImageDraw.Draw(test_img)
                        test_draw.text((10, 10), "한글테스트", font=test_font, fill='black')
                        
                        KOREAN_FONT_PATH = font_path
                        logger.info(f"✅ Korean font downloaded and verified: {font_path}")
                        return font_path
                except Exception as e:
                    logger.error(f"Font download attempt failed: {e}")
                    continue
        
        logger.error("❌ All font download attempts failed")
        return None
        
    except Exception as e:
        logger.error(f"❌ Font error: {e}")
        return None

def get_font(size, force_korean=True):
    """Get font with Korean support - IMPROVED"""
    global KOREAN_FONT_PATH, FONT_CACHE
    
    cache_key = f"{size}_{force_korean}"
    if cache_key in FONT_CACHE:
        return FONT_CACHE[cache_key]
    
    font = None
    
    if force_korean:
        # Try to download Korean font if not available
        if not KOREAN_FONT_PATH:
            download_korean_font()
        
        if KOREAN_FONT_PATH and os.path.exists(KOREAN_FONT_PATH):
            try:
                font = ImageFont.truetype(KOREAN_FONT_PATH, size)
                logger.info(f"✅ Korean font loaded size={size}")
            except Exception as e:
                logger.error(f"❌ Korean font loading failed: {e}")
                font = None
    
    if font is None:
        # Try system fonts with better Korean support
        system_fonts = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
        
        for font_path in system_fonts:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, size)
                    logger.info(f"✅ System font loaded: {font_path}")
                    break
                except:
                    continue
        
        if font is None:
            # Last resort - use default font but warn about Korean
            font = ImageFont.load_default()
            logger.warning("⚠️ Using default font - Korean will not display properly!")
    
    FONT_CACHE[cache_key] = font
    return font

def safe_draw_text(draw, position, text, font, fill):
    """Safely draw Korean text with better error handling"""
    try:
        if not text:
            return
        
        # Ensure text is properly encoded UTF-8
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        else:
            text = str(text)
        
        # Normalize unicode
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        
        text = text.strip()
        if not text:
            return
        
        # Try to draw text
        draw.text(position, text, font=font, fill=fill)
        
    except Exception as e:
        logger.error(f"Text drawing error: {e}")
        # Fallback - try to draw with ASCII only
        try:
            ascii_text = text.encode('ascii', 'replace').decode('ascii')
            draw.text(position, ascii_text, font=font, fill=fill)
        except Exception as e2:
            logger.error(f"Fallback text drawing also failed: {e2}")

def get_text_size(draw, text, font):
    """Get text size with compatibility for different PIL versions"""
    try:
        if not text or not font:
            return (0, 0)
        
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        
        # Normalize unicode
        import unicodedata
        text = unicodedata.normalize('NFC', str(text)).strip()
        
        if not text:
            return (0, 0)
        
        # Try new method first (PIL 8+)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        except AttributeError:
            # Fall back to old method
            return draw.textsize(text, font=font)
    except Exception as e:
        logger.error(f"Error getting text size: {e}")
        # Return reasonable default size based on character count
        return (len(text) * 20, 30)

def wrap_text(text, font, max_width, draw):
    """Wrap text to fit within max_width with better handling"""
    if not text or not font:
        return []
    
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    
    # Normalize unicode
    import unicodedata
    text = unicodedata.normalize('NFC', str(text)).strip()
    
    lines = []
    paragraphs = text.split('\n')
    
    for paragraph in paragraphs:
        if not paragraph:
            lines.append('')
            continue
        
        # For Korean text, we might need to split by characters instead of words
        words = paragraph.split()
        if not words:
            # If no spaces (common in Korean), split by characters
            words = list(paragraph)
        
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word]) if current_line else word
            try:
                width, _ = get_text_size(draw, test_line, font)
            except:
                width = len(test_line) * 20  # Fallback estimate
            
            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line) if len(current_line) > 1 else current_line[0])
                    current_line = [word]
                else:
                    # Word is too long, add it anyway
                    lines.append(word)
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line) if len(current_line) > 1 else current_line[0])
    
    return lines

def create_md_talk_section(text_content=None, width=1200):
    """Create MD TALK section with FIXED Korean support"""
    logger.info("🔤 Creating MD TALK section with FIXED Korean support")
    
    # Ensure Korean font is available
    download_korean_font()
    
    fixed_width = 1200
    fixed_height = 400
    
    left_margin = 100
    right_margin = 100
    top_margin = 60
    content_width = fixed_width - left_margin - right_margin
    
    # Get fonts with better error handling
    title_font = get_font(48, force_korean=True)
    body_font = get_font(28, force_korean=True)
    
    # Create white background image
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
    
    # Text content
    if text_content and text_content.strip():
        # Clean text - remove title if included
        text = text_content.replace('MD TALK', '').replace('MD Talk', '').strip()
    else:
        text = """이 제품은 일상에서도 부담없이 착용할 수 있는 편안한 디자인으로 매일의 스타일링에 포인트를 더해줍니다. 특별한 날은 물론 평범한 일상까지 모든 순간을 빛나게 만들어주는 당신만의 특별한 주얼리입니다."""
    
    # Ensure text is properly encoded
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    
    # Normalize unicode
    import unicodedata
    text = unicodedata.normalize('NFC', str(text)).strip()
    
    logger.info(f"MD TALK text (first 50 chars): {text[:50]}...")
    logger.info(f"Text encoding: UTF-8, Length: {len(text)}")
    
    # Wrap text
    wrapped_lines = wrap_text(text, body_font, content_width, draw)
    
    # Calculate positions
    line_height = 45
    title_bottom_margin = 60
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
            
            # Prevent text overflow
            if y_pos > fixed_height - 50:
                break
    
    logger.info(f"✅ MD TALK section created: {fixed_width}x{fixed_height}")
    return section_img

def create_design_point_section(text_content=None, width=1200):
    """Create DESIGN POINT section with FIXED Korean support"""
    logger.info("🔤 Creating DESIGN POINT section with FIXED Korean support")
    
    # Ensure Korean font is available
    download_korean_font()
    
    fixed_width = 1200
    fixed_height = 350
    
    left_margin = 100
    right_margin = 100
    top_margin = 60
    content_width = fixed_width - left_margin - right_margin
    
    # Get fonts with better error handling
    title_font = get_font(48, force_korean=True)
    body_font = get_font(24, force_korean=True)
    
    # Create white background image
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
    
    # Text content
    if text_content and text_content.strip():
        # Clean text - remove title if included
        text = text_content.replace('DESIGN POINT', '').replace('Design Point', '').strip()
    else:
        text = """남성 단품은 무광 텍스처와 유광 라인의 조화가 견고한 감성을 전하고 여자 단품은 파베 세팅과 섬세한 밀그레인의 디테일 화려하면서도 고급스러운 반짝임을 표현합니다"""
    
    # Ensure text is properly encoded
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    
    # Normalize unicode
    import unicodedata
    text = unicodedata.normalize('NFC', str(text)).strip()
    
    logger.info(f"DESIGN POINT text (first 50 chars): {text[:50]}...")
    logger.info(f"Text encoding: UTF-8, Length: {len(text)}")
    
    # Wrap text
    wrapped_lines = wrap_text(text, body_font, content_width, draw)
    
    # Calculate positions
    line_height = 40
    title_bottom_margin = 70
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
            
            # Prevent text overflow
            if y_pos > fixed_height - 50:
                break
    
    logger.info(f"✅ DESIGN POINT section created: {fixed_width}x{fixed_height}")
    return section_img

# The rest of the code remains the same...
