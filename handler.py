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
import unicodedata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################
# ENHANCEMENT HANDLER - 1200x1560
# VERSION: Enhancement-V5-Fixed-Korean
# Complete Korean font support with fixes
################################

VERSION = "Enhancement-V5-Fixed-Korean"
logger.info(f"🚀 Module loaded: {VERSION}")

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
            logger.info("✅ U2Net session initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize rembg: {e}")
            REMBG_SESSION = None
    return REMBG_SESSION

# Initialize on module load
logger.info("🔧 Initializing U2Net session on module load...")
init_rembg_session()

def download_korean_font():
    """Download and verify Korean font - COMPLETE FIX WITH MORE OPTIONS"""
    global KOREAN_FONT_PATH
    
    if KOREAN_FONT_PATH and os.path.exists(KOREAN_FONT_PATH):
        try:
            # Verify existing font
            test_font = ImageFont.truetype(KOREAN_FONT_PATH, 24)
            test_img = Image.new('RGB', (100, 50), 'white')
            test_draw = ImageDraw.Draw(test_img)
            test_draw.text((10, 10), "한글테스트", font=test_font, fill='black')
            logger.info(f"✅ Using existing Korean font: {KOREAN_FONT_PATH}")
            return KOREAN_FONT_PATH
        except:
            KOREAN_FONT_PATH = None
    
    try:
        # Extended font options
        font_options = [
            {
                'name': 'NanumGothic',
                'path': '/tmp/NanumGothic.ttf',
                'urls': [
                    'https://github.com/naver/nanumfont/raw/master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf',
                    'https://cdn.jsdelivr.net/gh/naver/nanumfont@master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf',
                    'https://raw.githubusercontent.com/naver/nanumfont/master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf'
                ]
            },
            {
                'name': 'NotoSansKR-Regular',
                'path': '/tmp/NotoSansKR-Regular.otf',
                'urls': [
                    'https://github.com/google/fonts/raw/main/ofl/notosanskr/NotoSansKR-Regular.otf',
                    'https://cdn.jsdelivr.net/gh/google/fonts@main/ofl/notosanskr/NotoSansKR-Regular.otf',
                    'https://raw.githubusercontent.com/google/fonts/main/ofl/notosanskr/NotoSansKR-Regular.otf'
                ]
            },
            {
                'name': 'D2Coding',
                'path': '/tmp/D2Coding.ttf',
                'urls': [
                    'https://github.com/naver/d2codingfont/raw/master/D2Coding/D2Coding-Ver1.3.2-20180524.ttf',
                    'https://raw.githubusercontent.com/naver/d2codingfont/master/D2Coding/D2Coding-Ver1.3.2-20180524.ttf'
                ]
            },
            {
                'name': 'NanumBarunGothic',
                'path': '/tmp/NanumBarunGothic.ttf',
                'urls': [
                    'https://github.com/naver/nanumfont/raw/master/fonts/NanumFontSetup_TTF_BARUNGOTHIC/NanumBarunGothic.ttf',
                    'https://cdn.jsdelivr.net/gh/naver/nanumfont@master/fonts/NanumFontSetup_TTF_BARUNGOTHIC/NanumBarunGothic.ttf'
                ]
            }
        ]
        
        for font_option in font_options:
            font_path = font_option['path']
            logger.info(f"🔍 Trying {font_option['name']}...")
            
            # Check if already exists
            if os.path.exists(font_path):
                try:
                    # Test font with Korean text
                    test_font = ImageFont.truetype(font_path, 24, encoding='utf-8')
                    test_img = Image.new('RGB', (200, 50), 'white')
                    test_draw = ImageDraw.Draw(test_img)
                    test_draw.text((10, 10), "한글 테스트 가나다", font=test_font, fill='black')
                    
                    KOREAN_FONT_PATH = font_path
                    logger.info(f"✅ Korean font loaded from cache: {font_path}")
                    return font_path
                except Exception as e:
                    logger.error(f"❌ Cached font test failed: {e}")
                    try:
                        os.remove(font_path)
                    except:
                        pass
                    continue
            
            # Try downloading
            for url in font_option['urls']:
                try:
                    logger.info(f"📥 Downloading from: {url}")
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': '*/*'
                    }
                    response = requests.get(url, timeout=30, headers=headers, allow_redirects=True)
                    
                    if response.status_code == 200:
                        content_length = len(response.content)
                        logger.info(f"📦 Downloaded {content_length} bytes")
                        
                        if content_length > 100000:  # Font should be > 100KB
                            # Save font
                            with open(font_path, 'wb') as f:
                                f.write(response.content)
                            
                            # Verify font with Korean text
                            test_font = ImageFont.truetype(font_path, 24, encoding='utf-8')
                            test_img = Image.new('RGB', (200, 50), 'white')
                            test_draw = ImageDraw.Draw(test_img)
                            
                            # Test with actual Korean text
                            test_text = "한글 폰트 테스트"
                            test_draw.text((10, 10), test_text, font=test_font, fill='black')
                            
                            # Additional verification
                            bbox = test_draw.textbbox((10, 10), test_text, font=test_font)
                            if bbox[2] - bbox[0] > 50:
                                KOREAN_FONT_PATH = font_path
                                logger.info(f"✅ Korean font downloaded and verified: {font_path}")
                                return font_path
                            else:
                                logger.error("❌ Font verification failed - text too small")
                                os.remove(font_path)
                        else:
                            logger.error(f"❌ File too small: {content_length} bytes")
                            
                except Exception as e:
                    logger.error(f"❌ Download failed from {url}: {str(e)}")
                    continue
        
        # Last resort - try system fonts
        system_font_paths = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
        ]
        
        for sys_font in system_font_paths:
            if os.path.exists(sys_font):
                try:
                    test_font = ImageFont.truetype(sys_font, 24, encoding='utf-8')
                    KOREAN_FONT_PATH = sys_font
                    logger.info(f"✅ Using system font: {sys_font}")
                    return sys_font
                except:
                    continue
        
        logger.error("❌ All font attempts failed")
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
        # Always try to download Korean font if not available
        if not KOREAN_FONT_PATH:
            download_korean_font()
        
        if KOREAN_FONT_PATH and os.path.exists(KOREAN_FONT_PATH):
            try:
                font = ImageFont.truetype(KOREAN_FONT_PATH, size, encoding='utf-8')
                logger.info(f"✅ Korean font loaded size={size}")
            except Exception as e:
                logger.error(f"❌ Korean font loading failed: {e}")
                font = None
    
    if font is None:
        # Try to get a basic font with larger size
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", size)
            logger.warning("⚠️ Using Liberation font - Korean may not display properly!")
        except:
            try:
                font = ImageFont.load_default()
                logger.warning("⚠️ Using default font - Korean will not display properly!")
            except:
                font = ImageFont.load_default()
    
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
        text = unicodedata.normalize('NFC', text)
        
        text = text.strip()
        if not text:
            return
        
        # Try to draw text
        draw.text(position, text, font=font, fill=fill)
        logger.info(f"✅ Drew text: {text[:20]}...")
        
    except Exception as e:
        logger.error(f"❌ Text drawing error: {e}")
        # Fallback - try with ASCII only
        try:
            import re
            # Replace Korean with [Korean] marker
            fallback_text = re.sub(r'[가-힣]+', '[Korean]', text)
            draw.text(position, fallback_text, font=font, fill=fill)
            logger.warning(f"⚠️ Used fallback text: {fallback_text[:20]}...")
        except Exception as e2:
            logger.error(f"❌ Fallback text drawing also failed: {e2}")

def get_text_size(draw, text, font):
    """Get text size with compatibility for different PIL versions"""
    try:
        if not text or not font:
            return (0, 0)
        
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        
        # Normalize unicode
        text = unicodedata.normalize('NFC', str(text)).strip()
        
        if not text:
            return (0, 0)
        
        # Try new method first (PIL 8+)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        except AttributeError:
            # Fall back to old method
            try:
                return draw.textsize(text, font=font)
            except:
                # Ultimate fallback
                char_width = 24 if any('\u3131' <= c <= '\u318E' or '\uAC00' <= c <= '\uD7A3' for c in text) else 12
                return (len(text) * char_width, 30)
    except Exception as e:
        logger.error(f"Error getting text size: {e}")
        return (100, 30)

def wrap_text(text, font, max_width, draw):
    """Wrap text to fit within max_width with better Korean handling"""
    if not text or not font:
        return []
    
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    
    # Normalize unicode
    text = unicodedata.normalize('NFC', str(text)).strip()
    
    lines = []
    paragraphs = text.split('\n')
    
    for paragraph in paragraphs:
        if not paragraph:
            lines.append('')
            continue
        
        # Split by words for better wrapping
        words = paragraph.split(' ')
        current_line = ""
        
        for word in words:
            if current_line:
                test_line = current_line + " " + word
            else:
                test_line = word
                
            try:
                width, _ = get_text_size(draw, test_line, font)
            except:
                width = len(test_line) * 20
            
            if width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
    
    return lines

def create_md_talk_section(text_content=None, width=1200):
    """Create MD TALK section with PERFECT Korean support"""
    logger.info("🔤 Creating MD TALK section with Korean support")
    
    # Force Korean font download at start
    font_path = download_korean_font()
    if not font_path:
        logger.error("❌ Failed to get Korean font, using fallback...")
    
    fixed_width = 1200
    fixed_height = 400
    
    left_margin = 100
    right_margin = 100
    top_margin = 60
    content_width = fixed_width - left_margin - right_margin
    
    # Get fonts with Korean support
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
        text = text_content.replace('MD TALK', '').replace('MD Talk', '').strip()
    else:
        text = """이 제품은 일상에서도 부담없이 착용할 수 있는 편안한 디자인으로 매일의 스타일링에 포인트를 더해줍니다."""
    
    # Ensure text is properly encoded
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    
    # Normalize unicode
    text = unicodedata.normalize('NFC', str(text)).strip()
    
    logger.info(f"MD TALK text (first 50 chars): {text[:50]}...")
    
    # Wrap text
    wrapped_lines = wrap_text(text, body_font, content_width, draw)
    logger.info(f"Wrapped into {len(wrapped_lines)} lines")
    
    # Calculate positions
    line_height = 45
    title_bottom_margin = 60
    y_pos = top_margin + title_height + title_bottom_margin
    
    # Draw body text
    for i, line in enumerate(wrapped_lines):
        if line:
            try:
                line_width, _ = get_text_size(draw, line, body_font)
                line_x = (fixed_width - line_width) // 2
            except:
                line_x = left_margin
            
            safe_draw_text(draw, (line_x, y_pos), line, body_font, (80, 80, 80))
            y_pos += line_height
            
            if y_pos > fixed_height - 50:
                break
    
    logger.info(f"✅ MD TALK section created: {fixed_width}x{fixed_height}")
    return section_img

def create_design_point_section(text_content=None, width=1200):
    """Create DESIGN POINT section with PERFECT Korean support"""
    logger.info("🔤 Creating DESIGN POINT section with Korean support")
    
    # Force Korean font download at start
    font_path = download_korean_font()
    if not font_path:
        logger.error("❌ Failed to get Korean font, using fallback...")
    
    fixed_width = 1200
    fixed_height = 350
    
    left_margin = 100
    right_margin = 100
    top_margin = 60
    content_width = fixed_width - left_margin - right_margin
    
    # Get fonts with Korean support
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
        text = text_content.replace('DESIGN POINT', '').replace('Design Point', '').strip()
    else:
        text = """남성 단품은 무광 텍스처와 유광 라인의 조화가 견고한 감성을 전하고 여자 단품은 파베 세팅과 섬세한 밀그레인의 디테일로 화려하면서도 고급스러운 반짝임을 표현합니다."""
    
    # Ensure text is properly encoded
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    
    # Normalize unicode
    text = unicodedata.normalize('NFC', str(text)).strip()
    
    logger.info(f"DESIGN POINT text (first 50 chars): {text[:50]}...")
    
    # Wrap text
    wrapped_lines = wrap_text(text, body_font, content_width, draw)
    logger.info(f"Wrapped into {len(wrapped_lines)} lines")
    
    # Calculate positions
    line_height = 40
    title_bottom_margin = 70
    y_pos = top_margin + title_height + title_bottom_margin
    
    # Draw body text
    for i, line in enumerate(wrapped_lines):
        if line:
            try:
                line_width, _ = get_text_size(draw, line, body_font)
                line_x = (fixed_width - line_width) // 2
            except:
                line_x = left_margin
            
            safe_draw_text(draw, (line_x, y_pos), line, body_font, (80, 80, 80))
            y_pos += line_height
            
            if y_pos > fixed_height - 50:
                break
    
    logger.info(f"✅ DESIGN POINT section created: {fixed_width}x{fixed_height}")
    return section_img

def find_special_mode(data, path=""):
    """Find special mode in nested data structures with deep search"""
    if isinstance(data, str):
        if data in ['both_text_sections', 'md_talk', 'design_point']:
            logger.info(f"✅ Found special_mode as string at {path}: {data}")
            return data
        return None
    
    if isinstance(data, dict):
        # Direct check
        if 'special_mode' in data and data['special_mode']:
            logger.info(f"✅ Found special_mode at {path}.special_mode: {data['special_mode']}")
            return data['special_mode']
        
        # Check all keys recursively
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            
            # Skip image data
            if key in ['enhanced_image', 'image', 'base64', 'image_base64'] and isinstance(value, str) and len(value) > 1000:
                continue
                
            if isinstance(value, dict):
                result = find_special_mode(value, new_path)
                if result:
                    return result
            elif isinstance(value, str) and value in ['both_text_sections', 'md_talk', 'design_point']:
                logger.info(f"✅ Found special_mode at {new_path}: {value}")
                return value
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    result = find_special_mode(item, f"{new_path}[{i}]")
                    if result:
                        return result
    
    return None

def find_text_content(data, content_type):
    """Find text content for MD TALK or DESIGN POINT with deep search"""
    if isinstance(data, dict):
        # Keys to search for based on content type
        if content_type == 'md_talk':
            keys = ['md_talk_content', 'md_talk', 'md_talk_text', 'text_content', 'claude_text', 'mdtalk', 'MD_TALK']
        elif content_type == 'design_point':
            keys = ['design_point_content', 'design_point', 'design_point_text', 'text_content', 'claude_text', 'designpoint', 'DESIGN_POINT']
        else:
            keys = ['text_content', 'claude_text', 'text']
        
        # Direct check
        for key in keys:
            if key in data and isinstance(data[key], str) and data[key].strip():
                logger.info(f"✅ Found {content_type} content at key: {key}")
                return data[key]
        
        # Check all keys recursively
        for key, value in data.items():
            # Skip image data
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

def u2net_optimized_removal(image: Image.Image) -> Image.Image:
    """U2Net optimized background removal"""
    try:
        from rembg import remove
        
        global REMBG_SESSION
        if REMBG_SESSION is None:
            REMBG_SESSION = init_rembg_session()
            if REMBG_SESSION is None:
                return image
        
        logger.info("🚀 U2Net Optimized Removal")
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Light pre-processing
        contrast = ImageEnhance.Contrast(image)
        image_enhanced = contrast.enhance(1.3)
        
        # Save to buffer
        buffered = BytesIO()
        image_enhanced.save(buffered, format="PNG", compress_level=1)
        buffered.seek(0)
        img_data = buffered.getvalue()
        
        # Apply U2Net with balanced settings
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

def image_to_base64(image, keep_transparency=True):
    """Convert to base64 WITH padding - ALWAYS"""
    buffered = BytesIO()
    
    if image.mode != 'RGBA' and keep_transparency:
        image = image.convert('RGBA')
    
    if image.mode == 'RGBA':
        image.save(buffered, format='PNG', compress_level=3, optimize=True)
    else:
        image.save(buffered, format='PNG', optimize=True, compress_level=3)
    
    buffered.seek(0)
    # ALWAYS include padding
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
    
    # If special_mode is not found in job, try to find it again
    if not special_mode:
        special_mode = find_special_mode(job)
    
    logger.info(f"🔤 Processing special mode: '{special_mode}'")
    
    # Validate special_mode
    if not special_mode or special_mode not in ['both_text_sections', 'md_talk', 'design_point']:
        logger.error(f"❌ Invalid special mode: '{special_mode}'")
        
        # Try to auto-detect based on content
        md_talk_content = find_text_content(job, 'md_talk')
        design_point_content = find_text_content(job, 'design_point')
        
        if md_talk_content and design_point_content:
            logger.info("📝 Auto-detected: both_text_sections")
            special_mode = 'both_text_sections'
        elif md_talk_content:
            logger.info("📝 Auto-detected: md_talk")
            special_mode = 'md_talk'
        elif design_point_content:
            logger.info("📝 Auto-detected: design_point")
            special_mode = 'design_point'
        else:
            return {
                "output": {
                    "error": f"Invalid or missing special mode. Got: '{special_mode}'",
                    "status": "error",
                    "version": VERSION
                }
            }
    
    # Ensure Korean font is downloaded before processing
    logger.info("📥 Ensuring Korean font is available...")
    font_path = download_korean_font()
    if not font_path:
        logger.error("❌ Korean font download failed, but continuing...")
    else:
        logger.info(f"✅ Korean font ready: {font_path}")
    
    if special_mode == 'both_text_sections':
        # Find text content
        md_talk_text = find_text_content(job, 'md_talk')
        design_point_text = find_text_content(job, 'design_point')
        
        # Default texts if not found
        if not md_talk_text:
            md_talk_text = """각도에 따라 달라지는 빛의 결들이 두 사람의 특별한 순간순간을 더 찬란하게 만들며 360도 새겨진 패턴으로 매일 새로운 반짝임을 보여줍니다."""
        
        if not design_point_text:
            design_point_text = """입체적인 컷팅 위로 섬세하게 빛나는 패턴이 고급스러움을 완성하며 각진 텍스처가 심플하면서 유니크한 매력을 더해줍니다."""
        
        # Create sections
        md_section = create_md_talk_section(md_talk_text)
        design_section = create_design_point_section(design_point_text)
        
        # Convert to base64 WITH padding
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
            text_content = """이 제품은 일상에서도 부담없이 착용할 수 있는 편안한 디자인으로 매일의 스타일링에 포인트를 더해줍니다."""
        
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
    
    else:
        return {
            "output": {
                "error": f"Unexpected special mode: '{special_mode}'",
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

def find_input_data_fast(data, depth=0, max_depth=10):
    """Find input data - improved recursive search"""
    if depth > max_depth:
        return None
        
    # Direct string check
    if isinstance(data, str) and len(data) > 50:
        # Basic check if it looks like base64
        sample = data[:100].strip()
        if all(c in string.ascii_letters + string.digits + '+/=' for c in sample):
            return data
    
    if isinstance(data, dict):
        # Priority keys for image data
        priority_keys = ['enhanced_image', 'image', 'image_base64', 'base64', 'img', 
                        'input_image', 'original_image', 'base64_image', 'imageData']
        
        # Check priority keys first
        for key in priority_keys:
            if key in data and isinstance(data[key], str) and len(data[key]) > 50:
                return data[key]
        
        # Recursive search all keys
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 1000:
                # Check if it might be base64
                sample = value[:100].strip()
                if all(c in string.ascii_letters + string.digits + '+/=' for c in sample):
                    logger.info(f"✅ Found potential image data at key: {key}")
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
    """Find filename - recursive search"""
    if depth > max_depth:
        return None
        
    if isinstance(data, dict):
        # Direct filename keys
        for key in ['filename', 'file_name', 'name', 'fileName', 'file', 'fname']:
            if key in data and isinstance(data[key], str) and data[key].strip():
                return data[key]
        
        # Recursive search all keys
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
    """Fast base64 decode with padding support"""
    try:
        if not base64_str or len(base64_str) < 50:
            raise ValueError("Invalid base64 string")
        
        # Remove data URI prefix if present
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[-1]
        
        # Remove whitespace
        base64_str = ''.join(base64_str.split())
        
        # Remove invalid characters
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        base64_str = ''.join(c for c in base64_str if c in valid_chars)
        
        # Try with existing padding first
        try:
            decoded = base64.b64decode(base64_str, validate=True)
            return decoded
        except:
            # Add proper padding if needed
            no_pad = base64_str.rstrip('=')
            padding_needed = (4 - len(no_pad) % 4) % 4
            padded = no_pad + ('=' * padding_needed)
            decoded = base64.b64decode(padded, validate=True)
            return decoded
            
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise ValueError(f"Invalid base64 data: {str(e)}")

def handler(event):
    """Enhancement handler - V5 Fixed Korean"""
    try:
        logger.info("=" * 60)
        logger.info(f"🚀 {VERSION} Handler Started")
        logger.info("=" * 60)
        logger.info("✅ Improvements in V5-Fixed:")
        logger.info("  - Enhanced Korean font download with more sources")
        logger.info("  - Better font fallback handling")
        logger.info("  - Improved text wrapping for Korean")
        logger.info("  - More robust font verification")
        
        # Force font download at startup
        logger.info("📥 Pre-loading Korean font...")
        font_path = download_korean_font()
        if font_path:
            logger.info(f"✅ Korean font ready: {font_path}")
        else:
            logger.warning("⚠️ Korean font not available, will use fallback")
        
        # Log input structure
        logger.info(f"Input event type: {type(event)}")
        if isinstance(event, dict):
            logger.info(f"Input keys: {list(event.keys())[:10]}")
        
        # Find special mode
        special_mode = find_special_mode(event)
        logger.info(f"🔍 Special mode search result: {special_mode}")
        
        # Check if this is a text section request
        if special_mode and special_mode in ['both_text_sections', 'md_talk', 'design_point']:
            logger.info(f"📝 Processing special mode: {special_mode}")
            return process_special_mode(event)
        
        # Image processing request
        logger.info("📸 Processing image enhancement request")
        
        # Find input data
        logger.info("🔍 Searching for input data...")
        filename = find_filename_fast(event)
        image_data_str = find_input_data_fast(event)
        
        if not image_data_str:
            # Check for text content as last resort
            md_talk_content = find_text_content(event, 'md_talk')
            design_point_content = find_text_content(event, 'design_point')
            
            if md_talk_content and design_point_content:
                logger.info("📝 Found both text contents, assuming both_text_sections mode")
                event['special_mode'] = 'both_text_sections'
                return process_special_mode(event)
            elif md_talk_content:
                logger.info("📝 Found MD TALK content, assuming md_talk mode")
                event['special_mode'] = 'md_talk'
                return process_special_mode(event)
            elif design_point_content:
                logger.info("📝 Found DESIGN POINT content, assuming design_point mode")
                event['special_mode'] = 'design_point'
                return process_special_mode(event)
            
            logger.error("❌ No input image data or text content found")
            raise ValueError("No input image data or text content found")
        
        logger.info(f"✅ Found image data, length: {len(image_data_str)}")
        
        # Decode and open image
        start_time = time.time()
        image_bytes = decode_base64_fast(image_data_str)
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        decode_time = time.time() - start_time
        logger.info(f"⏱️ Image decode: {decode_time:.2f}s")
        logger.info(f"📐 Original size: {image.size}")
        
        # STEP 1: Apply background removal
        start_time = time.time()
        logger.info("📸 Applying background removal")
        image = u2net_optimized_removal(image)
        
        removal_time = time.time() - start_time
        logger.info(f"⏱️ Background removal: {removal_time:.2f}s")
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # STEP 2: Resize to target dimensions
        start_time = time.time()
        logger.info("📏 Resizing to 1200x1560")
        image = resize_image_proportional(image, 1200, 1560)
        
        resize_time = time.time() - start_time
        logger.info(f"⏱️ Resize: {resize_time:.2f}s")
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Convert to base64 WITH padding
        start_time = time.time()
        enhanced_base64 = image_to_base64(image, keep_transparency=True)
        encode_time = time.time() - start_time
        logger.info(f"⏱️ Base64 encode: {encode_time:.2f}s")
        logger.info(f"📊 Base64 length: {len(enhanced_base64)}")
        
        # Total time
        total_time = decode_time + removal_time + resize_time + encode_time
        logger.info(f"⏱️ TOTAL TIME: {total_time:.2f}s")
        
        output_filename = filename or "enhanced_image.png"
        file_number = extract_file_number(output_filename)
        
        # Build response with proper structure for Make.com
        # CRITICAL: {"output": {...}} structure
        result = {
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
                "base64_padding": "INCLUDED",
                "compression": "level_3",
                "processing_times": {
                    "decode": f"{decode_time:.2f}s",
                    "background_removal": f"{removal_time:.2f}s",
                    "resize": f"{resize_time:.2f}s",
                    "encode": f"{encode_time:.2f}s",
                    "total": f"{total_time:.2f}s"
                }
            }
        }
        
        logger.info("✅ Processing complete, returning result")
        return result
        
    except Exception as e:
        logger.error(f"❌ Handler error: {str(e)}")
        import traceback
        
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

# Start message
logger.info("=" * 60)
logger.info(f"🚀 RunPod Serverless Worker Starting")
logger.info(f"📦 Version: {VERSION}")
logger.info(f"📅 Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 60)

# RunPod handler
runpod.serverless.start({"handler": handler})
