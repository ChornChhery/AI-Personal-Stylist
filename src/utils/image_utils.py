"""
Image Processing Utilities for Korean Fashion AI Personal Stylist
Week 1-2: Helper functions for image processing and manipulation
"""

import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Tuple, Optional, Union

def load_image(image_input: Union[str, bytes, np.ndarray]) -> Optional[np.ndarray]:
    """
    Load image from various input types
    
    Args:
        image_input: Image file path, bytes, or numpy array
        
    Returns:
        Image as numpy array or None if loading fails
    """
    try:
        if isinstance(image_input, str):
            # Load from file path
            image = cv2.imread(image_input)
            if image is None:
                # Try with PIL
                pil_image = Image.open(image_input)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
        elif isinstance(image_input, bytes):
            # Load from bytes
            nparr = np.frombuffer(image_input, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        elif isinstance(image_input, np.ndarray):
            # Already a numpy array
            image = image_input.copy()
            
        else:
            return None
            
        return image
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def resize_image(image: np.ndarray, max_size: Tuple[int, int] = (800, 800), 
                maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image while optionally maintaining aspect ratio
    
    Args:
        image: Input image
        max_size: Maximum (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    max_w, max_h = max_size
    
    if maintain_aspect:
        # Calculate scaling factor
        scale = min(max_w / w, max_h / h)
        
        if scale < 1.0:  # Only resize if image is larger
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        # Resize to exact dimensions
        image = cv2.resize(image, max_size, interpolation=cv2.INTER_AREA)
    
    return image

def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """
    Enhance image quality for better face analysis
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    enhanced = image.copy()
    
    # Convert to LAB color space for better processing
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    
    # Merge channels and convert back to BGR
    enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Apply gentle sharpening
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel * 0.1)
    
    # Slight noise reduction
    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    return enhanced

def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """
    Normalize lighting conditions for consistent analysis
    
    Args:
        image: Input image
        
    Returns:
        Lighting-normalized image
    """
    # Convert to YUV color space
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_channel, u_channel, v_channel = cv2.split(yuv)
    
    # Apply histogram equalization to Y channel
    y_eq = cv2.equalizeHist(y_channel)
    
    # Merge channels and convert back
    yuv_eq = cv2.merge([y_eq, u_channel, v_channel])
    normalized = cv2.cvtColor(yuv_eq, cv2.COLOR_YUV2BGR)
    
    return normalized

def detect_image_blur(image: np.ndarray, threshold: float = 100.0) -> Tuple[bool, float]:
    """
    Detect if image is blurry using Laplacian variance
    
    Args:
        image: Input image
        threshold: Blur threshold (lower = more blurry)
        
    Returns:
        Tuple of (is_sharp, blur_score)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = laplacian.var()
    
    is_sharp = blur_score > threshold
    
    return is_sharp, blur_score

def validate_image_quality(image: np.ndarray) -> dict:
    """
    Validate image quality for face analysis
    
    Args:
        image: Input image
        
    Returns:
        Quality assessment dictionary
    """
    assessment = {
        'is_valid': True,
        'issues': [],
        'recommendations': []
    }
    
    h, w = image.shape[:2]
    
    # Check image size
    if min(h, w) < 200:
        assessment['issues'].append('Image resolution too low')
        assessment['recommendations'].append('Use image with at least 200x200 pixels')
        assessment['is_valid'] = False
    
    # Check aspect ratio
    aspect_ratio = w / h
    if aspect_ratio > 3 or aspect_ratio < 0.33:
        assessment['issues'].append('Unusual aspect ratio')
        assessment['recommendations'].append('Use more square-like image aspect ratio')
    
    # Check blur
    is_sharp, blur_score = detect_image_blur(image)
    if not is_sharp:
        assessment['issues'].append('Image appears blurry')
        assessment['recommendations'].append('Use sharper image for better analysis')
    
    # Check brightness
    brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if brightness < 50:
        assessment['issues'].append('Image too dark')
        assessment['recommendations'].append('Use better lighting')
    elif brightness > 200:
        assessment['issues'].append('Image too bright/overexposed')
        assessment['recommendations'].append('Reduce lighting or exposure')
    
    # Add quality scores
    assessment['scores'] = {
        'resolution_score': min(1.0, min(h, w) / 400),
        'sharpness_score': min(1.0, blur_score / 200),
        'brightness_score': 1.0 - abs(brightness - 128) / 128
    }
    
    return assessment

def create_side_by_side_image(image1: np.ndarray, image2: np.ndarray, 
                            labels: Tuple[str, str] = None) -> np.ndarray:
    """
    Create side-by-side comparison image
    
    Args:
        image1: First image
        image2: Second image  
        labels: Optional labels for images
        
    Returns:
        Combined side-by-side image
    """
    # Resize images to same height
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    target_height = min(h1, h2)
    scale1 = target_height / h1
    scale2 = target_height / h2
    
    img1_resized = cv2.resize(image1, (int(w1 * scale1), target_height))
    img2_resized = cv2.resize(image2, (int(w2 * scale2), target_height))
    
    # Create combined image
    combined = np.hstack([img1_resized, img2_resized])
    
    # Add labels if provided
    if labels:
        label1, label2 = labels
        
        # Add text backgrounds
        cv2.rectangle(combined, (10, 10), (200, 40), (0, 0, 0), -1)
        cv2.rectangle(combined, (img1_resized.shape[1] + 10, 10), 
                     (img1_resized.shape[1] + 200, 40), (0, 0, 0), -1)
        
        # Add text
        cv2.putText(combined, label1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        cv2.putText(combined, label2, (img1_resized.shape[1] + 15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return combined

def image_to_base64(image: np.ndarray, format: str = 'JPEG') -> str:
    """
    Convert image to base64 string for web display
    
    Args:
        image: Input image
        format: Image format (JPEG, PNG)
        
    Returns:
        Base64 encoded image string
    """
    # Convert BGR to RGB
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format, quality=85 if format == 'JPEG' else None)
    
    # Encode to base64
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/{format.lower()};base64,{img_str}"

def base64_to_image(base64_string: str) -> Optional[np.ndarray]:
    """
    Convert base64 string to image
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        Image as numpy array
    """
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_string)
        
        # Convert to image
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
        
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None

def add_korean_watermark(image: np.ndarray, text: str = "Korean Fashion AI") -> np.ndarray:
    """
    Add subtle Korean-style watermark to image
    
    Args:
        image: Input image
        text: Watermark text
        
    Returns:
        Image with watermark
    """
    watermarked = image.copy()
    h, w = watermarked.shape[:2]
    
    # Position watermark in bottom-right
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    x = w - text_width - 10
    y = h - 10
    
    # Add semi-transparent background
    cv2.rectangle(watermarked, (x - 5, y - text_height - 5), 
                 (x + text_width + 5, y + 5), (255, 255, 255), -1)
    
    # Add text
    cv2.putText(watermarked, text, (x, y), font, font_scale, (100, 100, 100), thickness)
    
    return watermarked