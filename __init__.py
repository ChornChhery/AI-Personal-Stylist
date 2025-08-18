# src/__init__.py
"""
Korean Fashion AI Personal Stylist
Main source package
"""

__version__ = "0.1.0"
__author__ = "Korean Fashion AI Team"

# src/face_analysis/__init__.py
"""
Face Analysis Module
Contains face detection, shape analysis, and skin tone detection
"""

from .face_detector import FaceDetector
from .face_shape import FaceShapeAnalyzer
from .skin_tone import SkinToneAnalyzer

__all__ = ['FaceDetector', 'FaceShapeAnalyzer', 'SkinToneAnalyzer']

# src/utils/__init__.py
"""
Utilities Module
Helper functions for image processing and analysis
"""

from .image_utils import (
    load_image, resize_image, enhance_image_quality,
    validate_image_quality, image_to_base64, base64_to_image
)

__all__ = [
    'load_image', 'resize_image', 'enhance_image_quality',
    'validate_image_quality', 'image_to_base64', 'base64_to_image'
]

# src/korean_fashion/__init__.py
"""
Korean Fashion Module
Style matching and recommendations (Week 3-4)
"""

# This will be implemented in weeks 3-4
__all__ = []