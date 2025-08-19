"""
Configuration settings for Korean Fashion AI Personal Stylist
Week 1-2: Face Analysis & Skin Tone Detection Settings
"""


import os

# Project Settings
PROJECT_NAME = "Korean Fashion AI Personal Stylist"
VERSION = "0.1.0"
DEBUG = True

# Image Processing Settings
MAX_IMAGE_SIZE = (800, 800) # Resize large images for faster processing
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png','bmp']  # Supported image formats
MAX_FILE_SIZE_MB = 10  # Maximum file size in MB

# Face Detection Settings
FACE_DETECTION_CONFIDENCE = 0.7  # Confidence threshold for face detection
MIN_FACE_SIZE = (100, 100) # Minimum face size pixels for detection

# Face Shape Classification Settings
FACE_SHAPE_CATEGORIES = {
    'oval': 'Oval',
    'round': 'Round',
    'square': 'Square',
    'heart': 'Heart',
    'long': 'Long',
    'diamond': 'Diamond',
    'triangle': 'Triangle'
}

# Skin Tone Analysis Settings
SKIN_TONE_REGIONS = {
    'forehead': (0.3,0.15, 0.4, 0.25), #(x%, y%, width%, height%)
    'left_cheek': (0.15,0.4,0.2,0.25),
    'right_cheek': (0.65,0.4,0.2,0.25),
    'chin': (0.4,0.75,0.2,0.15),
    'nose': (0.4,0.35,0.2,0.2),
    'under_eye': (0.3,0.5,0.4,0.25)
}

# Color Tone Analysis Settings
KOREAN_COLOR_PALETTES = {
    'spring': {
        'colors': ['#FFB6C1', '#FF69B4', '#FF4500', '#FFD700', '#ADFF2F'],
        'description': 'Bright and warm colors suitable for spring skin tones.'
    },
    'summer': {
        'colors': ['#87CEEB', '#4682B4', '#6A5ACD', '#B0C4DE', '#F0E68C'],
        'description': 'Cool and soft colors suitable for summer skin tones.'
    },
    'autumn': {
        'colors': ['#D2691E', '#8B4513', '#FF6347', '#FFD700', '#8B0000'],
        'description': 'Warm and earthy colors suitable for autumn skin tones.'
    },
    'winter': {
        'colors': ['#000080', '#4682B4', '#B0C4DE', '#FF4500', '#FFFFFF'],
        'description': 'Cool and bold colors suitable for winter skin tones.'
    },
    'neutral': {
        'colors': ['#808080', '#A9A9A9', '#D3D3D3', '#F5F5F5', '#000000'],
        'description': 'Neutral colors suitable for all skin tones.'
    }
}

# Streamlit UI Settings
STREAMLIT_CONFIG = {
    'page_title': PROJECT_NAME,
    'page_icon': '🎨',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'theme': {
        'primaryColor': '#FF69B4',
        'backgroundColor': '#F0F0F0',
        'secondaryBackgroundColor': '#FFFFFF',
        'textColor': '#333333',
        'font': 'sans serif'
    }
}

# File Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'test_images')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STATIC_DIR = os.path.join(BASE_DIR, 'static')


# Create directories if they do not exist
for directory in [DATA_DIR, TEST_IMAGES_DIR, MODELS_DIR, STATIC_DIR]:
    os.makedirs(directory, exist_ok=True)