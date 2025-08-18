"""
Korean Fashion AI Personal Stylist - Main Streamlit Application
Week 1-2: Face Analysis & Skin Tone Detection Interface
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Import our modules
from src.face_analysis.face_detector import FaceDetector
from src.face_analysis.face_shape import FaceShapeAnalyzer
from src.face_analysis.skin_tone import SkinToneAnalyzer
from src.utils.image_utils import (
    load_image, resize_image, enhance_image_quality, 
    validate_image_quality, image_to_base64, create_side_by_side_image
)
from config.settings import STREAMLIT_CONFIG, SUPPORTED_FORMATS, MAX_FILE_SIZE_MB

# Configure Streamlit page
st.set_page_config(
    page_title=STREAMLIT_CONFIG['page_title'],
    page_icon=STREAMLIT_CONFIG['page_icon'],
    layout=STREAMLIT_CONFIG['layout'],
    initial_sidebar_state=STREAMLIT_CONFIG['initial_sidebar_state']
)

# Custom CSS for Korean aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .analysis-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .korean-tip {
        background: #f8f9ff;
        padding: 1rem;
        border-left: 4px solid #667eea;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .color-palette {
        display: flex;
        gap: 10px;
        margin: 1rem 0;
    }
    
    .color-swatch {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        border: 2px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

def initialize_analyzers():
    """Initialize all analysis modules"""
    if 'face_detector' not in st.session_state:
        st.session_state.face_detector = FaceDetector()
    if 'face_shape_analyzer' not in st.session_state:
        st.session_state.face_shape_analyzer = FaceShapeAnalyzer()
    if 'skin_tone_analyzer' not in st.session_state:
        st.session_state.skin_tone_analyzer = SkinToneAnalyzer()

def display_header():
    """Display application header"""
    st.markdown('<h1 class="main-header">🇰🇷 Korean Fashion AI Personal Stylist</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover your perfect Korean style with AI-powered face and skin tone analysis</p>', 
                unsafe_allow_html=True)

def upload_and_validate_image():
    """Handle image upload and validation"""
    st.subheader("📸 Upload Your Photo")
    
    uploaded_file = st.file_uploader(
        "Choose a clear selfie for analysis",
        type=SUPPORTED_FORMATS,
        help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}. Max size: {MAX_FILE_SIZE_MB}MB"
    )
    
    if uploaded_file is not None:
        # Check file size
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File size too large. Please upload an image smaller than {MAX_FILE_SIZE_MB}MB.")
            return None
            
        try:
            # Load and process image
            image_bytes = uploaded_file.read()
            image = load_image(image_bytes)
            
            if image is None:
                st.error("Could not load the image. Please try a different file.")
                return None
            
            # Validate image quality
            quality_assessment = validate_image_quality(image)
            
            # Display quality warnings if any
            if not quality_assessment['is_valid']:
                st.warning("⚠️ Image Quality Issues:")
                for issue in quality_assessment['issues']:
                    st.write(f"• {issue}")
                st.info("💡 Recommendations:")
                for rec in quality_assessment['recommendations']:
                    st.write(f"• {rec}")
                
                if st.button("Continue Anyway"):
                    st.session_state.continue_with_poor_quality = True
                
                if not st.session_state.get('continue_with_poor_quality', False):
                    return None
            
            # Resize and enhance image
            image = resize_image(image)
            enhanced_image = enhance_image_quality(image)
            
            return enhanced_image
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None
    
    return None

def analyze_face_and_display_results(image):
    """Perform complete face analysis and display results"""
    
    with st.spinner("🔍 Analyzing your facial features..."):
        # Face detection and analysis
        face_analysis = st.session_state.face_detector.analyze_face(image)
        
        if not face_analysis:
            st.error("❌ No face detected in the image. Please upload a clear photo with your face visible.")
            return
        
        st.success("✅ Face detected successfully!")
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        with col2:
            st.subheader("🎯 Face Detection")
            # Draw landmarks on image
            landmarks_image = st.session_state.face_detector.draw_landmarks(
                image, face_analysis['landmarks']
            )
            st.image(cv2.cvtColor(landmarks_image, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    # Face shape analysis
    with st.spinner("📐 Analyzing face shape..."):
        measurements = st.session_state.face_shape_analyzer.calculate_face_measurements(
            face_analysis['landmarks']
        )
        face_shape_result = st.session_state.face_shape_analyzer.classify_face_shape(measurements)
        korean_recommendations = st.session_state.face_shape_analyzer.get_korean_styling_recommendations(
            face_shape_result['shape']
        )
    
    # Skin tone analysis
    with st.spinner("🎨 Analyzing skin tone..."):
        skin_regions = st.session_state.skin_tone_analyzer.extract_skin_regions(
            face_analysis['cropped_face'], face_analysis['landmarks']
        )
        skin_analysis = st.session_state.skin_tone_analyzer.analyze_skin_color(skin_regions)
        makeup_recommendations = st.session_state.skin_tone_analyzer.get_korean_makeup_recommendations(skin_analysis)
    
    # Display results
    display_analysis_results(face_shape_result, korean_recommendations, skin_analysis, makeup_recommendations)

def display_analysis_results(face_shape_result, korean_recommendations, skin_analysis, makeup_recommendations):
    """Display comprehensive analysis results"""
    
    st.header("🎉 Your Korean Style Analysis Results")
    
    # Face Shape Results
    st.subheader("👤 Face Shape Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        face_shape = face_shape_result['shape']
        confidence = face_shape_result['confidence']
        korean_name = korean_recommendations['face_shape']
        
        st.markdown(f"""
        <div class="analysis-box">
            <h3>🎭 {korean_name}</h3>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
            <p>{korean_recommendations['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**💇‍♀️ Korean Hairstyle Recommendations:**")
        for hairstyle in korean_recommendations['recommended_hairstyles']:
            st.markdown(f"<div class='korean-tip'>• {hairstyle}</div>", unsafe_allow_html=True)
        
        st.markdown("**👓 Accessory Tips:**")
        for tip in korean_recommendations['accessory_tips'][:3]:  # Show first 3
            st.markdown(f"<div class='korean-tip'>• {tip}</div>", unsafe_allow_html=True)
    
    # Skin Tone Results
    st.subheader("🌈 Skin Tone & Color Analysis")
    
    if skin_analysis:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dominant_color = skin_analysis.get('dominant_color', {})
            if dominant_color:
                st.markdown("**🎨 Your Skin Tone**")
                hex_color = dominant_color['hex']
                st.markdown(f"""
                <div style="background-color: {hex_color}; width: 100px; height: 50px; 
                     border-radius: 10px; margin: 10px 0; border: 2px solid #ddd;"></div>
                <p><strong>Color:</strong> {hex_color}</p>
                """, unsafe_allow_html=True)
        
        with col2:
            undertone = skin_analysis.get('undertone', {})
            if undertone:
                primary_undertone = undertone['primary'].title()
                st.markdown("**🌡️ Undertone**")
                st.markdown(f"""
                <div class="analysis-box">
                    <h4>{primary_undertone}</h4>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            korean_season = skin_analysis.get('korean_season', {})
            if korean_season:
                season_name = korean_season['korean_name']
                st.markdown("**🍂 Korean Season**")
                st.markdown(f"""
                <div class="analysis-box">
                    <h4>{season_name}</h4>
                    <p>{korean_season['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Color Palette
        color_palette = skin_analysis.get('color_palette', {})
        if color_palette:
            st.markdown("**🎨 Your Perfect Color Palette**")
            recommended_colors = color_palette['recommended_colors']
            
            # Create color swatches
            color_html = '<div class="color-palette">'
            for color in recommended_colors[:8]:  # Show first 8 colors
                color_html += f'<div class="color-swatch" style="background-color: {color};" title="{color}"></div>'
            color_html += '</div>'
            
            st.markdown(color_html, unsafe_allow_html=True)
            
            # Show avoid colors
            avoid_colors = color_palette.get('avoid_colors', [])
            if avoid_colors:
                with st.expander("❌ Colors to Avoid"):
                    avoid_html = '<div class="color-palette">'
                    for color in avoid_colors[:6]:
                        avoid_html += f'<div class="color-swatch" style="background-color: {color};" title="{color}"></div>'
                    avoid_html += '</div>'
                    st.markdown(avoid_html, unsafe_allow_html=True)
    
    # Korean Makeup Recommendations
    st.subheader("💄 Korean Makeup Recommendations")
    
    if makeup_recommendations:
        tab1, tab2, tab3, tab4 = st.tabs(["💋 Lips", "🌸 Blush", "👁️ Eyes", "✨ Base"])
        
        with tab1:
            lip_colors = makeup_recommendations.get('lip_colors', [])
            st.markdown("**Perfect Korean Lip Colors for You:**")
            for color in lip_colors:
                st.markdown(f"<div class='korean-tip'>💋 {color}</div>", unsafe_allow_html=True)
        
        with tab2:
            blush_colors = makeup_recommendations.get('blush_colors', [])
            st.markdown("**Korean Blush Shades:**")
            for color in blush_colors:
                st.markdown(f"<div class='korean-tip'>🌸 {color}</div>", unsafe_allow_html=True)
        
        with tab3:
            eyeshadow_colors = makeup_recommendations.get('eyeshadow_colors', [])
            st.markdown("**Korean Eyeshadow Colors:**")
            for color in eyeshadow_colors:
                st.markdown(f"<div class='korean-tip'>👁️ {color}</div>", unsafe_allow_html=True)
        
        with tab4:
            base_tips = makeup_recommendations.get('base_makeup', [])
            st.markdown("**Korean Base Makeup Tips:**")
            for tip in base_tips:
                st.markdown(f"<div class='korean-tip'>✨ {tip}</div>", unsafe_allow_html=True)
    
    # Korean Styling Tips
    st.subheader("💫 Korean Styling Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎭 Face Shape Styling:**")
        for tip in korean_recommendations['korean_style_tips']:
            st.markdown(f"<div class='korean-tip'>• {tip}</div>", unsafe_allow_html=True)
    
    with col2:
        if makeup_recommendations:
            styling_tips = makeup_recommendations.get('styling_tips', [])
            st.markdown("**🌈 Color Styling:**")
            for tip in styling_tips:
                st.markdown(f"<div class='korean-tip'>• {tip}</div>", unsafe_allow_html=True)

def display_sidebar_info():
    """Display sidebar with additional information"""
    with st.sidebar:
        st.header("ℹ️ About This Analysis")
        
        st.markdown("""
        **What We Analyze:**
        - 👤 Face shape (5 categories)
        - 🎨 Skin tone & undertones  
        - 🌈 Korean seasonal colors
        - 💄 Makeup recommendations
        - 💇‍♀️ Korean hairstyles
        
        **Korean Beauty Standards:**
        - Based on K-beauty principles
        - Seasonal color analysis
        - Korean makeup techniques
        - K-fashion styling tips
        """)
        
        st.header("📝 Tips for Best Results")
        st.markdown("""
        - Use good lighting (natural light preferred)
        - Face should be clearly visible
        - Remove glasses if possible
        - Neutral expression works best
        - High resolution image (min 400x400px)
        """)
        
        st.header("🚀 Coming Soon")
        st.markdown("""
        - 👕 Wardrobe analysis
        - 🛍️ Shopping recommendations
        - 📱 Mobile app
        - 🎥 Virtual try-on
        """)

def main():
    """Main application function"""
    
    # Initialize analyzers
    initialize_analyzers()
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar_info()
    
    # Image upload and analysis
    image = upload_and_validate_image()
    
    if image is not None:
        analyze_face_and_display_results(image)
    else:
        # Show demo section when no image uploaded
        st.subheader("✨ What You'll Discover")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🎭 Face Shape Analysis**
            - Precise facial measurements
            - Korean beauty categorization
            - Personalized styling tips
            """)
        
        with col2:
            st.markdown("""
            **🌈 Color Analysis**
            - Skin tone detection
            - Korean seasonal colors
            - Perfect color palette
            """)
        
        with col3:
            st.markdown("""
            **💄 Korean Styling**
            - K-beauty makeup tips
            - Hairstyle recommendations
            - Fashion guidance
            """)
        
        st.info("👆 Upload your photo above to get started with your personalized Korean style analysis!")

if __name__ == "__main__":
    main()