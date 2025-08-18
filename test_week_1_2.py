"""
Test Script for Week 1-2 Implementation
Korean Fashion AI Personal Stylist - Face Analysis & Skin Tone Detection
"""

import cv2
import numpy as np
from src.face_analysis.face_detector import FaceDetector
from src.face_analysis.face_shape import FaceShapeAnalyzer
from src.face_analysis.skin_tone import SkinToneAnalyzer
from src.utils.image_utils import load_image, validate_image_quality, enhance_image_quality

def test_basic_functionality():
    """Test basic functionality without real image"""
    print("🚀 Testing Korean Fashion AI Personal Stylist - Week 1-2")
    print("=" * 60)
    
    try:
        # Test initialization
        print("📋 Testing module initialization...")
        face_detector = FaceDetector()
        face_shape_analyzer = FaceShapeAnalyzer()
        skin_tone_analyzer = SkinToneAnalyzer()
        print("✅ All modules initialized successfully!")
        
        # Test with synthetic image
        print("\n🖼️ Testing with synthetic test image...")
        
        # Create a simple test image (face-like shape)
        test_image = create_test_face_image()
        
        # Test image validation
        print("🔍 Testing image quality validation...")
        quality_assessment = validate_image_quality(test_image)
        print(f"   Image valid: {quality_assessment['is_valid']}")
        print(f"   Quality scores: {quality_assessment['scores']}")
        
        # Test face detection (will likely fail on synthetic image, but tests the pipeline)
        print("\n👤 Testing face detection pipeline...")
        face_analysis = face_detector.analyze_face(test_image)
        
        if face_analysis:
            print("✅ Face detection successful!")
            
            # Test face shape analysis
            print("📐 Testing face shape analysis...")
            measurements = face_shape_analyzer.calculate_face_measurements(face_analysis['landmarks'])
            face_shape_result = face_shape_analyzer.classify_face_shape(measurements)
            print(f"   Detected shape: {face_shape_result['shape']}")
            print(f"   Confidence: {face_shape_result['confidence']:.2f}")
            
            # Test Korean recommendations
            korean_recs = face_shape_analyzer.get_korean_styling_recommendations(face_shape_result['shape'])
            print(f"   Korean style: {korean_recs['face_shape']}")
            
            # Test skin tone analysis
            print("🎨 Testing skin tone analysis...")
            skin_regions = skin_tone_analyzer.extract_skin_regions(
                face_analysis['cropped_face'], face_analysis['landmarks']
            )
            
            if skin_regions:
                skin_analysis = skin_tone_analyzer.analyze_skin_color(skin_regions)
                print(f"   Skin regions extracted: {len(skin_regions)}")
                
                if skin_analysis:
                    undertone = skin_analysis.get('undertone', {}).get('primary', 'unknown')
                    korean_season = skin_analysis.get('korean_season', {}).get('season', 'unknown')
                    print(f"   Undertone: {undertone}")
                    print(f"   Korean season: {korean_season}")
            
        else:
            print("ℹ️ No face detected in synthetic image (expected)")
            print("   Testing color analysis functions...")
            
            # Test color analysis functions directly
            test_color_analysis()
        
        print("\n🎉 Week 1-2 basic functionality test completed!")
        print("💡 Ready to test with real images using Streamlit app")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("🔧 Check your dependencies and module imports")

def create_test_face_image():
    """Create a simple synthetic face-like image for testing"""
    # Create a 400x400 image with face-like colors
    image = np.ones((400, 400, 3), dtype=np.uint8) * 240  # Light background
    
    # Add face oval (simplified)
    center = (200, 200)
    axes = (80, 100)
    cv2.ellipse(image, center, axes, 0, 0, 360, (220, 180, 160), -1)  # Face color
    
    # Add eyes (dark spots)
    cv2.circle(image, (170, 180), 8, (50, 50, 50), -1)  # Left eye
    cv2.circle(image, (230, 180), 8, (50, 50, 50), -1)  # Right eye
    
    # Add nose (small triangle)
    nose_points = np.array([[200, 200], [195, 220], [205, 220]], dtype=np.int32)
    cv2.fillPoly(image, [nose_points], (200, 160, 140))
    
    # Add mouth (small line)
    cv2.line(image, (185, 250), (215, 250), (150, 100, 100), 3)
    
    return image

def test_color_analysis():
    """Test color analysis functions independently"""
    print("   🌈 Testing color analysis algorithms...")
    
    # Create sample skin color pixels
    skin_pixels = np.array([
        [220, 180, 160],  # Light warm skin
        [200, 160, 140],  # Medium warm skin
        [180, 140, 120],  # Deeper warm skin
    ])
    
    skin_tone_analyzer = SkinToneAnalyzer()
    
    # Test undertone analysis
    undertone_result = skin_tone_analyzer._analyze_undertones(skin_pixels)
    print(f"     Sample undertone analysis: {undertone_result['primary']}")
    
    # Test Korean season matching
    dominant_color = np.mean(skin_pixels, axis=0)
    season_result = skin_tone_analyzer._match_korean_season(dominant_color, undertone_result)
    print(f"     Sample Korean season: {season_result['season']}")
    
    print("   ✅ Color analysis functions working!")

def print_setup_instructions():
    """Print setup instructions for users"""
    print("\n📚 SETUP INSTRUCTIONS FOR WEEK 1-2")
    print("=" * 50)
    print("1. Create virtual environment:")
    print("   python -m venv korean_fashion_ai")
    print("   source korean_fashion_ai/bin/activate  # Linux/Mac")
    print("   korean_fashion_ai\\Scripts\\activate     # Windows")
    print()
    print("2. Install requirements:")
    print("   pip install -r requirements.txt")
    print()
    print("3. Create directory structure:")
    print("   mkdir -p data/test_images models static")
    print()
    print("4. Run this test:")
    print("   python test_week1_2.py")
    print()
    print("5. Run Streamlit app:")
    print("   streamlit run app.py")
    print()
    print("6. Test with real images:")
    print("   - Upload a clear selfie")
    print("   - Check face detection results")
    print("   - Review Korean styling recommendations")

def main():
    """Main test function"""
    print_setup_instructions()
    print("\n" + "="*60)
    test_basic_functionality()
    print("\n" + "="*60)
    print("🎯 NEXT STEPS:")
    print("- Test app with real photos")
    print("- Verify face detection accuracy")
    print("- Check Korean recommendations quality")
    print("- Prepare for Week 3-4: Korean Fashion Database")

if __name__ == "__main__":
    main()