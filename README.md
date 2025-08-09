# Korean Fashion AI Personal Stylist 🇰🇷✨

> *Your AI-powered personal stylist specializing in Korean fashion trends and personalized outfit recommendations*

## 📖 Overview

The Korean Fashion AI Personal Stylist is an intelligent system that analyzes your facial features, skin tone, and existing wardrobe to provide personalized Korean fashion recommendations. Whether you're new to K-fashion or looking to perfect your style, this AI helps you discover outfits that complement your unique features while staying true to Korean fashion aesthetics.

### 🌟 Key Features

- **Personal Analysis**: Advanced face shape and skin tone detection
- **Korean Style Matching**: Recommendations based on popular K-fashion trends
- **Wardrobe Scanner**: Analyze your existing clothes for styling potential
- **Color Harmony**: Personalized color palettes that complement your skin tone
- **Trend Integration**: Stay updated with current Korean fashion movements
- **Virtual Styling**: Get complete outfit recommendations with styling tips

## 🎯 Problem Statement

Many fashion enthusiasts struggle with:
- Understanding which Korean fashion styles suit their face shape and skin tone
- Making the most of their existing wardrobe
- Keeping up with constantly evolving K-fashion trends
- Finding the right color combinations for their complexion
- Translating K-pop and K-drama fashion inspiration into wearable outfits

## 💡 Solution

Our AI system provides personalized fashion recommendations by:

1. **Analyzing Your Features**: Using computer vision to determine face shape, skin tone, and undertones
2. **Understanding Korean Fashion**: Leveraging a comprehensive database of Korean fashion styles and trends
3. **Wardrobe Optimization**: Scanning your existing clothes to create new outfit combinations
4. **Personalized Recommendations**: Matching Korean fashion aesthetics with your unique characteristics

## 🔧 Technology Stack

### Machine Learning & AI
- **Computer Vision**: OpenCV, MediaPipe for face analysis
- **Deep Learning**: TensorFlow/Keras for fashion classification
- **Image Processing**: PIL, scikit-image for outfit analysis
- **Recommendation Engine**: Custom algorithms for style matching

### Backend
- **Language**: Python 3.8+
- **Core Libraries**: NumPy, Pandas, Scikit-learn
- **Image Processing**: OpenCV, PIL/Pillow
- **Data Analysis**: Matplotlib, Seaborn

### Frontend
- **Web Interface**: Streamlit for interactive dashboard
- **Visualization**: Custom CSS for Korean aesthetic UI
- **File Upload**: Multi-image support for wardrobe scanning

### Data Sources
- Korean Fashion Datasets (Kaggle)
- K-pop Idol Fashion Database
- Korean Street Fashion Collections
- Seasonal Color Analysis Data

## 🚀 Features in Detail

### 1. Personal Feature Analysis
```
✓ Face Shape Detection (Oval, Round, Square, Heart, Diamond)
✓ Skin Tone Analysis (Cool, Warm, Neutral undertones)
✓ Seasonal Color Palette Generation
✓ Body Type Estimation (Optional)
```

### 2. Korean Style Categories
- **K-Minimalism**: Clean lines, neutral colors, sophisticated simplicity
- **Korean Streetwear**: Oversized fits, layering, urban aesthetics  
- **Cute/Ulzzang Style**: Soft colors, feminine touches, youthful vibes
- **Korean Business**: Professional yet stylish workplace attire
- **K-pop Inspired**: Bold colors, statement pieces, trendy accessories

### 3. Wardrobe Analysis
```
✓ Clothing Item Recognition and Classification
✓ Color Extraction and Harmony Analysis
✓ Pattern and Texture Identification
✓ Style Compatibility Scoring
✓ Mix-and-Match Suggestions
```

### 4. Personalized Recommendations
- Daily outfit suggestions based on weather and occasion
- Color combinations that enhance your natural features
- Korean styling techniques and tips
- Shopping recommendations for missing wardrobe pieces
- Seasonal trend updates

## 📱 User Experience

### Step 1: Personal Analysis
Upload a clear selfie to receive your personalized style profile:
- Face shape analysis
- Skin tone determination
- Recommended Korean style categories

### Step 2: Style Preferences
Complete a brief questionnaire about:
- Lifestyle and occasions you dress for
- Preferred Korean fashion influences
- Comfort level with bold vs. subtle styles

### Step 3: Wardrobe Scan
Upload photos of your existing clothes to:
- Catalog your current wardrobe
- Identify styling opportunities
- Find missing pieces for complete looks

### Step 4: Get Styled!
Receive personalized recommendations including:
- Complete outfit combinations
- Korean styling techniques
- Color coordination tips
- Accessory suggestions

## 🎨 Sample Output

```
👤 Your Style Profile:
✓ Face Shape: Oval (versatile for most Korean hairstyles)
✓ Skin Tone: Cool undertone with Spring palette
✓ Recommended Style: Korean Minimalist with Cute accents

📅 Today's Recommendation:
👔 Cream oversized blazer + white cotton tee + wide-leg camel trousers
🎨 Add soft pink accessories (perfect for your cool undertones)
👜 Small crossbody bag in nude or blush pink
👟 White sneakers or loafers for Korean street style

💡 Korean Styling Tips:
- Tuck your tee loosely into trousers (Korean half-tuck style)
- Layer a delicate gold necklace for subtle elegance
- Try Korean gradient lip in coral pink
- Add a bucket hat for trendy Seoul street vibes

📸 Inspiration: Similar to BLACKPINK Rosé's off-duty style
```

## 🔮 Future Enhancements

- **Virtual Try-On**: AR integration for preview outfits
- **Shopping Integration**: Direct links to Korean fashion retailers
- **Social Features**: Share and rate outfit combinations
- **Seasonal Updates**: Automatic trend updates from Korean fashion weeks
- **Mobile App**: Dedicated mobile application with camera integration
- **Multi-Cultural Expansion**: Adapt to other Asian fashion styles

## 🎯 Target Audience

- **K-Fashion Enthusiasts**: People interested in Korean style but unsure where to start
- **Wardrobe Optimizers**: Those wanting to maximize their existing clothing
- **Busy Professionals**: Individuals needing quick, stylish outfit decisions
- **Korean Culture Fans**: K-pop and K-drama fans wanting to emulate their favorite stars
- **Fashion Students**: Learning about Korean fashion principles and color theory

## 🌟 Why This Project Matters

Korean fashion has become a global phenomenon, influencing style trends worldwide. However, many people struggle to adapt these trends to their personal features and lifestyle. This AI system bridges that gap by providing:

- **Personalized Guidance**: Not just trendy, but suitable for individual features
- **Cultural Authenticity**: Based on genuine Korean fashion principles
- **Practical Application**: Works with existing wardrobes
- **Accessibility**: Free alternative to expensive personal styling services
- **Educational Value**: Teaches fashion principles and color theory

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
OpenCV
TensorFlow
Streamlit
PIL/Pillow
NumPy, Pandas
Scikit-learn
```

### Installation
```bash
git clone https://github.com/yourusername/korean-fashion-ai-stylist
cd korean-fashion-ai-stylist
pip install -r requirements.txt
streamlit run app.py
```

## 🤝 Contributing

We welcome contributions! Areas where you can help:
- Expanding the Korean fashion database
- Improving color analysis algorithms
- Adding new style categories
- Enhancing the user interface
- Translation to multiple languages

## 📄 License

This project is licensed under the Chhery Chorn License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Korean Fashion Industry for inspiration
- Open-source computer vision community
- K-pop and K-drama styling teams
- Fashion color theory researchers
- Korean street fashion photographers

---

**Ready to discover your perfect Korean style?** 
Upload your photo and let our AI stylist guide you to fashion confidence! 💫

*Made with ❤️ for K-fashion lovers worldwide*




# Korean Fashion AI Stylist - Week 1-2 Development Plan

## 🎯 Week 1-2 Goals
- Set up project structure and environment
- Implement basic face detection and analysis
- Create skin tone detection algorithm
- Build face shape classification system
- Test with sample images and validate results

## 📅 Daily Breakdown

### **Day 1-2: Project Setup & Environment**
- [ ] Create project directory structure
- [ ] Set up virtual environment
- [ ] Install required libraries
- [ ] Create basic project files
- [ ] Test installation with simple OpenCV example

### **Day 3-4: Face Detection Foundation**
- [ ] Implement basic face detection using OpenCV
- [ ] Add MediaPipe face mesh for detailed landmarks
- [ ] Create face cropping and preprocessing functions
- [ ] Test with multiple face images

### **Day 5-7: Face Shape Analysis**
- [ ] Calculate key facial measurements
- [ ] Implement face shape classification algorithm
- [ ] Create face shape categories (oval, round, square, heart, diamond)
- [ ] Test and validate with sample images

### **Day 8-10: Skin Tone Detection**
- [ ] Implement skin tone extraction from face region
- [ ] Create color analysis algorithm
- [ ] Classify undertones (cool, warm, neutral)
- [ ] Generate seasonal color palette

### **Day 11-14: Integration & Testing**
- [ ] Combine face shape and skin tone analysis
- [ ] Create simple web interface with Streamlit
- [ ] Add image upload functionality
- [ ] Test entire pipeline with various images
- [ ] Debug and optimize performance

## 📁 Project Structure

```
korean_fashion_ai/
├── README.md
├── requirements.txt
├── app.py                          # Main Streamlit app
├── config/
│   └── settings.py                 # Configuration settings
├── src/
│   ├── __init__.py
│   ├── face_analysis/
│   │   ├── __init__.py
│   │   ├── face_detector.py        # Face detection logic
│   │   ├── face_shape.py           # Face shape analysis
│   │   └── skin_tone.py            # Skin tone detection
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py          # Image processing utilities
│   │   └── color_utils.py          # Color analysis utilities
│   └── korean_fashion/
│       ├── __init__.py
│       └── style_matcher.py        # Korean style matching (Week 3-4)
├── data/
│   ├── test_images/                # Sample images for testing
│   └── color_palettes/             # Color palette data
├── models/                         # Saved models (future)
├── static/                         # CSS and assets for Streamlit
└── tests/
    ├── __init__.py
    └── test_face_analysis.py       # Unit tests
```

## 🛠️ Technical Implementation Details

### Required Libraries
```python
opencv-python==4.8.1.78
mediapipe==0.10.3
numpy==1.24.3
pillow==10.0.1
streamlit==1.28.1
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
pandas==2.0.3
colorthief==0.2.1
```

### Key Algorithms to Implement

#### 1. Face Shape Classification
```python
# Measurements needed:
- Face width-to-height ratio
- Jawline angle
- Cheekbone width
- Forehead width
- Chin width and shape
```

#### 2. Skin Tone Analysis
```python
# Color extraction points:
- Forehead center
- Both cheeks
- Nose bridge
- Chin area
- Under-eye area
```

### Expected Deliverables by End of Week 2

1. **Working face detection system** that can identify faces in uploaded images
2. **Face shape classifier** that determines one of 5 face shapes with confidence score
3. **Skin tone analyzer** that extracts dominant skin color and determines undertones
4. **Basic web interface** where users can upload photos and get analysis results
5. **Test suite** with sample images showing accurate results

## 🎨 Week 1-2 User Interface Preview

The Streamlit app should include:
- **Photo Upload Area**: Drag & drop or file selection
- **Analysis Results Panel**: 
  - Detected face with landmarks
  - Face shape classification with confidence
  - Skin tone color palette
  - Undertone determination
- **Debug Info**: Processing steps and measurements (toggleable)

## 📊 Success Metrics for Week 1-2

- [ ] Face detection works on 95%+ of clear face photos
- [ ] Face shape classification achieves reasonable accuracy on test images
- [ ] Skin tone detection produces consistent results across different lighting
- [ ] Web interface is responsive and user-friendly
- [ ] Processing time under 3 seconds per image
- [ ] No crashes or errors with various image formats

## 🚀 Ready to Start?

Let's begin with Day 1-2: Project Setup! I'll help you create each file step by step.

**Next Steps:**
1. First, let's set up the project structure and requirements.txt
2. Then create the basic face detection functionality
3. Build upon that with face shape and skin tone analysis
4. Finally, integrate everything into a working Streamlit app

Are you ready to start coding? Let me know and I'll guide you through creating the first files!