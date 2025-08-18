"""
Skin Tone Analysis Module for Korean Fashion AI Personal Stylist
Week 1-2: Analyze skin tone and undertones for Korean fashion color recommendations
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
import colorsys

class SkinToneAnalyzer:
    def __init__(self):
        """
        Initialize skin tone analyzer with Korean beauty color standards
        """
        # Korean seasonal color analysis (based on Korean beauty standards)
        self.korean_seasons = {
            'spring': {
                'name': 'Spring (봄 웜톤)',
                'description': 'Warm, bright, light colors',
                'characteristics': ['Warm undertone', 'Clear complexion', 'Bright colors suit best'],
                'colors': ['#FFB6C1', '#F0E68C', '#98FB98', '#87CEEB', '#DDA0DD', '#FFA07A', '#20B2AA'],
                'avoid_colors': ['#000000', '#800000', '#000080', '#4B0082']
            },
            'summer': {
                'name': 'Summer (여름 쿨톤)',
                'description': 'Cool, soft, muted colors',
                'characteristics': ['Cool undertone', 'Soft complexion', 'Muted colors suit best'],
                'colors': ['#E6E6FA', '#F0F8FF', '#E0FFFF', '#F5FFFA', '#FFF8DC', '#B0C4DE', '#D8BFD8'],
                'avoid_colors': ['#FF4500', '#FF8C00', '#DAA520', '#DC143C']
            },
            'autumn': {
                'name': 'Autumn (가을 웜톤)',
                'description': 'Warm, deep, rich colors',
                'characteristics': ['Warm undertone', 'Rich complexion', 'Earth tones suit best'],
                'colors': ['#CD853F', '#D2691E', '#B22222', '#8B4513', '#A0522D', '#BC8F8F', '#F4A460'],
                'avoid_colors': ['#FF1493', '#00CED1', '#7B68EE', '#FF69B4']
            },
            'winter': {
                'name': 'Winter (겨울 쿨톤)',
                'description': 'Cool, clear, bold colors',
                'characteristics': ['Cool undertone', 'Clear complexion', 'Bold colors suit best'],
                'colors': ['#000000', '#FFFFFF', '#FF0000', '#0000FF', '#800080', '#008000', '#FF1493'],
                'avoid_colors': ['#F0E68C', '#DEB887', '#CD853F', '#D2691E']
            }
        }
        
        # Skin tone regions for analysis (relative coordinates)
        self.analysis_regions = {
            'forehead': (0.3, 0.15, 0.4, 0.25),
            'left_cheek': (0.15, 0.4, 0.2, 0.2),
            'right_cheek': (0.65, 0.4, 0.2, 0.2),
            'nose_bridge': (0.4, 0.35, 0.2, 0.15),
            'chin': (0.35, 0.7, 0.3, 0.15)
        }
        
        # Korean makeup base tone matching
        self.korean_base_tones = {
            'light_cool': {'name': '라이트 쿨', 'rgb': (245, 235, 225), 'hex': '#F5EBE1'},
            'light_warm': {'name': '라이트 웜', 'rgb': (250, 240, 220), 'hex': '#FAF0DC'},
            'medium_cool': {'name': '미디움 쿨', 'rgb': (225, 210, 195), 'hex': '#E1D2C3'},
            'medium_warm': {'name': '미디움 웜', 'rgb': (235, 215, 190), 'hex': '#EBD7BE'},
            'deep_cool': {'name': '딥 쿨', 'rgb': (200, 180, 160), 'hex': '#C8B4A0'},
            'deep_warm': {'name': '딥 웜', 'rgb': (210, 185, 155), 'hex': '#D2B99B'}
        }
    
    def extract_skin_regions(self, image: np.ndarray, landmarks: Dict) -> Dict[str, np.ndarray]:
        """
        Extract skin regions from face for analysis
        
        Args:
            image: Input face image
            landmarks: Facial landmarks
            
        Returns:
            Dictionary of skin region images
        """
        if 'all' not in landmarks or len(landmarks['all']) == 0:
            return {}
            
        h, w = image.shape[:2]
        skin_regions = {}
        
        # Use landmarks to define more precise skin regions
        points = np.array(landmarks['all'])
        
        # Extract regions based on facial landmarks
        regions = {
            'forehead': self._get_forehead_region(image, points),
            'left_cheek': self._get_cheek_region(image, points, 'left'),
            'right_cheek': self._get_cheek_region(image, points, 'right'),
            'nose_bridge': self._get_nose_region(image, points),
            'chin': self._get_chin_region(image, points)
        }
        
        # Filter out empty regions
        skin_regions = {name: region for name, region in regions.items() if region is not None}
        
        return skin_regions
    
    def _get_forehead_region(self, image: np.ndarray, points: np.ndarray) -> Optional[np.ndarray]:
        """Extract forehead region using landmarks"""
        try:
            # Use forehead landmarks (approximate area above eyebrows)
            forehead_points = [10, 151, 9, 10, 151, 195, 197, 196]
            valid_points = [points[i] for i in forehead_points if i < len(points)]
            
            if len(valid_points) < 4:
                return None
                
            # Create bounding box
            coords = np.array(valid_points)
            x_min, y_min = np.min(coords, axis=0)
            x_max, y_max = np.max(coords, axis=0)
            
            # Add some padding
            padding = int(min(x_max - x_min, y_max - y_min) * 0.2)
            y_min = max(0, y_min - padding)
            
            return image[int(y_min):int(y_max), int(x_min):int(x_max)]
            
        except Exception:
            return None
    
    def _get_cheek_region(self, image: np.ndarray, points: np.ndarray, side: str) -> Optional[np.ndarray]:
        """Extract cheek region using landmarks"""
        try:
            if side == 'left':
                # Left cheek landmarks
                cheek_points = [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206]
            else:
                # Right cheek landmarks  
                cheek_points = [345, 346, 347, 348, 349, 350, 355, 371, 266, 425, 426]
            
            valid_points = [points[i] for i in cheek_points if i < len(points)]
            
            if len(valid_points) < 4:
                return None
                
            coords = np.array(valid_points)
            x_min, y_min = np.min(coords, axis=0)
            x_max, y_max = np.max(coords, axis=0)
            
            return image[int(y_min):int(y_max), int(x_min):int(x_max)]
            
        except Exception:
            return None
    
    def _get_nose_region(self, image: np.ndarray, points: np.ndarray) -> Optional[np.ndarray]:
        """Extract nose bridge region using landmarks"""
        try:
            # Nose bridge landmarks
            nose_points = [6, 168, 8, 9, 10, 151, 195, 197, 196, 3]
            valid_points = [points[i] for i in nose_points if i < len(points)]
            
            if len(valid_points) < 4:
                return None
                
            coords = np.array(valid_points)
            x_min, y_min = np.min(coords, axis=0)
            x_max, y_max = np.max(coords, axis=0)
            
            return image[int(y_min):int(y_max), int(x_min):int(x_max)]
            
        except Exception:
            return None
    
    def _get_chin_region(self, image: np.ndarray, points: np.ndarray) -> Optional[np.ndarray]:
        """Extract chin region using landmarks"""
        try:
            # Chin area landmarks
            chin_points = [175, 176, 0, 17, 18, 200, 199, 175, 421, 418, 424, 422]
            valid_points = [points[i] for i in chin_points if i < len(points)]
            
            if len(valid_points) < 4:
                return None
                
            coords = np.array(valid_points)
            x_min, y_min = np.min(coords, axis=0)
            x_max, y_max = np.max(coords, axis=0)
            
            return image[int(y_min):int(y_max), int(x_min):int(x_max)]
            
        except Exception:
            return None
    
    def analyze_skin_color(self, skin_regions: Dict[str, np.ndarray]) -> Dict:
        """
        Analyze skin color from extracted regions
        
        Args:
            skin_regions: Dictionary of skin region images
            
        Returns:
            Skin color analysis results
        """
        if not skin_regions:
            return {}
            
        all_skin_pixels = []
        region_colors = {}
        
        # Extract colors from each region
        for region_name, region_image in skin_regions.items():
            if region_image is None or region_image.size == 0:
                continue
                
            # Convert to RGB if needed
            if len(region_image.shape) == 3:
                region_rgb = cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB)
            else:
                continue
                
            # Reshape to pixel array
            pixels = region_rgb.reshape(-1, 3)
            
            # Remove extreme values (shadows, highlights)
            pixels = self._filter_skin_pixels(pixels)
            
            if len(pixels) > 0:
                all_skin_pixels.extend(pixels)
                region_colors[region_name] = np.mean(pixels, axis=0)
        
        if not all_skin_pixels:
            return {}
            
        # Analyze overall skin tone
        all_pixels = np.array(all_skin_pixels)
        dominant_color = self._get_dominant_color(all_pixels)
        
        # Analyze undertones
        undertone_analysis = self._analyze_undertones(all_pixels)
        
        # Match to Korean seasonal colors
        seasonal_analysis = self._match_korean_season(dominant_color, undertone_analysis)
        
        # Match to Korean base tones
        base_tone_match = self._match_korean_base_tone(dominant_color)
        
        return {
            'dominant_color': {
                'rgb': tuple(dominant_color.astype(int)),
                'hex': self._rgb_to_hex(dominant_color)
            },
            'region_colors': {name: {
                'rgb': tuple(color.astype(int)),
                'hex': self._rgb_to_hex(color)
            } for name, color in region_colors.items()},
            'undertone': undertone_analysis,
            'korean_season': seasonal_analysis,
            'korean_base_tone': base_tone_match,
            'color_palette': self._generate_color_palette(seasonal_analysis)
        }
    
    def _filter_skin_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """Filter out non-skin pixels (shadows, highlights, etc.)"""
        # Remove very dark or very bright pixels
        brightness = np.mean(pixels, axis=1)
        mask = (brightness > 30) & (brightness < 240)
        
        # Remove pixels with extreme color values
        for channel in range(3):
            channel_mask = (pixels[:, channel] > 20) & (pixels[:, channel] < 250)
            mask = mask & channel_mask
        
        return pixels[mask]
    
    def _get_dominant_color(self, pixels: np.ndarray, n_clusters: int = 3) -> np.ndarray:
        """Get dominant color using K-means clustering"""
        if len(pixels) < n_clusters:
            return np.mean(pixels, axis=0)
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Return the cluster center with the most points
        labels = kmeans.labels_
        cluster_counts = np.bincount(labels)
        dominant_cluster = np.argmax(cluster_counts)
        
        return kmeans.cluster_centers_[dominant_cluster]
    
    def _analyze_undertones(self, pixels: np.ndarray) -> Dict:
        """Analyze skin undertones (warm/cool/neutral)"""
        avg_color = np.mean(pixels, axis=0)
        r, g, b = avg_color
        
        # Convert to different color spaces for undertone analysis
        hsv_color = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        hue = hsv_color[0] * 360
        
        # Analyze color ratios
        total = r + g + b
        red_ratio = r / total
        green_ratio = g / total
        blue_ratio = b / total
        
        # Korean undertone classification
        undertone_scores = {
            'warm': 0,
            'cool': 0,
            'neutral': 0
        }
        
        # Warm indicators (yellow/golden undertones)
        if red_ratio > 0.35 and green_ratio > 0.33:
            undertone_scores['warm'] += 1
        if 30 <= hue <= 60:  # Yellow-orange range
            undertone_scores['warm'] += 1
        if r > b and g > b:  # More red and green than blue
            undertone_scores['warm'] += 1
            
        # Cool indicators (pink/blue undertones)
        if blue_ratio > 0.32:
            undertone_scores['cool'] += 1
        if red_ratio > green_ratio and red_ratio > blue_ratio and red_ratio > 0.37:  # Pink undertones
            undertone_scores['cool'] += 1
        if 300 <= hue <= 360 or 0 <= hue <= 30:  # Pink-red range
            undertone_scores['cool'] += 1
            
        # Neutral indicators
        color_variance = np.var([red_ratio, green_ratio, blue_ratio])
        if color_variance < 0.001:  # Balanced ratios
            undertone_scores['neutral'] += 2
            
        # Determine primary undertone
        primary_undertone = max(undertone_scores, key=undertone_scores.get)
        
        # If scores are too close, classify as neutral
        max_score = max(undertone_scores.values())
        if max_score <= 1 or (max_score - min(undertone_scores.values())) <= 1:
            primary_undertone = 'neutral'
            
        return {
            'primary': primary_undertone,
            'scores': undertone_scores,
            'color_analysis': {
                'red_ratio': red_ratio,
                'green_ratio': green_ratio,
                'blue_ratio': blue_ratio,
                'hue': hue
            }
        }
    
    def _match_korean_season(self, dominant_color: np.ndarray, undertone_analysis: Dict) -> Dict:
        """Match skin tone to Korean seasonal color analysis"""
        r, g, b = dominant_color
        brightness = (r + g + b) / 3
        saturation = (max(r, g, b) - min(r, g, b)) / max(r, g, b) if max(r, g, b) > 0 else 0
        
        undertone = undertone_analysis['primary']
        
        # Korean seasonal classification logic
        season_scores = {}
        
        # Spring (봄 웜톤): Warm, bright, light
        spring_score = 0
        if undertone == 'warm':
            spring_score += 2
        if brightness > 180:  # Light skin
            spring_score += 1
        if saturation > 0.3:  # Clear colors
            spring_score += 1
        season_scores['spring'] = spring_score
        
        # Summer (여름 쿨톤): Cool, soft, muted
        summer_score = 0
        if undertone == 'cool':
            summer_score += 2
        if 120 < brightness < 200:  # Medium-light skin
            summer_score += 1
        if saturation < 0.4:  # Muted colors
            summer_score += 1
        season_scores['summer'] = summer_score
        
        # Autumn (가을 웜톤): Warm, deep, rich
        autumn_score = 0
        if undertone == 'warm':
            autumn_score += 2
        if brightness < 160:  # Deeper skin
            autumn_score += 1
        if saturation > 0.4:  # Rich colors
            autumn_score += 1
        season_scores['autumn'] = autumn_score
        
        # Winter (겨울 쿨톤): Cool, clear, bold
        winter_score = 0
        if undertone == 'cool':
            winter_score += 2
        if brightness < 140 or brightness > 200:  # Very light or deep
            winter_score += 1
        if saturation > 0.5:  # Bold colors
            winter_score += 1
        season_scores['winter'] = winter_score
        
        # Find best matching season
        best_season = max(season_scores, key=season_scores.get)
        confidence = season_scores[best_season] / 4.0  # Normalize to 0-1
        
        return {
            'season': best_season,
            'korean_name': self.korean_seasons[best_season]['name'],
            'description': self.korean_seasons[best_season]['description'],
            'confidence': confidence,
            'scores': season_scores,
            'characteristics': self.korean_seasons[best_season]['characteristics']
        }
    
    def _match_korean_base_tone(self, dominant_color: np.ndarray) -> Dict:
        """Match to Korean makeup base tones"""
        r, g, b = dominant_color
        
        best_match = None
        min_distance = float('inf')
        
        for tone_name, tone_info in self.korean_base_tones.items():
            tone_r, tone_g, tone_b = tone_info['rgb']
            
            # Calculate color distance
            distance = np.sqrt((r - tone_r)**2 + (g - tone_g)**2 + (b - tone_b)**2)
            
            if distance < min_distance:
                min_distance = distance
                best_match = tone_name
        
        if best_match:
            return {
                'match': best_match,
                'korean_name': self.korean_base_tones[best_match]['name'],
                'rgb': self.korean_base_tones[best_match]['rgb'],
                'hex': self.korean_base_tones[best_match]['hex'],
                'distance': min_distance
            }
        
        return {}
    
    def _generate_color_palette(self, seasonal_analysis: Dict) -> Dict:
        """Generate personalized color palette based on season"""
        season = seasonal_analysis.get('season', 'spring')
        
        if season in self.korean_seasons:
            season_info = self.korean_seasons[season]
            return {
                'recommended_colors': season_info['colors'],
                'avoid_colors': season_info['avoid_colors'],
                'season_name': season_info['name'],
                'palette_description': season_info['description']
            }
        
        return {}
    
    def _rgb_to_hex(self, rgb: np.ndarray) -> str:
        """Convert RGB to hex color code"""
        r, g, b = rgb.astype(int)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def get_korean_makeup_recommendations(self, skin_analysis: Dict) -> Dict:
        """
        Get Korean makeup recommendations based on skin analysis
        
        Args:
            skin_analysis: Results from analyze_skin_color
            
        Returns:
            Korean makeup recommendations
        """
        if not skin_analysis:
            return {}
        
        season = skin_analysis.get('korean_season', {}).get('season', 'spring')
        undertone = skin_analysis.get('undertone', {}).get('primary', 'neutral')
        base_tone = skin_analysis.get('korean_base_tone', {})
        
        recommendations = {
            'base_makeup': self._get_base_makeup_tips(base_tone, undertone),
            'lip_colors': self._get_korean_lip_recommendations(season, undertone),
            'blush_colors': self._get_korean_blush_recommendations(season, undertone),
            'eyeshadow_colors': self._get_korean_eyeshadow_recommendations(season, undertone),
            'styling_tips': self._get_korean_styling_tips(season)
        }
        
        return recommendations
    
    def _get_base_makeup_tips(self, base_tone: Dict, undertone: str) -> List[str]:
        """Get Korean base makeup recommendations"""
        tips = []
        
        if base_tone:
            korean_name = base_tone.get('korean_name', '')
            tips.append(f"Use Korean base in {korean_name} shade")
        
        if undertone == 'warm':
            tips.extend([
                "Choose Korean BB cream with yellow undertones",
                "Use peachy Korean blush",
                "Try Korean dewy finish foundation"
            ])
        elif undertone == 'cool':
            tips.extend([
                "Choose Korean BB cream with pink undertones",
                "Use Korean cushion foundation for natural glow",
                "Try rose-toned Korean blush"
            ])
        else:  # neutral
            tips.extend([
                "You can wear both warm and cool Korean base tones",
                "Try Korean gradient makeup techniques",
                "Experiment with different Korean cushion shades"
            ])
        
        return tips
    
    def _get_korean_lip_recommendations(self, season: str, undertone: str) -> List[str]:
        """Get Korean lip color recommendations"""
        lip_colors = {
            'spring': [
                "Coral pink Korean tint",
                "Peach Korean gradient lip",
                "Light berry Korean velvet tint",
                "Clear pink Korean lip gloss"
            ],
            'summer': [
                "Rose pink Korean tint",
                "Mauve Korean gradient lip",
                "Soft berry Korean lip stain",
                "Cool pink Korean lip balm"
            ],
            'autumn': [
                "Warm red Korean tint",
                "Brick red Korean gradient lip",
                "Deep orange Korean lip color",
                "Brown-red Korean velvet lip"
            ],
            'winter': [
                "True red Korean tint",
                "Deep berry Korean gradient lip",
                "Bold pink Korean lip color",
                "Classic red Korean matte lip"
            ]
        }
        
        return lip_colors.get(season, lip_colors['spring'])
    
    def _get_korean_blush_recommendations(self, season: str, undertone: str) -> List[str]:
        """Get Korean blush recommendations"""
        blush_colors = {
            'spring': ["Peachy coral", "Light apricot", "Warm pink", "Soft orange"],
            'summer': ["Rose pink", "Dusty pink", "Mauve", "Cool berry"],
            'autumn': ["Warm terracotta", "Deep peach", "Burnt orange", "Rich coral"],
            'winter': ["True pink", "Deep rose", "Berry", "Bold coral"]
        }
        
        return blush_colors.get(season, blush_colors['spring'])
    
    def _get_korean_eyeshadow_recommendations(self, season: str, undertone: str) -> List[str]:
        """Get Korean eyeshadow recommendations"""
        eyeshadow_colors = {
            'spring': ["Warm golden brown", "Peach shimmer", "Coral pink", "Light bronze"],
            'summer': ["Cool taupe", "Rose gold", "Dusty pink", "Soft purple"],
            'autumn': ["Deep brown", "Golden bronze", "Warm copper", "Rich orange"],
            'winter': ["Deep plum", "Silver", "Bold pink", "Classic black"]
        }
        
        return eyeshadow_colors.get(season, eyeshadow_colors['spring'])
    
    def _get_korean_styling_tips(self, season: str) -> List[str]:
        """Get general Korean styling tips for the season"""
        styling_tips = {
            'spring': [
                "Korean glass skin makeup with dewy finish",
                "Use Korean gradient lip technique",
                "Try Korean puppy eye makeup",
                "Focus on fresh, youthful Korean look"
            ],
            'summer': [
                "Korean no-makeup makeup look",
                "Use Korean cushion for natural coverage",
                "Try Korean straight brows",
                "Focus on elegant, refined Korean style"
            ],
            'autumn': [
                "Korean warm-toned makeup",
                "Use Korean contouring for definition",
                "Try Korean smoky eye with warm tones",
                "Focus on sophisticated Korean look"
            ],
            'winter': [
                "Korean bold lip with minimal eyes",
                "Use Korean highlighting for drama",
                "Try Korean cat eye makeup",
                "Focus on chic, modern Korean style"
            ]
        }
        
        return styling_tips.get(season, styling_tips['spring'])
    
    def visualize_skin_analysis(self, image: np.ndarray, skin_analysis: Dict, 
                              skin_regions: Dict = None) -> np.ndarray:
        """
        Create visualization of skin tone analysis
        
        Args:
            image: Original image
            skin_analysis: Skin analysis results
            skin_regions: Skin regions (optional)
            
        Returns:
            Annotated image with skin analysis
        """
        result_image = image.copy()
        h, w = result_image.shape[:2]
        
        # Create color palette bar
        palette_height = 60
        palette_y = h - palette_height - 10
        
        # Get color palette
        color_palette = skin_analysis.get('color_palette', {})
        recommended_colors = color_palette.get('recommended_colors', [])
        
        if recommended_colors:
            color_width = min(w // len(recommended_colors), 80)
            
            for i, hex_color in enumerate(recommended_colors):
                # Convert hex to BGR for OpenCV
                rgb = tuple(int(hex_color[1:][j:j+2], 16) for j in (0, 2, 4))
                bgr = (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR
                
                x1 = 10 + i * color_width
                x2 = x1 + color_width - 5
                y1 = palette_y
                y2 = palette_y + palette_height
                
                cv2.rectangle(result_image, (x1, y1), (x2, y2), bgr, -1)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 0), 2)
        
        # Add text information
        dominant_color = skin_analysis.get('dominant_color', {})
        korean_season = skin_analysis.get('korean_season', {})
        undertone = skin_analysis.get('undertone', {})
        
        text_lines = []
        if korean_season:
            text_lines.append(f"Korean Season: {korean_season.get('korean_name', 'Unknown')}")
        if undertone:
            text_lines.append(f"Undertone: {undertone.get('primary', 'Unknown').title()}")
        if dominant_color:
            hex_color = dominant_color.get('hex', '#000000')
            text_lines.append(f"Skin Tone: {hex_color}")
        
        # Add background for text
        if text_lines:
            text_bg_height = len(text_lines) * 25 + 10
            cv2.rectangle(result_image, (10, 10), (300, text_bg_height), (0, 0, 0), -1)
            
            for i, line in enumerate(text_lines):
                y_pos = 30 + i * 25
                cv2.putText(result_image, line, (15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image