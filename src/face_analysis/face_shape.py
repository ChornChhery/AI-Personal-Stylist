"""
Face Shape Analysis Module for Korean Fashion AI Personal Stylist
Week 1-2: Determine face shape from facial landmarks for Korean fashion styling
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import math

class FaceShapeAnalyzer:
    def __init__(self):
        """
        Initialize face shape analyzer with Korean beauty standards
        Korean face shape preferences and styling guidelines
        """
        # Define face shape categories with Korean beauty context
        self.face_shapes = {
            'oval': {
                'name': 'Oval (계란형)',
                'description': 'Balanced proportions, ideal for most Korean hairstyles',
                'korean_style_tips': [
                    'Perfect for Korean side-swept bangs',
                    'Suits both straight and wavy Korean hairstyles',
                    'Can wear bold Korean accessories'
                ]
            },
            'round': {
                'name': 'Round (둥근형)',
                'description': 'Soft curves, youthful appearance',
                'korean_style_tips': [
                    'Korean layered cuts add dimension',
                    'Avoid blunt bangs, try side-swept styles',
                    'Angular glasses frames balance roundness'
                ]
            },
            'square': {
                'name': 'Square (사각형)',
                'description': 'Strong jawline, angular features',
                'korean_style_tips': [
                    'Soft Korean waves soften angular features',
                    'Korean gradient lip makeup creates balance',
                    'Avoid straight-across bangs'
                ]
            },
            'heart': {
                'name': 'Heart (하트형)',
                'description': 'Wide forehead, narrow chin',
                'korean_style_tips': [
                    'Korean curtain bangs balance forehead',
                    'Focus blush on lower cheeks',
                    'Suits Korean bob cuts'
                ]
            },
            'diamond': {
                'name': 'Diamond (다이아몬드형)',
                'description': 'Wide cheekbones, narrow forehead and chin',
                'korean_style_tips': [
                    'Korean face-framing layers are ideal',
                    'Highlight eyes with Korean makeup techniques',
                    'Avoid center parts'
                ]
            }
        }
        
        # Measurement ratios for face shape classification
        self.shape_ratios = {
            'width_height': {'oval': (0.75, 0.85), 'round': (0.85, 1.0), 
                           'square': (0.8, 0.95), 'heart': (0.7, 0.8), 'diamond': (0.65, 0.75)},
            'jaw_cheek': {'oval': (0.8, 0.95), 'round': (0.9, 1.0), 
                         'square': (0.9, 1.1), 'heart': (0.6, 0.8), 'diamond': (0.7, 0.85)},
            'forehead_cheek': {'oval': (0.85, 0.95), 'round': (0.8, 0.9), 
                              'square': (0.85, 0.95), 'heart': (1.0, 1.2), 'diamond': (0.7, 0.85)}
        }
    
    def calculate_face_measurements(self, landmarks: Dict) -> Dict:
        """
        Calculate key facial measurements from landmarks
        
        Args:
            landmarks: Facial landmark coordinates
            
        Returns:
            Dictionary of facial measurements
        """
        measurements = {}
        
        if 'all' not in landmarks or len(landmarks['all']) < 468:
            return measurements
            
        # Convert landmarks to numpy array for easier processing
        points = np.array(landmarks['all'])
        
        # Key landmark indices for measurements (MediaPipe 468 landmarks)
        # Face outline points
        face_top = points[10]      # Top of face
        face_bottom = points[152]  # Bottom of chin
        left_cheek = points[234]   # Left cheek
        right_cheek = points[454]  # Right cheek
        
        # Jawline points
        left_jaw = points[172]     # Left jaw
        right_jaw = points[397]    # Right jaw
        
        # Forehead points (approximate)
        forehead_left = points[21]
        forehead_right = points[251]
        
        # Calculate face dimensions
        face_height = self._calculate_distance(face_top, face_bottom)
        face_width = self._calculate_distance(left_cheek, right_cheek)
        jaw_width = self._calculate_distance(left_jaw, right_jaw)
        forehead_width = self._calculate_distance(forehead_left, forehead_right)
        
        # Calculate ratios
        measurements = {
            'face_height': face_height,
            'face_width': face_width,
            'jaw_width': jaw_width,
            'forehead_width': forehead_width,
            'width_height_ratio': face_width / face_height if face_height > 0 else 0,
            'jaw_cheek_ratio': jaw_width / face_width if face_width > 0 else 0,
            'forehead_cheek_ratio': forehead_width / face_width if face_width > 0 else 0,
            'face_center': ((face_top[0] + face_bottom[0]) // 2, (face_top[1] + face_bottom[1]) // 2)
        }
        
        # Additional Korean beauty measurements
        measurements.update(self._calculate_korean_proportions(landmarks))
        
        return measurements
    
    def _calculate_distance(self, point1: Tuple, point2: Tuple) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _calculate_korean_proportions(self, landmarks: Dict) -> Dict:
        """
        Calculate proportions important in Korean beauty standards
        
        Args:
            landmarks: Facial landmark coordinates
            
        Returns:
            Korean beauty measurements
        """
        if 'all' not in landmarks:
            return {}
            
        points = np.array(landmarks['all'])
        
        # Korean beauty golden ratios
        korean_measurements = {}
        
        try:
            # Eye to eyebrow distance (Korean makeup important)
            left_eye_top = points[159]
            left_eyebrow = points[70]
            eye_brow_distance = self._calculate_distance(left_eye_top, left_eyebrow)
            
            # Nose bridge to tip ratio (Korean beauty standard)
            nose_bridge = points[6]
            nose_tip = points[1]
            nose_length = self._calculate_distance(nose_bridge, nose_tip)
            
            # Lip ratio (important for Korean gradient lip)
            upper_lip = points[13]
            lower_lip = points[14]
            lip_thickness = self._calculate_distance(upper_lip, lower_lip)
            
            korean_measurements.update({
                'eye_brow_distance': eye_brow_distance,
                'nose_length': nose_length,
                'lip_thickness': lip_thickness,
                'korean_beauty_score': self._calculate_korean_beauty_score(points)
            })
            
        except (IndexError, TypeError):
            # Handle cases where specific landmarks aren't available
            pass
            
        return korean_measurements
    
    def _calculate_korean_beauty_score(self, points: np.ndarray) -> float:
        """
        Calculate a beauty score based on Korean beauty standards
        This is a simplified version for educational purposes
        """
        try:
            # Korean beauty often emphasizes:
            # 1. Small face ratio
            # 2. V-shaped jawline
            # 3. Large eyes relative to face
            # 4. Balanced proportions
            
            face_top = points[10]
            face_bottom = points[152]
            left_face = points[234]
            right_face = points[454]
            
            face_area = abs((face_top[1] - face_bottom[1]) * (right_face[0] - left_face[0]))
            
            # Eye area approximation
            left_eye_area = self._calculate_eye_area(points, 'left')
            right_eye_area = self._calculate_eye_area(points, 'right')
            total_eye_area = left_eye_area + right_eye_area
            
            # Calculate eye-to-face ratio (Korean beauty prefers larger eyes)
            eye_face_ratio = total_eye_area / face_area if face_area > 0 else 0
            
            # V-line jaw calculation
            jaw_angle = self._calculate_jaw_angle(points)
            v_line_score = max(0, (180 - jaw_angle) / 180)  # Sharper jaw = higher score
            
            # Combine factors for Korean beauty score (0-1 scale)
            beauty_score = (eye_face_ratio * 0.4 + v_line_score * 0.3 + 
                          self._calculate_symmetry_score(points) * 0.3)
            
            return min(1.0, beauty_score)
            
        except Exception:
            return 0.5  # Default neutral score
    
    def _calculate_eye_area(self, points: np.ndarray, side: str) -> float:
        """Calculate approximate eye area"""
        try:
            if side == 'left':
                eye_points = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173]
            else:  # right
                eye_points = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466]
            
            eye_coords = points[eye_points]
            # Simple bounding box area calculation
            min_x, min_y = np.min(eye_coords, axis=0)
            max_x, max_y = np.max(eye_coords, axis=0)
            return (max_x - min_x) * (max_y - min_y)
            
        except Exception:
            return 0
    
    def _calculate_jaw_angle(self, points: np.ndarray) -> float:
        """Calculate jaw angle for V-line assessment"""
        try:
            # Use jawline points to calculate angle
            left_jaw = points[172]
            chin = points[152]
            right_jaw = points[397]
            
            # Calculate vectors
            vec1 = np.array([left_jaw[0] - chin[0], left_jaw[1] - chin[1]])
            vec2 = np.array([right_jaw[0] - chin[0], right_jaw[1] - chin[1]])
            
            # Calculate angle between vectors
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
            
            return angle
            
        except Exception:
            return 120  # Default moderate angle
    
    def _calculate_symmetry_score(self, points: np.ndarray) -> float:
        """Calculate facial symmetry score"""
        try:
            # Compare left and right side distances
            center_x = np.mean(points[:, 0])
            
            # Sample points for symmetry check
            symmetry_pairs = [
                (234, 454),  # Cheeks
                (172, 397),  # Jaw
                (33, 362),   # Eyes outer corners
            ]
            
            symmetry_scores = []
            for left_idx, right_idx in symmetry_pairs:
                left_dist = abs(points[left_idx][0] - center_x)
                right_dist = abs(points[right_idx][0] - center_x)
                
                if max(left_dist, right_dist) > 0:
                    symmetry = 1 - abs(left_dist - right_dist) / max(left_dist, right_dist)
                    symmetry_scores.append(symmetry)
            
            return np.mean(symmetry_scores) if symmetry_scores else 0.5
            
        except Exception:
            return 0.5
    
    def classify_face_shape(self, measurements: Dict) -> Dict:
        """
        Classify face shape based on measurements
        
        Args:
            measurements: Facial measurements dictionary
            
        Returns:
            Face shape classification with confidence scores
        """
        if not measurements:
            return {'shape': 'unknown', 'confidence': 0.0, 'scores': {}}
            
        # Extract key ratios
        width_height = measurements.get('width_height_ratio', 0)
        jaw_cheek = measurements.get('jaw_cheek_ratio', 0)
        forehead_cheek = measurements.get('forehead_cheek_ratio', 0)
        
        # Calculate confidence scores for each face shape
        shape_scores = {}
        
        for shape, ratios in self.shape_ratios.items():
            score = 0
            count = 0
            
            # Width-height ratio score
            if 'width_height' in ratios:
                min_ratio, max_ratio = ratios['width_height']
                if min_ratio <= width_height <= max_ratio:
                    score += 1 - abs(width_height - (min_ratio + max_ratio) / 2) / ((max_ratio - min_ratio) / 2)
                count += 1
            
            # Jaw-cheek ratio score
            if 'jaw_cheek' in ratios:
                min_ratio, max_ratio = ratios['jaw_cheek']
                if min_ratio <= jaw_cheek <= max_ratio:
                    score += 1 - abs(jaw_cheek - (min_ratio + max_ratio) / 2) / ((max_ratio - min_ratio) / 2)
                count += 1
            
            # Forehead-cheek ratio score
            if 'forehead_cheek' in ratios:
                min_ratio, max_ratio = ratios['forehead_cheek']
                if min_ratio <= forehead_cheek <= max_ratio:
                    score += 1 - abs(forehead_cheek - (min_ratio + max_ratio) / 2) / ((max_ratio - min_ratio) / 2)
                count += 1
            
            # Average score for this shape
            shape_scores[shape] = score / count if count > 0 else 0
        
        # Find best matching shape
        best_shape = max(shape_scores, key=shape_scores.get) if shape_scores else 'oval'
        best_confidence = shape_scores.get(best_shape, 0.0)
        
        # If confidence is too low, default to oval
        if best_confidence < 0.3:
            best_shape = 'oval'
            best_confidence = 0.5
        
        return {
            'shape': best_shape,
            'confidence': best_confidence,
            'scores': shape_scores,
            'measurements': measurements
        }
    
    def get_korean_styling_recommendations(self, face_shape: str) -> Dict:
        """
        Get Korean fashion and beauty recommendations for face shape
        
        Args:
            face_shape: Detected face shape
            
        Returns:
            Korean styling recommendations
        """
        if face_shape not in self.face_shapes:
            face_shape = 'oval'  # Default fallback
            
        shape_info = self.face_shapes[face_shape]
        
        return {
            'face_shape': shape_info['name'],
            'description': shape_info['description'],
            'korean_style_tips': shape_info['korean_style_tips'],
            'recommended_hairstyles': self._get_hairstyle_recommendations(face_shape),
            'makeup_tips': self._get_makeup_recommendations(face_shape),
            'accessory_tips': self._get_accessory_recommendations(face_shape)
        }
    
    def _get_hairstyle_recommendations(self, face_shape: str) -> List[str]:
        """Get Korean hairstyle recommendations"""
        hairstyles = {
            'oval': [
                'Korean layered cut (레이어드 컷)',
                'Side-swept Korean bangs',
                'Korean bob with soft waves',
                'Long straight hair with face-framing layers'
            ],
            'round': [
                'Korean wolf cut for dimension',
                'Long layers with side part',
                'Asymmetrical Korean bob',
                'Korean curtain bangs'
            ],
            'square': [
                'Soft Korean waves',
                'Korean shag cut',
                'Long layers past shoulders',
                'Side-swept bangs with texture'
            ],
            'heart': [
                'Korean curtain bangs',
                'Chin-length bob',
                'Korean perm for lower half',
                'Layered cut with volume at bottom'
            ],
            'diamond': [
                'Korean face-framing layers',
                'Side-parted long hair',
                'Soft Korean bangs',
                'Layered bob with texture'
            ]
        }
        return hairstyles.get(face_shape, hairstyles['oval'])
    
    def _get_makeup_recommendations(self, face_shape: str) -> List[str]:
        """Get Korean makeup recommendations"""
        makeup = {
            'oval': [
                'Korean gradient lip in any color',
                'Natural dewy skin base',
                'Straight Korean eyebrows',
                'Subtle aegyo-sal (under-eye bags)'
            ],
            'round': [
                'Contouring to add definition',
                'Korean gradient lip in deeper tones',
                'Slightly arched eyebrows',
                'Focus on eye makeup to elongate'
            ],
            'square': [
                'Soft Korean gradient lip',
                'Rounded eyebrow shape',
                'Blush on apples of cheeks',
                'Soft, blended contour'
            ],
            'heart': [
                'Focus blush on lower cheeks',
                'Korean gradient lip in coral/pink',
                'Soft, straight eyebrows',
                'Highlight bridge of nose'
            ],
            'diamond': [
                'Highlight cheekbones',
                'Korean tinted lip balm',
                'Soft, rounded eyebrows',
                'Eye makeup to draw attention upward'
            ]
        }
        return makeup.get(face_shape, makeup['oval'])
    
    def _get_accessory_recommendations(self, face_shape: str) -> List[str]:
        """Get Korean accessory recommendations"""
        accessories = {
            'oval': [
                'Any Korean hair accessories',
                'Statement earrings',
                'Korean bucket hat',
                'Oversized glasses frames'
            ],
            'round': [
                'Angular Korean hair clips',
                'Long drop earrings',
                'Korean beret',
                'Cat-eye glasses frames'
            ],
            'square': [
                'Soft, rounded Korean hair accessories',
                'Circular earrings',
                'Korean soft cap',
                'Round glasses frames'
            ],
            'heart': [
                'Korean hair clips at sides',
                'Bottom-heavy earrings',
                'Wide-brimmed Korean hat',
                'Glasses with lower focal point'
            ],
            'diamond': [
                'Korean hair accessories at temples',
                'Stud or small hoop earrings',
                'Korean headband',
                'Glasses that add width to forehead'
            ]
        }
        return accessories.get(face_shape, accessories['oval'])
    
    def visualize_face_shape(self, image: np.ndarray, landmarks: Dict, 
                           face_shape_result: Dict) -> np.ndarray:
        """
        Create visualization of face shape analysis
        
        Args:
            image: Original image
            landmarks: Facial landmarks
            face_shape_result: Face shape classification result
            
        Returns:
            Annotated image with face shape analysis
        """
        result_image = image.copy()
        
        if 'all' not in landmarks:
            return result_image
            
        # Draw face outline
        points = np.array(landmarks['all'])
        face_outline_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        outline_points = []
        for idx in face_outline_indices:
            if idx < len(points):
                outline_points.append(tuple(points[idx].astype(int)))
        
        if outline_points:
            # Draw face outline
            cv2.polylines(result_image, [np.array(outline_points)], True, (0, 255, 0), 2)
        
        # Add text annotations
        face_shape = face_shape_result.get('shape', 'unknown')
        confidence = face_shape_result.get('confidence', 0.0)
        korean_name = self.face_shapes.get(face_shape, {}).get('name', face_shape)
        
        # Add text with background
        text_lines = [
            f"Face Shape: {korean_name}",
            f"Confidence: {confidence:.2f}",
            f"Korean Style: Ready!"
        ]
        
        y_offset = 30
        for line in text_lines:
            # Create text background
            (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result_image, (10, y_offset - text_height - 5), 
                         (10 + text_width + 10, y_offset + 5), (0, 0, 0), -1)
            
            # Add text
            cv2.putText(result_image, line, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        return result_image