
"""
Face Detection Module for Korean Fashion AI Personal Stylist
Week 1-2: Core face detection and landmark extraction using OpenCV and MediaPipe
"""


import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict
from PIL import Image


class FaceDetector:
    def __init__(self, detection_confidence: float = 0.7):
        """
        Initialize face detector with MediaPipe and OpenCV
        
        Args:
            detection_confidence: Minimum confidence for face detection (0.0 to 1.0)
        """
        self.detection_confidence = detection_confidence

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils


        # Initialize face detection model
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection = 1, # Use model 1 for better accuracy
            min_detection_confidence = self.detection_confidence
        )


        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode = True,
            max_num_faces = 1, # Only detect one face or focus on one face at a time
            refine_landmarks = True,
            min_detection_confidence = self.detection_confidence
        )


        # Define key facial landmarks indices for face shape analysis
        self.landmark_indices = {
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'nose': [1, 2, 5, 6, 19, 20, 94, 125, 141, 235, 236, 237, 238, 239, 240, 241, 242],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'jawline': [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
            'forehead': [10, 151, 9, 10, 151, 195, 197, 196, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 292, 308, 415, 310, 311, 312, 13, 82, 81, 80, 78]
        }

    def preprocess_image(self, image:np.ndarray) -> np.ndarray:
        """
        Preprocess image for face detection
        Args:
            image: Input image as a numpy array (BGR format)
        Returns:
            Preprocessed image as a numpy array (RGB format)
        """

        # Covert BGR to RGB if using OpenCV
        if len(image.shape) == 3 and image.shape[2] ==3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

    def detect_face(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect face in the image using MediaPipe Face Detection
        Detect face in image and return bounding box and landmarks
        Args:
            image: Input image as a numpy array
        Returns:
            Dictionary with face detection results or None if no face detected
        """
        
        image_rgb = self.preprocess_image(image)
        results = self.face_detection.process(image_rgb)

        if not results.detections:
            return None
        
        # Get the first (most confident) face detection result
        detection = results.detections[0]

        # Extract bounding box coordinates
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = image.shape

        # Ensure bounding box is within image bounds
        x = max(0,x)
        y = max(0,y)
        width = min(width, w-x)
        height = min(height, h-y)


        return {
            'bbox': (x,y, width, height),
            'confidence': detection.score[0],
            'relative_bbox': (bbox.xmin, bbox.ymin, bbox.width, bbox.height),
        }
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """
        Extract detailed facial landmarks using MediaPipe Face Mesh
        Args:
            image: Input image as a numpy array
        Returns:
            Dictionary with facial landmarks coordinates or None if no face detected
        """

        image_rgb = self.preprocess_image(image)
        results = self.face_mesh.process(image_rgb)

        if not results.detections:
            return None
        
        # Get landmarks for the first face
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape

        # Convert normalized coordinates to pixel coordinates
        landmarks = {}

        for name, indices in self.landmark_indices.items():
            landmarks[name] = []
            for idx in indices:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks[name].append((x, y))

        # Add all landmarks as raw data for detailed analysis
        all_landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            all_landmarks.append((x, y))
            
        landmarks['all'] = all_landmarks

        return landmarks
    
    def crop_face(self, image: np.ndarray, padding: float = 0.2) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Crop face region from image with padding
        
        Args:
            image: Input image as numpy array
            padding: Additional padding around face (0.0 to 1.0)
            
        Returns:
            Tuple of (cropped_face, face_info) or None if no face found
        """

        face_detection = self.detect_face(image)

        if not face_detection:
            return None
    
        x, y, width, height = face_detection['bbox']

        # Add padding
        pad_x = int(width * padding)
        pad_y = int(height * padding)

        # Calculate expanded bounding box
        x_start = max(0, x-pad_x)
        y_start = max(0, y-pad_y)
        x_end = min(image.shape[1], x+ width + pad_x)
        y_end = min(image.shape[0], y + height + pad_y)


        # Crop the face region
        cropped_face = image[y_start:y_end, x_start:x_end]

        # Return cropped face and adjusted coordinates
        face_info = {
            'original_bbox': face_detection['bbox'],
            'cropped_bbox': (x_start, y_start, x_end-x_start, y_end- y_start),
            'confidence': face_detection['confidence']
        }

        return cropped_face, face_info
    

    def analyze_face(self, image: np.ndarry) -> Optional[Dict]:
        """
        Complete face analysis including detection, landmarks, and cropping
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Complete face analysis results
        """

        # Detect face and extract landmarks
        face_detection = self.detect_face(image)
        if not face_detection:
            return None
        
        # Extract landmarks
        landmarks = self.extract_landmarks(image)
        if not landmarks:
            return None

        # Crop face
        crop_result = self.crop_face(image)
        if not crop_result:
            return None
        
        cropped_face, face_info = crop_result

        return {
            'detection': face_detection,
            'landmarks': landmarks,
            'cropped_face': cropped_face,
            'face_info': face_info,
            'original_shape': image.shape
        }
    
    def draw_landmarks(self, image: np.ndarray, landmarks: Dict, colors: Dict = None) -> np.ndarray:
        """
        Draw facial landmarks on image for visualization
        
        Args:
            image: Input image
            landmarks: Landmark coordinates
            colors: Colors for different landmark groups
            
        Returns:
            Image with drawn landmarks
        """

        if colors is None:
            colors = {
                'face_oval': (255, 0, 0),      # Red
                'left_eye': (0, 255, 0),       # Green
                'right_eye': (0, 255, 0),      # Green
                'nose': (0, 0, 255),           # Blue
                'mouth': (255, 255, 0),        # Yellow
                'jawline': (255, 0, 255),      # Magenta
                'forehead': (0, 255, 255)      # Cyan
            }

        result_image = image.copy()

        for landmark_name, points in landmarks.items():
            if landmark_name == 'all':
                continue  # Skip raw landmarks for drawing

            color = colors.get(landmark_name, (128,128,128))

            for point in points:
                cv2.circle(result_image, point, 2, color, -1)

            return result_image
        

    def __del__(self):
        """Clean up resources"""

        if hasattr(self, 'face_detection'):
            self.face_detection.close()

        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
