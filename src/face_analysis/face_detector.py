
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
        
        # Get the first (most confident) face detection result
        detection = results.detections[0]

        # Extract bounding box coordinates
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = image.shape

        # Convert relative coordinates to absolute pixel coordinates
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        # Ensure bounding box is within image bounds
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)

        return {
            'bbox': (x,y,width, height),
            'confidence': detection.score[0],
            'relative_bbox': (bbox.xmin, bbox.ymin, bbox.width, bbox.height)
        }