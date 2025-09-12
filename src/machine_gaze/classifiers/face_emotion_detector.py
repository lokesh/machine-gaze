"""
Face detection and emotion classification for Machine Gaze.

This module implements face detection using MediaPipe and emotion classification
using a pre-trained TensorFlow model. It follows the same classifier interface
as other detectors for seamless integration.

Detects 7 basic emotions: happy, sad, angry, fear, surprise, disgust, neutral
"""

import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

from ..core.base_classifier import BaseClassifier, Detection

logger = logging.getLogger(__name__)


class FaceEmotionDetector(BaseClassifier):
    """
    Face detection and emotion classification using MediaPipe + TensorFlow.
    
    This classifier:
    1. Detects faces using MediaPipe Face Detection
    2. Crops face regions
    3. Classifies emotions using a pre-trained model
    4. Returns detections with emotion labels
    """
    
    # Standard 7 emotion categories
    EMOTION_LABELS = [
        'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize face emotion detector.
        
        Args:
            config: Configuration dictionary with:
                - face_confidence: Minimum confidence for face detection (0.5)
                - emotion_confidence: Minimum confidence for emotion prediction (0.3)
                - model_path: Path to emotion model (optional, uses built-in if None)
                - face_padding: Padding around detected faces (0.2)
                - min_face_size: Minimum face size in pixels (30)
        """
        super().__init__(config)
        
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available. Install with: pip install mediapipe")
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        # Face detection settings
        self.face_confidence = config.get('face_confidence', 0.5)
        self.face_padding = config.get('face_padding', 0.2)
        self.min_face_size = config.get('min_face_size', 30)
        
        # Emotion classification settings
        self.emotion_confidence = config.get('emotion_confidence', 0.3)
        self.model_path = config.get('model_path', None)
        
        # MediaPipe and TensorFlow components (loaded in load_model)
        self.mp_face_detection = None
        self.face_detector = None
        self.emotion_model = None
        
        logger.info("Initialized Face Emotion Detector")
    
    def load_model(self) -> None:
        """
        Load MediaPipe face detection and emotion classification models.
        """
        try:
            # Initialize MediaPipe Face Detection
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 1 for full range, 0 for short range
                min_detection_confidence=self.face_confidence
            )
            logger.info("MediaPipe Face Detection loaded successfully")
            
            # Load emotion classification model
            if self.model_path and Path(self.model_path).exists():
                # Load custom model if provided
                self.emotion_model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Loaded custom emotion model from: {self.model_path}")
            else:
                # Create a simple CNN model for emotion detection
                # In a production system, you'd load a pre-trained model
                self.emotion_model = self._create_simple_emotion_model()
                logger.info("Created simple emotion classification model")
            
            logger.info("Face Emotion Detector models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Face Emotion Detector models: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect faces and classify emotions in a frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of Detection objects with face bounding boxes and emotion labels
        """
        if self.face_detector is None or self.emotion_model is None:
            raise RuntimeError("Models not loaded. Call load_model() first.")
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            
            # Detect faces
            results = self.face_detector.process(rgb_frame)
            
            detections = []
            if results.detections:
                for detection in results.detections:
                    # Extract face bounding box
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert relative coordinates to absolute pixels
                    x = int(bbox.xmin * width)
                    y = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)
                    
                    # Add padding around face
                    pad_x = int(w * self.face_padding)
                    pad_y = int(h * self.face_padding)
                    
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(width, x + w + pad_x)
                    y2 = min(height, y + h + pad_y)
                    
                    # Skip faces that are too small
                    if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                        continue
                    
                    # Extract face region for emotion classification
                    face_region = frame[y1:y2, x1:x2]
                    emotion_label, emotion_confidence = self._classify_emotion(face_region)
                    
                    # Only include if emotion confidence is high enough
                    if emotion_confidence >= self.emotion_confidence:
                        face_detection = Detection(
                            bbox=(x1, y1, x2, y2),
                            class_name=f"face_{emotion_label}",
                            confidence=float(emotion_confidence),
                            metadata={
                                'detector': 'face_emotion',
                                'emotion': emotion_label,
                                'emotion_confidence': emotion_confidence,
                                'face_confidence': detection.score[0],
                                'face_size': (x2 - x1, y2 - y1)
                            }
                        )
                        detections.append(face_detection)
            
            logger.debug(f"Face emotion detector found {len(detections)} faces with emotions")
            return detections
            
        except Exception as e:
            logger.error(f"Error in face emotion detection: {e}")
            return []
    
    def _classify_emotion(self, face_region: np.ndarray) -> tuple[str, float]:
        """
        Classify emotion in a face region.
        
        Args:
            face_region: Cropped face image
            
        Returns:
            Tuple of (emotion_label, confidence)
        """
        try:
            # Preprocess face for emotion model
            # Resize to model input size (typically 48x48 for emotion models)
            face_resized = cv2.resize(face_region, (48, 48))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            face_normalized = face_gray.astype(np.float32) / 255.0
            
            # Add batch dimension
            face_input = np.expand_dims(face_normalized, axis=[0, -1])
            
            # Predict emotion
            predictions = self.emotion_model.predict(face_input, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx])
            
            emotion_label = self.EMOTION_LABELS[emotion_idx]
            return emotion_label, confidence
            
        except Exception as e:
            logger.error(f"Error classifying emotion: {e}")
            return "neutral", 0.0
    
    def _create_simple_emotion_model(self) -> tf.keras.Model:
        """
        Create a simple CNN model for emotion classification.
        
        In production, you'd load a pre-trained model like FER2013.
        This is a minimal model for demonstration.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(48, 48, 1)),
            
            # First conv block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            # Second conv block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            # Third conv block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            # Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.EMOTION_LABELS), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Initialize with random weights (in production, load pre-trained weights)
        logger.warning("Using untrained emotion model - predictions will be random!")
        logger.info("For production use, load a pre-trained emotion model like FER2013")
        
        return model
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for face detection.
        MediaPipe handles most preprocessing internally.
        """
        return frame
    
    def get_supported_emotions(self) -> List[str]:
        """Get list of supported emotion labels."""
        return self.EMOTION_LABELS.copy()


def create_face_emotion_detector(config_override: Optional[Dict[str, Any]] = None) -> FaceEmotionDetector:
    """
    Create a face emotion detector with default configuration.
    
    Args:
        config_override: Optional config overrides
        
    Returns:
        Configured FaceEmotionDetector instance
    """
    default_config = {
        'enabled': True,
        'face_confidence': 0.5,
        'emotion_confidence': 0.3,
        'face_padding': 0.2,
        'min_face_size': 30,
        'confidence_threshold': 0.3,  # Overall detection threshold
        'model_path': None  # Use built-in model
    }
    
    if config_override:
        default_config.update(config_override)
    
    return FaceEmotionDetector(default_config)
