"""
Classifiers module for Machine Gaze.

This module contains all classifier implementations and handles
automatic registration with the global classifier registry.
"""

from ..core.classifier_registry import register_classifier

# Import all available classifiers
from .yolo_world_detector import YOLOWorldDetector
from .face_emotion_detector import FaceEmotionDetector

# Register classifiers with the global registry
register_classifier('yolo_world', YOLOWorldDetector)
register_classifier('face_emotion', FaceEmotionDetector)

# Export for convenience
__all__ = [
    'YOLOWorldDetector',
    'FaceEmotionDetector',
]
