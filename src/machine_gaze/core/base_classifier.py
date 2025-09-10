"""
Base classifier interface for the Machine Gaze system.

This module defines the abstract base class that all classifiers must implement.
The design follows the Strategy pattern to allow easy swapping and extension
of different classification algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import cv2


class Detection:
    """
    Represents a single detection result from a classifier.
    
    This standardized format allows different classifiers to return
    results in a consistent way for the rendering pipeline.
    """
    
    def __init__(
        self,
        bbox: Tuple[int, int, int, int],  # (x1, y1, x2, y2)
        class_name: str,
        confidence: float,
        track_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.bbox = bbox
        self.class_name = class_name
        self.confidence = confidence
        self.track_id = track_id
        self.metadata = metadata or {}
    
    def __repr__(self):
        return (f"Detection(bbox={self.bbox}, class='{self.class_name}', "
                f"conf={self.confidence:.2f}, track_id={self.track_id})")


class BaseClassifier(ABC):
    """
    Abstract base class for all classifiers in the Machine Gaze system.
    
    This provides a consistent interface that allows the core pipeline
    to work with any type of classifier (object detection, emotion
    recognition, gesture detection, etc.) without knowing the specifics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the classifier with configuration.
        
        Args:
            config: Dictionary containing classifier-specific configuration
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.name = self.__class__.__name__
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the classifier's model and any required resources.
        
        This is called once during initialization and should handle
        model loading, device selection, and any preprocessing setup.
        """
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single frame.
        
        Args:
            frame: Input image as numpy array (BGR format)
            
        Returns:
            List of Detection objects found in the frame
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if this classifier is enabled."""
        return self.enabled
    
    def get_name(self) -> str:
        """Get the classifier name."""
        return self.name
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Optional preprocessing step for the frame.
        
        Default implementation returns the frame unchanged.
        Subclasses can override for classifier-specific preprocessing.
        """
        return frame
    
    def postprocess_detections(self, detections: List[Detection]) -> List[Detection]:
        """
        Optional postprocessing step for detections.
        
        Default implementation filters by confidence threshold.
        Subclasses can override for additional filtering or enhancement.
        """
        return [d for d in detections if d.confidence >= self.confidence_threshold]
