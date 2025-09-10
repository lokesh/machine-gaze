"""
YOLO-World object detector implementation for Machine Gaze.

This module implements the YOLO-World open-vocabulary object detector,
which allows detection of custom object classes specified via text
without requiring model retraining.

YOLO-World was chosen because:
- Open-vocabulary detection (custom classes via text)
- Good speed/accuracy balance for offline processing  
- Excellent integration with Ultralytics ecosystem
- Strong M1 Mac support via MPS
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from ultralytics import YOLOWorld
    YOLO_WORLD_AVAILABLE = True
except ImportError:
    YOLO_WORLD_AVAILABLE = False
    YOLOWorld = None

from ..core.base_classifier import BaseClassifier, Detection

logger = logging.getLogger(__name__)


class YOLOWorldDetector(BaseClassifier):
    """
    YOLO-World object detector implementation.
    
    This classifier uses YOLO-World's open-vocabulary capabilities
    to detect custom object classes specified in the configuration.
    Perfect for the Machine Gaze project's diverse target objects.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize YOLO-World detector.
        
        Args:
            config: Configuration dictionary with the following keys:
                - model_size: 's', 'm', 'l', or 'x' (default: 's')
                - classes: List of class names to detect
                - confidence_threshold: Minimum confidence for detections
                - device: 'auto', 'cpu', 'mps', or specific GPU (default: 'auto')
                - input_size: Input image size (default: 640)
        """
        super().__init__(config)
        
        if not YOLO_WORLD_AVAILABLE:
            raise ImportError(
                "ultralytics not available. Install with: pip install ultralytics"
            )
        
        # Model configuration
        self.model_size = config.get('model_size', 's')
        self.device = config.get('device', 'auto')
        self.input_size = config.get('input_size', 640)
        
        # Classes to detect
        self.classes = config.get('classes', [])
        if not self.classes:
            raise ValueError("YOLO-World detector requires 'classes' list in config")
        
        # Model will be loaded in load_model()
        self.model: Optional[YOLOWorld] = None
        
        logger.info(f"Initialized YOLO-World detector for classes: {self.classes}")
    
    def load_model(self) -> None:
        """
        Load the YOLO-World model and set vocabulary.
        
        This downloads the model weights on first use and configures
        the open-vocabulary detection with the specified classes.
        """
        try:
            # Load model - this will download weights if not cached
            model_name = f"yolov8{self.model_size}-world.pt"
            logger.info(f"Loading YOLO-World model: {model_name}")
            
            self.model = YOLOWorld(model_name)
            
            # Set the vocabulary for open-vocabulary detection
            self.model.set_classes(self.classes)
            logger.info(f"Set YOLO-World vocabulary to: {self.classes}")
            
            # Configure device
            if self.device == 'auto':
                # Auto-detect best device for M1 Mac
                import torch
                if torch.backends.mps.is_available():
                    device = 'mps'
                    logger.info("Using MPS acceleration for M1 Mac")
                elif torch.cuda.is_available():
                    device = 'cuda'
                    logger.info("Using CUDA acceleration")
                else:
                    device = 'cpu'
                    logger.info("Using CPU inference")
            else:
                device = self.device
            
            self.model.to(device)
            logger.info(f"YOLO-World model loaded successfully on device: {device}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO-World model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLO-World detection on a frame.
        
        Args:
            frame: Input image as numpy array (BGR format)
            
        Returns:
            List of Detection objects found in the frame
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Run inference
            results = self.model(frame, imgsz=self.input_size, verbose=False)
            
            detections = []
            for result in results:
                # Extract detection data
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    # Convert to Detection objects
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        if conf >= self.confidence_threshold:
                            # Get class name from the vocabulary
                            class_name = self.classes[cls_id] if cls_id < len(self.classes) else f"class_{cls_id}"
                            
                            detection = Detection(
                                bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                                class_name=class_name,
                                confidence=float(conf),
                                metadata={
                                    'class_id': cls_id,
                                    'detector': 'yolo_world'
                                }
                            )
                            detections.append(detection)
            
            logger.debug(f"YOLO-World found {len(detections)} detections")
            return detections
            
        except Exception as e:
            logger.error(f"Error during YOLO-World detection: {e}")
            return []
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for YOLO-World.
        
        YOLO-World handles preprocessing internally, so we just
        ensure the frame is in the correct format.
        """
        # Ensure frame is BGR (OpenCV default)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return frame
        else:
            raise ValueError(f"Expected BGR image, got shape: {frame.shape}")
    
    def get_supported_classes(self) -> List[str]:
        """Get list of classes this detector is configured for."""
        return self.classes.copy()
    
    def update_classes(self, new_classes: List[str]) -> None:
        """
        Update the vocabulary for detection.
        
        Args:
            new_classes: New list of class names to detect
        """
        self.classes = new_classes
        if self.model is not None:
            self.model.set_classes(self.classes)
            logger.info(f"Updated YOLO-World vocabulary to: {self.classes}")


def create_machine_gaze_detector(config_override: Optional[Dict[str, Any]] = None) -> YOLOWorldDetector:
    """
    Create a YOLO-World detector configured for Machine Gaze target objects.
    
    This is a convenience function that sets up the detector with the
    specific object classes mentioned in the GAMEPLAN.txt.
    
    Args:
        config_override: Optional config overrides
        
    Returns:
        Configured YOLOWorldDetector instance
    """
    # Default configuration for Machine Gaze objects
    default_config = {
        'model_size': 's',  # Start with small model for faster processing
        'confidence_threshold': 0.3,  # Lower threshold to catch more objects
        'device': 'auto',
        'input_size': 640,
        'enabled': True,
        'classes': [
            # Vehicles
            'car',
            'pickup truck', 
            'box truck',
            'semi truck',
            'truck',
            'vehicle',
            
            # Infrastructure
            'tree',
            'street light',
            'lamppost',
            'traffic light',
            'stoplight',
            
            # Mobility devices
            'walker',
            'rollator',
            'mobility walker', 
            'walking frame',
            'stroller',
            'wheelchair',
            
            # People (for context)
            'person',
            'pedestrian'
        ]
    }
    
    # Apply any overrides
    if config_override:
        default_config.update(config_override)
    
    return YOLOWorldDetector(default_config)
