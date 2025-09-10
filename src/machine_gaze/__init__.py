"""
Machine Gaze - Computer Vision Analysis for Pre-recorded Video.

A modular system for object detection and emotion analysis with
annotated video output. Designed for research and art projects.
"""

from .core.classifier_registry import get_registry, register_classifier
from .core.video_processor import VideoProcessor
from .utils.config_loader import ConfigLoader

# Import classifiers to trigger registration
from . import classifiers

__version__ = "0.1.0"
__all__ = [
    'get_registry',
    'register_classifier', 
    'VideoProcessor',
    'ConfigLoader'
]
