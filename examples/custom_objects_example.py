#!/usr/bin/env python3
"""
Example: Adding custom objects to YOLO-World detector programmatically.

This shows how to create a detector with custom object types
without modifying configuration files.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from machine_gaze.classifiers.yolo_world_detector import YOLOWorldDetector

def create_custom_detector():
    """Create a YOLO-World detector with custom objects."""
    
    # Define your custom object list
    custom_objects = [
        # Your specific use case objects
        "bicycle",
        "motorcycle", 
        "stop sign",
        "fire hydrant",
        "bench",
        "dog",
        "cat",
        "bird",
        "backpack",
        "handbag",
        
        # Still include original Machine Gaze objects
        "person",
        "car",
        "tree",
        "wheelchair",
        "walker"
    ]
    
    # Create detector config
    config = {
        'model_size': 's',
        'confidence_threshold': 0.3,
        'device': 'auto',
        'input_size': 640,
        'enabled': True,
        'classes': custom_objects
    }
    
    # Create and return detector
    detector = YOLOWorldDetector(config)
    print(f"Created detector for {len(custom_objects)} object types:")
    for obj in custom_objects:
        print(f"  - {obj}")
    
    return detector

def update_detector_classes_dynamically():
    """Example: Update classes after detector creation."""
    
    # Create detector with initial classes
    initial_config = {
        'classes': ['person', 'car', 'tree'],
        'confidence_threshold': 0.3,
        'model_size': 's'
    }
    
    detector = YOLOWorldDetector(initial_config)
    print("Initial classes:", detector.get_supported_classes())
    
    # Add more classes
    new_classes = ['person', 'car', 'tree', 'bicycle', 'dog', 'stop sign']
    detector.update_classes(new_classes)
    print("Updated classes:", detector.get_supported_classes())
    
    return detector

if __name__ == "__main__":
    print("=== Custom Objects Example ===")
    create_custom_detector()
    
    print("\n=== Dynamic Update Example ===")
    update_detector_classes_dynamically()
