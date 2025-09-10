#!/usr/bin/env python3
"""
Test script to verify Machine Gaze setup and dependencies.

This script checks that all components can be imported and
basic functionality works before processing actual videos.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML imported successfully")
    except ImportError as e:
        print(f"✗ PyYAML import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLOWorld
        print("✓ Ultralytics YOLOWorld imported successfully")
    except ImportError as e:
        print(f"✗ Ultralytics import failed: {e}")
        print("  Install with: pip install ultralytics")
        return False
    
    return True


def test_machine_gaze_imports():
    """Test Machine Gaze module imports."""
    print("\nTesting Machine Gaze imports...")
    
    try:
        from machine_gaze import VideoProcessor, ConfigLoader, get_registry
        print("✓ Core Machine Gaze modules imported")
    except ImportError as e:
        print(f"✗ Machine Gaze import failed: {e}")
        return False
    
    try:
        from machine_gaze.classifiers import YOLOWorldDetector
        print("✓ YOLOWorld detector imported")
    except ImportError as e:
        print(f"✗ YOLOWorld detector import failed: {e}")
        return False
    
    return True


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from machine_gaze import ConfigLoader
        config = ConfigLoader.load_config()
        print("✓ Default configuration loaded")
        
        # Check key sections
        if 'classifiers' in config:
            print("✓ Classifiers section found in config")
        else:
            print("✗ Classifiers section missing from config")
            return False
        
        if 'yolo_world' in config['classifiers']:
            print("✓ YOLO-World classifier configured")
        else:
            print("✗ YOLO-World classifier not configured")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False


def test_classifier_registry():
    """Test classifier registry functionality."""
    print("\nTesting classifier registry...")
    
    try:
        from machine_gaze import get_registry
        
        registry = get_registry()
        registered_types = registry.get_registered_types()
        print(f"✓ Registry created, registered types: {registered_types}")
        
        if 'yolo_world' in registered_types:
            print("✓ YOLO-World classifier registered")
        else:
            print("✗ YOLO-World classifier not registered")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Classifier registry test failed: {e}")
        return False


def test_yolo_world_creation():
    """Test YOLO-World classifier creation (without model loading)."""
    print("\nTesting YOLO-World classifier creation...")
    
    try:
        from machine_gaze.classifiers.yolo_world_detector import create_machine_gaze_detector
        
        # Create detector with minimal config to avoid downloading model
        config_override = {
            'confidence_threshold': 0.5,
            'device': 'cpu',  # Force CPU to avoid MPS issues during testing
            'model_size': 's'
        }
        
        detector = create_machine_gaze_detector(config_override)
        print("✓ YOLO-World detector created successfully")
        print(f"✓ Target classes: {len(detector.get_supported_classes())} classes configured")
        
        # Don't call load_model() as it would download the weights
        return True
        
    except Exception as e:
        print(f"✗ YOLO-World detector creation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Machine Gaze Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_machine_gaze_imports,
        test_config_loading,
        test_classifier_registry,
        test_yolo_world_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Machine Gaze is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt") 
        print("2. Test with a video: python main.py videos/sarah.MOV output/test.mp4")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
