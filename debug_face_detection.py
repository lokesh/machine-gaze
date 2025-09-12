#!/usr/bin/env python3
"""
Debug script to test face detection on individual frames.

This helps diagnose why face emotion detection isn't working.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from machine_gaze.classifiers.face_emotion_detector import FaceEmotionDetector
from machine_gaze import ConfigLoader

def test_face_detection_on_frame(video_path: str, frame_number: int = 50):
    """
    Test face detection on a specific frame from a video.
    
    Args:
        video_path: Path to video file
        frame_number: Frame to extract and test (default: 50)
    """
    print(f"Testing face detection on frame {frame_number} from {video_path}")
    
    # Open video and extract frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Seek to specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        return
    
    print(f"Frame shape: {frame.shape}")
    
    # Test with different face detection settings
    test_configs = [
        {
            "name": "Very Permissive",
            "face_confidence": 0.1,
            "emotion_confidence": 0.1,
            "min_face_size": 10,
            "face_padding": 0.3
        },
        {
            "name": "Default Settings",
            "face_confidence": 0.5,
            "emotion_confidence": 0.3,
            "min_face_size": 30,
            "face_padding": 0.2
        },
        {
            "name": "High Confidence",
            "face_confidence": 0.8,
            "emotion_confidence": 0.5,
            "min_face_size": 50,
            "face_padding": 0.1
        }
    ]
    
    for config in test_configs:
        print(f"\n--- Testing {config['name']} ---")
        print(f"Config: {config}")
        
        # Create detector with test config
        detector_config = {
            'enabled': True,
            'confidence_threshold': 0.1,
            **{k: v for k, v in config.items() if k != 'name'}
        }
        
        try:
            detector = FaceEmotionDetector(detector_config)
            detector.load_model()
            
            # Run detection
            detections = detector.detect(frame)
            print(f"Found {len(detections)} face detections")
            
            for i, detection in enumerate(detections):
                print(f"  Face {i+1}: {detection}")
                
            # Save debug image if faces found
            if detections:
                debug_frame = frame.copy()
                for detection in detections:
                    x1, y1, x2, y2 = detection.bbox
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = f"{detection.class_name}: {detection.confidence:.2f}"
                    cv2.putText(debug_frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                output_path = f"debug_faces_{config['name'].lower().replace(' ', '_')}.jpg"
                cv2.imwrite(output_path, debug_frame)
                print(f"  Saved debug image: {output_path}")
        
        except Exception as e:
            print(f"  Error: {e}")

def test_mediapipe_directly(video_path: str, frame_number: int = 50):
    """
    Test MediaPipe face detection directly without our wrapper.
    """
    print(f"\n--- Testing MediaPipe directly ---")
    
    try:
        import mediapipe as mp
        
        # Open video and extract frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Could not read frame")
            return
        
        # Test different MediaPipe settings
        test_settings = [
            {"model_selection": 0, "min_detection_confidence": 0.1},  # Short range, very low conf
            {"model_selection": 0, "min_detection_confidence": 0.5},  # Short range, default conf
            {"model_selection": 1, "min_detection_confidence": 0.1},  # Full range, very low conf
            {"model_selection": 1, "min_detection_confidence": 0.5},  # Full range, default conf
        ]
        
        mp_face_detection = mp.solutions.face_detection
        
        for i, settings in enumerate(test_settings):
            print(f"\nMediaPipe Test {i+1}: {settings}")
            
            with mp_face_detection.FaceDetection(**settings) as face_detection:
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)
                
                if results.detections:
                    print(f"  Found {len(results.detections)} faces")
                    
                    # Draw detections
                    debug_frame = frame.copy()
                    height, width = frame.shape[:2]
                    
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * width)
                        y = int(bbox.ymin * height)
                        w = int(bbox.width * width)
                        h = int(bbox.height * height)
                        
                        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                        
                        score = detection.score[0]
                        label = f"Face: {score:.2f}"
                        cv2.putText(debug_frame, label, (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        print(f"    Face: confidence={score:.3f}, bbox=({x},{y},{x+w},{y+h}), size=({w}x{h})")
                    
                    output_path = f"debug_mediapipe_{i+1}.jpg"
                    cv2.imwrite(output_path, debug_frame)
                    print(f"  Saved: {output_path}")
                else:
                    print("  No faces detected")
                    
    except Exception as e:
        print(f"MediaPipe test error: {e}")

def main():
    """Run face detection debugging."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug face detection")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--frame", type=int, default=50, help="Frame number to test (default: 50)")
    
    args = parser.parse_args()
    
    print("Face Detection Debug Tool")
    print("=" * 50)
    
    # Test our face detection implementation
    test_face_detection_on_frame(args.video, args.frame)
    
    # Test MediaPipe directly  
    test_mediapipe_directly(args.video, args.frame)
    
    print("\n" + "=" * 50)
    print("Debug complete. Check the generated debug images.")

if __name__ == "__main__":
    main()
