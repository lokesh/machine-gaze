"""
Video processing pipeline for the Machine Gaze system.

This module handles video input/output and orchestrates the detection
pipeline. It's designed to work with the ClassifierRegistry to process
videos through multiple classifiers and render annotated output.
"""

import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any
from .classifier_registry import ClassifierRegistry, Detection

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Main video processing pipeline that coordinates detection and rendering.
    
    This class handles:
    - Video input/output operations
    - Frame-by-frame processing through classifiers
    - Rendering detections onto frames
    - Progress tracking and error handling
    """
    
    def __init__(self, registry: ClassifierRegistry, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the video processor.
        
        Args:
            registry: ClassifierRegistry instance with loaded classifiers
            config: Optional configuration for video processing settings
        """
        self.registry = registry
        self.config = config or {}
        
        # Video processing settings
        self.output_fps = self.config.get('output_fps', None)  # None = use input fps
        self.output_codec = self.config.get('output_codec', 'mp4v')
        self.output_quality = self.config.get('output_quality', 0.9)
        
        # Rendering settings
        self.font_scale = self.config.get('font_scale', 0.6)
        self.font_thickness = self.config.get('font_thickness', 2)
        self.bbox_thickness = self.config.get('bbox_thickness', 2)
        self.text_color = self.config.get('text_color', (255, 255, 255))  # White
        self.bbox_color = self.config.get('bbox_color', (0, 255, 0))  # Green
        
        # Overlay-only mode settings
        self.overlay_only = self.config.get('overlay_only', False)
        self.background_color = self.config.get('background_color', (0, 0, 0))  # Black background
        
        # Progress callback
        self.progress_callback: Optional[Callable[[int, int], None]] = None
    
    def set_progress_callback(self, callback: Callable[[int, int], None]) -> None:
        """
        Set a callback function to receive progress updates.
        
        Args:
            callback: Function that takes (current_frame, total_frames) as arguments
        """
        self.progress_callback = callback
    
    def process_video(self, input_path: str, output_path: str) -> bool:
        """
        Process a video file through the detection pipeline.
        
        Args:
            input_path: Path to input video file
            output_path: Path for output annotated video
            
        Returns:
            True if processing completed successfully, False otherwise
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            logger.error(f"Input video not found: {input_path}")
            return False
        
        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {input_path}")
            return False
        
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
            
            # Set up output video writer
            output_fps = self.output_fps if self.output_fps else fps
            fourcc = cv2.VideoWriter_fourcc(*self.output_codec)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
            if not out.isOpened():
                logger.error(f"Could not create output video: {output_path}")
                return False
            
            # Process frames
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame through classifiers
                detections = self.registry.process_frame(frame)
                
                # Render detections onto frame
                annotated_frame = self.render_detections(frame, detections)
                
                # Write annotated frame
                out.write(annotated_frame)
                
                frame_count += 1
                
                # Progress callback
                if self.progress_callback:
                    self.progress_callback(frame_count, total_frames)
                
                # Log progress occasionally
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames")
            
            logger.info(f"Processing complete: {frame_count} frames written to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return False
            
        finally:
            cap.release()
            if 'out' in locals():
                out.release()
    
    def render_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Render detection results onto a frame.
        
        Args:
            frame: Input frame as numpy array
            detections: List of Detection objects to render
            
        Returns:
            Frame with rendered detections
        """
        if self.overlay_only:
            # Create a blank frame with background color
            output_frame = np.full_like(frame, self.background_color, dtype=np.uint8)
        else:
            # Work on a copy to avoid modifying the original
            output_frame = frame.copy()
        
        for detection in detections:
            self._draw_detection(output_frame, detection)
        
        return output_frame
    
    def _draw_detection(self, frame: np.ndarray, detection: Detection) -> None:
        """
        Draw a single detection on the frame.
        
        Args:
            frame: Frame to draw on (modified in place)
            detection: Detection object to render
        """
        x1, y1, x2, y2 = detection.bbox
        
        # Choose color based on classifier or use default
        color = self._get_color_for_class(detection.class_name)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.bbox_thickness)
        
        # Prepare label text
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        if detection.track_id is not None:
            label += f" (ID: {detection.track_id})"
        
        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
        )
        
        # Draw text background rectangle
        text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
        cv2.rectangle(
            frame,
            (x1, text_y - text_height - baseline),
            (x1 + text_width, text_y + baseline),
            color,
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            frame,
            label,
            (x1, text_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            self.text_color,
            self.font_thickness
        )
    
    def _get_color_for_class(self, class_name: str) -> tuple:
        """
        Get a consistent color for a given class name.
        
        Args:
            class_name: Name of the detected class
            
        Returns:
            BGR color tuple
        """
        # Simple hash-based color assignment for consistency
        hash_value = hash(class_name)
        
        # Generate RGB values from hash
        r = (hash_value & 0xFF0000) >> 16
        g = (hash_value & 0x00FF00) >> 8
        b = hash_value & 0x0000FF
        
        # Ensure colors are bright enough to be visible
        r = max(r, 100)
        g = max(g, 100)
        b = max(b, 100)
        
        # Return as BGR for OpenCV
        return (b, g, r)
    
    def process_single_frame(self, frame: np.ndarray) -> tuple[np.ndarray, List[Detection]]:
        """
        Process a single frame and return annotated frame and detections.
        
        Useful for testing or real-time processing.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (annotated_frame, detections)
        """
        detections = self.registry.process_frame(frame)
        annotated_frame = self.render_detections(frame, detections)
        return annotated_frame, detections
