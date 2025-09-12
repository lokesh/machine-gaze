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
from ..utils.detection_utils import associate_faces_with_people, filter_overlapping_detections
from ..tracking import ByteTracker, TrackSmoother
from ..tracking.track_smoother import SmoothedTrackState

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
        
        # Detection enhancement settings
        self.enable_face_person_association = self.config.get('face_person_association', True)
        self.enable_nms = self.config.get('non_max_suppression', True)
        self.nms_threshold = self.config.get('nms_threshold', 0.5)
        
        # Tracking settings
        tracking_config = self.config.get('tracking', {})
        self.enable_tracking = tracking_config.get('enabled', False)
        self.show_track_ids = tracking_config.get('show_track_ids', True)
        self.show_trajectories = tracking_config.get('show_trajectories', False)
        self.trajectory_length = tracking_config.get('trajectory_length', 10)
        
        # Initialize tracking components
        if self.enable_tracking:
            self.tracker = ByteTracker(
                high_thresh=tracking_config.get('high_thresh', 0.6),
                low_thresh=tracking_config.get('low_thresh', 0.1),
                new_track_thresh=tracking_config.get('new_track_thresh', 0.7),
                track_buffer=tracking_config.get('track_buffer', 30),
                match_thresh=tracking_config.get('match_thresh', 0.8)
            )
            
            self.track_smoother = TrackSmoother(
                temporal_window=tracking_config.get('temporal_window', 5),
                bbox_smoothing=tracking_config.get('bbox_smoothing', True),
                confidence_smoothing=tracking_config.get('confidence_smoothing', True),
                class_voting=tracking_config.get('class_voting', True)
            )
            
            # Track visualization
            self.track_colors = {}  # track_id -> color mapping
            self.track_trajectories = {}  # track_id -> list of center points
        else:
            self.tracker = None
            self.track_smoother = None
        
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
                
                # Enhance detections with post-processing
                enhanced_detections = self._enhance_detections(detections)
                
                # Apply tracking if enabled
                final_detections = enhanced_detections
                if self.enable_tracking and self.tracker is not None:
                    track_states = self.tracker.update(enhanced_detections)
                    
                    if self.track_smoother is not None:
                        smoothed_tracks = self.track_smoother.smooth_tracks(track_states)
                        # Convert smoothed tracks back to detections
                        final_detections = self._tracks_to_detections(smoothed_tracks)
                    else:
                        # Convert track states back to detections
                        final_detections = self._tracks_to_detections(track_states)
                
                # Render detections onto frame
                annotated_frame = self.render_detections(frame, final_detections)
                
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
        
        # Choose color - use track color if tracking is enabled and track_id exists
        if self.enable_tracking and detection.track_id is not None:
            color = self._get_track_color(detection.track_id)
            
            # Update trajectory
            if self.show_trajectories:
                self._update_track_trajectory(detection.track_id, detection.bbox)
                self._draw_track_trajectory(frame, detection.track_id, color)
        else:
            color = self._get_color_for_class(detection.class_name)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.bbox_thickness)
        
        # Prepare label text
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        if detection.track_id is not None and self.show_track_ids:
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
    
    def _enhance_detections(self, detections: List[Detection]) -> List[Detection]:
        """
        Enhance detection results with post-processing.
        
        Args:
            detections: Raw detections from classifiers
            
        Returns:
            Enhanced detections with face-person associations and NMS applied
        """
        enhanced = detections
        
        # Associate faces with people if enabled
        if self.enable_face_person_association:
            enhanced = associate_faces_with_people(enhanced)
            logger.debug(f"Face-person association: {len(detections)} -> {len(enhanced)} detections")
        
        # Apply Non-Maximum Suppression if enabled
        if self.enable_nms:
            enhanced = filter_overlapping_detections(enhanced, self.nms_threshold)
            logger.debug(f"NMS applied: reduced overlapping detections")
        
        return enhanced
    
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
        enhanced_detections = self._enhance_detections(detections)
        
        # Apply tracking if enabled
        final_detections = enhanced_detections
        if self.enable_tracking and self.tracker is not None:
            track_states = self.tracker.update(enhanced_detections)
            
            if self.track_smoother is not None:
                smoothed_tracks = self.track_smoother.smooth_tracks(track_states)
                # Convert smoothed tracks back to detections
                final_detections = self._tracks_to_detections(smoothed_tracks)
            else:
                # Convert track states back to detections
                final_detections = self._tracks_to_detections(track_states)
        
        annotated_frame = self.render_detections(frame, final_detections)
        return annotated_frame, final_detections
    
    def _tracks_to_detections(self, track_states) -> List[Detection]:
        """
        Convert track states back to Detection objects.
        
        Args:
            track_states: List of TrackState or SmoothedTrackState objects
            
        Returns:
            List of Detection objects with track IDs
        """
        detections = []
        for track in track_states:
            detection = Detection(
                bbox=track.bbox,
                class_name=track.class_name,
                confidence=track.confidence,
                track_id=track.track_id,
                metadata=track.metadata
            )
            detections.append(detection)
        return detections
    
    def _get_track_color(self, track_id: int) -> tuple:
        """
        Get consistent color for a track ID.
        
        Args:
            track_id: Track identifier
            
        Returns:
            BGR color tuple
        """
        if track_id not in self.track_colors:
            # Generate deterministic color from track ID
            np.random.seed(track_id)
            color = tuple(np.random.randint(50, 255, 3).tolist())
            self.track_colors[track_id] = color
        
        return self.track_colors[track_id]
    
    def _update_track_trajectory(self, track_id: int, bbox: tuple):
        """
        Update trajectory for a track.
        
        Args:
            track_id: Track identifier
            bbox: Bounding box (x1, y1, x2, y2)
        """
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        
        if track_id not in self.track_trajectories:
            self.track_trajectories[track_id] = []
        
        self.track_trajectories[track_id].append((center_x, center_y))
        
        # Keep only recent trajectory points
        if len(self.track_trajectories[track_id]) > self.trajectory_length:
            self.track_trajectories[track_id].pop(0)
    
    def _draw_track_trajectory(self, frame: np.ndarray, track_id: int, color: tuple):
        """
        Draw trajectory for a track.
        
        Args:
            frame: Frame to draw on
            track_id: Track identifier
            color: Color for trajectory
        """
        if track_id not in self.track_trajectories:
            return
        
        trajectory = self.track_trajectories[track_id]
        if len(trajectory) < 2:
            return
        
        # Draw trajectory lines
        for i in range(1, len(trajectory)):
            pt1 = trajectory[i-1]
            pt2 = trajectory[i]
            
            # Fade older points
            alpha = i / len(trajectory)
            faded_color = tuple(int(c * alpha) for c in color)
            
            cv2.line(frame, pt1, pt2, faded_color, 2)
        
        # Draw trajectory points
        for i, point in enumerate(trajectory):
            alpha = (i + 1) / len(trajectory)
            radius = int(3 * alpha)
            faded_color = tuple(int(c * alpha) for c in color)
            cv2.circle(frame, point, radius, faded_color, -1)
