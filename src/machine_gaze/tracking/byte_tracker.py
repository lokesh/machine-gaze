"""
ByteTracker implementation for object tracking.

Provides temporal consistency by tracking objects across frames using the ByteTrack algorithm.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque

from ..core.base_classifier import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    """State information for a tracked object."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_name: str
    confidence: float
    age: int  # Number of frames this track has existed
    hit_streak: int  # Number of consecutive frames with detections
    time_since_update: int  # Frames since last detection
    velocity: Tuple[float, float]  # dx, dy per frame
    metadata: Dict


class SimpleKalmanFilter:
    """
    Simple Kalman filter for tracking bounding box center and velocity.
    
    State vector: [center_x, center_y, velocity_x, velocity_y]
    """
    
    def __init__(self):
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.covariance = np.eye(4) * 1000  # Initial uncertainty
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=float)
        
        # Measurement matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        
        # Process noise
        self.Q = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 100, 0],
            [0, 0, 0, 100]
        ], dtype=float)
        
        # Measurement noise
        self.R = np.array([
            [10, 0],
            [0, 10]
        ], dtype=float)
    
    def predict(self):
        """Predict next state."""
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        return self.state[:2]  # Return predicted position
    
    def update(self, measurement):
        """Update with measurement (center_x, center_y)."""
        measurement = np.array(measurement)
        
        # Innovation
        y = measurement - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        # Update
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance
        
        return self.state[:2]  # Return updated position


class Track:
    """Individual track for a detected object."""
    
    def __init__(self, detection: Detection, track_id: int):
        self.track_id = track_id
        self.class_name = detection.class_name
        self.age = 0
        self.hit_streak = 1
        self.time_since_update = 0
        self.metadata = detection.metadata or {}
        
        # Initialize Kalman filter
        self.kalman = SimpleKalmanFilter()
        center_x = (detection.bbox[0] + detection.bbox[2]) / 2
        center_y = (detection.bbox[1] + detection.bbox[3]) / 2
        self.kalman.state[:2] = [center_x, center_y]
        
        # Store bbox dimensions
        self.width = detection.bbox[2] - detection.bbox[0]
        self.height = detection.bbox[3] - detection.bbox[1]
        
        # History for smoothing
        self.confidence_history = deque([detection.confidence], maxlen=10)
        self.bbox_history = deque([detection.bbox], maxlen=5)
    
    def predict(self):
        """Predict next position."""
        self.age += 1
        self.time_since_update += 1
        predicted_center = self.kalman.predict()
        
        # Convert back to bbox format
        x1 = int(predicted_center[0] - self.width / 2)
        y1 = int(predicted_center[1] - self.height / 2)
        x2 = int(predicted_center[0] + self.width / 2)
        y2 = int(predicted_center[1] + self.height / 2)
        
        return (x1, y1, x2, y2)
    
    def update(self, detection: Detection):
        """Update track with new detection."""
        self.time_since_update = 0
        self.hit_streak += 1
        
        # Update Kalman filter
        center_x = (detection.bbox[0] + detection.bbox[2]) / 2
        center_y = (detection.bbox[1] + detection.bbox[3]) / 2
        self.kalman.update([center_x, center_y])
        
        # Update dimensions (with smoothing)
        new_width = detection.bbox[2] - detection.bbox[0]
        new_height = detection.bbox[3] - detection.bbox[1]
        self.width = 0.7 * self.width + 0.3 * new_width
        self.height = 0.7 * self.height + 0.3 * new_height
        
        # Update history
        self.confidence_history.append(detection.confidence)
        self.bbox_history.append(detection.bbox)
        
        # Update metadata if available
        if detection.metadata:
            self.metadata.update(detection.metadata)
    
    def get_state(self) -> TrackState:
        """Get current track state."""
        # Get smoothed position
        predicted_bbox = self.predict()
        self.time_since_update -= 1  # Undo the increment from predict()
        
        # Get smoothed confidence
        smoothed_confidence = np.mean(list(self.confidence_history))
        
        # Calculate velocity
        velocity = (0.0, 0.0)
        if hasattr(self.kalman, 'state'):
            velocity = (float(self.kalman.state[2]), float(self.kalman.state[3]))
        
        return TrackState(
            track_id=self.track_id,
            bbox=predicted_bbox,
            class_name=self.class_name,
            confidence=float(smoothed_confidence),
            age=self.age,
            hit_streak=self.hit_streak,
            time_since_update=self.time_since_update,
            velocity=velocity,
            metadata=self.metadata
        )


class ByteTracker:
    """
    ByteTracker implementation for multi-object tracking.
    
    Provides temporal consistency by tracking objects across frames.
    """
    
    def __init__(self, 
                 high_thresh: float = 0.6,
                 low_thresh: float = 0.1,
                 new_track_thresh: float = 0.7,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8):
        """
        Initialize ByteTracker.
        
        Args:
            high_thresh: High confidence threshold for tracking
            low_thresh: Low confidence threshold for potential matches
            new_track_thresh: Threshold for creating new tracks
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IoU threshold for matching detections to tracks
        """
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self.tracked_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []
        
        self.frame_id = 0
        self.track_id_count = 0
        
        logger.info(f"Initialized ByteTracker with thresholds: high={high_thresh}, low={low_thresh}")
    
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def associate_detections_to_tracks(self, detections: List[Detection], 
                                     tracks: List[Track]) -> Tuple[List[Tuple[int, int]], 
                                                                  List[int], 
                                                                  List[int]]:
        """
        Associate detections to tracks using IoU matching.
        
        Returns:
            matches: List of (detection_idx, track_idx) pairs
            unmatched_detections: List of detection indices
            unmatched_tracks: List of track indices
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for d, detection in enumerate(detections):
            for t, track in enumerate(tracks):
                predicted_bbox = track.predict()
                track.time_since_update -= 1  # Undo increment from predict
                iou_matrix[d, t] = self.calculate_iou(detection.bbox, predicted_bbox)
        
        # Find matches using simple greedy matching
        matches = []
        matched_det_indices = set()
        matched_track_indices = set()
        
        # Sort by IoU in descending order
        iou_pairs = []
        for d in range(len(detections)):
            for t in range(len(tracks)):
                if iou_matrix[d, t] > self.match_thresh:
                    iou_pairs.append((iou_matrix[d, t], d, t))
        
        iou_pairs.sort(reverse=True)
        
        for iou, d, t in iou_pairs:
            if d not in matched_det_indices and t not in matched_track_indices:
                matches.append((d, t))
                matched_det_indices.add(d)
                matched_track_indices.add(t)
        
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_det_indices]
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_track_indices]
        
        return matches, unmatched_detections, unmatched_tracks
    
    def update(self, detections: List[Detection]) -> List[TrackState]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections for current frame
            
        Returns:
            List of current track states
        """
        self.frame_id += 1
        
        # Separate high and low confidence detections
        high_conf_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_conf_dets = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]
        
        # Predict all tracks
        for track in self.tracked_tracks:
            track.predict()
        
        # First association: high confidence detections with tracked tracks
        matches, unmatched_dets, unmatched_tracks = self.associate_detections_to_tracks(
            high_conf_dets, self.tracked_tracks)
        
        # Update matched tracks
        for det_idx, track_idx in matches:
            self.tracked_tracks[track_idx].update(high_conf_dets[det_idx])
        
        # Handle unmatched tracks - move to lost
        for track_idx in unmatched_tracks:
            track = self.tracked_tracks[track_idx]
            if track.time_since_update < self.track_buffer:
                self.lost_tracks.append(track)
        
        # Remove unmatched tracks from tracked list
        self.tracked_tracks = [self.tracked_tracks[i] for i in range(len(self.tracked_tracks)) 
                              if i not in unmatched_tracks]
        
        # Second association: remaining detections with lost tracks
        if len(self.lost_tracks) > 0 and len(unmatched_dets) > 0:
            remaining_dets = [high_conf_dets[i] for i in unmatched_dets]
            matches2, unmatched_dets2, unmatched_lost = self.associate_detections_to_tracks(
                remaining_dets, self.lost_tracks)
            
            # Recover matched lost tracks
            for det_idx, track_idx in matches2:
                track = self.lost_tracks[track_idx]
                track.update(remaining_dets[det_idx])
                self.tracked_tracks.append(track)
            
            # Update unmatched detections
            unmatched_dets = [unmatched_dets[i] for i in unmatched_dets2]
            
            # Remove recovered tracks from lost list
            self.lost_tracks = [self.lost_tracks[i] for i in range(len(self.lost_tracks)) 
                               if i not in [match[1] for match in matches2]]
        
        # Third association: low confidence detections with remaining tracks
        if len(low_conf_dets) > 0:
            all_remaining_tracks = self.tracked_tracks + self.lost_tracks
            matches3, _, _ = self.associate_detections_to_tracks(low_conf_dets, all_remaining_tracks)
            
            for det_idx, track_idx in matches3:
                if track_idx < len(self.tracked_tracks):
                    self.tracked_tracks[track_idx].update(low_conf_dets[det_idx])
                else:
                    lost_idx = track_idx - len(self.tracked_tracks)
                    track = self.lost_tracks[lost_idx]
                    track.update(low_conf_dets[det_idx])
                    self.tracked_tracks.append(track)
                    self.lost_tracks.remove(track)
        
        # Create new tracks for high confidence unmatched detections
        for det_idx in unmatched_dets:
            detection = high_conf_dets[det_idx]
            if detection.confidence >= self.new_track_thresh:
                self.track_id_count += 1
                new_track = Track(detection, self.track_id_count)
                self.tracked_tracks.append(new_track)
        
        # Clean up old lost tracks
        self.lost_tracks = [track for track in self.lost_tracks 
                           if track.time_since_update < self.track_buffer]
        
        # Get current track states
        track_states = []
        for track in self.tracked_tracks:
            if track.hit_streak >= 2:  # Only return tracks that have been matched at least twice
                track_states.append(track.get_state())
        
        logger.debug(f"Frame {self.frame_id}: {len(detections)} detections -> {len(track_states)} tracks")
        
        return track_states
    
    def reset(self):
        """Reset tracker state."""
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.frame_id = 0
        self.track_id_count = 0
        logger.info("ByteTracker reset")
