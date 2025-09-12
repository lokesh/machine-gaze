"""
Track smoother for reducing detection flickering and implementing temporal voting.

Provides smoothing algorithms to make tracking more stable and reduce noise.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import deque, defaultdict, Counter
from dataclasses import dataclass

from .byte_tracker import TrackState

logger = logging.getLogger(__name__)


@dataclass
class SmoothedTrackState:
    """Smoothed track state with temporal consistency."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float
    age: int
    hit_streak: int
    time_since_update: int
    velocity: Tuple[float, float]
    metadata: Dict
    stability_score: float  # How stable this track is (0-1)


class TemporalVoter:
    """
    Implements temporal voting using majority vote over sliding window.
    
    Smooths class predictions and confidence scores across time.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize temporal voter.
        
        Args:
            window_size: Size of sliding window for voting
        """
        self.window_size = window_size
        self.class_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.confidence_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.metadata_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
        logger.debug(f"Initialized TemporalVoter with window_size={window_size}")
    
    def update(self, track_id: int, class_name: str, confidence: float, metadata: Dict):
        """Update voting history for a track."""
        self.class_history[track_id].append(class_name)
        self.confidence_history[track_id].append(confidence)
        self.metadata_history[track_id].append(metadata or {})
    
    def get_voted_class(self, track_id: int) -> Tuple[str, float]:
        """
        Get majority voted class and average confidence.
        
        Returns:
            (voted_class_name, averaged_confidence)
        """
        if track_id not in self.class_history:
            return "", 0.0
        
        classes = list(self.class_history[track_id])
        confidences = list(self.confidence_history[track_id])
        
        if not classes:
            return "", 0.0
        
        # Majority vote for class
        class_counter = Counter(classes)
        voted_class = class_counter.most_common(1)[0][0]
        
        # Average confidence for the voted class
        voted_confidences = [conf for cls, conf in zip(classes, confidences) if cls == voted_class]
        avg_confidence = np.mean(voted_confidences) if voted_confidences else 0.0
        
        return voted_class, float(avg_confidence)
    
    def get_voted_metadata(self, track_id: int) -> Dict:
        """Get merged metadata from voting window."""
        if track_id not in self.metadata_history:
            return {}
        
        # Merge all metadata, with more recent taking precedence
        merged_metadata = {}
        for metadata in self.metadata_history[track_id]:
            if metadata:
                merged_metadata.update(metadata)
        
        return merged_metadata
    
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """Remove history for tracks that are no longer active."""
        all_track_ids = set(self.class_history.keys())
        active_set = set(active_track_ids)
        
        for track_id in all_track_ids - active_set:
            if track_id in self.class_history:
                del self.class_history[track_id]
            if track_id in self.confidence_history:
                del self.confidence_history[track_id]
            if track_id in self.metadata_history:
                del self.metadata_history[track_id]


class KalmanBBoxSmoother:
    """
    Kalman filter for smoothing bounding box positions.
    
    Uses a more sophisticated model than the simple tracker's Kalman filter.
    """
    
    def __init__(self):
        """Initialize Kalman filter for bbox smoothing."""
        # State: [center_x, center_y, width, height, vel_x, vel_y, vel_w, vel_h]
        self.state = np.zeros(8)
        self.covariance = np.eye(8) * 100
        
        # State transition matrix
        self.F = np.eye(8)
        self.F[0, 4] = 1  # x += vel_x
        self.F[1, 5] = 1  # y += vel_y
        self.F[2, 6] = 1  # w += vel_w
        self.F[3, 7] = 1  # h += vel_h
        
        # Measurement matrix (observe position and size)
        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1  # observe center_x
        self.H[1, 1] = 1  # observe center_y
        self.H[2, 2] = 1  # observe width
        self.H[3, 3] = 1  # observe height
        
        # Process noise (allow for acceleration)
        self.Q = np.eye(8) * 0.1
        self.Q[4:, 4:] *= 10  # Higher noise for velocity
        
        # Measurement noise
        self.R = np.eye(4) * 5
        
        self.initialized = False
    
    def update(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Update with new bounding box and return smoothed result."""
        # Convert bbox to measurement
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        measurement = np.array([center_x, center_y, width, height])
        
        if not self.initialized:
            # Initialize with first measurement
            self.state[:4] = measurement
            self.initialized = True
            return bbox
        
        # Predict
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        # Update
        y = measurement - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(8) - K @ self.H) @ self.covariance
        
        # Convert back to bbox
        smoothed_center_x = self.state[0]
        smoothed_center_y = self.state[1]
        smoothed_width = max(10, self.state[2])  # Minimum width
        smoothed_height = max(10, self.state[3])  # Minimum height
        
        x1 = int(smoothed_center_x - smoothed_width / 2)
        y1 = int(smoothed_center_y - smoothed_height / 2)
        x2 = int(smoothed_center_x + smoothed_width / 2)
        y2 = int(smoothed_center_y + smoothed_height / 2)
        
        return (x1, y1, x2, y2)


class TrackSmoother:
    """
    Main track smoother that combines temporal voting and Kalman smoothing.
    
    Reduces flickering and improves temporal consistency of tracking results.
    """
    
    def __init__(self, 
                 temporal_window: int = 5,
                 bbox_smoothing: bool = True,
                 confidence_smoothing: bool = True,
                 class_voting: bool = True):
        """
        Initialize track smoother.
        
        Args:
            temporal_window: Size of temporal window for voting
            bbox_smoothing: Enable Kalman smoothing for bounding boxes
            confidence_smoothing: Enable confidence score smoothing
            class_voting: Enable temporal voting for class predictions
        """
        self.temporal_window = temporal_window
        self.bbox_smoothing = bbox_smoothing
        self.confidence_smoothing = confidence_smoothing
        self.class_voting = class_voting
        
        # Initialize components
        if self.class_voting:
            self.temporal_voter = TemporalVoter(temporal_window)
        
        if self.bbox_smoothing:
            self.kalman_smoothers: Dict[int, KalmanBBoxSmoother] = {}
        
        if self.confidence_smoothing:
            self.confidence_history: Dict[int, deque] = defaultdict(
                lambda: deque(maxlen=temporal_window))
        
        # Track stability scoring
        self.stability_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=temporal_window))
        
        logger.info(f"Initialized TrackSmoother: window={temporal_window}, "
                   f"bbox_smooth={bbox_smoothing}, conf_smooth={confidence_smoothing}, "
                   f"class_vote={class_voting}")
    
    def calculate_stability_score(self, track_id: int, bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculate stability score for a track based on bbox consistency.
        
        Returns score between 0 (unstable) and 1 (very stable).
        """
        if track_id not in self.stability_history:
            return 0.5  # Neutral score for new tracks
        
        history = list(self.stability_history[track_id])
        if len(history) < 2:
            return 0.5
        
        # Calculate bbox variation
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        current_features = np.array([center_x, center_y, width, height])
        
        # Compare with recent history
        variations = []
        for hist_bbox in history[-3:]:  # Last 3 frames
            hist_center_x = (hist_bbox[0] + hist_bbox[2]) / 2
            hist_center_y = (hist_bbox[1] + hist_bbox[3]) / 2
            hist_width = hist_bbox[2] - hist_bbox[0]
            hist_height = hist_bbox[3] - hist_bbox[1]
            
            hist_features = np.array([hist_center_x, hist_center_y, hist_width, hist_height])
            
            # Normalized variation
            variation = np.linalg.norm(current_features - hist_features) / np.linalg.norm(hist_features + 1e-6)
            variations.append(variation)
        
        # Convert variation to stability (lower variation = higher stability)
        avg_variation = np.mean(variations)
        stability_score = max(0.0, min(1.0, 1.0 - avg_variation * 5))  # Scale factor of 5
        
        return stability_score
    
    def smooth_tracks(self, track_states: List[TrackState]) -> List[SmoothedTrackState]:
        """
        Apply smoothing to track states.
        
        Args:
            track_states: Raw track states from tracker
            
        Returns:
            Smoothed track states
        """
        smoothed_tracks = []
        active_track_ids = [track.track_id for track in track_states]
        
        for track in track_states:
            track_id = track.track_id
            
            # Update histories
            if self.class_voting:
                self.temporal_voter.update(track_id, track.class_name, track.confidence, track.metadata)
            
            if self.confidence_smoothing:
                self.confidence_history[track_id].append(track.confidence)
            
            self.stability_history[track_id].append(track.bbox)
            
            # Apply smoothing
            smoothed_bbox = track.bbox
            if self.bbox_smoothing:
                if track_id not in self.kalman_smoothers:
                    self.kalman_smoothers[track_id] = KalmanBBoxSmoother()
                smoothed_bbox = self.kalman_smoothers[track_id].update(track.bbox)
            
            # Apply temporal voting
            smoothed_class = track.class_name
            smoothed_confidence = track.confidence
            smoothed_metadata = track.metadata
            
            if self.class_voting:
                voted_class, voted_confidence = self.temporal_voter.get_voted_class(track_id)
                if voted_class:
                    smoothed_class = voted_class
                    smoothed_confidence = voted_confidence
                smoothed_metadata = self.temporal_voter.get_voted_metadata(track_id)
            
            # Apply confidence smoothing
            if self.confidence_smoothing and track_id in self.confidence_history:
                conf_history = list(self.confidence_history[track_id])
                smoothed_confidence = float(np.mean(conf_history))
            
            # Calculate stability score
            stability_score = self.calculate_stability_score(track_id, track.bbox)
            
            # Create smoothed track state
            smoothed_track = SmoothedTrackState(
                track_id=track_id,
                bbox=smoothed_bbox,
                class_name=smoothed_class,
                confidence=smoothed_confidence,
                age=track.age,
                hit_streak=track.hit_streak,
                time_since_update=track.time_since_update,
                velocity=track.velocity,
                metadata=smoothed_metadata,
                stability_score=stability_score
            )
            
            smoothed_tracks.append(smoothed_track)
        
        # Cleanup old tracks
        if self.class_voting:
            self.temporal_voter.cleanup_old_tracks(active_track_ids)
        
        # Remove old Kalman smoothers
        if self.bbox_smoothing:
            old_track_ids = set(self.kalman_smoothers.keys()) - set(active_track_ids)
            for track_id in old_track_ids:
                del self.kalman_smoothers[track_id]
        
        # Remove old confidence history
        if self.confidence_smoothing:
            old_track_ids = set(self.confidence_history.keys()) - set(active_track_ids)
            for track_id in old_track_ids:
                del self.confidence_history[track_id]
        
        logger.debug(f"Smoothed {len(track_states)} tracks -> {len(smoothed_tracks)} smoothed tracks")
        
        return smoothed_tracks
    
    def reset(self):
        """Reset smoother state."""
        if hasattr(self, 'temporal_voter'):
            self.temporal_voter = TemporalVoter(self.temporal_window)
        
        if hasattr(self, 'kalman_smoothers'):
            self.kalman_smoothers.clear()
        
        if hasattr(self, 'confidence_history'):
            self.confidence_history.clear()
        
        self.stability_history.clear()
        
        logger.info("TrackSmoother reset")
