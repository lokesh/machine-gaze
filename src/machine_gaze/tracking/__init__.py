"""
Object tracking module for MACHINE GAZE.

This module provides temporal consistency and tracking capabilities using ByteTrack.
"""

from .byte_tracker import ByteTracker
from .track_smoother import TrackSmoother

__all__ = ["ByteTracker", "TrackSmoother"]
