"""
Utility functions for working with detection results.

Provides functions to merge, filter, and enhance detection results
from multiple classifiers.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from ..core.base_classifier import Detection


def calculate_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box (x1, y1, x2, y2)
        bbox2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def is_bbox_inside(inner_bbox: Tuple[int, int, int, int], outer_bbox: Tuple[int, int, int, int], threshold: float = 0.7) -> bool:
    """
    Check if one bounding box is inside another.
    
    Args:
        inner_bbox: Potentially inner bounding box
        outer_bbox: Potentially outer bounding box  
        threshold: Minimum overlap ratio to consider "inside"
        
    Returns:
        True if inner_bbox is mostly inside outer_bbox
    """
    x1_inner, y1_inner, x2_inner, y2_inner = inner_bbox
    x1_outer, y1_outer, x2_outer, y2_outer = outer_bbox
    
    # Calculate overlap area
    x1_overlap = max(x1_inner, x1_outer)
    y1_overlap = max(y1_inner, y1_outer)
    x2_overlap = min(x2_inner, x2_outer)
    y2_overlap = min(y2_inner, y2_outer)
    
    if x2_overlap <= x1_overlap or y2_overlap <= y1_overlap:
        return False
    
    overlap_area = (x2_overlap - x1_overlap) * (y2_overlap - y1_overlap)
    inner_area = (x2_inner - x1_inner) * (y2_inner - y1_inner)
    
    return (overlap_area / inner_area) >= threshold if inner_area > 0 else False


def associate_faces_with_people(detections: List[Detection]) -> List[Detection]:
    """
    Associate face emotion detections with person detections.
    
    This function finds faces that are inside person bounding boxes
    and creates enhanced person detections that include emotion information.
    
    Args:
        detections: List of all detections from various classifiers
        
    Returns:
        Enhanced list of detections with face-person associations
    """
    # Separate different types of detections
    person_detections = []
    face_detections = []
    other_detections = []
    
    for detection in detections:
        if detection.class_name == "person":
            person_detections.append(detection)
        elif detection.class_name.startswith("face_"):
            face_detections.append(detection)
        else:
            other_detections.append(detection)
    
    enhanced_detections = []
    associated_faces = set()
    
    # Associate faces with people
    for person in person_detections:
        best_face = None
        best_overlap = 0.0
        best_face_idx = -1
        
        for i, face in enumerate(face_detections):
            if i in associated_faces:
                continue
                
            # Check if face is inside person bounding box
            if is_bbox_inside(face.bbox, person.bbox, threshold=0.5):
                overlap = calculate_iou(face.bbox, person.bbox)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_face = face
                    best_face_idx = i
        
        if best_face:
            # Create enhanced person detection with emotion
            emotion = best_face.metadata.get('emotion', 'unknown')
            emotion_confidence = best_face.metadata.get('emotion_confidence', 0.0)
            
            enhanced_person = Detection(
                bbox=person.bbox,
                class_name=f"person_{emotion}",
                confidence=person.confidence,
                track_id=person.track_id,
                metadata={
                    **person.metadata,
                    'has_emotion': True,
                    'emotion': emotion,
                    'emotion_confidence': emotion_confidence,
                    'face_bbox': best_face.bbox,
                    'face_confidence': best_face.metadata.get('face_confidence', 0.0)
                }
            )
            enhanced_detections.append(enhanced_person)
            associated_faces.add(best_face_idx)
        else:
            # Person without detected face emotion
            enhanced_person = Detection(
                bbox=person.bbox,
                class_name="person",
                confidence=person.confidence,
                track_id=person.track_id,
                metadata={
                    **person.metadata,
                    'has_emotion': False
                }
            )
            enhanced_detections.append(enhanced_person)
    
    # Add unassociated face detections (faces without corresponding people)
    for i, face in enumerate(face_detections):
        if i not in associated_faces:
            enhanced_detections.append(face)
    
    # Add all other detections unchanged
    enhanced_detections.extend(other_detections)
    
    return enhanced_detections


def filter_overlapping_detections(detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    """
    Filter out overlapping detections using Non-Maximum Suppression.
    
    Args:
        detections: List of detections to filter
        iou_threshold: IoU threshold for considering detections as overlapping
        
    Returns:
        Filtered list of detections
    """
    if not detections:
        return detections
    
    # Sort by confidence (highest first)
    sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    
    filtered = []
    suppressed = set()
    
    for i, detection in enumerate(sorted_detections):
        if i in suppressed:
            continue
        
        filtered.append(detection)
        
        # Suppress overlapping detections of the same class
        for j, other_detection in enumerate(sorted_detections[i+1:], i+1):
            if j in suppressed:
                continue
                
            # Only suppress same class detections
            if detection.class_name == other_detection.class_name:
                iou = calculate_iou(detection.bbox, other_detection.bbox)
                if iou > iou_threshold:
                    suppressed.add(j)
    
    return filtered


def get_detection_stats(detections: List[Detection]) -> Dict[str, int]:
    """
    Get statistics about detections by class.
    
    Args:
        detections: List of detections
        
    Returns:
        Dictionary mapping class names to counts
    """
    stats = {}
    for detection in detections:
        class_name = detection.class_name
        stats[class_name] = stats.get(class_name, 0) + 1
    
    return stats


def merge_detection_lists(detection_lists: List[List[Detection]]) -> List[Detection]:
    """
    Merge multiple lists of detections into a single list.
    
    Args:
        detection_lists: List of detection lists to merge
        
    Returns:
        Merged list of all detections
    """
    merged = []
    for detection_list in detection_lists:
        merged.extend(detection_list)
    
    return merged
