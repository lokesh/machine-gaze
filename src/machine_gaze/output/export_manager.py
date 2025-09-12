"""
Export manager for MACHINE GAZE output formats.

Supports multiple output formats including video, image sequences,
detection data, and analysis reports.
"""

import logging
import json
import csv
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import time

from ..core.base_classifier import Detection

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Data for a single frame."""
    frame_number: int
    timestamp: float
    detections: List[Detection]
    frame_path: Optional[str] = None  # Path to saved frame image
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "detections": [self._detection_to_dict(d) for d in self.detections],
            "frame_path": self.frame_path
        }
    
    def _detection_to_dict(self, detection: Detection) -> Dict[str, Any]:
        """Convert detection to dictionary."""
        # Convert numpy types to native Python types for JSON serialization
        bbox = [int(x) for x in detection.bbox]
        confidence = float(detection.confidence)
        track_id = int(detection.track_id) if detection.track_id is not None else None
        
        return {
            "bbox": bbox,
            "class_name": detection.class_name,
            "confidence": confidence,
            "track_id": track_id,
            "metadata": detection.metadata
        }


class ExportManager:
    """
    Manages multiple output formats for MACHINE GAZE.
    
    Supports:
    - Video output (MP4, AVI, MOV)
    - Image sequences (PNG, JPG)
    - Detection data (JSON, CSV)
    - Analysis reports (HTML, PDF)
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize export manager.
        
        Args:
            output_dir: Base directory for all outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Output format settings
        self.video_codec = 'mp4v'
        self.image_format = 'png'
        self.image_quality = 95
        
        # Data collection
        self.frame_data: List[FrameData] = []
        self.session_start_time = time.time()
        
        # Output files
        self.video_writer = None
        self.output_files = []
        
        logger.info(f"Initialized ExportManager with output directory: {output_dir}")
    
    def configure_video_output(self, codec: str = 'mp4v', fps: float = 30.0, 
                             quality: int = 0) -> None:
        """
        Configure video output settings.
        
        Args:
            codec: Video codec (mp4v, XVID, etc.)
            fps: Frames per second
            quality: Video quality (0 = lossless, higher = more compression)
        """
        self.video_codec = codec
        self.video_fps = fps
        self.video_quality = quality
        logger.info(f"Configured video output: {codec} @ {fps}fps")
    
    def configure_image_output(self, format: str = 'png', quality: int = 95) -> None:
        """
        Configure image output settings.
        
        Args:
            format: Image format (png, jpg, tiff)
            quality: Image quality for lossy formats
        """
        self.image_format = format.lower()
        self.image_quality = quality
        logger.info(f"Configured image output: {format} (quality: {quality})")
    
    def start_video_output(self, output_path: str, width: int, height: int, 
                          fps: float = None) -> bool:
        """
        Start video output.
        
        Args:
            output_path: Path for output video file
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second (uses configured value if None)
            
        Returns:
            True if successful, False otherwise
        """
        if fps is None:
            fps = getattr(self, 'video_fps', 30.0)
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if self.video_writer.isOpened():
                self.output_files.append(output_path)
                logger.info(f"Started video output: {output_path}")
                return True
            else:
                logger.error(f"Failed to open video writer for {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start video output: {e}")
            return False
    
    def write_frame(self, frame: np.ndarray, detections: List[Detection], 
                   frame_number: int, save_image: bool = False) -> bool:
        """
        Write a frame to all configured outputs.
        
        Args:
            frame: Frame image as numpy array
            detections: List of detections for this frame
            frame_number: Frame number
            save_image: Whether to save individual frame image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            timestamp = time.time() - self.session_start_time
            frame_path = None
            
            # Save individual frame image if requested
            if save_image:
                frame_filename = f"frame_{frame_number:06d}.{self.image_format}"
                frame_path = str(self.output_dir / "frames" / frame_filename)
                
                # Ensure frames directory exists
                Path(frame_path).parent.mkdir(exist_ok=True)
                
                # Save frame
                if self.image_format == 'jpg':
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
                elif self.image_format == 'png':
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                else:
                    cv2.imwrite(frame_path, frame)
            
            # Write to video
            if self.video_writer is not None:
                self.video_writer.write(frame)
            
            # Store frame data
            frame_data = FrameData(
                frame_number=frame_number,
                timestamp=timestamp,
                detections=detections,
                frame_path=frame_path
            )
            self.frame_data.append(frame_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write frame {frame_number}: {e}")
            return False
    
    def finish_video_output(self) -> bool:
        """
        Finish and close video output.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                logger.info("Video output finished")
            return True
        except Exception as e:
            logger.error(f"Failed to finish video output: {e}")
            return False
    
    def export_detection_data(self, format: str = 'json', filename: str = None) -> str:
        """
        Export detection data in specified format.
        
        Args:
            format: Output format ('json', 'csv')
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"detections_{timestamp}.{format}"
        
        output_path = self.output_dir / filename
        
        try:
            if format.lower() == 'json':
                self._export_json(output_path)
            elif format.lower() == 'csv':
                self._export_csv(output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.output_files.append(str(output_path))
            logger.info(f"Exported detection data to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to export detection data: {e}")
            raise
    
    def _export_json(self, output_path: Path) -> None:
        """Export data as JSON."""
        data = {
            "session_info": {
                "start_time": float(self.session_start_time),
                "total_frames": int(len(self.frame_data)),
                "output_directory": str(self.output_dir)
            },
            "frames": [frame.to_dict() for frame in self.frame_data]
        }
        
        # Custom JSON encoder to handle numpy types
        def convert_numpy(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=convert_numpy)
    
    def _export_csv(self, output_path: Path) -> None:
        """Export data as CSV."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'frame_number', 'timestamp', 'detection_id', 'bbox_x1', 'bbox_y1', 
                'bbox_x2', 'bbox_y2', 'class_name', 'confidence', 'track_id'
            ])
            
            # Data rows
            for frame in self.frame_data:
                for i, detection in enumerate(frame.detections):
                    writer.writerow([
                        frame.frame_number,
                        frame.timestamp,
                        i,
                        detection.bbox[0],
                        detection.bbox[1],
                        detection.bbox[2],
                        detection.bbox[3],
                        detection.class_name,
                        detection.confidence,
                        detection.track_id
                    ])
    
    def generate_analysis_report(self, format: str = 'html', filename: str = None) -> str:
        """
        Generate analysis report.
        
        Args:
            format: Report format ('html', 'txt')
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to generated report
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"analysis_report_{timestamp}.{format}"
        
        output_path = self.output_dir / filename
        
        try:
            if format.lower() == 'html':
                self._generate_html_report(output_path)
            elif format.lower() == 'txt':
                self._generate_text_report(output_path)
            else:
                raise ValueError(f"Unsupported report format: {format}")
            
            self.output_files.append(str(output_path))
            logger.info(f"Generated analysis report: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to generate analysis report: {e}")
            raise
    
    def _generate_html_report(self, output_path: Path) -> None:
        """Generate HTML analysis report."""
        # Calculate statistics
        stats = self._calculate_statistics()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MACHINE GAZE - Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-box {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
        .class-table {{ border-collapse: collapse; width: 100%; }}
        .class-table th, .class-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .class-table th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 MACHINE GAZE - Analysis Report</h1>
        <p>Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stats">
        <div class="stat-box">
            <h3>📊 Overall Statistics</h3>
            <p><strong>Total Frames:</strong> {stats['total_frames']}</p>
            <p><strong>Total Detections:</strong> {stats['total_detections']}</p>
            <p><strong>Average Detections/Frame:</strong> {stats['avg_detections_per_frame']:.2f}</p>
            <p><strong>Session Duration:</strong> {stats['session_duration']:.1f} seconds</p>
        </div>
        
        <div class="stat-box">
            <h3>🎯 Detection Classes</h3>
            <p><strong>Unique Classes:</strong> {len(stats['class_counts'])}</p>
            <p><strong>Most Common:</strong> {stats['most_common_class']}</p>
            <p><strong>Average Confidence:</strong> {stats['avg_confidence']:.3f}</p>
        </div>
    </div>
    
    <h2>📋 Class Detection Summary</h2>
    <table class="class-table">
        <tr>
            <th>Class Name</th>
            <th>Total Detections</th>
            <th>Average Confidence</th>
            <th>Frames Present</th>
        </tr>
"""
        
        for class_name, class_stats in stats['class_details'].items():
            html_content += f"""
        <tr>
            <td>{class_name}</td>
            <td>{class_stats['count']}</td>
            <td>{class_stats['avg_confidence']:.3f}</td>
            <td>{class_stats['frames_present']}</td>
        </tr>
"""
        
        html_content += """
    </table>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_text_report(self, output_path: Path) -> None:
        """Generate text analysis report."""
        stats = self._calculate_statistics()
        
        report = f"""
MACHINE GAZE - Analysis Report
Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

OVERALL STATISTICS:
  Total Frames: {stats['total_frames']}
  Total Detections: {stats['total_detections']}
  Average Detections/Frame: {stats['avg_detections_per_frame']:.2f}
  Session Duration: {stats['session_duration']:.1f} seconds

DETECTION CLASSES:
  Unique Classes: {len(stats['class_counts'])}
  Most Common: {stats['most_common_class']}
  Average Confidence: {stats['avg_confidence']:.3f}

CLASS DETAILS:
"""
        
        for class_name, class_stats in stats['class_details'].items():
            report += f"""
  {class_name}:
    Total Detections: {class_stats['count']}
    Average Confidence: {class_stats['avg_confidence']:.3f}
    Frames Present: {class_stats['frames_present']}
"""
        
        with open(output_path, 'w') as f:
            f.write(report)
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate statistics from collected data."""
        if not self.frame_data:
            return {
                'total_frames': 0,
                'total_detections': 0,
                'avg_detections_per_frame': 0,
                'session_duration': 0,
                'class_counts': {},
                'most_common_class': 'None',
                'avg_confidence': 0,
                'class_details': {}
            }
        
        total_frames = len(self.frame_data)
        all_detections = []
        for frame in self.frame_data:
            all_detections.extend(frame.detections)
        
        total_detections = len(all_detections)
        avg_detections_per_frame = total_detections / total_frames if total_frames > 0 else 0
        
        # Class statistics
        class_counts = {}
        class_confidences = {}
        class_frames = {}
        
        for frame in self.frame_data:
            frame_classes = set()
            for detection in frame.detections:
                class_name = detection.class_name
                
                # Count detections
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                # Track confidences
                if class_name not in class_confidences:
                    class_confidences[class_name] = []
                class_confidences[class_name].append(detection.confidence)
                
                # Track frames with this class
                frame_classes.add(class_name)
            
            # Count frames for each class
            for class_name in frame_classes:
                class_frames[class_name] = class_frames.get(class_name, 0) + 1
        
        # Most common class
        most_common_class = max(class_counts.keys(), key=class_counts.get) if class_counts else 'None'
        
        # Average confidence
        all_confidences = [d.confidence for d in all_detections]
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        # Class details
        class_details = {}
        for class_name in class_counts:
            class_details[class_name] = {
                'count': class_counts[class_name],
                'avg_confidence': sum(class_confidences[class_name]) / len(class_confidences[class_name]),
                'frames_present': class_frames[class_name]
            }
        
        session_duration = self.frame_data[-1].timestamp if self.frame_data else 0
        
        return {
            'total_frames': total_frames,
            'total_detections': total_detections,
            'avg_detections_per_frame': avg_detections_per_frame,
            'session_duration': session_duration,
            'class_counts': class_counts,
            'most_common_class': most_common_class,
            'avg_confidence': avg_confidence,
            'class_details': class_details
        }
    
    def get_output_files(self) -> List[str]:
        """Get list of all generated output files."""
        return self.output_files.copy()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.finish_video_output()
        logger.info("ExportManager cleanup complete")
