# 🤖 MACHINE GAZE - User Guide

*Advanced Computer Vision Pipeline for Object Detection, Emotion Recognition, and Tracking*

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Installation](#-installation)
3. [Basic Usage](#-basic-usage)
4. [Configuration System](#-configuration-system)
5. [Advanced Features](#-advanced-features)
6. [Batch Processing](#-batch-processing)
7. [Multiple Output Formats](#-multiple-output-formats)
8. [Tracking & Smoothing](#-tracking--smoothing)
9. [Troubleshooting](#-troubleshooting)
10. [API Reference](#-api-reference)

---

## 🚀 Quick Start

Process a video with tracking and emotion detection in 30 seconds:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Process a video with default settings
python main.py input_video.mp4 output_video.mp4

# 3. Enable advanced tracking with emotions
python main.py input_video.mp4 output_video.mp4 --config config/tracking_config.yaml
```

**That's it!** Your processed video will include:
- ✅ Object detection (people, cars, etc.)
- ✅ Face emotion recognition  
- ✅ Temporal tracking with smooth IDs
- ✅ Motion trajectories

---

## 📦 Installation

### Prerequisites
- **Python 3.9+**
- **macOS with M1/M2** (recommended) or other platforms
- **4GB+ RAM** for video processing

### Install Dependencies

```bash
# Clone or download the MACHINE GAZE project
cd machine-gaze/

# Install Python packages
pip install -r requirements.txt

# Verify installation
python test_setup.py
```

### GPU Acceleration (Optional)
- **M1/M2 Mac**: MPS acceleration automatically enabled
- **NVIDIA GPU**: Install CUDA toolkit for GPU acceleration
- **CPU only**: Works fine, just slower

---

## 🎮 Basic Usage

### Simple Video Processing

```bash
# Basic object detection
python main.py input.mp4 output.mp4

# With specific configuration
python main.py input.mp4 output.mp4 --config config/emotion_detection.yaml

# Overlay-only mode (no original video background)
python main.py input.mp4 output.mp4 --overlay-only

# Debug mode with detailed logging
python main.py input.mp4 output.mp4 --debug
```

### Example Outputs

| Command | What You Get |
|---------|--------------|
| `python main.py video.mp4 out.mp4` | Basic object detection on video |
| `python main.py video.mp4 out.mp4 --config config/emotion_detection.yaml` | Objects + face emotions |
| `python main.py video.mp4 out.mp4 --config config/tracking_config.yaml` | Full tracking with motion trails |
| `python main.py video.mp4 out.mp4 --overlay-only` | Clean overlay without original video |

---

## ⚙️ Configuration System

MACHINE GAZE uses YAML configuration files for easy customization.

### Quick Configuration Management

```bash
# List available templates
python config_manager_cli.py list

# Create config from template
python config_manager_cli.py create emotion_analysis my_config.yaml

# Interactive config creation
python config_manager_cli.py interactive

# Validate existing config
python config_manager_cli.py validate config/my_config.yaml

# Add classifier to existing config
python config_manager_cli.py add-classifier config/my_config.yaml face_emotion
```

### Built-in Configuration Templates

| Template | Description | Use Case |
|----------|-------------|----------|
| `basic_detection` | Simple object detection | General purpose |
| `emotion_analysis` | Objects + emotion recognition | Social media, content analysis |
| `advanced_tracking` | Full tracking with trails | Security, sports analysis |
| `security_monitoring` | High-accuracy person tracking | Surveillance, safety |

### Custom Configuration Example

```yaml
# my_custom_config.yaml
video_processor:
  overlay_only: false
  tracking:
    enabled: true
    show_track_ids: true
    show_trajectories: true

classifiers:
  yolo_world:
    enabled: true
    confidence_threshold: 0.6
    classes: ["person", "car", "dog", "cat"]
  
  face_emotion:
    enabled: true
    face_confidence: 0.5
    emotion_confidence: 0.3
```

---

## 🔬 Advanced Features

### 1. Multi-Object Tracking

Track objects across frames with persistent IDs:

```bash
# Enable tracking
python main.py video.mp4 output.mp4 --config config/tracking_config.yaml
```

**Features:**
- 🎯 **Persistent Track IDs**: Objects keep same ID across frames
- 🌈 **Color-coded Visualization**: Each track has unique color
- 📈 **Motion Trajectories**: See where objects have been
- 🎚️ **Temporal Smoothing**: Reduces detection flickering

### 2. Emotion Recognition

Detect and track emotions on faces:

```bash
# Emotion detection
python main.py video.mp4 output.mp4 --config config/emotion_detection.yaml
```

**Detected Emotions:**
- 😊 Happy
- 😢 Sad  
- 😡 Angry
- 😱 Fear
- 😮 Surprise
- 🤢 Disgust
- 😐 Neutral

### 3. Custom Object Classes

Add your own objects to detect:

```yaml
# In your config file
classifiers:
  yolo_world:
    classes: ["person", "dog", "cat", "bicycle", "tree", "your_custom_object"]
```

### 4. Advanced Video Processing

```bash
# Multiple classifiers active
python main.py video.mp4 output.mp4 --config config/emotion_analysis.yaml

# Custom background color for overlay
python main.py video.mp4 output.mp4 --config config/white_background.yaml

# Extended object detection
python main.py video.mp4 output.mp4 --config config/extended_objects.yaml
```

---

## 🔄 Batch Processing

Process multiple videos efficiently with the batch processor.

### Directory Processing

```bash
# Process all videos in a directory
python batch_processor.py directory input_videos/ output_videos/ config/tracking_config.yaml

# With parallel processing (2 workers)
python batch_processor.py directory input_videos/ output_videos/ config/emotion_detection.yaml --workers 2
```

### CSV Batch Jobs

```bash
# Create example CSV
python batch_processor.py create-example

# Edit example_batch.csv with your video paths
# Then process batch
python batch_processor.py csv example_batch.csv --workers 3
```

**Example batch CSV:**
```csv
input_path,output_path,config_path,job_id
videos/video1.mp4,output/video1_processed.mp4,config/tracking_config.yaml,video1_tracking
videos/video2.mp4,output/video2_processed.mp4,config/emotion_detection.yaml,video2_emotion
```

### Single Video in Batch Mode

```bash
# Process single video with batch system
python batch_processor.py single input.mp4 output.mp4 config/tracking_config.yaml
```

---

## 📊 Multiple Output Formats

Generate videos, data exports, and analysis reports.

### Advanced Processing Script

```bash
# Multiple output formats
python advanced_main.py input.mp4 \
  --config config/tracking_config.yaml \
  --video-output processed.mp4 \
  --data-export json,csv \
  --analysis-report html,txt \
  --image-sequence png

# Custom output directory
python advanced_main.py input.mp4 \
  --config config/emotion_detection.yaml \
  --video-output my_video.mp4 \
  --data-export json \
  --analysis-report html \
  --output-dir my_analysis/
```

### Available Output Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| **Video** | `.mp4`, `.avi`, `.mov` | Processed video with annotations |
| **Data** | `.json`, `.csv` | Detection data for analysis |
| **Reports** | `.html`, `.txt` | Statistical analysis reports |
| **Images** | `.png`, `.jpg`, `.tiff` | Individual frame sequences |

### Example Outputs

```bash
# After processing, you get:
output_directory/
├── processed_video.mp4      # Annotated video
├── detections.json         # All detection data
├── detections.csv          # Tabular detection data  
├── analysis_report.html    # Interactive HTML report
├── analysis_report.txt     # Text summary
└── frames/                 # Individual frame images
    ├── frame_000001.png
    ├── frame_000002.png
    └── ...
```

---

## 🎯 Tracking & Smoothing

Advanced temporal processing for stable, smooth results.

### Tracking Features

**ByteTracker Integration:**
- Multi-object tracking across frames
- Recovery of temporarily lost objects
- Configurable confidence thresholds

**Temporal Smoothing:**
- Kalman filtering for smooth bounding boxes
- Confidence score averaging
- Class prediction voting

### Tracking Configuration

```yaml
tracking:
  enabled: true
  
  # ByteTracker parameters
  high_thresh: 0.6           # High confidence detections
  low_thresh: 0.1            # Low confidence recovery
  new_track_thresh: 0.7      # New track creation
  track_buffer: 30           # Frames to keep lost tracks
  match_thresh: 0.8          # IoU threshold for matching
  
  # Visualization
  show_track_ids: true       # Display track IDs
  show_trajectories: true    # Motion trails
  trajectory_length: 15      # Trail length
  
  # Smoothing
  temporal_window: 5         # Voting window size
  bbox_smoothing: true       # Smooth bounding boxes
  confidence_smoothing: true # Average confidence scores
  class_voting: true         # Temporal class voting
```

### Performance Tuning

| Use Case | Recommended Settings |
|----------|---------------------|
| **Real-time** | `temporal_window: 3`, `bbox_smoothing: true` |
| **High accuracy** | `high_thresh: 0.8`, `temporal_window: 7` |
| **Fast processing** | `tracking.enabled: false` |
| **Crowded scenes** | `match_thresh: 0.9`, `track_buffer: 50` |

---

## 🔧 Troubleshooting

### Common Issues

#### "No module named 'cv2'"
```bash
# Install OpenCV
pip install opencv-python
```

#### "Numpy not available" 
```bash
# Downgrade numpy for compatibility
pip install "numpy<2.0"
```

#### "CLIP library not found"
```bash
# Install CLIP manually
pip install git+https://github.com/ultralytics/CLIP.git
```

#### Face detection not working
```bash
# Use debug mode to diagnose
python debug_face_detection.py your_video.mp4 --frame 50

# Try more permissive settings
python config_manager_cli.py create emotion_analysis permissive_config.yaml
# Then edit face_confidence: 0.1, emotion_confidence: 0.1
```

#### Low processing speed
```bash
# Disable tracking for faster processing
python main.py video.mp4 output.mp4 --config config/basic_detection.yaml

# Reduce video resolution or use smaller models
```

### Debug Mode

```bash
# Enable verbose logging
python main.py video.mp4 output.mp4 --debug

# Check specific frame
python debug_face_detection.py video.mp4 --frame 100

# Validate configuration
python config_manager_cli.py validate config/your_config.yaml
```

### Performance Optimization

```bash
# Use MPS acceleration on M1 Mac (automatic)
# Reduce confidence thresholds for faster processing
# Use basic_detection config for maximum speed
# Process smaller videos first to test setup
```

---

## 📚 API Reference

### Core Classes

#### VideoProcessor
```python
from machine_gaze import VideoProcessor, get_registry, ConfigLoader

# Initialize
config = ConfigLoader().load_config("config/tracking_config.yaml")
registry = get_registry()
registry.setup_from_config(config)
processor = VideoProcessor(registry, config.get('video_processor', {}))

# Process video
success = processor.process_video("input.mp4", "output.mp4")

# Process single frame
annotated_frame, detections = processor.process_single_frame(frame)
```

#### Configuration Manager
```python
from machine_gaze.utils.config_manager import ConfigManager

# Initialize
manager = ConfigManager("config/")

# Create config from template
manager.create_config_from_template("emotion_analysis", "my_config.yaml")

# Add classifier
manager.add_classifier_to_config("my_config.yaml", "face_emotion")

# Validate
validation = manager.validate_config("my_config.yaml")
```

#### Export Manager
```python
from machine_gaze.output.export_manager import ExportManager

# Initialize
exporter = ExportManager("output_dir/")

# Configure outputs
exporter.configure_video_output("mp4v", 30.0)
exporter.start_video_output("output.mp4", 1920, 1080)

# Export data
exporter.export_detection_data("json", "detections.json")
exporter.generate_analysis_report("html", "report.html")
```

### Detection Format

```python
# Detection object structure
class Detection:
    bbox: tuple              # (x1, y1, x2, y2)
    class_name: str         # "person", "face_happy", etc.
    confidence: float       # 0.0 to 1.0
    track_id: int          # Unique track identifier
    metadata: dict         # Additional information
```

### Configuration Structure

```yaml
video_processor:
  overlay_only: bool
  background_color: [R, G, B]
  tracking:
    enabled: bool
    # ... tracking parameters

classifiers:
  classifier_name:
    enabled: bool
    confidence_threshold: float
    # ... classifier-specific parameters

logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
```

---

## 🎯 Examples & Use Cases

### Security & Surveillance
```bash
# High-accuracy person tracking
python main.py security_feed.mp4 monitored_output.mp4 --config config/security_monitoring.yaml
```

### Social Media Analysis
```bash
# Emotion detection for content analysis
python main.py social_video.mp4 emotion_analysis.mp4 --config config/emotion_analysis.yaml
```

### Sports Analysis
```bash
# Player tracking with motion trails
python main.py game_footage.mp4 player_tracking.mp4 --config config/advanced_tracking.yaml
```

### Accessibility Applications
```bash
# Wheelchair/walker detection
python config_manager_cli.py create basic_detection accessibility_config.yaml
# Edit to include: "wheelchair", "walker", "cane"
python main.py accessibility_video.mp4 output.mp4 --config accessibility_config.yaml
```

### Batch Processing for Research
```bash
# Process entire dataset
python batch_processor.py directory research_videos/ analyzed_videos/ config/emotion_analysis.yaml --workers 4
```

---

## 🚀 Next Steps

- **Explore different configurations** using the config manager
- **Try batch processing** for multiple videos
- **Generate analysis reports** to understand your data
- **Customize object classes** for your specific use case
- **Integrate tracking** for temporal consistency

**Happy detecting! 🎉**

---

*MACHINE GAZE - Making computer vision accessible and powerful for everyone.*
