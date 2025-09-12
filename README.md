# 🤖 MACHINE GAZE

*Advanced Computer Vision Pipeline for Object Detection, Emotion Recognition, and Tracking*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![M1 Mac Optimized](https://img.shields.io/badge/M1%20Mac-optimized-red.svg)](README.md)

---

## ✨ Features

🎯 **Multi-Object Detection** - People, vehicles, animals, and custom objects  
😊 **Emotion Recognition** - Real-time facial emotion analysis  
🎪 **Advanced Tracking** - Persistent object IDs with motion trails  
⚡ **M1 Mac Optimized** - Native MPS acceleration for Apple Silicon  
🔧 **Highly Configurable** - YAML-based configuration system  
📊 **Multiple Output Formats** - Video, JSON, CSV, HTML reports  
🚀 **Batch Processing** - Process multiple videos efficiently  
🎨 **Professional Visualization** - Clean overlays and tracking displays  

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Process a video with default settings
python main.py input.mp4 output.mp4

# Enable advanced tracking with emotions
python main.py input.mp4 output.mp4 --config config/tracking_config.yaml --overlay-only
```

**Result:** Professional video analysis with object detection, emotion recognition, and temporal tracking!

---

## 🎬 Demo

| Original | MACHINE GAZE Output |
|----------|-------------------|
| ![Input Video](docs/input_example.png) | ![Output Video](docs/output_example.png) |

*Example: Person tracking with emotion recognition and motion trails*

---

## 📦 Installation

### Prerequisites
- Python 3.9+
- 4GB+ RAM for video processing
- M1/M2 Mac recommended (MPS acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/machine-gaze.git
cd machine-gaze

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_setup.py
```

### Optional: GPU Acceleration
- **M1/M2 Mac**: MPS automatically enabled ✅
- **NVIDIA GPU**: Install CUDA toolkit
- **CPU only**: Works fine, just slower

---

## 🎮 Usage Examples

### Basic Processing
```bash
# Simple object detection
python main.py video.mp4 output.mp4

# With emotion recognition
python main.py video.mp4 output.mp4 --config config/emotion_detection.yaml

# Overlay-only mode (clean background)
python main.py video.mp4 output.mp4 --overlay-only
```

### Advanced Tracking
```bash
# Full tracking with motion trails
python main.py video.mp4 output.mp4 --config config/tracking_config.yaml

# Security monitoring setup
python main.py security_feed.mp4 monitored.mp4 --config config/security_monitoring.yaml
```

### Batch Processing
```bash
# Process entire directory
python batch_processor.py directory input_videos/ output_videos/ config/tracking_config.yaml

# Process from CSV list
python batch_processor.py csv video_list.csv --workers 3
```

### Multiple Output Formats
```bash
# Generate video, data, and reports
python advanced_main.py input.mp4 \
  --config config/emotion_detection.yaml \
  --video-output processed.mp4 \
  --data-export json,csv \
  --analysis-report html
```

---

## ⚙️ Configuration

MACHINE GAZE uses a powerful YAML-based configuration system.

### Built-in Templates

| Template | Description | Use Case |
|----------|-------------|----------|
| `basic_detection` | Simple object detection | General purpose |
| `emotion_analysis` | Objects + emotion recognition | Social media analysis |
| `advanced_tracking` | Full tracking with trails | Security, sports |
| `security_monitoring` | High-accuracy person tracking | Surveillance |

### Quick Configuration
```bash
# List available templates
python config_manager_cli.py list

# Create from template
python config_manager_cli.py create emotion_analysis my_config.yaml

# Interactive creation
python config_manager_cli.py interactive
```

### Example Configuration
```yaml
video_processor:
  overlay_only: true
  tracking:
    enabled: true
    show_track_ids: true
    show_trajectories: true

classifiers:
  yolo_world:
    enabled: true
    confidence_threshold: 0.6
    classes: ["person", "car", "bicycle"]
  
  face_emotion:
    enabled: true
    face_confidence: 0.5
    emotion_confidence: 0.3
```

---

## 🏗️ Architecture

MACHINE GAZE follows a modular, extensible architecture:

```
📁 machine_gaze/
├── 🔧 core/                 # Core processing pipeline
│   ├── classifier_registry.py   # Classifier management
│   ├── video_processor.py       # Video processing
│   └── base_classifier.py       # Abstract base class
├── 🎯 classifiers/          # Detection modules
│   ├── yolo_world_detector.py   # Object detection
│   └── face_emotion_detector.py # Emotion recognition
├── 🎪 tracking/             # Tracking system
│   ├── byte_tracker.py          # ByteTracker implementation
│   └── track_smoother.py        # Temporal smoothing
├── 📊 output/               # Output formats
│   └── export_manager.py        # Multi-format export
└── 🛠️ utils/                # Utilities
    ├── config_loader.py         # Configuration loading
    └── config_manager.py        # Configuration management
```

### Key Design Principles

🔌 **Modular**: Easy to add new classifiers and features  
⚙️ **Configurable**: YAML-based configuration system  
🚀 **Performant**: Optimized for M1 Mac and GPU acceleration  
📊 **Observable**: Comprehensive logging and debugging  
🎯 **Extensible**: Plugin architecture for custom features  

---

## 🔧 Development

### Project Structure
```
machine-gaze/
├── src/machine_gaze/        # Main package
├── config/                  # Configuration files
├── videos/                  # Input videos
├── output/                  # Processed outputs
├── tests/                   # Unit tests
├── main.py                  # Basic CLI interface
├── advanced_main.py         # Advanced processing
├── batch_processor.py       # Batch processing
├── config_manager_cli.py    # Configuration management
└── requirements.txt         # Dependencies
```

### Adding New Classifiers

1. **Create classifier class:**
```python
from machine_gaze.core.base_classifier import BaseClassifier, Detection

class MyClassifier(BaseClassifier):
    def load_model(self):
        # Load your model
        pass
    
    def detect(self, frame):
        # Return List[Detection]
        pass
```

2. **Register in `classifiers/__init__.py`:**
```python
from .my_classifier import MyClassifier
get_registry().register('my_classifier', MyClassifier)
```

3. **Add to configuration:**
```yaml
classifiers:
  my_classifier:
    enabled: true
    confidence_threshold: 0.5
    # Your parameters here
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_video_processor.py

# With coverage
python -m pytest --cov=machine_gaze tests/
```

---

## 📊 Performance

### Benchmarks (M1 Mac Pro)

| Video Resolution | Objects | Tracking | FPS | Memory |
|------------------|---------|----------|-----|--------|
| 1080p | Basic | No | ~15 FPS | ~2GB |
| 1080p | + Emotions | No | ~12 FPS | ~2.5GB |
| 1080p | + Emotions | Yes | ~10 FPS | ~3GB |
| 4K | + Emotions | Yes | ~4 FPS | ~4GB |

### Optimization Tips

🚀 **For Speed:**
- Use `basic_detection` config
- Disable tracking: `tracking.enabled: false`
- Reduce confidence thresholds
- Process smaller videos

🎯 **For Accuracy:**
- Use `security_monitoring` config
- Increase confidence thresholds
- Enable temporal smoothing
- Use larger models

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/your-username/machine-gaze.git
cd machine-gaze

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests
python -m pytest
```

### Contribution Guidelines

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas for Contribution

- 🔧 **New Classifiers**: Gesture recognition, pose estimation, etc.
- 🎨 **Visualization**: Enhanced overlay styles and animations
- 🚀 **Performance**: Optimization and profiling
- 📚 **Documentation**: Examples, tutorials, API docs
- 🧪 **Testing**: Unit tests, integration tests, benchmarks

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Ultralytics](https://ultralytics.com/)** - YOLO-World implementation
- **[MediaPipe](https://mediapipe.dev/)** - Face detection framework
- **[ByteTrack](https://github.com/ifzhang/ByteTrack)** - Multi-object tracking algorithm
- **Apple** - M1/M2 MPS acceleration support

---

## 📞 Support

- 📖 **Documentation**: [User Guide](USER_GUIDE.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-username/machine-gaze/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-username/machine-gaze/discussions)
- 📧 **Email**: machine-gaze@example.com

---

## 🗺️ Roadmap

### Current Phase: Polish & Usability ✅
- ✅ Advanced CLI tools
- ✅ Configuration management
- ✅ Multiple output formats
- ✅ Comprehensive documentation

### Planned Features
- 🔮 **Real-time Processing**: Webcam and live video support
- 🎭 **Gesture Recognition**: Hand and body gesture detection
- 🏃 **Pose Estimation**: Human pose and activity recognition
- ☁️ **Cloud Integration**: Scalable cloud processing
- 📱 **Mobile Support**: iOS and Android deployment
- 🤖 **Custom Models**: Easy integration of custom trained models

---

**Made with ❤️ for the computer vision community**

*MACHINE GAZE - Making advanced computer vision accessible to everyone.*

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/your-username/machine-gaze?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/machine-gaze?style=social)
![GitHub issues](https://img.shields.io/github/issues/your-username/machine-gaze)
![GitHub last commit](https://img.shields.io/github/last-commit/your-username/machine-gaze)

**Transform your videos with AI-powered computer vision. Get started today! 🚀**
