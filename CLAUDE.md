# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Machine Gaze is an offline computer-vision pipeline that annotates pre-recorded video clips (research/art project, typically 15–30s clips). It runs object detection (YOLO-World), optional face emotion recognition, and ByteTrack-based tracking, then renders annotated video plus optional JSON/CSV/HTML exports. Designed for local M1 Mac with MPS acceleration; there are no real-time constraints (overnight batch processing is acceptable). See `GAMEPLAN.txt` for the full design rationale and phase history.

## Commands

```bash
# Install (Python 3.9+; M1/M2 Mac auto-uses MPS)
pip install -r requirements.txt

# Verify dependencies and pipeline can initialize
python test_setup.py

# Basic processing (uses config/default_config.yaml when --config omitted)
python main.py input.mp4 output.mp4
python main.py input.mp4 output.mp4 --config config/tracking_config.yaml --overlay-only --debug

# Advanced multi-format output (--config is REQUIRED here)
python advanced_main.py input.mp4 --config config/emotion_detection.yaml \
  --video-output processed.mp4 --data-export json,csv --analysis-report html

# Batch processing (subcommands: single | csv | directory | create-example)
python batch_processor.py directory input_videos/ output_videos/ config/tracking_config.yaml --workers 3
python batch_processor.py csv example_batch.csv --workers 3
python batch_processor.py create-example     # writes an example batch CSV

# Configuration management (subcommands: list | create | interactive | validate | add-classifier | export-docs)
python config_manager_cli.py list
python config_manager_cli.py create emotion_analysis my_config.yaml
```

There is no formal test suite — `tests/` is empty and `test_setup.py` is an import/smoke check, not pytest tests. `black`, `flake8`, and `mypy` are listed in `requirements.txt` but no project config exists for them.

## Architecture

The package lives in `src/machine_gaze/`. Each entrypoint script (`main.py`, `advanced_main.py`, `batch_processor.py`) does `sys.path.insert(0, "src")` before importing, so the package does not need to be pip-installed.

Pipeline data flows in one direction, with `Detection` as the universal currency between stages:

1. **`ConfigLoader`** (`utils/config_loader.py`) loads a YAML config into a plain dict. There is no schema validation here — config keys map directly onto kwargs throughout the pipeline, so a typo silently disables a feature rather than erroring.

2. **`ClassifierRegistry`** (`core/classifier_registry.py`) is a global singleton (`get_registry()`). Classifiers register their *type* at import time, then `setup_from_config(config)` instantiates and activates the ones with `enabled: true` under the `classifiers:` config key. `process_frame()` fans a frame out to every active classifier and concatenates their `Detection` lists. Classifier errors are caught and logged per-classifier — one failing classifier does not abort the frame.

3. **`VideoProcessor`** (`core/video_processor.py`) is the orchestrator. Per frame it calls `registry.process_frame()` → `_enhance_detections()` (face↔person association + NMS, both in `utils/detection_utils.py`) → optional tracking → `render_detections()`. When tracking is enabled, `ByteTracker.update()` then `TrackSmoother.smooth_tracks()` run, and the resulting track states are converted *back* into `Detection` objects via `_tracks_to_detections()` so the renderer only ever deals with `Detection`. `overlay_only` mode draws onto a solid background instead of the source frame.

4. **`ExportManager`** (`output/export_manager.py`) is used only by `advanced_main.py` to write JSON/CSV/HTML/image-sequence outputs.

Key contracts:
- **`Detection`** (`core/base_classifier.py`) — `bbox=(x1,y1,x2,y2)`, `class_name`, `confidence`, optional `track_id`, `metadata` dict. Bbox is integer pixel corners; OpenCV colors are BGR tuples throughout.
- **`BaseClassifier`** (`core/base_classifier.py`) — abstract `load_model()` + `detect(frame) -> List[Detection]`. `load_model()` is called once at instantiation when the classifier is enabled.

### Adding a classifier

1. Subclass `BaseClassifier` in `src/machine_gaze/classifiers/`, implementing `load_model()` and `detect()`.
2. Register it in `classifiers/__init__.py` via `register_classifier('your_name', YourClass)` — this file is imported by the package `__init__`, which is what makes registration happen automatically.
3. Add a `classifiers.your_name:` block (with `enabled: true`) to the YAML config. The config sub-dict is passed verbatim as the classifier's `config` arg.

The two built-in classifiers are `yolo_world` (`YOLOWorldDetector`) and `face_emotion` (`FaceEmotionDetector`).

## Models, devices, and configs

- **YOLO-World weights** auto-download on first run (e.g. `yolov8s-world.pt`, already vendored in repo root). `model_size` (`s`/`m`/`l`/`x`) selects the variant. Target detection classes are open-vocabulary strings set via `model.set_classes()` from the config's `classes:` list — change what's detected by editing that list, no retraining.
- **Device** defaults to `auto`: MPS → CUDA → CPU. Override with `device: cpu|mps|cuda` in the `yolo_world` config.
- **`face_emotion` is disabled by default** in `config/default_config.yaml`; enable it via a config that sets `enabled: true` (e.g. `config/emotion_detection.yaml`).
- Ready-made configs live in `config/`: `default_config.yaml`, `tracking_config.yaml`, `emotion_detection.yaml`, `extended_objects.yaml`, `white_background.yaml`, etc. `config_manager.py` (`utils/`) also generates configs from named templates.

## Conventions

- Configuration is the primary extension mechanism — prefer adding/adjusting YAML over hardcoding. Most tunables (thresholds, rendering style, tracking params, overlay mode) are config-driven and read with `.get(key, default)`, so unknown keys are ignored silently.
- The `--overlay-only` CLI flag overrides `video_processor.overlay_only`; `--debug` forces `logging.level` to `DEBUG`.
- Output directories are created automatically (`output/`, `advanced_output/`, `batch_output/` contain prior run artifacts and are not source).
