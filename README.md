# Machine Gaze

An offline computer-vision pipeline that annotates pre-recorded video clips. It
runs open-vocabulary object detection (YOLO-World), optional face emotion
recognition, and ByteTrack-based tracking, then renders annotated video plus
optional JSON/CSV/HTML exports.

This is a research / art project. It is built for a local M1/M2 Mac with MPS
acceleration and has **no real-time constraints** — overnight batch processing
of short clips (typically 15–30s) is the intended workflow. See `GAMEPLAN.txt`
for the full design rationale and phase history.

## Install

```bash
# Python 3.9+  (M1/M2 Macs auto-use MPS)
pip install -r requirements.txt

# Verify dependencies and that the pipeline can initialize
python test_setup.py
```

YOLO-World weights (`yolov8s-world.pt`) are vendored in the repo root and also
auto-download on first run if missing.

## Usage

```bash
# Basic processing (uses config/default_config.yaml when --config is omitted)
python main.py input.mp4 output.mp4
python main.py input.mp4 output.mp4 --config config/tracking_config.yaml --overlay-only --debug

# Grid quantization — the artistic transform (see below)
python main.py input.mp4 output/grid.mp4 --config config/grid.yaml

# Advanced multi-format output (--config is REQUIRED here)
python advanced_main.py input.mp4 --config config/emotion_detection.yaml \
  --video-output processed.mp4 --data-export json,csv --analysis-report html

# Batch processing (subcommands: single | csv | directory | create-example)
python batch_processor.py directory input_videos/ output_videos/ config/tracking_config.yaml --workers 3
python batch_processor.py create-example     # writes an example batch CSV

# Configuration management (list | create | interactive | validate | add-classifier | export-docs)
python config_manager_cli.py list
python config_manager_cli.py create emotion_analysis my_config.yaml
```

CLI flags: `--overlay-only` forces detections to draw on a solid background
instead of the source frame; `--debug` forces `DEBUG` logging.

## Grid quantization

The grid renderer imposes a fixed lattice (default 24 rows × 36 columns) on the
frame and snaps every detection onto it — the machine's compulsion to
categorize and place, flattening an organic scene into a rigid grid. Enable it
with a `grid:` block in the config (see `config/grid.yaml`); when enabled it
takes over the final draw step. Tracking is recommended so subjects "click"
coherently between cells rather than jittering per frame.

| `mode`        | Effect |
|---------------|--------|
| `snap_box`    | Snap each box's edges to grid lines, keeping its rough size. |
| `snap_cell`   | Collapse each subject to the single cell at its center — the most abstract "everything is a token" view. |
| `pixel`       | Resample the real image content inside each detection into its snapped cell block — subjects move cell-to-cell like sprites on graph paper. |
| `typographic` | Replace each subject with its class label, set to fill its snapped cell. |
| `census`      | Heatmap of how many detections occupy each cell — the video as a data visualization of itself. |
| `ghost`       | Snapped boxes on a canvas that never fully clears; vacated cells decay slowly, accumulating the trace of everything seen. |

Key grid options: `rows`, `cols`, `mode`, `show_gridlines`, `gridline_color`,
`background_color`, `min_cells`, `ghost_decay`.

## Configuration

Configuration is the primary extension mechanism — prefer adjusting YAML over
hardcoding. Most tunables (thresholds, rendering style, tracking params, overlay
mode, grid) are read with `.get(key, default)`, so unknown keys are ignored. The
config loader now **warns** on unrecognized keys in known sections to catch
typos (a misspelled key silently disables a feature otherwise).

Ready-made configs in `config/`:

| File | Purpose |
|------|---------|
| `default_config.yaml` | Standard object detection. |
| `tracking_config.yaml` | ByteTrack tracking + temporal smoothing + trails. |
| `emotion_detection.yaml` | Enables the `face_emotion` classifier. |
| `extended_objects.yaml` | Wider open-vocabulary class list. |
| `white_background.yaml` | Overlay-only on a white background. |
| `grid.yaml` | Grid quantization (see above). |

- **Detection classes** are open-vocabulary strings under `classifiers.yolo_world.classes` —
  change what's detected by editing that list, no retraining.
- **Device** defaults to `auto` (MPS → CUDA → CPU); override with `device:` in the `yolo_world` config.
- **`face_emotion` is disabled by default**; enable it via `emotion_detection.yaml` or `enabled: true`.
- **Output codec** defaults to `avc1` (H.264) under the top-level `output:` block,
  and falls back to `mp4v` automatically if the local OpenCV build can't open an H.264 writer.

## Architecture

The package lives in `src/machine_gaze/`. Each entrypoint does
`sys.path.insert(0, "src")`, so the package does not need to be pip-installed.
Data flows one direction, with `Detection` as the universal currency between
stages:

1. **`ConfigLoader`** (`utils/config_loader.py`) loads YAML into a dict, validates known keys (warn-only), and merges the top-level `output:`/`grid:` blocks into the video-processor config.
2. **`ClassifierRegistry`** (`core/classifier_registry.py`) — a global singleton. Classifiers register their *type* at import time; `setup_from_config()` instantiates the ones with `enabled: true`. `process_frame()` fans a frame to every active classifier and concatenates their `Detection` lists (one failing classifier doesn't abort the frame).
3. **`VideoProcessor`** (`core/video_processor.py`) orchestrates per frame: `process_frame()` → `_enhance_detections()` (face↔person association + NMS) → optional tracking → final render. When the grid is enabled, `GridRenderer` (`core/grid_renderer.py`) performs the final draw instead of the plain box renderer.
4. **`ExportManager`** (`output/export_manager.py`) — JSON/CSV/HTML/image-sequence exports, used by `advanced_main.py`.

Key contracts:
- **`Detection`** (`core/base_classifier.py`) — `bbox=(x1,y1,x2,y2)` integer pixel corners, `class_name`, `confidence`, optional `track_id`, `metadata`. OpenCV colors are BGR tuples throughout.
- **`BaseClassifier`** — abstract `load_model()` + `detect(frame) -> List[Detection]`.

### Adding a classifier

1. Subclass `BaseClassifier` in `src/machine_gaze/classifiers/`, implementing `load_model()` and `detect()`.
2. Register it in `classifiers/__init__.py` via `register_classifier('your_name', YourClass)` (this file is imported by the package `__init__`, which triggers registration).
3. Add a `classifiers.your_name:` block with `enabled: true` to the YAML config — the sub-dict is passed verbatim as the classifier's `config`.

Built-in classifiers: `yolo_world` (`YOLOWorldDetector`) and `face_emotion` (`FaceEmotionDetector`).

## Notes

- There is no formal test suite. `tests/` is empty; `test_setup.py` is an import/smoke check, not pytest tests. `black`/`flake8`/`mypy` are listed in `requirements.txt` but no project config exists for them.
- `face_emotion` uses MediaPipe and is tuned with low thresholds by default for permissive detection — raise `face_confidence`/`emotion_confidence` if you get false positives.
- `output/`, `advanced_output/`, `batch_output/` hold prior run artifacts and are git-ignored.
