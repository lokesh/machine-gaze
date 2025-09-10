Use an open-vocabulary detector for most objects, plus two small add-ons:

Detector (one pass): OWLv2 / GroundingDINO / YOLO-World with your custom phrases → boxes for car, truck, tree, street light, walker, stroller….

Emotion (“sad face”): tiny face detector + emotion classifier on just the face crops (fast, accurate offline).

“Street” (region, not an object): semantic segmentation to label the road/street area in each frame.

That gives you high-detail labels without real-time constraints and runs well on an M1 Mac; the same pipeline scales cleanly to cloud GPUs if you want speed.

Workflow (accuracy-first, offline)
0) Define your label set

Create a phrase list with synonyms for recall:

car, pickup truck, box truck, semi truck

tree

street light, lamppost

walker, rollator, mobility walker, walking frame (and optionally “pedestrian” if you meant a person walking)

stroller, wheelchair (often co-occurs with walker scenes)

traffic light, stoplight (if that’s what “light” meant)

(Keep “sad face” out of this list; it’s handled by the emotion head below.)
Also add negative prompts later in post-processing (e.g., filter out “traffic sign” when you only want lights).

1) Open-vocabulary detection (boxes + text labels)

Pick one:

OWLv2 (highest recall, heavier),

GroundingDINO (excellent phrase grounding, robust),

YOLO-World (fast, YOLO-style ergonomics).

Run per frame with your phrase list → get boxes labeled with the matched phrase. Controls that matter:

score_threshold per phrase (you can make “walker” stricter than “car”).

nms_iou (0.4–0.6 typical).

img_size (↑ for small objects; 1280–1536 is a sweet spot offline).

Temporal smoothing: maintain a short history (e.g., last 5–9 frames). Only display a label if it’s seen ≥2/5 frames; decay slowly to avoid flicker.

2) “Sad face” (face crops only)

Detect faces inside person boxes (or across the full frame if people aren’t requested).

Run an emotion classifier (7–8 classes; return the top label+score).

Only attach “sad face” when score ≥ threshold and the face is ≥ a minimum size (prevents false positives on tiny faces).

Smooth per track ID (majority vote over a sliding window).

3) “Street” (region, not a box)

Use a semantic segmenter (e.g., DeepLabv3+, UPerNet, or any modern scene parser) to get road/street masks.

Post-process: fill small holes, keep the largest connected component, and render as a single polygon/outline or a translucent region (since a “street” box isn’t meaningful).

4) Tracking (optional but recommended)

Track detections with ByteTrack / BoT-SORT to get stable IDs per person/vehicle.

Color boxes by track ID; accumulate motion trails (nice visually).

5) Rendering

Overlay cut: draw boxes + labels on the original frames.

Boxes-only cut: blank or white canvas; draw the same boxes/labels (and street polygon) with motion trails.

Optional: attach original audio via ffmpeg mux.

Local on M1 Mac (no real-time needed)

Environment

python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu     # MPS comes with macOS PyTorch build
pip install opencv-python numpy
# Choose one detector stack:
# GroundingDINO:
pip install groundingdino transformers timm onnxruntime
# or OWLv2 (via transformers) or YOLO-World (via ultralytics ecosystem)
pip install ultralytics


Notes

Set device="mps" in PyTorch for Apple GPU; some ops may fall back to CPU—fine for offline.

Use higher input sizes (1280–1536) and even test-time augmentation (flip) if you want extra recall.

For emotion, tiny CNNs on face crops are very fast; you can run them on CPU.