#!/usr/bin/env python3
"""Scratch: render multiple grid modes from a single detection+tracking pass."""
import sys, time; sys.path.insert(0, "src")
import cv2, logging
logging.disable(logging.INFO)
from machine_gaze import get_registry, ConfigLoader
from machine_gaze.core.video_processor import VideoProcessor
from machine_gaze.core.grid_renderer import GridRenderer

INPUT = "videos/kiran.MOV"
MODES = ["pixel", "typographic", "ghost"]

config = ConfigLoader.load_config("config/grid.yaml")
reg = get_registry(); reg.setup_from_config(config)

# One processor for detection + tracking (grid disabled here).
vc = ConfigLoader.video_processor_config(config)
vc["grid"] = {"enabled": False}
vp = VideoProcessor(reg, vc)

cap = cv2.VideoCapture(INPUT)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(3)); h = int(cap.get(4))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"{INPUT}: {w}x{h} {total} frames @ {fps:.1f}fps", flush=True)

# One grid renderer + writer per mode (each gets its own state, e.g. ghost trail).
renderers, writers = {}, {}
for m in MODES:
    gc = dict(config["grid"]); gc["mode"] = m; gc["show_gridlines"] = True
    renderers[m] = GridRenderer(gc, vp)
    path = f"output/kiran_{m}.mp4"
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not out.isOpened():
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    writers[m] = out

t0 = time.time(); n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    dets = vp._enhance_detections(reg.process_frame(frame))
    if vp.enable_tracking and vp.tracker is not None:
        tracks = vp.tracker.update(dets)
        if vp.track_smoother is not None:
            tracks = vp.track_smoother.smooth_tracks(tracks)
        dets = vp._tracks_to_detections(tracks)
    for m in MODES:
        writers[m].write(renderers[m].render(frame, dets))
    n += 1
    if n % 50 == 0:
        rate = n / (time.time() - t0)
        print(f"  {n}/{total} frames ({rate:.1f} fps, ETA {(total-n)/rate:.0f}s)", flush=True)

cap.release()
for out in writers.values():
    out.release()
print(f"DONE {n} frames in {time.time()-t0:.0f}s -> " + ", ".join(f"output/kiran_{m}.mp4" for m in MODES), flush=True)
