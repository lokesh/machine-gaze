"""
Grid rendering modes for Machine Gaze.

This module imposes a fixed lattice (default 24 rows x 36 columns) on the
frame and quantizes every Detection onto it. It is the project's primary
*artistic* transform: the machine's compulsion to categorize and place,
flattening an organic scene into a rigid grid.

A ``GridRenderer`` is constructed by ``VideoProcessor`` when a ``grid:`` block
with ``enabled: true`` is present in the config. The processor delegates the
whole frame to ``GridRenderer.render()`` instead of its normal
``render_detections()``. The renderer holds a back-reference to the processor
so it can reuse the existing per-class / per-track colors and box drawing.

Modes (``grid.mode``):
- ``snap_box``     -- snap each bbox's edges to the nearest grid lines, keep
                      roughly the real size, then draw boxes as usual.
- ``snap_cell``    -- collapse each detection to the single cell containing its
                      center. The most abstract "everything is a token" view.
- ``pixel``        -- resample the actual image content inside each detection
                      into its snapped cell block, so subjects click cell to
                      cell like sprites on graph paper.
- ``typographic``  -- replace each subject with its class label, set to fill
                      its snapped cell block (concrete-poetry of "person/car").
- ``census``       -- a heatmap of how many detections occupy each cell; the
                      video becomes a data visualization of itself.
- ``ghost``        -- like ``snap_box`` but the canvas is never cleared; vacated
                      cells decay slowly, accumulating the trace of everything
                      the machine has seen in the clip.

All modes optionally draw the grid lines themselves (``show_gridlines``).
"""

import cv2
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .classifier_registry import Detection

logger = logging.getLogger(__name__)

VALID_MODES = ("snap_box", "snap_cell", "pixel", "typographic", "census", "ghost")


class GridRenderer:
    """Quantizes detections onto a fixed grid and renders them."""

    def __init__(self, config: Dict[str, Any], processor: "VideoProcessor"):  # noqa: F821
        self.processor = processor
        self.config = config or {}

        self.enabled = self.config.get("enabled", False)
        self.rows = max(1, int(self.config.get("rows", 24)))
        self.cols = max(1, int(self.config.get("cols", 36)))

        self.mode = self.config.get("mode", "snap_box")
        if self.mode not in VALID_MODES:
            logger.warning(
                "Unknown grid.mode '%s'; falling back to 'snap_box'. Valid modes: %s",
                self.mode, ", ".join(VALID_MODES),
            )
            self.mode = "snap_box"

        # Grid line overlay
        self.show_gridlines = self.config.get("show_gridlines", False)
        self.gridline_color = tuple(self.config.get("gridline_color", (40, 40, 40)))
        self.gridline_thickness = int(self.config.get("gridline_thickness", 1))

        # Background for canvas-replacing modes (pixel/typographic/census/ghost).
        # Falls back to the processor's overlay background.
        bg = self.config.get("background_color", None)
        self.background_color = (
            tuple(bg) if bg is not None else tuple(processor.background_color)
        )

        # Minimum size, in cells, that a snapped box may collapse to.
        self.min_cells = max(1, int(self.config.get("min_cells", 1)))

        # ghost mode: per-frame decay of the persistent layer (0..1).
        self.ghost_decay = float(self.config.get("ghost_decay", 0.92))
        self._ghost: Optional[np.ndarray] = None

        # census mode colormap (OpenCV COLORMAP_* constant).
        self.census_colormap = int(
            self.config.get("census_colormap", cv2.COLORMAP_INFERNO)
        )

        if self.enabled:
            logger.info(
                "Grid rendering enabled: %dx%d cells, mode=%s",
                self.cols, self.rows, self.mode,
            )

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def _cell_size(self, w: int, h: int) -> Tuple[float, float]:
        return w / self.cols, h / self.rows

    def _snap_box(self, bbox, cw: float, ch: float, w: int, h: int) -> Tuple[int, int, int, int]:
        """Snap bbox edges to the nearest grid lines, enforcing a minimum size."""
        x1, y1, x2, y2 = bbox
        gx1 = round(x1 / cw)
        gy1 = round(y1 / ch)
        gx2 = round(x2 / cw)
        gy2 = round(y2 / ch)

        # Enforce minimum span (in cells) without drifting off-frame.
        if gx2 - gx1 < self.min_cells:
            gx2 = gx1 + self.min_cells
        if gy2 - gy1 < self.min_cells:
            gy2 = gy1 + self.min_cells

        # Clamp to grid bounds, shifting the box in if it overran.
        gx1 = max(0, min(gx1, self.cols - self.min_cells))
        gy1 = max(0, min(gy1, self.rows - self.min_cells))
        gx2 = max(gx1 + self.min_cells, min(gx2, self.cols))
        gy2 = max(gy1 + self.min_cells, min(gy2, self.rows))

        return (
            int(round(gx1 * cw)), int(round(gy1 * ch)),
            int(round(gx2 * cw)), int(round(gy2 * ch)),
        )

    def _snap_cell(self, bbox, cw: float, ch: float) -> Tuple[int, int, int, int]:
        """Collapse a detection to the single cell containing its center."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        col = max(0, min(int(cx // cw), self.cols - 1))
        row = max(0, min(int(cy // ch), self.rows - 1))
        return (
            int(round(col * cw)), int(round(row * ch)),
            int(round((col + 1) * cw)), int(round((row + 1) * ch)),
        )

    @staticmethod
    def _clamp_bbox(bbox, w: int, h: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(x1 + 1, min(int(x2), w))
        y2 = max(y1 + 1, min(int(y2), h))
        return x1, y1, x2, y2

    def _snapped_detection(self, det: Detection, cw, ch, w, h, collapse: bool) -> Detection:
        bbox = self._snap_cell(det.bbox, cw, ch) if collapse else self._snap_box(det.bbox, cw, ch, w, h)
        return Detection(
            bbox=bbox,
            class_name=det.class_name,
            confidence=det.confidence,
            track_id=det.track_id,
            metadata=det.metadata,
        )

    # ------------------------------------------------------------------ #
    # Canvas helpers
    # ------------------------------------------------------------------ #
    def _solid_canvas(self, frame: np.ndarray) -> np.ndarray:
        return np.full_like(frame, self.background_color, dtype=np.uint8)

    def _box_base(self, frame: np.ndarray) -> np.ndarray:
        """Base canvas for box-drawing modes: source frame, or background in overlay mode."""
        if self.processor.overlay_only:
            return self._solid_canvas(frame)
        return frame.copy()

    def _draw_gridlines(self, frame: np.ndarray, cw: float, ch: float, w: int, h: int) -> None:
        if not self.show_gridlines:
            return
        for c in range(1, self.cols):
            x = int(round(c * cw))
            cv2.line(frame, (x, 0), (x, h), self.gridline_color, self.gridline_thickness)
        for r in range(1, self.rows):
            y = int(round(r * ch))
            cv2.line(frame, (0, y), (w, y), self.gridline_color, self.gridline_thickness)

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def render(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        h, w = frame.shape[:2]
        cw, ch = self._cell_size(w, h)

        if self.mode == "snap_box":
            out = self._render_boxes(frame, detections, cw, ch, w, h, collapse=False)
        elif self.mode == "snap_cell":
            out = self._render_boxes(frame, detections, cw, ch, w, h, collapse=True)
        elif self.mode == "pixel":
            out = self._render_pixel(frame, detections, cw, ch, w, h)
        elif self.mode == "typographic":
            out = self._render_typographic(frame, detections, cw, ch, w, h)
        elif self.mode == "census":
            out = self._render_census(frame, detections, cw, ch, w, h)
        elif self.mode == "ghost":
            out = self._render_ghost(frame, detections, cw, ch, w, h)
        else:  # pragma: no cover - guarded in __init__
            out = self._render_boxes(frame, detections, cw, ch, w, h, collapse=False)

        return out

    # ------------------------------------------------------------------ #
    # Modes
    # ------------------------------------------------------------------ #
    def _render_boxes(self, frame, detections, cw, ch, w, h, collapse: bool) -> np.ndarray:
        out = self._box_base(frame)
        for det in detections:
            snapped = self._snapped_detection(det, cw, ch, w, h, collapse)
            self.processor._draw_detection(out, snapped)
        self._draw_gridlines(out, cw, ch, w, h)
        return out

    def _render_pixel(self, frame, detections, cw, ch, w, h) -> np.ndarray:
        out = self._solid_canvas(frame)
        # Draw larger subjects first so smaller ones land on top instead of being buried.
        ordered = sorted(
            detections,
            key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
            reverse=True,
        )
        for det in ordered:
            sx1, sy1, sx2, sy2 = self._clamp_bbox(det.bbox, w, h)
            patch = frame[sy1:sy2, sx1:sx2]
            if patch.size == 0:
                continue
            dx1, dy1, dx2, dy2 = self._snap_box(det.bbox, cw, ch, w, h)
            dw, dh = dx2 - dx1, dy2 - dy1
            if dw <= 0 or dh <= 0:
                continue
            out[dy1:dy2, dx1:dx2] = cv2.resize(patch, (dw, dh), interpolation=cv2.INTER_AREA)
        self._draw_gridlines(out, cw, ch, w, h)
        return out

    def _render_typographic(self, frame, detections, cw, ch, w, h) -> np.ndarray:
        out = self._solid_canvas(frame)
        for det in detections:
            x1, y1, x2, y2 = self._snap_box(det.bbox, cw, ch, w, h)
            color = self.processor._get_color_for_class(det.class_name)
            label = det.class_name
            scale, thickness, (tw, th) = self._fit_text(label, x2 - x1, y2 - y1)
            tx = x1 + max(0, ((x2 - x1) - tw) // 2)
            ty = y1 + ((y2 - y1) + th) // 2
            cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thickness, cv2.LINE_AA)
        self._draw_gridlines(out, cw, ch, w, h)
        return out

    def _render_census(self, frame, detections, cw, ch, w, h) -> np.ndarray:
        counts = np.zeros((self.rows, self.cols), dtype=np.float32)
        for det in detections:
            x1, y1, x2, y2 = self._snap_box(det.bbox, cw, ch, w, h)
            c1 = max(0, min(int(round(x1 / cw)), self.cols - 1))
            c2 = max(c1 + 1, min(int(round(x2 / cw)), self.cols))
            r1 = max(0, min(int(round(y1 / ch)), self.rows - 1))
            r2 = max(r1 + 1, min(int(round(y2 / ch)), self.rows))
            counts[r1:r2, c1:c2] += 1.0

        peak = counts.max()
        if peak > 0:
            norm = (counts / peak * 255.0).astype(np.uint8)
        else:
            norm = counts.astype(np.uint8)
        # Upscale blockily so each cell is a flat tile, then colorize.
        big = cv2.resize(norm, (w, h), interpolation=cv2.INTER_NEAREST)
        out = cv2.applyColorMap(big, self.census_colormap)
        self._draw_gridlines(out, cw, ch, w, h)
        return out

    def _render_ghost(self, frame, detections, cw, ch, w, h) -> np.ndarray:
        # Current frame's snapped boxes drawn on pure black, regardless of overlay setting.
        current = np.zeros((h, w, 3), dtype=np.uint8)
        for det in detections:
            snapped = self._snapped_detection(det, cw, ch, w, h, collapse=False)
            self.processor._draw_detection(current, snapped)

        if self._ghost is None or self._ghost.shape != current.shape:
            self._ghost = np.zeros((h, w, 3), dtype=np.float32)

        # Decay the accumulated trace, then let the current frame light cells back up.
        self._ghost *= self.ghost_decay
        self._ghost = np.maximum(self._ghost, current.astype(np.float32))

        out = self._ghost.astype(np.uint8)
        self._draw_gridlines(out, cw, ch, w, h)
        return out

    # ------------------------------------------------------------------ #
    # Text fitting
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fit_text(text: str, box_w: int, box_h: int) -> Tuple[float, int, Tuple[int, int]]:
        """Pick a font scale/thickness so ``text`` fits inside the box with padding."""
        if not text:
            return 0.4, 1, (0, 0)
        target_w = max(1, int(box_w * 0.85))
        target_h = max(1, int(box_h * 0.6))
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.3
        best_scale = scale
        best_size = (0, 0)
        # Grow the scale until the text would overflow either dimension.
        while scale < 12.0:
            thickness = max(1, int(scale * 1.5))
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            if tw > target_w or th > target_h:
                break
            best_scale, best_size = scale, (tw, th)
            scale += 0.1
        thickness = max(1, int(best_scale * 1.5))
        return best_scale, thickness, best_size
