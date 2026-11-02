"""Work area from the ArUco board, with no ground truth (replaces the oracle object box).

The object stands inside the ring of markers. Each detected marker gives its distance (known side
length and calibration), hence the expected image radius of the target's bounding sphere. The area
is the bounding box of the marker centres, grown by that radius. Markers are painted over with the
surrounding paper colour so they do not attract matches.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DICT = cv2.aruco.DICT_4X4_50
PAINT_GROW = 1.3   # painted quad, in marker sizes (covers the black border and blur)
RING_GROW = 1.8    # outer edge of the ring the paper colour is sampled from


@dataclass
class Area:
    box: tuple[float, float, float, float]  # full-resolution pixels
    radius_px: float                         # expected image radius of the bounding sphere
    image: np.ndarray                        # BGR, markers painted over
    n_markers: int


def _grow(q: np.ndarray, k: float) -> np.ndarray:
    m = q.mean(0)
    return np.round(m + k * (q - m)).astype(np.int32)


def find_area(image_bgr: np.ndarray, K: np.ndarray, dist: np.ndarray, marker_m: float,
              radius_m: float) -> Area | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return None
    quads = [c.reshape(4, 2).astype(np.float64) for c in corners]
    h = marker_m / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    ranges = []
    for q in quads:
        ok, _, tvec = cv2.solvePnP(obj, q, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            ranges.append(float(np.linalg.norm(tvec)))
    if not ranges:
        return None
    radius_px = K[0, 0] * radius_m / float(np.median(ranges))

    H, W = gray.shape
    c = np.array([q.mean(0) for q in quads])
    x0, y0 = np.maximum(c.min(0) - radius_px, 0)
    x1, y1 = np.minimum(c.max(0) + radius_px, [W, H])

    painted = image_bgr.copy()
    for q in quads:
        outer = _grow(q, RING_GROW)
        bx0, by0 = np.maximum(outer.min(0), 0)
        bx1, by1 = np.minimum(outer.max(0) + 1, [W, H])
        off = np.array([bx0, by0], np.int32)
        inner = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
        cv2.fillConvexPoly(inner, _grow(q, PAINT_GROW) - off, 1)
        ring = np.zeros_like(inner)
        cv2.fillConvexPoly(ring, outer - off, 1)
        inner, ring = inner.astype(bool), ring.astype(bool) & ~inner.astype(bool)
        src = image_bgr[by0:by1, bx0:bx1]
        painted[by0:by1, bx0:bx1][inner] = np.median(src[ring], axis=0) if ring.any() else 255
    return Area((float(x0), float(y0), float(x1), float(y1)), radius_px, painted, len(quads))
