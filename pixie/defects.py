"""Assembly check: which parts of the target does the photo confirm, at the estimated pose?

Inputs are the target model, the image and the pose estimate, nothing else. The target is
rendered at the estimated pose in an undistorted crop; its geometric edges (silhouette, depth
steps, creases) are owned by the part on their near side. An edge pixel is confirmed when the
photo has a gradient of the same orientation within `tol_px`. A missing brick leaves its edges
unconfirmed (what shows instead sits a brick height away); a brick added in front hides them.

Part support = mean confirmation of its edge pixels. Defect evidence is pooled over parts within
`region_mm` of each other, since a wrong brick spans several parts; the image score is the worst
region. Over several views of one scene, support is pooled per part before the same steps.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .parts import Parts
from .render import Rasterizer

S = 512
CROP_MARGIN = 1.2
DEPTH_STEP = 0.002    # m, depth jump that makes an occluding edge
CREASE_COS = np.cos(np.radians(30))
N_BINS = 8            # edge orientation bins over 180 deg


@dataclass
class Params:
    """Tuned on synthetic defects only (scripts/tune_defects.py), then frozen; the defaults are the
    frozen values (docs/method.md)."""
    tol_px: int = 2          # search radius for a confirming photo edge
    norm_q: float = 50.0     # percentile of edge responses that counts as full confirmation
    min_edge: int = 20       # edge pixels for a part to be scored
    region_mm: float = 0.0   # pooling radius (0 = each part on its own)

    @staticmethod
    def load(path: Path) -> "Params":
        return Params(**json.loads(Path(path).read_text()))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))


@dataclass
class Evidence:
    n_edge: np.ndarray     # (P,) edge pixels owned by each part (0 = not visible)
    support: np.ndarray    # (P,) sum of confirmations; support / n_edge is in [0, 1]
    edges: np.ndarray      # S x S part id of the edge pixel, -1 elsewhere (for figures)
    conf: np.ndarray       # S x S confirmation of the edge pixel


# --- views -------------------------------------------------------------------------------------

def view_box(vertices: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray):
    p = vertices @ R.T + t
    uv = p[:, :2] / p[:, 2:] * K[[0, 1], [0, 1]] + K[:2, 2]
    return (*uv.min(0), *uv.max(0))


def crop_intrinsics(K: np.ndarray, box) -> np.ndarray:
    """Pinhole intrinsics of the S x S undistorted crop around `box` (undistorted pixels)."""
    x0, y0, x1, y1 = box
    L = CROP_MARGIN * max(x1 - x0, y1 - y0)
    s, ox, oy = S / L, (x0 + x1) / 2 - L / 2, (y0 + y1) / 2 - L / 2
    return np.array([[s * K[0, 0], 0, s * (K[0, 2] - ox)], [0, s * K[1, 1], s * (K[1, 2] - oy)], [0, 0, 1]])


def undistorted_crop(image_bgr: np.ndarray, K: np.ndarray, dist: np.ndarray, Kc: np.ndarray) -> np.ndarray:
    """S x S RGB view through the crop camera Kc (no distortion), resampled from the photo."""
    u, v = np.meshgrid(np.arange(S, dtype=np.float64), np.arange(S, dtype=np.float64))
    rays = np.stack([(u - Kc[0, 2]) / Kc[0, 0], (v - Kc[1, 2]) / Kc[1, 1], np.ones_like(u)], -1)
    src, _ = cv2.projectPoints(rays.reshape(-1, 3), np.zeros(3), np.zeros(3), K, dist)
    src = src.reshape(S, S, 2)
    scale = Kc[0, 0] / K[0, 0]                         # crop px per full-resolution px
    sigma = max(0.5 / scale - 0.5, 0.0)
    pad = int(3 * sigma) + 2
    H, W = image_bgr.shape[:2]
    x0, y0 = np.maximum(np.floor(src.reshape(-1, 2).min(0)).astype(int) - pad, 0)
    x1, y1 = np.minimum(np.ceil(src.reshape(-1, 2).max(0)).astype(int) + pad, [W - 1, H - 1])
    roi = image_bgr[y0:y1 + 1, x0:x1 + 1]
    if sigma > 0:
        roi = cv2.GaussianBlur(roi, (0, 0), sigma)
    m = (src - [x0, y0]).astype(np.float32)
    out = cv2.remap(roi, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# --- edges -------------------------------------------------------------------------------------

def _orientation(stack: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Gradient magnitude and orientation (mod pi) of a multi-channel image, structure tensor."""
    J = np.zeros((3,) + stack.shape[:2], np.float32)
    for c in range(stack.shape[2]):
        gx = cv2.Sobel(stack[..., c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(stack[..., c], cv2.CV_32F, 0, 1, ksize=3)
        J += np.stack([gx * gx, gx * gy, gy * gy])
    J = np.stack([cv2.GaussianBlur(j, (0, 0), sigma) for j in J])
    tr, det = J[0] + J[2], J[0] * J[2] - J[1] ** 2
    lam = tr / 2 + np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    theta = (0.5 * np.arctan2(2 * J[1], J[0] - J[2])) % np.pi
    return np.sqrt(lam), theta


def render_edges(ras: Rasterizer, parts: Parts, R, t, Kc) -> tuple[np.ndarray, np.ndarray]:
    """Part id on each geometric edge pixel of the target render (-1 elsewhere) and the edge
    normal orientation (mod pi)."""
    r = ras.render(R, t, Kc, S, S)
    m, d, n = r.mask, r.depth, r.normal
    edge = torch.zeros_like(m)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        mq = torch.roll(m, (dy, dx), (0, 1))
        dq = torch.roll(d, (dy, dx), (0, 1))
        nq = torch.roll(n, (dy, dx), (0, 1))
        e = ~mq | (dq - d > DEPTH_STEP) | ((n * nq).sum(-1) < CREASE_COS)
        edge |= m & e
    edge[[0, -1], :] = False
    edge[:, [0, -1]] = False
    fp = torch.as_tensor(parts.face_part, device=r.face.device)
    owner = torch.where(edge, fp[r.face.clamp_min(0)], torch.full_like(r.face, -1)).cpu().numpy()
    blurred = cv2.GaussianBlur((owner >= 0).astype(np.float32), (0, 0), 1.5)
    _, theta = _orientation(blurred[..., None], 2.0)
    return owner, theta


def evidence(photo_rgb: np.ndarray, owner: np.ndarray, theta_r: np.ndarray, n_parts: int,
             p: Params) -> Evidence:
    img = cv2.GaussianBlur(photo_rgb.astype(np.float32) / 255, (0, 0), 1.0)
    mag, theta_i = _orientation(img, 1.0)
    bins = (theta_i / np.pi * N_BINS).astype(int) % N_BINS
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.tol_px + 1,) * 2)
    on = owner >= 0
    rbin = (theta_r[on] / np.pi * N_BINS).astype(int) % N_BINS
    val = np.zeros(on.sum(), np.float32)
    for b in range(N_BINS):
        # photo gradients within one bin of b, spread over tol_px
        near = (bins == b) | (bins == (b + 1) % N_BINS) | (bins == (b - 1) % N_BINS)
        D = cv2.dilate(np.where(near, mag, 0).astype(np.float32), kernel)
        sel = rbin == b
        val[sel] = D[on][sel]
    norm = max(float(np.percentile(val, p.norm_q)), 1e-6) if len(val) else 1.0
    c = np.clip(val / norm, 0, 1)
    ids = owner[on]
    conf = np.zeros(owner.shape, np.float32)
    conf[on] = c
    return Evidence(np.bincount(ids, minlength=n_parts).astype(float),
                    np.bincount(ids, weights=c, minlength=n_parts), owner, conf)


# --- scores ------------------------------------------------------------------------------------

def neighbours(parts: Parts, region_mm: float) -> np.ndarray:
    d = np.linalg.norm(parts.centroid[:, None] - parts.centroid[None], axis=-1)
    return d <= region_mm / 1000


def part_scores(n_edge: np.ndarray, support: np.ndarray, nb: np.ndarray, p: Params) -> np.ndarray:
    """Defect score per part, pooled over its region; nan where the region is not seen."""
    miss = n_edge - support
    num, den = nb @ miss, nb @ n_edge
    out = np.where(den >= p.min_edge, num / np.maximum(den, 1e-9), np.nan)
    out[n_edge < p.min_edge] = np.nan
    return out


def image_score(scores: np.ndarray) -> float:
    return float(np.nanmax(scores)) if np.isfinite(scores).any() else float("nan")
