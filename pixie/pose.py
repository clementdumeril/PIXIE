"""PIXIE-style pose from one RGB image and the untextured target model.

1. crop the query around the object (the paper assumes the detection is given),
2. match it with SuperPoint + LightGlue (MINIMA weights) against every reference render,
3. lift the reference keypoints to 3D with the render's point map,
4. RANSAC PnP on the best-matching references (1000 iterations, 5 px),
5. view re-selection: also try the reference closest to the estimated pose, keep the better.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .references import S, Reference

PNP_ITERS = 1000
PNP_PX = 5.0          # RANSAC threshold, in crop pixels (crop resized to S)
MIN_INLIER_RATIO = 0.5
TRY_TOP = 5           # references tried, by number of matches, before falling back
CROP_MARGIN = 1.2


@dataclass
class Crop:
    image: np.ndarray   # S x S RGB
    origin: np.ndarray  # full-resolution pixel of crop pixel (0, 0)
    scale: float        # crop px per full-resolution px

    def to_full(self, uv: np.ndarray) -> np.ndarray:
        return uv / self.scale + self.origin


def _warp(image_bgr: np.ndarray, ox: float, oy: float, s: float, size: tuple[int, int]) -> Crop:
    # crop pixel (u, v) <- full pixel (u / s + ox, v / s + oy); pre-blur against aliasing
    src = cv2.GaussianBlur(image_bgr, (0, 0), max(0.5 / s - 0.5, 0.01)) if s < 1 else image_bgr
    A = np.array([[s, 0, -s * ox + 0.5 * s - 0.5], [0, s, -s * oy + 0.5 * s - 0.5]])
    out = cv2.warpAffine(src, A, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Crop(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), np.array([ox + 0.5 - 0.5 / s, oy + 0.5 - 0.5 / s]), s)


def crop(image_bgr: np.ndarray, box: tuple[float, float, float, float]) -> Crop:
    """Square crop around the object box, with margin, resized to S x S."""
    x0, y0, x1, y1 = box
    L = CROP_MARGIN * max(x1 - x0, y1 - y0)
    return _warp(image_bgr, (x0 + x1) / 2 - L / 2, (y0 + y1) / 2 - L / 2, S / L, (S, S))


def crop_scaled(image_bgr: np.ndarray, box: tuple[float, float, float, float], s: float) -> Crop:
    """The box itself, resampled at s crop px per full-resolution px."""
    x0, y0, x1, y1 = box
    return _warp(image_bgr, x0, y0, s, (round((x1 - x0) * s), round((y1 - y0) * s)))


@dataclass
class Result:
    R: np.ndarray | None
    t: np.ndarray | None
    ref: int = -1
    n_matches: int = 0
    n_inliers: int = 0
    reselected: bool = False
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _rot_angle(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)))


class Estimator:
    def __init__(self, extractor, matcher, refs: list[Reference], K: np.ndarray, dist: np.ndarray,
                 center: np.ndarray, device):
        self.extractor, self.matcher, self.refs = extractor, matcher, refs
        self.K, self.dist, self.center, self.device = K, dist, center, device

    @torch.inference_mode()
    def match_all(self, c: Crop):
        img = torch.from_numpy(c.image).permute(2, 0, 1).float().div(255).to(self.device)
        q = self.extractor.extract(img, resize=None)
        out = []
        for ref in self.refs:
            m = self.matcher({"image0": q, "image1": ref.feats})["matches"][0]
            out.append(m.cpu().numpy())
        return q["keypoints"][0].cpu().numpy(), out

    def _pnp(self, c: Crop, qkp: np.ndarray, m: np.ndarray, ref: Reference):
        if len(m) < 6:
            return None
        uv = c.to_full(qkp[m[:, 0]]).astype(np.float64)
        uv = cv2.undistortPoints(uv[:, None], self.K, self.dist, P=self.K)[:, 0]
        xyz = ref.xyz[torch.as_tensor(m[:, 1])].cpu().numpy().astype(np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(xyz, uv, self.K, None, iterationsCount=PNP_ITERS,
                                                 reprojectionError=PNP_PX / c.scale, confidence=0.999)
        if not ok or inl is None:
            return None
        return cv2.Rodrigues(rvec)[0], tvec.ravel(), len(inl)

    def estimate(self, c: Crop) -> Result:
        qkp, matches = self.match_all(c)
        counts = np.array([len(m) for m in matches])
        best = Result(None, None, counts=counts)
        for i in np.argsort(-counts)[:TRY_TOP]:
            p = self._pnp(c, qkp, matches[i], self.refs[i])
            if p and p[2] > best.n_inliers:
                best = Result(p[0], p[1], int(i), int(counts[i]), p[2], counts=counts)
                if p[2] >= MIN_INLIER_RATIO * counts[i]:
                    break
        if best.R is None:
            return best
        # view re-selection: the reference whose viewpoint is closest to the estimate, as seen
        # from a camera centred on the object (the reference cameras are)
        cc = best.R @ self.center + best.t
        z = cc / np.linalg.norm(cc)
        axis = np.cross(z, [0, 0, 1])
        ang = np.arccos(np.clip(z[2], -1, 1))
        Ra = cv2.Rodrigues(axis / max(np.linalg.norm(axis), 1e-12) * ang)[0]
        Reff = Ra @ best.R
        same_mode = [j for j, r in enumerate(self.refs) if r.mode == self.refs[best.ref].mode]
        j = min(same_mode, key=lambda j: _rot_angle(self.refs[j].R, Reff))
        if j != best.ref:
            p = self._pnp(c, qkp, matches[j], self.refs[j])
            if p and p[2] > best.n_inliers:
                best = Result(p[0], p[1], j, int(counts[j]), p[2], True, counts)
        return best
