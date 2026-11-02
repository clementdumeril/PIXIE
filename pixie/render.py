"""Minimal GPU rasteriser in PyTorch (no OpenGL: the GPU machine is reached over SSH).

render() returns, per pixel, the depth, the camera-frame normal and the model-frame 3D point of
the nearest surface. The 3D point map is what turns 2D matches on a reference render into
2D-3D correspondences.

Pixel centres are at integer coordinates, as in OpenCV's projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

FACE_BITS = 25                 # face id packed in the low bits of the z-buffer key
MAX_CANDIDATES = 8_000_000     # pixel candidates processed at once


@dataclass
class Render:
    depth: torch.Tensor    # (H, W), 0 where empty
    normal: torch.Tensor   # (H, W, 3), camera frame, facing the camera
    xyz: torch.Tensor      # (H, W, 3), model frame
    mask: torch.Tensor     # (H, W) bool
    face: torch.Tensor     # (H, W) long, -1 where empty


class Rasterizer:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, face_normals: np.ndarray, device):
        self.device = device
        self.v = torch.as_tensor(vertices, dtype=torch.float32, device=device)
        self.f = torch.as_tensor(faces, dtype=torch.long, device=device)
        self.fn = torch.as_tensor(face_normals, dtype=torch.float32, device=device)
        assert len(self.f) < 2 ** FACE_BITS

    @torch.inference_mode()
    def render(self, R: np.ndarray, t: np.ndarray, K: np.ndarray, H: int, W: int) -> Render:
        dev = self.device
        R_ = torch.as_tensor(R, dtype=torch.float32, device=dev)
        t_ = torch.as_tensor(t, dtype=torch.float32, device=dev)
        vc = self.v @ R_.T + t_
        z = vc[:, 2]
        uv = torch.stack([K[0, 0] * vc[:, 0] / z + K[0, 2], K[1, 1] * vc[:, 1] / z + K[1, 2]], 1)
        P, Z = uv[self.f], z[self.f]                               # (F, 3, 2), (F, 3)
        x0 = P[..., 0].min(1).values.ceil().clamp(0, W - 1).long()
        x1 = P[..., 0].max(1).values.floor().clamp(0, W - 1).long()
        y0 = P[..., 1].min(1).values.ceil().clamp(0, H - 1).long()
        y1 = P[..., 1].max(1).values.floor().clamp(0, H - 1).long()
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        a = P[:, 0]
        area = _cross(P[:, 1] - a, P[:, 2] - a)
        keep = (Z > 1e-6).all(1) & (bw > 0) & (bh > 0) & (area.abs() > 1e-9)
        keep &= (P[..., 0].max(1).values >= 0) & (P[..., 0].min(1).values <= W - 1)
        keep &= (P[..., 1].max(1).values >= 0) & (P[..., 1].min(1).values <= H - 1)
        fids = keep.nonzero().squeeze(1)

        INF = torch.iinfo(torch.long).max
        zbuf = torch.full((H * W,), INF, dtype=torch.long, device=dev)
        counts = (bw * bh)[fids]
        start = 0
        while start < len(fids):
            # take as many faces as fit in the candidate budget (at least one)
            csum = torch.cumsum(counts[start:], 0)
            n = max(int(torch.searchsorted(csum, MAX_CANDIDATES)), 1)
            fi = fids[start:start + n]
            c = counts[start:start + n]
            start += n
            local = torch.repeat_interleave(torch.arange(len(fi), device=dev), c)
            off = torch.arange(int(c.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(c, 0) - c, c)
            f = fi[local]
            px = x0[f] + off % bw[f]
            py = y0[f] + torch.div(off, bw[f], rounding_mode="floor")
            w = _bary(P[f], px.float(), py.float(), area[f])
            inside = (w >= 0).all(1)
            f, px, py, w = f[inside], px[inside], py[inside], w[inside]
            depth = 1.0 / (w / Z[f]).sum(1)
            key = (depth * 1e6).long() * (2 ** FACE_BITS) + f      # micrometres, then face id
            zbuf.scatter_reduce_(0, py * W + px, key, reduce="amin")

        hit = zbuf < INF
        pix = hit.nonzero().squeeze(1)
        f = zbuf[pix] % (2 ** FACE_BITS)
        px, py = (pix % W).float(), torch.div(pix, W, rounding_mode="floor").float()
        w = _bary(P[f], px, py, area[f])
        wp = w / Z[f]                                             # perspective-correct weights
        depth = 1.0 / wp.sum(1)
        wp = wp * depth[:, None]
        xyz = (wp[:, :, None] * self.v[self.f[f]]).sum(1)
        n = self.fn[f] @ R_.T
        pc = (wp[:, :, None] * vc[self.f[f]]).sum(1)
        n = torch.where(((n * pc).sum(1) > 0)[:, None], -n, n)   # face the camera

        out_d = torch.zeros(H * W, device=dev)
        out_n = torch.zeros(H * W, 3, device=dev)
        out_x = torch.zeros(H * W, 3, device=dev)
        out_f = torch.full((H * W,), -1, dtype=torch.long, device=dev)
        out_d[pix], out_n[pix], out_x[pix], out_f[pix] = depth, n, xyz, f
        return Render(out_d.view(H, W), out_n.view(H, W, 3), out_x.view(H, W, 3), hit.view(H, W),
                      out_f.view(H, W))


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _bary(P: torch.Tensor, px: torch.Tensor, py: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    p = torch.stack([px, py], 1)
    w0 = _cross(P[:, 1] - p, P[:, 2] - p) / area
    w1 = _cross(P[:, 2] - p, P[:, 0] - p) / area
    return torch.stack([w0, w1, 1 - w0 - w1], 1)


# --- reference viewpoints -------------------------------------------------------------------

def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def view_directions(n_fib: int = 115) -> np.ndarray:
    """6 axis-aligned views + n_fib Fibonacci views (PIXIE: 6 + 115 = 121)."""
    axes = np.concatenate([np.eye(3), -np.eye(3)])
    return np.concatenate([axes, fibonacci_sphere(n_fib)])


def look_at(direction: np.ndarray, roll: float) -> np.ndarray:
    """World->camera rotation (OpenCV axes) for a camera on `direction` looking at the origin,
    rotated by `roll` radians about its optical axis."""
    zc = -direction / np.linalg.norm(direction)
    up = np.array([0.0, 0.0, 1.0]) if abs(zc[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    xc = np.cross(zc, up)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.stack([xc, yc, zc])
    c, s = np.cos(roll), np.sin(roll)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R


# --- images fed to the matcher --------------------------------------------------------------

def colorize(r: Render, mode: str) -> np.ndarray:
    """uint8 RGB image of a render: 'normal' (camera-frame normals) or 'depth' (turbo colormap
    over the object's own depth range). Background is black."""
    m = r.mask.cpu().numpy()
    if mode == "normal":
        img = ((r.normal.cpu().numpy() * np.array([1, -1, -1]) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    elif mode == "depth":
        d = r.depth.cpu().numpy()
        lo, hi = d[m].min(), d[m].max()
        g = (255 * (1 - (d - lo) / max(hi - lo, 1e-9))).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(cv2.applyColorMap(g, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(mode)
    img[~m] = 0
    return img
