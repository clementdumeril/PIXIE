"""Reference set of a target model: renders from 121 directions x 8 in-plane rolls, SuperPoint
features on each render, and the model-frame 3D point under every keypoint."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .render import Rasterizer, colorize, look_at, view_directions

S = 512           # render size (px)
FILL_PX = 2       # keypoints this close to the silhouette take the nearest surface point
DIST_RADII = 6    # camera distance in bounding-sphere radii
FILL_RATIO = 0.45 # bounding-sphere radius in units of S


@dataclass
class Reference:
    R: np.ndarray      # model -> reference camera rotation (object centred)
    mode: str
    feats: dict        # SuperPoint output, batch of 1, on the GPU
    xyz: torch.Tensor  # (n, 3) model-frame point of every keypoint


def reference_camera(mesh):
    center = mesh.bounds.mean(0)
    radius = float(np.linalg.norm(mesh.vertices - center, axis=1).max())
    dist = DIST_RADII * radius
    f = FILL_RATIO * S * dist / radius
    K = np.array([[f, 0, (S - 1) / 2], [0, f, (S - 1) / 2], [0, 0, 1]])
    return center, dist, K


def _filled_xyz(xyz: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Extend the point map a few pixels past the silhouette (mean of valid neighbours), so
    keypoints detected right on an outline still get a 3D point."""
    x = (xyz * mask[..., None]).permute(2, 0, 1)[None]
    m = mask[None, None].float()
    for _ in range(FILL_PX):
        s = F.avg_pool2d(x * 1, 3, 1, 1, count_include_pad=False)
        c = F.avg_pool2d(m, 3, 1, 1, count_include_pad=False)
        new = (m == 0) & (c > 0)
        x = torch.where(new, s / c.clamp_min(1e-9), x)
        m = torch.where(new, torch.ones_like(m), m)
    return x[0].permute(1, 2, 0), m[0, 0] > 0


@torch.inference_mode()
def build(mesh, extractor, device, modes=("normal",), rolls: int = 8) -> list[Reference]:
    ras = Rasterizer(mesh.vertices, mesh.faces, np.array(mesh.face_normals), device)
    center, dist, K = reference_camera(mesh)
    refs = []
    for d in view_directions():
        for k in range(rolls):
            R = look_at(d, 2 * np.pi * k / rolls)
            r = ras.render(R, -R @ (center + dist * d), K, S, S)
            xyz, valid = _filled_xyz(r.xyz, r.mask)
            for mode in modes:
                img = torch.from_numpy(colorize(r, mode)).permute(2, 0, 1).float().div(255).to(device)
                f = extractor.extract(img, resize=None)
                kp = f["keypoints"][0].round().long()
                kp[:, 0].clamp_(0, S - 1)
                kp[:, 1].clamp_(0, S - 1)
                ok = valid[kp[:, 1], kp[:, 0]]
                f = {k_: (v[:, ok] if k_ in ("keypoints", "keypoint_scores", "descriptors") else v)
                     for k_, v in f.items()}
                refs.append(Reference(R, mode, f, xyz[kp[ok, 1], kp[ok, 0]]))
    return refs
