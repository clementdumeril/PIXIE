"""Synthetic defects for tuning the assembly check. Uses only the target model, never the
dataset's images or as-built meshes.

A defect is made on the target itself: remove the parts in a brick-sized box of the LEGO grid
together with everything resting on it ("missing"), put a copy of such a box elsewhere on top
("extra"), or both with the same box ("moved"). The result is rendered as a crude photo: one flat
colour per brick-sized run of grid cells, Lambertian shading plus a soft highlight, a paper
background with printed lines and a few loose bricks, then blur, noise and JPEG.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import trimesh

from .defects import S
from .parts import Parts
from .render import Rasterizer, look_at

PITCH = 0.008     # stud pitch, m
PLATE = 0.0032    # plate height, m
STUD = np.array([1.6, 4.8, 4.8]) / 1000
KINDS = ("none", "missing", "extra", "moved")
KIND_P = (0.3, 0.4, 0.15, 0.15)
PALETTE = np.array([[201, 26, 9], [0, 85, 191], [242, 205, 55], [35, 120, 65], [255, 255, 255],
                    [27, 42, 52], [160, 165, 169], [254, 138, 24], [165, 202, 24], [228, 205, 158],
                    [5, 157, 158], [160, 110, 185], [88, 42, 18], [252, 151, 172]], float) / 255


@dataclass
class Target:
    mesh: trimesh.Trimesh
    parts: Parts
    up: int              # +1 if model +z is up, else -1
    cell: np.ndarray     # (P, 3) grid cell of each part: x, y in studs, height in plates
    height: np.ndarray   # (P,) part centroid height above the lowest part, m
    center: np.ndarray
    base: np.ndarray     # (P,) part whose colour each part takes: the brick under a stud, else itself


def target(mesh: trimesh.Trimesh, parts: Parts) -> Target:
    vs = [mesh.vertices[np.unique(mesh.faces[parts.face_part == p])] for p in range(len(parts))]
    lo3, hi3 = np.array([v.min(0) for v in vs]), np.array([v.max(0) for v in vs])
    studs = np.isclose(np.sort(hi3 - lo3, 1), STUD, atol=2e-4).all(1)
    z = parts.centroid[:, 2]
    up = 1 if z[studs].mean() > z.mean() else -1
    top, bottom = np.maximum(up * lo3[:, 2], up * hi3[:, 2]), np.minimum(up * lo3[:, 2], up * hi3[:, 2])
    base = np.arange(len(parts))
    for s in np.flatnonzero(studs):
        c = parts.centroid[s, :2]
        under = (~studs & (lo3[:, :2] <= c + 1e-4).all(1) & (hi3[:, :2] >= c - 1e-4).all(1)
                 & (top <= bottom[s] + 5e-4))
        if under.any():
            base[s] = np.flatnonzero(under)[np.argmax(top[under])]
    h = up * z
    h = h - h.min()
    lo = parts.centroid[:, :2].min(0)
    cell = np.column_stack([np.floor((parts.centroid[:, :2] - lo) / PITCH), np.floor(h / PLATE)]).astype(int)
    return Target(mesh, parts, up, cell, h, mesh.bounds.mean(0), base)


@dataclass
class Defect:
    kind: str
    removed: np.ndarray           # (P,) bool, target parts absent from the synthetic assembly
    extra: np.ndarray             # indices of target parts copied ...
    offset: np.ndarray            # ... with this translation (m)


def _box(rng, T: Target, seed: int) -> np.ndarray:
    size = rng.choice([1, 2, 4], 2)
    start = T.cell[seed, :2] - rng.integers(0, size)
    layers = rng.choice([1, 3])
    z0 = T.cell[seed, 2] - rng.integers(0, layers)
    foot = ((T.cell[:, :2] >= start) & (T.cell[:, :2] < start + size)).all(1)
    return foot & (T.cell[:, 2] >= z0)          # the box and everything resting on it


def defect(rng, T: Target, visible: np.ndarray) -> Defect:
    kind = rng.choice(KINDS, p=KIND_P)
    P = len(T.parts)
    removed, extra, offset = np.zeros(P, bool), np.zeros(0, int), np.zeros(3)
    if kind == "none":
        return Defect(kind, removed, extra, offset)
    box = _box(rng, T, rng.choice(np.flatnonzero(visible)))
    if kind in ("missing", "moved"):
        removed = box
    if kind in ("extra", "moved"):
        extra = np.flatnonzero(box)
        while True:
            dxy = rng.integers(-3, 4, 2)
            if dxy.any():
                break
        moved = T.cell[extra, :2] + dxy
        under = np.isin(T.cell[:, 0] * 10000 + T.cell[:, 1], moved[:, 0] * 10000 + moved[:, 1]) & ~removed
        top = T.height[under].max() if under.any() else 0.0
        dh = top - T.height[extra].min() + PLATE
        offset = np.array([dxy[0] * PITCH, dxy[1] * PITCH, T.up * dh])
    return Defect(kind, removed, extra, offset)


def _colours(rng, T: Target) -> np.ndarray:
    """Colour per part: one colour per run of 2 or 4 cells along x or y, per brick layer. Heights
    are taken 1 mm down so that studs and top faces share the colour of the brick under them."""
    out = np.zeros((len(T.parts), 3))
    brick = {}
    layers = np.floor(np.maximum(T.height - 0.001, 0) / (3 * PLATE)).astype(int)
    for p, (x, y, _) in enumerate(T.cell):
        layer = layers[p]
        along_x = (layer % 2 == 0)
        run = 2 + 2 * (hash((layer, y if along_x else x)) % 2)
        key = (layer, y, x // run) if along_x else (layer, x, y // run)
        if key not in brick:
            brick[key] = PALETTE[rng.integers(len(PALETTE))]
        out[p] = brick[key]
    return out[T.base]


def assembly(T: Target, d: Defect, colours: np.ndarray):
    """Vertices, faces and per-face colour of the synthetic assembly."""
    fp = T.parts.face_part
    keep = ~d.removed[fp]
    V, F, C = [T.mesh.vertices], [T.mesh.faces[keep]], [colours[fp[keep]]]
    if len(d.extra):
        sel = np.isin(fp, d.extra)
        F.append(T.mesh.faces[sel] + len(T.mesh.vertices))
        V.append(T.mesh.vertices + d.offset)
        C.append(colours[fp[sel]])
    return np.concatenate(V), np.concatenate(F), np.concatenate(C)


def sample_pose(rng, T: Target):
    """Camera above the board: elevation 20-75 deg, 25-50 cm away, near-upright."""
    e, a = np.radians(rng.uniform(20, 75)), rng.uniform(0, 2 * np.pi)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), T.up * np.sin(e)])
    roll = np.radians(rng.uniform(-20, 20)) + (np.pi if T.up < 0 else 0)
    R = look_at(d, roll)
    return R, -R @ (T.center + rng.uniform(0.25, 0.5) * d)


def perturb(rng, R, t, rot_deg: float = 0.3, trans_mm=(0.5, 0.5, 2.0)):
    """Pose estimate with errors of the size the estimator makes (a few crop pixels)."""
    axis = rng.normal(size=3)
    rv = axis / np.linalg.norm(axis) * np.radians(abs(rng.normal(0, rot_deg)))
    return cv2.Rodrigues(rv)[0] @ R, t + rng.normal(0, trans_mm) / 1000


def _background(rng) -> np.ndarray:
    base = rng.uniform(0.8, 0.95) * np.array([1, rng.uniform(0.97, 1.0), rng.uniform(0.92, 1.0)])
    g = np.linspace(rng.uniform(0.9, 1.0), rng.uniform(0.9, 1.0), S)[:, None, None]
    img = np.ones((S, S, 3)) * base * g
    for _ in range(rng.integers(3, 9)):                       # printed lines
        p0, p1 = rng.integers(0, S, 2), rng.integers(0, S, 2)
        cv2.line(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), (0.15,) * 3, int(rng.integers(1, 3)))
    for _ in range(rng.integers(0, 4)):                       # loose bricks
        c = rng.integers(0, S, 2)
        box = cv2.boxPoints(((float(c[0]), float(c[1])), tuple(float(v) for v in rng.uniform(30, 120, 2)),
                             float(rng.uniform(0, 180))))
        cv2.fillConvexPoly(img, box.astype(np.int32), tuple(float(v) for v in PALETTE[rng.integers(len(PALETTE))]))
    return img


def photo(rng, T: Target, d: Defect, R, t, Kc, device) -> np.ndarray:
    """S x S RGB uint8 picture of the synthetic assembly at the true pose (R, t)."""
    V, F, C = assembly(T, d, _colours(rng, T))
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    r = Rasterizer(V, F, fn, device).render(R, t, Kc, S, S)
    m = r.mask.cpu().numpy()
    n = r.normal.cpu().numpy()
    f = r.face.cpu().numpy()
    light = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-1, -0.3), -1.0])   # towards the light: up, front
    light /= np.linalg.norm(light)
    lam = np.clip((n * light).sum(-1), 0, 1)
    half = light + np.array([0, 0, -1.0])
    half /= np.linalg.norm(half)
    spec = np.clip((n * half).sum(-1), 0, 1) ** 30
    alb = C[np.maximum(f, 0)]
    amb = rng.uniform(0.3, 0.5)
    shaded = alb * (amb + (1 - amb) * lam)[..., None] + 0.25 * spec[..., None]
    img = np.where(m[..., None], shaded, _background(rng))
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.0))
    img = np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1)
    ok, buf = cv2.imencode(".jpg", (img[..., ::-1] * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(80, 96))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].copy()
