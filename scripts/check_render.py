"""Check the rasteriser: render the target with the gt pose over the (undistorted) photo, show
the reference renders of one viewpoint, and time a full reference set.

Usage: python scripts/check_render.py <dataset_dir> [model]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pixie.data import camera, samples, target_mesh  # noqa: E402
from pixie.render import Rasterizer, colorize, look_at, view_directions  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"
SCALE = 0.25


def main() -> None:
    root = Path(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else "model1"
    dev = torch.device("cuda")
    mesh = target_mesh(root, model)
    cam = camera(root, model)
    ras = Rasterizer(mesh.vertices, mesh.faces, mesh.face_normals, dev)

    s = samples(root, model)[0]
    img = cv2.undistort(cv2.imread(str(s.image)), cam.K, cam.dist)
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    K = cam.K.copy()
    K[:2] *= SCALE
    K[:2, 2] += 0.5 * SCALE - 0.5     # pixel-centre convention after resizing
    R, t = s.pose()
    r = ras.render(R, t, K, *img.shape[:2])
    col = cv2.cvtColor(colorize(r, "normal"), cv2.COLOR_RGB2BGR)
    m = r.mask.cpu().numpy()
    blend = img.copy()
    blend[m] = (0.5 * img[m] + 0.5 * col[m]).astype(np.uint8)
    ys, xs = np.nonzero(m)
    pad = 40
    box = np.s_[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
    cv2.imwrite(str(OUT / f"render_gt_{model}.jpg"), np.hstack([img[box], blend[box]]))
    print(f"gt render: {m.sum()} object pixels")

    # reference renders, PIXIE layout: 121 directions x 8 rolls, object centred
    S = 512
    center = mesh.bounds.mean(0)
    radius = np.linalg.norm(mesh.vertices - center, axis=1).max()
    dist = 6 * radius
    f = 0.45 * S * dist / radius
    Kr = np.array([[f, 0, (S - 1) / 2], [0, f, (S - 1) / 2], [0, 0, 1]])
    dirs = view_directions()
    tiles = []
    for roll in (0, np.pi / 4):
        for mode in ("normal", "depth"):
            Rr = look_at(dirs[10], roll)
            tr = -Rr @ (center + dist * dirs[10])
            tiles.append(colorize(ras.render(Rr, tr, Kr, S, S), mode))
    cv2.imwrite(str(OUT / f"render_refs_{model}.jpg"), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for d in dirs:
        for k in range(8):
            Rr = look_at(d, k * np.pi / 4)
            ras.render(Rr, -Rr @ (center + dist * d), Kr, S, S)
            n += 1
    torch.cuda.synchronize()
    print(f"{n} reference renders at {S}px: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
