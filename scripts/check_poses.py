"""Sanity check of the ground-truth convention: project the target mesh with the gt pose onto a
few images (blue: target model, orange: as-built model of defective scenes) and print the mesh
extents. Writes out/check_<model>_<scene>.jpg.

Usage: python scripts/check_poses.py <dataset_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pixie.data import camera, load_mesh, samples, target_mesh  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"


def draw(img, mesh, R, t, cam, color):
    # sharp edges only, so the outline stays readable
    sharp = mesh.face_adjacency_angles > np.radians(30)
    edges = mesh.face_adjacency_edges[sharp]
    rvec, _ = cv2.Rodrigues(R)
    p, _ = cv2.projectPoints(mesh.vertices, rvec, t, cam.K, cam.dist)
    p = p.reshape(-1, 2)
    for a, b in edges:
        cv2.line(img, tuple(np.round(p[a]).astype(int)), tuple(np.round(p[b]).astype(int)), color, 2)


def main() -> None:
    root = Path(sys.argv[1])
    OUT.mkdir(exist_ok=True)
    for model in sorted(p.name for p in root.iterdir() if p.is_dir()):
        mesh = target_mesh(root, model)
        cam = camera(root, model)
        ext = mesh.bounds[1] - mesh.bounds[0]
        print(f"{model}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces, extent {np.round(ext, 4)}, "
              f"{len(mesh.split(only_watertight=False))} connected parts, K fx {cam.K[0, 0]:.0f}")
        ss = samples(root, model)
        print(f"  {len(ss)} images with gt, scenes: {sorted({s.scene for s in ss})}")
        seen = set()
        for s in ss:
            if s.kind in seen:
                continue
            seen.add(s.kind)
            R, t = s.pose()
            img = cv2.imread(str(s.image))
            raw = img.copy()
            draw(img, mesh, R, t, cam, (214, 120, 42))
            built = s.gt.parent / "model.obj"
            if built.exists():
                draw(img, load_mesh(built), R, t, cam, (52, 104, 235))
            print(f"  {s.scene}/{s.image.stem}: t = {np.round(t, 3)} m, depth {t[2]:.3f}")
            # crop around the projected mesh so the overlay is readable
            p, _ = cv2.projectPoints(mesh.vertices, cv2.Rodrigues(R)[0], t, cam.K, cam.dist)
            p = p.reshape(-1, 2)
            (x0, y0), (x1, y1) = p.min(0), p.max(0)
            m = 0.4 * max(x1 - x0, y1 - y0)
            h, w = img.shape[:2]
            box = np.s_[max(int(y0 - m), 0):min(int(y1 + m), h), max(int(x0 - m), 0):min(int(x1 + m), w)]
            cv2.imwrite(str(OUT / f"check_{model}_{s.scene}.jpg"), np.hstack([raw[box], img[box]]))


if __name__ == "__main__":
    main()
