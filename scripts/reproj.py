"""2D alignment of estimated poses: mean reprojection distance of target vertices, est vs gt,
in pixels of the S x S object crop. Tells how precisely a render at the estimated pose lines up
with the photo, which bounds what per-part checks can resolve. Analysis only.

Usage: python scripts/reproj.py <dataset_dir> <run.csv>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pixie import pose, references  # noqa: E402
from pixie.data import camera, samples, target_mesh  # noqa: E402


def main() -> None:
    root, path = Path(sys.argv[1]), Path(sys.argv[2])
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = {}
    for model in sorted({r["model"] for r in rows}):
        mesh, cam = target_mesh(root, model), camera(root, model)
        pts = np.ascontiguousarray(mesh.vertices[:: max(len(mesh.vertices) // 2000, 1)], float)
        gt = {s.image.stem: s.pose() for s in samples(root, model)}
        for r in (r for r in rows if r["model"] == model and r["rvec"]):
            Rg, tg = gt[r["image"]]
            pg, _ = cv2.projectPoints(pts, cv2.Rodrigues(Rg)[0], tg, cam.K, cam.dist)
            pe, _ = cv2.projectPoints(pts, np.array(r["rvec"].split(), float), np.array(r["tvec"].split(), float),
                                      cam.K, cam.dist)
            pg, pe = pg.reshape(-1, 2), pe.reshape(-1, 2)
            side = pose.CROP_MARGIN * np.ptp(pg, 0).max()
            out.setdefault(r["kind"], []).append(np.linalg.norm(pe - pg, axis=1).mean() * references.S / side)
    for k, v in out.items():
        v = np.array(v)
        print(f"{k:30s} n {len(v):4d}  median {np.median(v):6.1f} px  p75 {np.percentile(v, 75):6.1f}  "
              f"< 5 px {100 * np.mean(v < 5):5.1f} %  < 10 px {100 * np.mean(v < 10):5.1f} %")


if __name__ == "__main__":
    main()
