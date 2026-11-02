"""Run the PIXIE-style estimator on the dataset and write one CSV row per image.

--box board (default, no ground truth): find the marker-board area, estimate a coarse pose in it
with the normal-map references, then crop around the target projected with that pose.
--box oracle: crop around the target projected with the gt pose. The dataset rules forbid this
at inference; it is kept only to reproduce the first runs.

Usage: python scripts/run_eval.py <dataset_dir> <weights.pth> <out.csv>
           [--models model1 ...] [--per-scene N] [--modes normal depth] [--box board|oracle]
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pixie import awake, detect, matching, pose, references  # noqa: E402
from pixie.data import camera, marker_size, samples, target_mesh  # noqa: E402

FIELDS = ["model", "scene", "kind", "image", "add_cm", "rot_deg", "trans_cm", "ref", "mode",
          "n_matches", "n_inliers", "reselected", "seconds", "rvec", "tvec",
          "n_markers", "coarse_add_cm", "coarse_rot_deg"]
COARSE_KEYPOINTS = 4096   # the board area is several times the object


def project_box(mesh, R, t, cam) -> tuple[float, float, float, float]:
    p, _ = cv2.projectPoints(mesh.vertices, cv2.Rodrigues(R)[0], t, cam.K, cam.dist)
    p = p.reshape(-1, 2)
    return (*p.min(0), *p.max(0))


def board_box(img, mesh, cam, coarse: pose.Estimator, marker_m: float, radius_m: float):
    """Stage 1: coarse pose in the board area. Returns (box or None, painted image, area, result)."""
    area = detect.find_area(img, cam.K, cam.dist, marker_m, radius_m)
    if area is None:
        return None, img, None, None
    s = references.FILL_RATIO * references.S / area.radius_px
    r = coarse.estimate(pose.crop_scaled(area.image, area.box, s))
    box = project_box(mesh, r.R, r.t, cam) if r.R is not None else None
    return box, area.image, area, r


def errors(pts, R, t, Rg, tg) -> tuple[float, float, float]:
    add = np.linalg.norm((pts @ R.T + t) - (pts @ Rg.T + tg), axis=1).mean()
    return 100 * add, np.degrees(pose._rot_angle(R, Rg)), 100 * np.linalg.norm(t - tg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--models", nargs="+", default=["model1", "model2", "model3", "model4"])
    ap.add_argument("--per-scene", type=int, default=0, help="images per scene, 0 = all")
    ap.add_argument("--modes", nargs="+", default=["normal"])
    ap.add_argument("--box", choices=["board", "oracle"], default="board")
    a = ap.parse_args()
    awake.keep_awake()

    dev = torch.device("cuda")
    extractor, matcher = matching.load(a.ckpt, dev)
    coarse_extractor = matching.superpoint(dev, COARSE_KEYPOINTS)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.out.exists():
        done = {(r["model"], r["image"]) for r in csv.DictReader(open(a.out, encoding="utf-8"))}
    new = not a.out.exists()
    with open(a.out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        for model in a.models:
            mesh = target_mesh(a.root, model)
            cam = camera(a.root, model)
            t0 = time.perf_counter()
            refs = references.build(mesh, extractor, dev, tuple(a.modes))
            nk = np.mean([len(r.xyz) for r in refs])
            print(f"{model}: {len(refs)} references, {nk:.0f} keypoints each, {time.perf_counter() - t0:.0f} s",
                  flush=True)
            center, _, _ = references.reference_camera(mesh)
            radius = float(np.linalg.norm(mesh.vertices - center, axis=1).max())
            marker_m = marker_size(a.root, model)
            est = pose.Estimator(extractor, matcher, refs, cam.K, cam.dist, center, dev)
            coarse = pose.Estimator(coarse_extractor, matcher, [r for r in refs if r.mode == "normal"],
                                    cam.K, cam.dist, center, dev)
            pts = mesh.vertices
            ss = samples(a.root, model)
            if a.per_scene:
                by = {}
                for s in ss:
                    by.setdefault(s.scene, []).append(s)
                ss = [s for v in by.values() for s in v[:: max(len(v) // a.per_scene, 1)][: a.per_scene]]
            for s in ss:
                if (model, s.image.stem) in done:
                    continue
                Rg, tg = s.pose()   # scoring only, except for --box oracle
                t1 = time.perf_counter()
                img = cv2.imread(str(s.image))
                area = rc = None
                if a.box == "oracle":
                    box = project_box(mesh, Rg, tg, cam)
                else:
                    box, img, area, rc = board_box(img, mesh, cam, coarse, marker_m, radius)
                r = est.estimate(pose.crop(img, box)) if box is not None else pose.Result(None, None)
                dt = time.perf_counter() - t1
                row = {"model": model, "scene": s.scene, "kind": s.kind, "image": s.image.stem,
                       "seconds": f"{dt:.1f}", "n_matches": r.n_matches, "n_inliers": r.n_inliers,
                       "reselected": int(r.reselected), "ref": r.ref,
                       "mode": refs[r.ref].mode if r.ref >= 0 else "",
                       "n_markers": area.n_markers if area else 0,
                       "coarse_add_cm": "nan", "coarse_rot_deg": "nan"}
                if rc is not None and rc.R is not None:
                    e = errors(pts, rc.R, rc.t, Rg, tg)
                    row.update(coarse_add_cm=f"{e[0]:.3f}", coarse_rot_deg=f"{e[1]:.2f}")
                if r.R is None:
                    row.update(add_cm="nan", rot_deg="nan", trans_cm="nan", rvec="", tvec="")
                else:
                    e = errors(pts, r.R, r.t, Rg, tg)
                    row.update(add_cm=f"{e[0]:.3f}", rot_deg=f"{e[1]:.2f}", trans_cm=f"{e[2]:.3f}",
                               rvec=" ".join(f"{x:.6f}" for x in cv2.Rodrigues(r.R)[0].ravel()),
                               tvec=" ".join(f"{x:.6f}" for x in r.t))
                w.writerow(row)
                fh.flush()
                print(f"  {s.scene:32s} {s.image.stem}: ADD {row['add_cm']:>8s} cm  rot {row['rot_deg']:>7s} deg  "
                      f"{r.n_inliers}/{r.n_matches} inliers  coarse ADD {row['coarse_add_cm']:>8s}  "
                      f"{dt:.1f} s", flush=True)
            # free this model's references before building the next: on 8 GB the two together
            # spill into shared memory and every image gets several times slower
            del est, coarse, refs
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
