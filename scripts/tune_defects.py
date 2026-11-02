"""Tune the assembly check on synthetic defects, then freeze its parameters.

Reads only the target models and the camera intrinsics: no dataset image, pose or label.

Usage: python scripts/tune_defects.py <dataset_dir> <out_dir> [--n 120] [--seed 0]
Writes <out_dir>/defect_params.json and a few synth_*.jpg examples.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pixie import awake, defects, synth  # noqa: E402
from pixie.data import camera, target_mesh  # noqa: E402
from pixie.metrics import auroc  # noqa: E402
from pixie.parts import parts as split_parts  # noqa: E402
from pixie.render import Rasterizer  # noqa: E402

GRID_EVIDENCE = {"tol_px": [2, 3, 5], "norm_q": [50.0, 75.0, 90.0]}
GRID_SCORE = {"min_edge": [10, 20], "region_mm": [0.0, 8.0, 16.0, 24.0]}


def figure(img, ev: defects.Evidence) -> np.ndarray:
    vis = img[..., ::-1].copy()
    on = ev.edges >= 0
    c = ev.conf[on]
    vis[on] = np.column_stack([np.zeros_like(c), 255 * c, 255 * (1 - c)])   # green confirmed, red not
    return np.hstack([img[..., ::-1], vis])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--models", nargs="+", default=["model1", "model2", "model3", "model4"])
    ap.add_argument("--n", type=int, default=120, help="synthetic views per model")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    awake.keep_awake()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    ev_grid = list(itertools.product(*GRID_EVIDENCE.values()))
    samples = []   # (model, removed, defective, {ev_key: (n_edge, support)})
    nbs = {}
    px_err = []
    for model in a.models:
        t0 = time.perf_counter()
        mesh = target_mesh(a.root, model)
        prt = split_parts(mesh)
        T = synth.target(mesh, prt)
        K = camera(a.root, model).K
        ras = Rasterizer(mesh.vertices, mesh.faces, np.array(mesh.face_normals), dev)
        nbs[model] = {r: defects.neighbours(prt, r) for r in GRID_SCORE["region_mm"]}
        pts = np.asarray(mesh.vertices[:: max(len(mesh.vertices) // 1000, 1)], float)
        kinds = []
        for i in range(a.n):
            R, t = synth.sample_pose(rng, T)
            Re, te = synth.perturb(rng, R, t)
            Kc = defects.crop_intrinsics(K, defects.view_box(mesh.vertices, Re, te, K))
            own_true, _ = defects.render_edges(ras, prt, R, t, Kc)
            visible = np.bincount(own_true[own_true >= 0], minlength=len(prt)) >= 15
            d = synth.defect(rng, T, visible)
            img = synth.photo(rng, T, d, R, t, Kc, dev)
            owner, theta = defects.render_edges(ras, prt, Re, te, Kc)
            evs = {}
            for tol, q in ev_grid:
                ev = defects.evidence(img, owner, theta, len(prt), defects.Params(tol_px=tol, norm_q=q))
                evs[(tol, q)] = (ev.n_edge, ev.support)
                if i < 3 and (tol, q) == (3, 75.0):
                    cv2.imwrite(str(a.out / f"synth_{model}_{i}_{d.kind}.jpg"), figure(img, ev))
            samples.append((model, d.removed, d.kind != "none", evs))
            kinds.append(d.kind)
            p0 = (pts @ R.T + t) @ Kc.T
            p1 = (pts @ Re.T + te) @ Kc.T
            px_err.append(np.linalg.norm(p0[:, :2] / p0[:, 2:] - p1[:, :2] / p1[:, 2:], axis=1).mean())
        print(f"{model}: {len(prt)} parts, up {T.up:+d}, {a.n} views {dict(zip(*np.unique(kinds, return_counts=True)))}, "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    print(f"simulated pose error: median {np.median(px_err):.1f} px, p75 {np.percentile(px_err, 75):.1f} px")

    results = []
    for (tol, q), (min_edge, region) in itertools.product(ev_grid, itertools.product(*GRID_SCORE.values())):
        p = defects.Params(tol_px=tol, norm_q=q, min_edge=min_edge, region_mm=region)
        part_s, part_y, img_s, img_y = [], [], [], []
        for model, removed, bad, evs in samples:
            n_edge, sup = evs[(tol, q)]
            sc = defects.part_scores(n_edge, sup, nbs[model][region], p)
            part_s.append(sc)
            part_y.append(removed)
            img_s.append(defects.image_score(sc))
            img_y.append(bad)
        pa = auroc(np.concatenate(part_s), np.concatenate(part_y))
        ia = auroc(img_s, img_y)
        results.append(((pa + ia) / 2, pa, ia, p))
    results.sort(key=lambda r: -r[0])
    print(f"{'tol':>4s} {'q':>5s} {'min':>4s} {'region':>6s}  {'part AUROC':>10s}  {'image AUROC':>11s}")
    for m, pa, ia, p in results[:12]:
        print(f"{p.tol_px:4d} {p.norm_q:5.0f} {p.min_edge:4d} {p.region_mm:6.0f}  {pa:10.3f}  {ia:11.3f}")
    results[0][3].save(a.out / "defect_params.json")
    print("frozen:", results[0][3])


if __name__ == "__main__":
    main()
