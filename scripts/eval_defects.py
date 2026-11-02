"""Evaluate the frozen assembly check on the PIXIE images, from the poses of a run CSV.

Inference uses the target model, the image and the estimated pose. The as-built meshes and the gt
poses are read only to score (or, with --gt-pose, to measure how much pose error costs).

  image level: AUROC of the image score, defective scenes (assembly_mistake*) vs correct ones
  scene level: part support pooled over all views of a scene, then the same score
  part level:  AUROC / AP of part scores for parts absent from the as-built model, over the
               parts seen in defective scenes

Usage: python scripts/eval_defects.py <dataset_dir> <run.csv> <params.json> <out_dir> [--gt-pose]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pixie import awake, defects  # noqa: E402
from pixie.data import camera, load_mesh, samples, target_mesh  # noqa: E402
from pixie.metrics import auroc, average_precision  # noqa: E402
from pixie.parts import missing, parts as split_parts  # noqa: E402
from pixie.render import Rasterizer  # noqa: E402

KINDS = ("color", "clutter", "assembly_mistake", "assembly_mistake_with_clutter")


def figure(photo, ev: defects.Evidence) -> np.ndarray:
    vis = photo[..., ::-1].copy()
    on = ev.edges >= 0
    c = ev.conf[on]
    vis[on] = np.column_stack([np.zeros_like(c), 255 * c, 255 * (1 - c)])
    return np.hstack([photo[..., ::-1], vis])


def summary(name: str, img_rows: list[dict], scenes: dict) -> None:
    bad = lambda k: k.startswith("assembly_mistake")
    s = np.array([r["score"] for r in img_rows])
    y = np.array([bad(r["kind"]) for r in img_rows])
    clutter = np.array(["clutter" in r["kind"] for r in img_rows])
    print(f"\n{name}")
    print(f"  image AUROC  all {auroc(s, y):.3f}   no clutter {auroc(s[~clutter], y[~clutter]):.3f}   "
          f"clutter {auroc(s[clutter], y[clutter]):.3f}   (n {len(s)}, no pose {np.isnan(s).sum()})")
    for k in KINDS:
        sel = np.array([r["kind"] == k for r in img_rows])
        print(f"    {k:30s} median score {np.nanmedian(s[sel]):.3f}")
    ss = np.array([v["score"] for v in scenes.values()])
    sy = np.array([bad(v["kind"]) for v in scenes.values()])
    print(f"  scene AUROC  {auroc(ss, sy):.3f}  ({sy.sum()} defective / {(~sy).sum()} correct scenes)")
    ps = np.concatenate([v["parts"] for v in scenes.values() if bad(v["kind"])])
    py = np.concatenate([v["missing"] for v in scenes.values() if bad(v["kind"])])
    ips = np.concatenate([r["parts"] for r in img_rows if bad(r["kind"])])
    ipy = np.concatenate([r["missing"] for r in img_rows if bad(r["kind"])])
    print(f"  part AUROC   per image {auroc(ips, ipy):.3f} (AP {average_precision(ips, ipy):.3f}, "
          f"base rate {np.nanmean(np.where(np.isfinite(ips), ipy, np.nan)):.3f})   "
          f"per scene {auroc(ps, py):.3f} (AP {average_precision(ps, py):.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("run", type=Path)
    ap.add_argument("params", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--gt-pose", action="store_true", help="analysis: check at the gt pose")
    a = ap.parse_args()
    awake.keep_awake()
    a.out.mkdir(parents=True, exist_ok=True)
    p = defects.Params.load(a.params)
    print(p)
    dev = torch.device("cuda")
    rows = list(csv.DictReader(open(a.run, encoding="utf-8")))
    img_rows, scenes, figs, nbs = [], {}, set(), {}
    for model in sorted({r["model"] for r in rows}):
        mesh, cam = target_mesh(a.root, model), camera(a.root, model)
        prt = split_parts(mesh)
        nb = nbs[model] = defects.neighbours(prt, p.region_mm)
        ras = Rasterizer(mesh.vertices, mesh.faces, np.array(mesh.face_normals), dev)
        by_stem = {s.image.stem: s for s in samples(a.root, model)}
        labels = {}
        for r in (r for r in rows if r["model"] == model):
            s = by_stem[r["image"]]
            if s.scene not in labels:   # evaluation only
                labels[s.scene] = missing(prt, split_parts(load_mesh(a.root / model / "gt" / s.scene / "model.obj")))
            sc = scenes.setdefault(f"{model}/{s.scene}", {"kind": s.kind, "missing": labels[s.scene],
                                                         "n_edge": np.zeros(len(prt)), "support": np.zeros(len(prt))})
            if a.gt_pose:
                R, t = s.pose()
            elif r["rvec"]:
                R, t = cv2.Rodrigues(np.array(r["rvec"].split(), float))[0], np.array(r["tvec"].split(), float)
            else:
                img_rows.append({**r, "score": np.nan, "parts": np.full(len(prt), np.nan), "missing": labels[s.scene]})
                continue
            Kc = defects.crop_intrinsics(cam.K, defects.view_box(mesh.vertices, R, t, cam.K))
            photo = defects.undistorted_crop(cv2.imread(str(s.image)), cam.K, cam.dist, Kc)
            owner, theta = defects.render_edges(ras, prt, R, t, Kc)
            ev = defects.evidence(photo, owner, theta, len(prt), p)
            ps = defects.part_scores(ev.n_edge, ev.support, nb, p)
            sc["n_edge"] += ev.n_edge
            sc["support"] += ev.support
            img_rows.append({**r, "score": defects.image_score(ps), "parts": ps, "missing": labels[s.scene]})
            if (model, s.kind) not in figs:
                figs.add((model, s.kind))
                cv2.imwrite(str(a.out / f"check_{model}_{s.scene}_{s.image.stem}.jpg"), figure(photo, ev))
        print(f"{model}: {sum(r['model'] == model for r in img_rows)} images", flush=True)
    for key, v in scenes.items():
        v["parts"] = defects.part_scores(v["n_edge"], v["support"], nbs[key.split("/")[0]], p)
        v["score"] = defects.image_score(v["parts"])
    summary(f"{a.run.name}{' at gt pose' if a.gt_pose else ''}", img_rows, scenes)
    with open(a.out / "images.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "scene", "kind", "image", "score"])
        for r in img_rows:
            w.writerow([r["model"], r["scene"], r["kind"], r["image"], f"{r['score']:.4f}"])
    with open(a.out / "scenes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "kind", "score", "n_missing_seen", "n_seen"])
        for k, v in scenes.items():
            seen = np.isfinite(v["parts"])
            w.writerow([k, v["kind"], f"{v['score']:.4f}", int((v["missing"] & seen).sum()), int(seen.sum())])


if __name__ == "__main__":
    main()
