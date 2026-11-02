"""PIXIE dataset access: target mesh, camera, images and ground-truth poses.

Layout (github.com/ZEISS/PIXIE-dataset):
  dataset/modelN/modelN.obj                     fully assembled target (the only mesh used at inference)
  dataset/modelN/calibration/camera_calibration.json
  dataset/modelN/data/<scene>/IMG_xxxx.JPG
  dataset/modelN/gt/<scene>/IMG_xxxx.json       rvec, tvec (model -> camera), evaluation only
  dataset/modelN/gt/<scene>/model.obj           as-built mesh of a defective scene, evaluation only
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh

SCENE_TYPES = ("assembly_mistake_with_clutter", "assembly_mistake", "clutter", "color")


def scene_type(scene: str) -> str:
    return next(t for t in SCENE_TYPES if scene.startswith(t))


@dataclass
class Camera:
    K: np.ndarray       # 3x3, full-resolution pixels
    dist: np.ndarray    # OpenCV distortion coefficients


@dataclass
class Sample:
    model: str
    scene: str
    image: Path
    gt: Path

    @property
    def kind(self) -> str:
        return scene_type(self.scene)

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Ground-truth (R, t), model frame -> camera frame. Evaluation only."""
        g = json.loads(self.gt.read_text())
        R, _ = cv2.Rodrigues(np.asarray(g["rvec"], float))
        return R, np.asarray(g["tvec"], float)


def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh", process=False)
    return trimesh.Trimesh(m.vertices, m.faces, process=True)   # merge duplicate vertices


def target_mesh(root: Path, model: str) -> trimesh.Trimesh:
    return load_mesh(root / model / f"{model}.obj")


def camera(root: Path, model: str) -> Camera:
    c = json.loads((root / model / "calibration" / "camera_calibration.json").read_text())
    return Camera(np.asarray(c["intrinsics_mtx"], float), np.asarray(c["dist_coeffs"], float).ravel())


def marker_size(root: Path, model: str) -> float:
    """ArUco side length (m). Only this is read from modelN_poses.json: its 'model' entry places
    the object on the board, which is ground truth."""
    return json.loads((root / model / f"{model}_poses.json").read_text())["marker_size_mm"] / 1000


def samples(root: Path, model: str) -> list[Sample]:
    out = []
    for sc in sorted((root / model / "data").iterdir()):
        for img in sorted(sc.glob("*.JPG")):
            gt = root / model / "gt" / sc.name / f"{img.stem}.json"
            if gt.exists():
                out.append(Sample(model, sc.name, img, gt))
    return out
