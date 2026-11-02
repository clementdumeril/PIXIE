"""Parts of a target mesh: its connected components (stud, tube, wall... of a brick), with a
geometry key that identifies the same part in another mesh of the same model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from trimesh.graph import connected_component_labels


@dataclass
class Parts:
    face_part: np.ndarray   # (F,) part of every face
    centroid: np.ndarray    # (P, 3) model frame, m
    keys: list[bytes]       # vertex set rounded to 0.01 mm

    def __len__(self) -> int:
        return len(self.centroid)


def parts(mesh: trimesh.Trimesh) -> Parts:
    lab = connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    cen, keys = [], []
    for p in range(lab.max() + 1):
        v = mesh.vertices[np.unique(mesh.faces[lab == p])]
        cen.append(v.mean(0))
        keys.append(np.unique(np.round(v * 1e5).astype(np.int64), axis=0).tobytes())
    return Parts(lab, np.array(cen), keys)


def missing(target: Parts, built: Parts) -> np.ndarray:
    """Bool per target part: absent from the as-built mesh (removed or moved). Evaluation only."""
    have = set(built.keys)
    return np.array([k not in have for k in target.keys])
