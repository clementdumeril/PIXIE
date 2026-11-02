"""Compare each scene's as-built mesh (gt/model.obj) with the target, component by component.

Evaluation only: says which target components are absent from the as-built model and which
as-built components are not in the target. A component counts as present when the as-built mesh
has one with the same vertex set (rounded to 0.01 mm).

Usage: python scripts/defect_labels.py <dataset_dir> [out.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pixie.data import load_mesh, target_mesh  # noqa: E402


def components(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    return mesh.split(only_watertight=False)


def key(c: trimesh.Trimesh) -> bytes:
    v = np.round(np.asarray(c.vertices) * 1e5).astype(np.int64)
    return np.unique(v, axis=0).tobytes()


def main() -> None:
    root = Path(sys.argv[1])
    out = {}
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        model = mdir.name
        tcomp = components(target_mesh(root, model))
        tkeys = [key(c) for c in tcomp]
        tset = set(tkeys)
        print(f"{model}: {len(tcomp)} target components, "
              f"extent {np.ptp(target_mesh(root, model).vertices, 0) * 100} cm")
        for sdir in sorted(p for p in (mdir / "gt").iterdir() if p.is_dir()):
            built = components(load_mesh(sdir / "model.obj"))
            bkeys = [key(c) for c in built]
            bset = set(bkeys)
            missing = [i for i, k in enumerate(tkeys) if k not in bset]
            extra = [i for i, k in enumerate(bkeys) if k not in tset]
            vol_m = sum(abs(tcomp[i].volume) for i in missing) * 1e6
            vol_e = sum(abs(built[i].volume) for i in extra) * 1e6
            ious = [json.load(open(j))["iou3d"] for j in sdir.glob("*.json") if "iou3d" in json.load(open(j))]
            print(f"  {sdir.name:32s} built {len(built):4d}  missing {len(missing):3d} ({vol_m:6.2f} cm3)  "
                  f"extra {len(extra):3d} ({vol_e:6.2f} cm3)  iou3d {np.mean(ious) if ious else float('nan'):.3f}")
            out[f"{model}/{sdir.name}"] = {"missing": missing, "n_extra": len(extra),
                                           "missing_cm3": vol_m, "extra_cm3": vol_e}
    if len(sys.argv) > 2:
        json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
