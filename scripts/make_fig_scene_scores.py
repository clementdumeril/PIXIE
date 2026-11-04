"""Scene-level assembly check: per-scene scores and AUROC, from docs/defect_scenes.csv.

Usage: python scripts/make_fig_scene_scores.py [docs/defect_scenes.csv] [docs/scene_scores.svg]

Prints the scene AUROC (defective = assembly_mistake*, correct = color / clutter) and writes a strip plot
of the scores per scene type. Reads only the committed CSV; no dataset or GPU needed.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("color", "clutter", "assembly_mistake", "assembly_mistake_with_clutter")
LABELS = ("color\n(correct)", "clutter\n(correct)", "mistake\n(defective)", "mistake + clutter\n(defective)")


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a defective scene scores higher than a correct one (ties count half)."""
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "defect_scenes.csv"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs" / "scene_scores.svg"
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    score = {k: np.array([float(r["score"]) for r in rows if r["kind"] == k]) for k in KINDS}
    pos = np.concatenate([score[k] for k in KINDS if k.startswith("assembly_mistake")])
    neg = np.concatenate([score[k] for k in KINDS if not k.startswith("assembly_mistake")])
    a = auroc(pos, neg)
    print(f"{len(pos)} defective / {len(neg)} correct scenes, scene AUROC {a:.3f}")

    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(7.0, 3.6), facecolor="white")
    rng = np.random.default_rng(0)
    for i, k in enumerate(KINDS):
        defective = k.startswith("assembly_mistake")
        x = i + rng.uniform(-0.12, 0.12, len(score[k]))
        ax.scatter(x, score[k], s=36, marker="^" if defective else "o",
                   color="#c2410c" if defective else "#2563eb", edgecolor="black", linewidth=0.4, zorder=3)
        ax.hlines(np.median(score[k]), i - 0.25, i + 0.25, color="black", linewidth=1.2, zorder=2)
    ax.set_xticks(range(len(KINDS)), LABELS)
    ax.set_ylabel("scene score\n(unconfirmed edges, worst part)")
    ax.set_title(f"Assembly check, one point per scene (n = {len(rows)}), AUROC {a:.3f}", fontsize=11)
    ax.grid(axis="y", color="#e5e7eb", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white", metadata={"Date": None} if out.suffix == ".svg" else None)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
