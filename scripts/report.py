"""Summarise run_eval CSVs per scenario type, in the paper's metrics (median ADD, share < 1 cm).

Usage: python scripts/report.py out/full_both.csv [out/full_normal.csv ...]
"""

from __future__ import annotations

import csv
import sys

import numpy as np

KINDS = ("color", "clutter", "assembly_mistake", "assembly_mistake_with_clutter")
PAPER = {"all": (0.36, 73.5)}   # PIXIE on this dataset: median ADD (cm), % < 1 cm


def line(name: str, rows: list[dict]) -> str:
    add = np.array([float(r["add_cm"]) for r in rows])          # nan = no pose, counts as failure
    rot = np.array([float(r["rot_deg"]) for r in rows])
    sec = np.array([float(r["seconds"]) for r in rows])
    ok = np.nan_to_num(add, nan=np.inf)
    return (f"  {name:30s} {len(rows):4d}  {np.median(ok):7.3f}  {100 * np.mean(ok < 1):6.1f}  "
            f"{100 * np.mean(ok < 2):6.1f}  {np.nanmedian(rot):6.2f}  {100 * np.mean(np.nan_to_num(rot, nan=180) > 90):5.1f}"
            f"  {np.mean(sec):5.1f}")


def main() -> None:
    for path in sys.argv[1:]:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        print(f"\n{path}")
        print(f"  {'':30s} {'n':>4s}  {'ADD med':>7s}  {'<1cm %':>6s}  {'<2cm %':>6s}  {'rot med':>6s}  {'>90°%':>5s}  {'s/img':>5s}")
        print(line("all", rows) + f"   (paper: {PAPER['all'][0]} cm, {PAPER['all'][1]} %)")
        for k in KINDS:
            print(line(k, [r for r in rows if r["kind"] == k]))
        for m in sorted({r["model"] for r in rows}):
            print(line(m, [r for r in rows if r["model"] == m]))


if __name__ == "__main__":
    main()
