# PIXIE reimplemented: 6D pose of a LEGO assembly from RGB photos, plus an assembly check

![Overview: photo and CAD model, board-based localisation, coarse and fine pose from reference renders, CAD edges checked against the photo](docs/figures/overview.png)

A from-scratch reimplementation of PIXIE (Carl Zeiss AG): the zero-shot 6D pose of a LEGO assembly from an RGB photo and its CAD model, with no training on the object (SuperPoint + LightGlue matching against renders of the CAD model, then PnP).

- Pose: 0.32 cm median ADD on the 695 photos of the public PIXIE dataset, without a given bounding box, against 0.36 cm in the paper, which assumes the 2D detection is given.
- Assembly check, added here: it flags misassembled scenes at 0.92 AUROC on 48 real scenes when all views of a scene are pooled (0.85 from a single photo). It was tuned only on synthetic defects.

[Paper (arXiv 2607.16015)](https://arxiv.org/abs/2607.16015) · [Dataset](https://github.com/ZEISS/PIXIE-dataset) · [Detailed results](docs/RESULTS.md) · [Method and settings](docs/method.md)

## How it works

- Render the CAD model from 968 viewpoints as normal and depth maps, with a small PyTorch rasteriser.
- Find the work area from the ArUco board on the table and estimate a coarse pose there. Ground-truth poses are never read.
- Match the photo crop to the renders with SuperPoint + LightGlue (MINIMA weights), then solve RANSAC PnP on the best matches.
- Assembly check: render the CAD model at the estimated pose and test, part by part, whether the photo has a matching edge. A missing brick leaves its edges unconfirmed.
- The check's 4 parameters were tuned on synthetic defects and frozen before any real image was checked.

## Run it

Run on one NVIDIA RTX 4060 (8 GB). The evaluation scripts require CUDA.

```bash
pip install torch opencv-contrib-python trimesh scipy numpy matplotlib "lightglue @ git+https://github.com/cvg/LightGlue"
```

Data and weights: the [PIXIE dataset](https://github.com/ZEISS/PIXIE-dataset) and MINIMA's `sp_lg` LightGlue checkpoint from [LSXI7/MINIMA](https://github.com/LSXI7/MINIMA).

```bash
python scripts/run_eval.py <dataset>/dataset <minima_lightglue.pth> out/board_both.csv --modes normal depth --box board   # pose, about 20 s per image
python scripts/report.py out/board_both.csv                                       # ADD metrics per scenario and model
python scripts/tune_defects.py <dataset>/dataset out/tune --n 120 --seed 0        # synthetic defects -> out/tune/defect_params.json
python scripts/eval_defects.py <dataset>/dataset out/board_both.csv out/tune/defect_params.json out/defects_board   # scene, image, part AUROC
python scripts/make_fig_scene_scores.py                                           # docs/scene_scores.svg, no dataset or GPU needed
```

## Limitations

- 4 LEGO models. Studs make matching and edge checks easier than smooth industrial parts would.
- Clutter is the weak spot of both the pose (75 % < 1 cm with defects + clutter) and the check (image AUROC 0.76).
- Finding the object relies on this dataset's ArUco board; another setup needs another detector.
- About 20 s per image on an RTX 4060, against 3.6 s on an RTX 4090 in the paper. One run, no spread across seeds.

## Citation, credits and license

Written here: the rasteriser, reference set, pose estimator, board localisation, assembly check, synthetic-defect tuning and evaluation. Reused: the method (PIXIE paper), the dataset (ZEISS, BSD-3-Clause), and pretrained SuperPoint, LightGlue and MINIMA weights. Nothing is trained in this repository.

Please cite the original method:

```bibtex
@misc{jungemeyer2026pixie,
  author = {Jungemeyer, Leon and Maga{\~n}a, Alejandro and Mohan, Gautham and Karl, Matthias and Werdehausen, Daniel},
  title  = {{PIXIE}: A Zero-Shot texture-invariant 6D pose estimation framework for unseen objects with assembly defects},
  year   = {2026},
  eprint = {2607.16015},
  archivePrefix = {arXiv}
}
```

PIXIE method and dataset: Carl Zeiss AG (dataset under BSD-3-Clause; the images in `docs/` come from it). SuperPoint: Magic Leap (its weights carry a non-commercial licence). LightGlue: ETH Zurich CVG (Apache-2.0). MINIMA weights: their authors (repository under Apache-2.0).

Code: MIT, see [`LICENSE`](LICENSE). Dataset images and third-party weights keep their own licences. Clément Dumeril · [clement.dumeril.net](https://clement.dumeril.net)
