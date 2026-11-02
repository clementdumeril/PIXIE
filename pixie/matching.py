"""SuperPoint + LightGlue with the MINIMA cross-modal weights (MINIMA's `sp_lg` setting)."""

from __future__ import annotations

from pathlib import Path

import torch
from lightglue import LightGlue, SuperPoint

N_LAYERS = 9


def superpoint(device, max_keypoints: int = 2048) -> SuperPoint:
    return SuperPoint(max_num_keypoints=max_keypoints, detection_threshold=0.0005,
                      nms_radius=4, remove_borders=4).eval().to(device)


def load(ckpt: Path, device, max_keypoints: int = 2048):
    extractor = superpoint(device, max_keypoints)
    matcher = LightGlue(features="superpoint", depth_confidence=0.95, width_confidence=0.99,
                        filter_threshold=0.1, weights=None).eval().to(device)
    # MINIMA ships the old LightGlue state-dict layout; same renaming as its load_model.py
    state = torch.load(ckpt, map_location=device)
    for i in range(N_LAYERS):
        for a, b in ((f"self_attn.{i}", f"transformers.{i}.self_attn"),
                     (f"cross_attn.{i}", f"transformers.{i}.cross_attn")):
            state = {k.replace(a, b): v for k, v in state.items()}
    missing, unexpected = matcher.load_state_dict(state, strict=False)
    assert set(missing) <= {"confidence_thresholds"} and not unexpected, (missing, unexpected)
    return extractor, matcher
