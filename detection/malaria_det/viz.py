"""Visualisation : dessin des boîtes (vérité terrain ou prédictions) pour contrôle qualité."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from malaria_det.config import CLASS_NAMES

# BGR — infecté en rouge vif (attire l'œil), sain en vert, globule blanc en bleu
COLORS = {0: (60, 180, 60), 1: (0, 0, 255), 2: (255, 120, 0)}


def draw_boxes(img: np.ndarray, boxes_cls, scores=None, thickness: int = 2, label: bool = True) -> np.ndarray:
    out = img.copy()
    bc = np.asarray(boxes_cls, dtype=np.float64).reshape(-1, 5)
    for i, (x1, y1, x2, y2, c) in enumerate(bc):
        color = COLORS.get(int(c), (200, 200, 200))
        th = thickness * (2 if int(c) == 1 else 1)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, th)
        if label and (int(c) != 0 or scores is not None):
            txt = CLASS_NAMES[int(c)].replace("rbc_", "")
            if scores is not None:
                txt += f" {scores[i]:.2f}"
            cv2.putText(out, txt, (int(x1), max(0, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def read_yolo_labels(path: Path, w: int, h: int) -> np.ndarray:
    rows = [l.split() for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    out = []
    for c, cx, cy, bw, bh in rows:
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        out.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, int(c)])
    return np.asarray(out, dtype=np.float64).reshape(-1, 5)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dessine les labels YOLO d'une tuile")
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    img = cv2.imread(str(a.image))
    lbl = Path(str(a.image).replace("/images/", "/labels/")).with_suffix(".txt")
    boxes = read_yolo_labels(lbl, img.shape[1], img.shape[0])
    cv2.imwrite(str(a.out), draw_boxes(img, boxes))
    print(f"{len(boxes)} boîtes -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
