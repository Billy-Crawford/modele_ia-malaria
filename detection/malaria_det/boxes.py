"""Opérations géométriques sur boîtes (format xyxy). NumPy pur, sans dépendance au framework.

Ce module est la RÉFÉRENCE algorithmique pour l'implémentation Dart côté mobile.
"""

from __future__ import annotations

import numpy as np


def as_boxes(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1, 4)


def area(b: np.ndarray) -> np.ndarray:
    b = as_boxes(b)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def iou_matrix(a, b) -> np.ndarray:
    """IoU entre deux ensembles de boîtes : (N,4) x (M,4) -> (N,M)."""
    a, b = as_boxes(a), as_boxes(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def ioa_matrix(a, b) -> np.ndarray:
    """Intersection sur aire de `a` : (N,M). Utile pour les zones ignorées."""
    a, b = as_boxes(a), as_boxes(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return inter / np.maximum(area(a)[:, None], 1e-12)


def xywh_to_xyxy(xywh) -> np.ndarray:
    x = np.asarray(xywh, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(x)
    out[:, 0] = x[:, 0] - x[:, 2] / 2
    out[:, 1] = x[:, 1] - x[:, 3] / 2
    out[:, 2] = x[:, 0] + x[:, 2] / 2
    out[:, 3] = x[:, 1] + x[:, 3] / 2
    return out


def nms_cluster(boxes, scores, iou_thr: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """NMS glouton indépendant de la classe, qui retourne aussi les GROUPES supprimés.

    Retourne (keep, clusters) où clusters[k] contient les indices (dans `boxes`) de toutes les
    boîtes fusionnées dans keep[k] (y compris keep[k] lui-même).

    Pourquoi des groupes : on détecte d'abord la *cellule* (localisation), puis on décide de son
    statut infecté/sain en agrégeant les probabilités de toutes les propositions qui la couvrent.
    Cela évite qu'une proposition "saine" à score élevé masque une proposition "infectée" sur la
    même cellule — un faux négatif est cliniquement plus grave qu'un faux positif.
    """
    boxes = as_boxes(boxes)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = len(boxes)
    if n == 0:
        return np.zeros(0, dtype=np.int64), []
    order = np.argsort(-scores, kind="stable")
    b = boxes[order]
    areas = area(b)
    alive = np.ones(n, dtype=bool)
    keep: list[int] = []
    clusters: list[np.ndarray] = []
    for i in range(n):
        if not alive[i]:
            continue
        alive[i] = False
        rest = np.nonzero(alive[i + 1 :])[0] + i + 1
        members = [i]
        if len(rest):
            ix1 = np.maximum(b[i, 0], b[rest, 0])
            iy1 = np.maximum(b[i, 1], b[rest, 1])
            ix2 = np.minimum(b[i, 2], b[rest, 2])
            iy2 = np.minimum(b[i, 3], b[rest, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-12)
            sup = rest[iou >= iou_thr]
            alive[sup] = False
            members.extend(sup.tolist())
        keep.append(int(order[i]))
        clusters.append(order[np.asarray(members, dtype=np.int64)])
    return np.asarray(keep, dtype=np.int64), clusters
