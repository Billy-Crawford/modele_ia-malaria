"""Configuration centrale du pipeline de détection Malaria AI (option B : détection d'objets).

Toutes les constantes partagées entre préparation des données, entraînement, évaluation,
inférence et application mobile sont définies ICI et nulle part ailleurs.
Toute modification de géométrie (échelle, tuile, recouvrement) doit être répercutée
dans l'application Flutter (voir docs/MOBILE_CONTRACT.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DETECTION_ROOT = PACKAGE_ROOT.parent
MODEL_REPO_ROOT = DETECTION_ROOT.parent

# ---------------------------------------------------------------------------
# Chemins par défaut (relatifs au dépôt `model/`)
# ---------------------------------------------------------------------------
DEFAULT_RAW_DIR = MODEL_REPO_ROOT / "data" / "external" / "nlm_pf" / "extracted"
DEFAULT_COCO_DIR = MODEL_REPO_ROOT / "data" / "external" / "mira_coco"
DEFAULT_PROCESSED_DIR = MODEL_REPO_ROOT / "data" / "processed" / "nlm_pf_yolo"
DEFAULT_RUNS_DIR = DETECTION_ROOT / "runs"
DEFAULT_REPORTS_DIR = DETECTION_ROOT / "reports"

# ---------------------------------------------------------------------------
# Classes (l'ordre est un CONTRAT avec le modèle exporté et l'app mobile)
# ---------------------------------------------------------------------------
CLASS_NAMES: tuple[str, ...] = ("rbc_uninfected", "rbc_infected", "wbc")
CLASS_NAMES_FR: tuple[str, ...] = ("Globule rouge sain", "Globule rouge infecté", "Globule blanc")
CLS_UNINFECTED = 0
CLS_INFECTED = 1
CLS_WBC = 2
NUM_CLASSES = len(CLASS_NAMES)

# Catégories COCO (annotations MIRA-Vision) -> classes YOLO. 4 = "ambiguous" -> zone ignorée.
MIRA_TO_YOLO: dict[int, int] = {1: CLS_UNINFECTED, 2: CLS_INFECTED, 3: CLS_WBC}
MIRA_AMBIGUOUS_ID = 4


@dataclass(frozen=True)
class GeometryConfig:
    """Géométrie canonique : on ramène chaque image à une échelle où un globule rouge ~62 px.

    - Les images NLM font 5312x2988 avec des hématies de ~125 px -> scale 0.5.
    - Tuiles 640x640 (taille d'entrée native YOLO) avec un recouvrement de 160 px, supérieur
      au plus grand objet (globule blanc p95 ~145 px à l'échelle 0.5). Garantie : tout objet
      de taille <= recouvrement est ENTIÈREMENT visible dans au moins une tuile.
    """

    target_rbc_px: float = 62.0
    reference_scale: float = 0.5
    tile: int = 640
    overlap: int = 160
    min_visible: float = 0.5  # fraction minimale visible pour garder une boîte tronquée (entraînement)
    edge_margin: int = 4  # px : une détection touchant un bord INTERNE de tuile est rejetée (inférence)


@dataclass(frozen=True)
class PostprocessConfig:
    """Post-traitement de référence (doit être reproduit à l'identique côté mobile)."""

    score_thr: float = 0.25  # "cellness" = max des probabilités de classe
    inf_thr: float = 0.50  # probabilité minimale pour déclarer une hématie infectée
    nms_iou: float = 0.50
    max_candidates: int = 4000  # par tuile, avant NMS


GEOMETRY = GeometryConfig()
POSTPROCESS = PostprocessConfig()

# Référence clinique (OMS) : la parasitémie sur frottis mince se compte sur >= 1000 hématies.
WHO_MIN_RBC_THIN_SMEAR = 1000
# OMS (Guidelines for malaria, critères de paludisme grave) : hyperparasitémie P. falciparum > 10 %.
WHO_HYPERPARASITEMIA_PCT = 10.0
