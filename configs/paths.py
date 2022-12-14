from pathlib import Path
from typing import Final

# ========================= VQA =========================
d_ROOT: Final[Path] = Path(__file__).parent.parent
d_LOGS: Final[Path] = d_ROOT / "logs"
d_DATA: Final[Path] = d_ROOT / "data"

# ===================== VQA & VQA-E ======================
d_VQA: Final[Path] = d_DATA / "vqa"
d_VQAE: Final[Path] = d_DATA / "vqa-e"

# ========================= COCO  ========================
d_COCO: Final[Path] = d_DATA / "COCO"
d_COCO_IMAGE: Final[Path] = d_COCO / "images"
d_COCO_FEATURE: Final[Path] = d_COCO / "feature_36"
d_COCO_BBOX: Final[Path] = d_COCO / "bbox_36"
d_COCO_GRAPH: Final[Path] = d_COCO / "graph_36"

# ========================= GloVe ========================
d_GLOVE: Final[Path] = d_DATA / "glove.6B"

# ======================= Generated ======================
d_ANNOTATIONS: Final[Path] = d_DATA / "annotations"
f_CANDIDATE_ANSWERS: Final[Path] = d_DATA / "candidate_answers.txt"
f_GOLVE_VOCABULARIES: Final[Path] = d_DATA / "glove_vocabularies.txt"
