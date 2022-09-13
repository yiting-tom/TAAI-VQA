from pathlib import Path
from typing import Final

ROOT: Final[Path] = Path(__file__).parent.parent
LOGS: Final[Path] = ROOT / 'logs'
DATA: Final[Path] = ROOT / 'data'
COCO: Final[Path] = DATA / 'COCO'