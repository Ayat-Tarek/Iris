
import re
from pathlib import Path
import cv2
import numpy as np
from typing import Tuple

def ensure_dir(p: str) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

def next_index(out_dir: Path) -> int:
    """
    Scan for files like image{N}_segmented.png and return N+1 (or 1 if none).
    """
    patt = re.compile(r"image(\d+)_segmented\.png$", re.IGNORECASE)
    max_idx = 0
    for f in out_dir.glob("image*_segmented.png"):
        m = patt.search(f.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1

def save_images(index:int, segmented_bgr: np.ndarray, normalized_gray: np.ndarray, out_dir: Path) -> Tuple[str, str]:
    seg_path = out_dir / f"image{index}_segmented.png"
    nor_path = out_dir / f"image{index}_normalized.png"
    cv2.imwrite(str(seg_path), segmented_bgr)
    cv2.imwrite(str(nor_path), normalized_gray)
    return str(seg_path), str(nor_path)
