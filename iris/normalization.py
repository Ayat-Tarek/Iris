import cv2
import numpy as np
from typing import Tuple

def rubber_sheet(gray: np.ndarray,
                 cx:int, cy:int, r_pupil:int, r_iris:int,
                 radial_res:int = 64, angular_res:int = 360) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classic rubber-sheet: map annulus [pupil, iris] to (radial_res x angular_res).
    Returns (normalized, mask). Normalized is uint8 grayscale.
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    thetas = np.linspace(0, 2*np.pi, angular_res, endpoint=False).astype(np.float32)
    rs = np.linspace(0, 1, radial_res, endpoint=True).astype(np.float32)[:, None]  # (R,1)

    # radii per theta
    r_line = (1 - rs) * r_pupil + rs * r_iris  # (R,1)
    xs = cx + r_line * np.cos(thetas)[None, :] # (R,T)
    ys = cy + r_line * np.sin(thetas)[None, :] # (R,T)

    # remap needs float32
    map_x = xs.astype(np.float32)
    map_y = ys.astype(np.float32)

    norm = cv2.remap(gray, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    # build a validity mask (inside image bounds)
    h, w = gray.shape[:2]
    valid = (map_x >= 0) & (map_x < w-1) & (map_y >= 0) & (map_y < h-1)
    mask = (valid.astype(np.uint8)) * 255
    return norm, mask
