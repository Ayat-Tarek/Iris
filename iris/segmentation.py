# segmentation.py  — Pure Wildes-style (edge → Hough voting for circles)
import cv2
import numpy as np
from typing import Tuple, Optional

# =========================
# Tunables
# =========================
GAUSS_K = (5, 5)         # small blur to stabilize edges
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

CANNY_LO = 30            # you can autoset from median if needed
CANNY_HI = 120

PUPIL_R_MIN_FRAC = 1/50  # pupil radius range as fractions of min(H,W)
PUPIL_R_MAX_FRAC = 1/6
IRIS_R_MIN_SCALE = 1.5   # iris radius starts at >= 1.5 * r_pupil
IRIS_R_MAX_FRAC = 0.48   # as fraction of min(H,W)

R_STEP = 1               # radius step (pixels)
CENTER_NEAR_PUPIL = 3    # (pixels) constrain iris center near pupil center
EDGE_SUBSAMPLE = 1       # process every Nth edge pixel for speed (1 = use all)
VOTE_BOTH_NORMAL_DIRS = True  # vote both inward/outward normals for robustness

# =========================
# Helpers
# =========================
def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def _enhance(gray: np.ndarray) -> np.ndarray:
    g = _to_gray(gray)
    g = cv2.GaussianBlur(g, GAUSS_K, 1.0)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    g = clahe.apply(g)
    # Light specular inpaint (optional)
    spec = cv2.inRange(g, 240, 255)
    if np.any(spec):
        g = cv2.inpaint(g, spec, 3, cv2.INPAINT_TELEA)
    return g

def _binary_edges(enhanced: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(enhanced, CANNY_LO, CANNY_HI, L2gradient=True)
    return edges

def _gradients(enhanced: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy

def _radius_range_for_pupil(h: int, w: int) -> np.ndarray:
    m = min(h, w)
    rmin = max(6, int(np.floor(m * PUPIL_R_MIN_FRAC)))
    rmax = max(rmin + 5, int(np.ceil(m * PUPIL_R_MAX_FRAC)))
    return np.arange(rmin, rmax + 1, R_STEP, dtype=int)

def _radius_range_for_iris(h: int, w: int, r_pupil: int) -> np.ndarray:
    m = min(h, w)
    rmin = max(r_pupil + 8, int(np.ceil(IRIS_R_MIN_SCALE * r_pupil)))
    rmax = max(rmin + 8, int(np.floor(m * IRIS_R_MAX_FRAC)))
    return np.arange(rmin, rmax + 1, R_STEP, dtype=int)

def _mean_inside(gray: np.ndarray, cx: int, cy: int, r: int) -> float:
    H, W = gray.shape[:2]
    Y, X = np.ogrid[:H, :W]
    mask = (X - cx)**2 + (Y - cy)**2 <= r*r
    if not np.any(mask):
        return 255.0
    return float(np.mean(gray[mask]))

def _vote_centers(edges: np.ndarray,
                  gx: np.ndarray,
                  gy: np.ndarray,
                  radii: np.ndarray,
                  center_hint: Optional[Tuple[int, int]] = None,
                  center_tol: int = 0) -> Tuple[int, int, int, float]:
    """
    Core Wildes Hough voting:
      - edges: binary edge map (uint8 0/255)
      - gx, gy: gradients (float32)
      - radii: 1D array of candidate radii
      - center_hint: (cx,cy) to restrict votes near this point (for iris stage)
      - center_tol: allowed +/- pixels from hint (0 => no restriction)
    Returns best (cx, cy, r, score).
    """
    H, W = edges.shape[:2]
    # Accumulator per radius: to save memory, we process one radius at a time.
    # (You can optimize by tiling or GPU later.)
    ys, xs = np.nonzero(edges)
    if EDGE_SUBSAMPLE > 1 and len(xs) > 0:
        xs = xs[::EDGE_SUBSAMPLE]
        ys = ys[::EDGE_SUBSAMPLE]

    best_score = -1.0
    best = (W//2, H//2, radii[0])

    # Precompute gradient directions (unit normals)
    # Normal direction = +/- (gx, gy) normalized
    mag = np.hypot(gx, gy) + 1e-6
    nx = gx / mag
    ny = gy / mag

    # Clamp function
    def _in_bounds(xc, yc):
        return (0 <= xc < W) and (0 <= yc < H)

    for r in radii:
        acc = np.zeros((H, W), dtype=np.uint16)

        # Vote from each edge point
        for x, y in zip(xs, ys):
            fx = nx[y, x]
            fy = ny[y, x]
            if fx == 0.0 and fy == 0.0:
                continue

            # candidate center along normal (outward)
            cx1 = int(round(x - r * fx))
            cy1 = int(round(y - r * fy))

            # optional inward vote (helps when gradient orientation is noisy)
            if VOTE_BOTH_NORMAL_DIRS:
                cx2 = int(round(x + r * fx))
                cy2 = int(round(y + r * fy))
            else:
                cx2 = cx1
                cy2 = cy1

            # restrict near hint if provided (iris stage)
            if center_hint is not None and center_tol > 0:
                hx, hy = center_hint
                if abs(cx1 - hx) > center_tol or abs(cy1 - hy) > center_tol:
                    pass  # skip this vote if out of band
                else:
                    if _in_bounds(cx1, cy1):
                        acc[cy1, cx1] += 1
                if VOTE_BOTH_NORMAL_DIRS:
                    if abs(cx2 - hx) <= center_tol and abs(cy2 - hy) <= center_tol and _in_bounds(cx2, cy2):
                        acc[cy2, cx2] += 1
            else:
                if _in_bounds(cx1, cy1):
                    acc[cy1, cx1] += 1
                if VOTE_BOTH_NORMAL_DIRS and _in_bounds(cx2, cy2):
                    acc[cy2, cx2] += 1

        # Peak in accumulator is the best center for this radius
        idx = np.argmax(acc)
        peak = int(acc.flat[idx])
        cy, cx = divmod(idx, W)

        # Score = peak normalized by number of edge samples (roughly)
        denom = max(1, len(xs))
        score = peak / float(denom)

        if score > best_score:
            best_score = score
            best = (cx, cy, r)

    cx, cy, r = best
    return int(cx), int(cy), int(r), float(best_score)

# =========================
# Public API (unchanged)
# =========================
def find_pupil(gray: np.ndarray) -> Tuple[int, int, int]:
    """
    Wildes-style: edges -> gradient-guided circular Hough voting for (cx,cy,r).
    Picks radius with highest center-accumulator peak; tiebreak by darkest interior.
    """
    enh = _enhance(gray)
    edges = _binary_edges(enh)
    gx, gy = _gradients(enh)

    H, W = enh.shape[:2]
    pupil_radii = _radius_range_for_pupil(H, W)

    # Vote for centers across candidate radii
    cx, cy, r, score = _vote_centers(edges, gx, gy, pupil_radii)

    # Optional tie-break: among +/- 2 pixels around (cx,cy,r), pick darkest interior
    # (robust to tiny off-by-one peaks)
    best_triplet = (cx, cy, r)
    best_mu = _mean_inside(enh, cx, cy, r)
    for dcx in (-1, 0, 1):
        for dcy in (-1, 0, 1):
            for dr in (-1, 0, 1):
                rc = r + dr
                xc = cx + dcx
                yc = cy + dcy
                if rc <= 4:
                    continue
                mu = _mean_inside(enh, xc, yc, rc)
                if mu < best_mu:
                    best_mu = mu
                    best_triplet = (xc, yc, rc)

    cx, cy, r = best_triplet
    return int(cx), int(cy), int(r)

def find_iris(gray: np.ndarray, cx: int, cy: int, r_pupil: int) -> int:
    """
    Wildes-style: same center (within a small tolerance). Search radii > pupil
    and pick the one whose center votes peak near (cx,cy).
    """
    enh = _enhance(gray)
    edges = _binary_edges(enh)
    gx, gy = _gradients(enh)

    H, W = enh.shape[:2]
    iris_radii = _radius_range_for_iris(H, W, r_pupil)

    # Constrain center near pupil center (same eye center model)
    cx_i, cy_i, r_i, score = _vote_centers(
        edges, gx, gy, iris_radii,
        center_hint=(cx, cy),
        center_tol=CENTER_NEAR_PUPIL
    )

    # Optionally refine r by small local search (keep same center)
    best_r = r_i
    best_score = score
    for dr in (-2, -1, 1, 2):
        rr = r_i + dr
        if rr <= r_pupil + 4 or rr >= int(min(H, W) * IRIS_R_MAX_FRAC):
            continue
        # Recompute a mini vote at fixed center: count edge hits on this circle
        # (fast check)
        thetas = np.linspace(0, 2*np.pi, 720, endpoint=False)
        xs = np.clip(np.round(cx + rr * np.cos(thetas)).astype(int), 0, W-1)
        ys = np.clip(np.round(cy + rr * np.sin(thetas)).astype(int), 0, H-1)
        hits = (edges[ys, xs] > 0).sum() / 720.0
        if hits > best_score:
            best_score = hits
            best_r = rr

    return int(best_r)

def draw_segmentation_overlay(bgr: np.ndarray, cx:int, cy:int, r_pupil:int, r_iris:int) -> np.ndarray:
    out = bgr.copy()
    cv2.circle(out, (int(cx), int(cy)), int(r_iris), (255, 0, 0), 2)   # blue iris
    cv2.circle(out, (int(cx), int(cy)), int(r_pupil), (0, 255, 0), 2)  # green pupil
    return out
