from pathlib import Path
import cv2
import numpy as np
from typing import Tuple, Dict, List
import json
import os

from .segmentation import find_pupil, find_iris, draw_segmentation_overlay
from .normalization import rubber_sheet
from .utils import ensure_dir, next_index, save_images
from .feature_extraction import encode_iris


def preprocess_iris_image(image_path: str) -> np.ndarray:
    """
    Grayscale → median blur → CLAHE → inpaint reflections → normalize [0,255].
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    # reflection cleanup
    reflection_mask = cv2.inRange(enhanced, 240, 255)
    cleaned = cv2.inpaint(enhanced, reflection_mask, 3, cv2.INPAINT_TELEA)
    normalized = cv2.normalize(cleaned, None, 0, 255, cv2.NORM_MINMAX)
    return normalized


def run_full_pipeline(
    image_path: str,
    out_root: str = "outputs",
    radial_res: int = 64,
    angular_res: int = 360,
) -> Tuple[str, str, Tuple[int, int, int, int]]:
    """
    Full end-to-end pipeline for ONE image:
      preprocessing → segmentation → normalization → iris code.
    Returns:
        (segmented_path, normalized_path, (cx, cy, r_pupil, r_iris))
    """

    out_dir = ensure_dir(out_root)

    # --- Step 1: Preprocess image ---
    pre = preprocess_iris_image(image_path)

    # --- Step 2: Segmentation (Wildes) ---
    cx, cy, r_pupil = find_pupil(pre)
    r_iris = find_iris(pre, cx, cy, r_pupil)

    # --- Step 3: Visualization overlay ---
    pre_bgr = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)
    overlay = draw_segmentation_overlay(pre_bgr, cx, cy, r_pupil, r_iris)

    # --- Step 4: Normalization (rubber sheet) ---
    norm, _ = rubber_sheet(
        pre, cx, cy, r_pupil, r_iris,
        radial_res=radial_res,
        angular_res=angular_res,
    )

    # --- Step 5: Feature extraction (iris code) ---
    iris_code = encode_iris(norm)

    # --- Step 6: Save results (so GUI can display them) ---
    stem = Path(image_path).stem
    seg_path = out_dir / f"{stem}_segmented.png"
    norm_path = out_dir / f"{stem}_normalized.png"

    cv2.imwrite(str(seg_path), overlay)
    cv2.imwrite(str(norm_path), norm)

    # optional: save code to a temporary json for debugging
    # Path(out_dir, f"{stem}_code.npy").write_bytes(iris_code.tobytes())

    return str(seg_path), str(norm_path), (cx, cy, r_pupil, r_iris)


def store_iris_code(iris_code, path: str = "iris_codes/iris_codes.json", person_id: str | None = None):
    """
    Save an iris code to a JSON database file.

    If `person_id` is provided, use it as JSON key (e.g., "000_L_000").
    Otherwise an automatic person_XXX id is generated (legacy behavior).

    Returns the key used to store the code.
    """
    path_p = Path(path)
    path_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "r") as fh:
            db = json.load(fh)
    except FileNotFoundError:
        db = {}
    except json.JSONDecodeError:
        print(f"Warning: JSON decode error while reading {path}; starting with an empty database.")
        db = {}

    if person_id:
        key = str(person_id)
        if key in db:
            print(f"Warning: overwriting existing iris code for '{key}' in {path}")
    else:
        if len(db) == 0:
            next_id = 1
        else:
            existing_ids = [int(k.split("_")[1]) for k in db.keys() if k.startswith("person_")]
            next_id = (max(existing_ids) + 1) if existing_ids else 1
        key = f"person_{next_id:03d}"

    db[key] = iris_code.tolist()

    with open(path, "w") as f:
        json.dump(db, f, indent=2)

    print(f"Iris code saved as '{key}' in {path}")
    return key

def build_iris_codes_dataset_per_eye(
    dataset_dir: str = "CASIA-Iris-Thousand",
    train_codes_path: str = "iris_codes/iris_codes_train.json",
    test_codes_path: str = "iris_codes/iris_codes_test.json",
    subjects_limit: int = 100,
    gallery_k_per_eye: int = 8,
    radial_res: int = 64,
    angular_res: int = 360,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Per-eye builder with deterministic gallery/probe split:
      - First `subjects_limit` subjects (sorted).
      - Eyes 'L' and 'R' handled separately.
      - First `gallery_k_per_eye` images -> TRAIN (gallery).
      - Remaining images -> TEST (probe).
    Keys stored as: "<subject>_<eye>_<idx:03d>", e.g., "000_L_000".

    Returns: mapping[subject][eye]['train'|'test'] -> list of stored keys
    """
    mapping: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    dataset_p = Path(dataset_dir)

    Path(train_codes_path).parent.mkdir(parents=True, exist_ok=True)
    Path(test_codes_path).parent.mkdir(parents=True, exist_ok=True)

    if not dataset_p.exists() or not dataset_p.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    person_dirs = sorted([p for p in dataset_p.iterdir() if p.is_dir()])[:subjects_limit]
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

    def collect_sorted_images(folder: Path) -> List[Path]:
        imgs: List[Path] = []
        if folder.exists() and folder.is_dir():
            for e in exts:
                imgs.extend(sorted(folder.glob(e)))
        return imgs

    total_train, total_test = 0, 0

    for person_dir in person_dirs:
        subj = person_dir.name
        mapping[subj] = {"L": {"train": [], "test": []},
                         "R": {"train": [], "test": []}}

        for eye in ("L", "R"):
            eye_dir = person_dir / eye
            imgs = collect_sorted_images(eye_dir)
            if not imgs:
                print(f"[skip] No images for {subj}/{eye}")
                continue

            train_imgs = imgs[:gallery_k_per_eye]
            test_imgs = imgs[gallery_k_per_eye:]
            print(f"[info] {subj}/{eye}: total={len(imgs)}  train={len(train_imgs)}  test={len(test_imgs)}")

            k = 0
            # TRAIN (gallery)
            for img_p in train_imgs:
                try:
                    pre = preprocess_iris_image(str(img_p))
                    cx, cy, r_pupil = find_pupil(pre)
                    r_iris = find_iris(pre, cx, cy, r_pupil)
                    norm, _ = rubber_sheet(pre, cx, cy, r_pupil, r_iris,
                                           radial_res=radial_res, angular_res=angular_res)
                    iris_code = encode_iris(norm)
                    key = f"{subj}_{eye}_{k:03d}"; k += 1
                    stored_key = store_iris_code(iris_code, path=train_codes_path, person_id=key)
                    mapping[subj][eye]["train"].append(stored_key)
                    total_train += 1
                except Exception as e:
                    print(f"  ! [train] {subj}/{eye} {img_p.name}: {e}")

            # TEST (probe)
            for img_p in test_imgs:
                try:
                    pre = preprocess_iris_image(str(img_p))
                    cx, cy, r_pupil = find_pupil(pre)
                    r_iris = find_iris(pre, cx, cy, r_pupil)
                    norm, _ = rubber_sheet(pre, cx, cy, r_pupil, r_iris,
                                           radial_res=radial_res, angular_res=angular_res)
                    iris_code = encode_iris(norm)
                    key = f"{subj}_{eye}_{k:03d}"; k += 1
                    stored_key = store_iris_code(iris_code, path=test_codes_path, person_id=key)
                    mapping[subj][eye]["test"].append(stored_key)
                    total_test += 1
                except Exception as e:
                    print(f"  ! [test]  {subj}/{eye} {img_p.name}: {e}")

    print("Finished per-eye gallery/probe split")
    print(f"  Subjects: {len(person_dirs)}")
    print(f"  Train (gallery): {total_train} -> {train_codes_path}")
    print(f"  Test  (probe)  : {total_test} -> {test_codes_path}")
    return mapping


def run_dataset_segmentation_only(
    dataset_dir: str,
    out_root: str = "outputs",
    radial_res: int = 64,
    angular_res: int = 360,
):
    """
    Run preprocessing + segmentation + normalization
    for ALL images in the dataset directory.

    Saves:
      - segmented overlay
      - normalized image

    NO iris codes, NO train/test split.
    """

    dataset_p = Path(dataset_dir)
    if not dataset_p.exists():
        raise FileNotFoundError(dataset_dir)

    out_dir = ensure_dir(out_root)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

    total = 0
    failed = 0

    for person_dir in sorted(dataset_p.iterdir()):
        if not person_dir.is_dir():
            continue

        for eye_dir in sorted(person_dir.iterdir()):
            if not eye_dir.is_dir():
                continue

            for ext in exts:
                for img_path in sorted(eye_dir.glob(ext)):
                    try:
                        stem = img_path.stem
                        seg_path = out_dir / f"{stem}_segmented.png"
                        norm_path = out_dir / f"{stem}_normalized.png"

                        # --- Skip if already processed (checkpoint resume) ---
                        if seg_path.exists() and norm_path.exists():
                            print(f"[skip] {img_path.name} already processed.")
                            continue

                        # --- Preprocess ---
                        pre = preprocess_iris_image(str(img_path))

                        # --- Segmentation ---
                        cx, cy, r_pupil = find_pupil(pre)
                        r_iris = find_iris(pre, cx, cy, r_pupil)

                        # --- Overlay ---
                        pre_bgr = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)
                        overlay = draw_segmentation_overlay(pre_bgr, cx, cy, r_pupil, r_iris)

                        # --- Normalization ---
                        norm, _ = rubber_sheet(
                            pre, cx, cy, r_pupil, r_iris,
                            radial_res=radial_res,
                            angular_res=angular_res,
                        )

                        # --- Save results ---
                        cv2.imwrite(str(seg_path), overlay)
                        cv2.imwrite(str(norm_path), norm)
                        total += 1

                    except Exception as e:
                        failed += 1
                        print(f"[FAIL] {img_path}: {e}")

    print("====================================")
    print("Dataset segmentation finished")
    print(f"  Total processed: {total}")
    print(f"  Failed:          {failed}")
    print(f"  Output folder:   {out_dir}")
