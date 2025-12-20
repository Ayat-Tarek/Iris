import sys
import os
import cv2
import numpy as np
import json
from scipy.spatial.distance import hamming as scipy_hamming

from pathlib import Path
from typing import Tuple, Optional

from iris.pipeline import (
    run_full_pipeline,
    preprocess_iris_image,
    run_dataset_segmentation_only,
)


from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QTextEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QTabWidget, QGroupBox, QFormLayout,
    QSizePolicy, QFrame
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from iris.feature_extraction import encode_iris


def compute_template(image_path: str) -> np.ndarray:
    """
    Compute iris code using the existing encode_iris function.
    Returns binary iris code as numpy array.
    """
    try:
        _, normalized_path, _ = run_full_pipeline(image_path, out_root="outputs")
        norm = cv2.imread(normalized_path, cv2.IMREAD_GRAYSCALE)
        if norm is None:
            raise RuntimeError("Cannot read normalized image.")
        iris_code = encode_iris(norm)
        return iris_code
    except Exception as e:
        raise RuntimeError(f"Failed to compute iris code: {str(e)}")


def compare_templates(t1: np.ndarray, t2: np.ndarray) -> Tuple[float, float]:
    """
    Compare two iris codes using Hamming distance (lower = better).
    Returns (hamming_distance, similarity_percentage).
    """
    return compute_hamming_distance(t1, t2)


def search_dataset_for_best_match(query_template: np.ndarray,
                                  codes_file: str = "iris_codes/iris_codes_train.json"):
    """
    Linear search over stored codes; returns best match tuple:
    (subject_key, "Unknown", best_hamming, best_similarity) or None.
    """
    if not os.path.exists(codes_file):
        print("❌ No saved iris codes found. Run enrollment first.")
        return None

    with open(codes_file, "r") as f:
        saved = json.load(f)

    best_subject = None
    best_hamming = 1.0
    best_similarity = 0.0

    for subject_id, template_list in saved.items():
        stored_template = np.array(template_list, dtype=np.uint8)
        hd, sim = compare_templates(query_template, stored_template)
        if hd < best_hamming:
            best_hamming = hd
            best_similarity = sim
            best_subject = subject_id

    if best_subject is None:
        return None

    # Side label not tracked here
    return best_subject, "Unknown", best_hamming, best_similarity


def compute_hamming_distance(code1: np.ndarray, code2: np.ndarray) -> Tuple[float, float]:
    """
    Hamming distance (fraction of differing bits) and similarity.
    """
    # ensure same length
    n = min(code1.size, code2.size)
    if n == 0:
        return 1.0, 0.0
    a = code1.ravel()[:n].astype(np.uint8)
    b = code2.ravel()[:n].astype(np.uint8)
    hd = float(scipy_hamming(a, b))  # in [0,1]
    similarity = max(0.0, 100.0 * (1.0 - hd))
    return round(hd, 4), round(similarity, 2)


# =========================
# GUI
# =========================
class IrisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Iris Recognition — GUI")
        self.setMinimumSize(920, 600)
        self._dataset_root = "CASIA-Iris-Thousand"  # default dataset path
        self._output_dir = "outputs"

        self.setStyleSheet(self._qss())
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        header = QLabel("Iris Recognition")
        header.setObjectName("headerLabel")
        header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(header)

        tabs = QTabWidget()
        tabs.addTab(self._build_verification_tab(), "Verification")
        tabs.addTab(self._build_identification_tab(), "Identification")
        tabs.setObjectName("mainTabs")
        layout.addWidget(tabs)

        footer = QLabel("")
        footer.setObjectName("footerLabel")
        layout.addWidget(footer)

        self.setLayout(layout)

    # -------- Verification Tab --------
    def _build_verification_tab(self):
        tab = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(18)

        left_group = QGroupBox("Verification")
        left_layout = QFormLayout()
        left_layout.setLabelAlignment(Qt.AlignLeft)
        left_layout.setFormAlignment(Qt.AlignTop)

        self.verify_subject_input = QLineEdit()
        # keys look like: 000_L_000
        self.verify_subject_input.setPlaceholderText("Enter subject key (e.g. 000_L_000)")
        left_layout.addRow("Subject key:", self.verify_subject_input)

        self.verify_image_path = QLineEdit()
        self.verify_image_path.setReadOnly(True)
        left_layout.addRow("Image file:", self.verify_image_path)

        btn_row = QHBoxLayout()
        btn_upload = QPushButton("Upload Image")
        btn_upload.clicked.connect(self._on_verify_upload)
        btn_verify = QPushButton("Run Verification")
        btn_verify.clicked.connect(self._on_run_verification)
        btn_row.addWidget(btn_upload)
        btn_row.addWidget(btn_verify)
        left_layout.addRow(btn_row)

        self.verify_result_box = QTextEdit()
        self.verify_result_box.setReadOnly(True)
        self.verify_result_box.setFixedHeight(160)
        left_layout.addRow("Result:", self.verify_result_box)

        left_group.setLayout(left_layout)
        left_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        main_layout.addWidget(left_group, 1)

        right_group = QGroupBox("Preview & Stats")
        right_layout = QVBoxLayout()
        self.verify_image_label = QLabel()
        self.verify_image_label.setFixedSize(380, 280)
        self.verify_image_label.setFrameShape(QFrame.Box)
        self.verify_image_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.verify_image_label, alignment=Qt.AlignTop)

        stats_box = QGroupBox("Scores")
        stats_layout = QFormLayout()
        self.verify_hamming_label = QLabel("-")
        self.verify_similarity_label = QLabel("-")
        self.verify_matched_subject_label = QLabel("-")
        self.verify_eer_label = QLabel("-")
        self.verify_far_label = QLabel("-")
        self.verify_frr_label = QLabel("-")
        self.verify_acc_label = QLabel("-")
        stats_layout.addRow("Hamming:", self.verify_hamming_label)
        stats_layout.addRow("Similarity %:", self.verify_similarity_label)
        stats_layout.addRow("Matched Subject:", self.verify_matched_subject_label)
        stats_layout.addRow("EER:", self.verify_eer_label)
        stats_layout.addRow("FAR:", self.verify_far_label)
        stats_layout.addRow("FRR:", self.verify_frr_label)
        stats_layout.addRow("Accuracy:", self.verify_acc_label)
        stats_box.setLayout(stats_layout)
        right_layout.addWidget(stats_box)
        right_group.setLayout(right_layout)
        right_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        main_layout.addWidget(right_group, 0)

        tab.setLayout(main_layout)
        return tab

    # -------- Identification Tab --------
    def _build_identification_tab(self):
        tab = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(18)

        left_group = QGroupBox("Identification")
        left_layout = QVBoxLayout()

        control_row = QHBoxLayout()
        self.ident_image_path = QLineEdit()
        self.ident_image_path.setReadOnly(True)
        control_row.addWidget(self.ident_image_path)

        btn_upload = QPushButton("Upload Image")
        btn_upload.clicked.connect(self._on_ident_upload)
        control_row.addWidget(btn_upload)
        left_layout.addLayout(control_row)

        btn_identify = QPushButton("Run Identification")
        btn_identify.clicked.connect(self._on_run_identification)
        left_layout.addWidget(btn_identify)

        self.ident_result_box = QTextEdit()
        self.ident_result_box.setReadOnly(True)
        self.ident_result_box.setFixedHeight(160)
        left_layout.addWidget(self.ident_result_box)

        ds_label = QLabel(f"Dataset root: {self._dataset_root}")
        ds_label.setObjectName("mutedLabel")
        left_layout.addWidget(ds_label)

        left_group.setLayout(left_layout)
        left_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        main_layout.addWidget(left_group, 1)

        right_group = QGroupBox("Best Match")
        right_layout = QVBoxLayout()

        self.ident_query_image = QLabel()
        self.ident_query_image.setFixedSize(380, 160)
        self.ident_query_image.setFrameShape(QFrame.Box)
        self.ident_query_image.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.ident_query_image)

        stats_box = QGroupBox("Match Info")
        stats_layout = QFormLayout()
        self.ident_match_subject_label = QLabel("-")
        self.ident_match_side_label = QLabel("-")
        self.ident_match_hamming_label = QLabel("-")
        self.ident_match_similarity_label = QLabel("-")
        self.ident_eer_label = QLabel("-")
        self.ident_far_label = QLabel("-")
        self.ident_frr_label = QLabel("-")
        self.ident_acc_label = QLabel("-")
        stats_layout.addRow("Subject:", self.ident_match_subject_label)
        stats_layout.addRow("Hamming:", self.ident_match_hamming_label)
        stats_layout.addRow("Similarity %:", self.ident_match_similarity_label)
        stats_layout.addRow("EER:", self.ident_eer_label)
        stats_layout.addRow("FAR:", self.ident_far_label)
        stats_layout.addRow("FRR:", self.ident_frr_label)
        stats_layout.addRow("Accuracy:", self.ident_acc_label)
        stats_box.setLayout(stats_layout)
        right_layout.addWidget(stats_box)

        right_group.setLayout(right_layout)
        right_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        main_layout.addWidget(right_group, 0)

        tab.setLayout(main_layout)
        return tab

    # ---------- Handlers ----------
    def _on_verify_upload(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select iris image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.verify_image_path.setText(path)
            try:
                seg_path, norm_path, _ = run_full_pipeline(path, out_root=self._output_dir)
                # show preprocessed+segmented overlay
                self._display_pixmap(self.verify_image_label, seg_path)
                self._verify_last_normalized = norm_path
                self.verify_result_box.setPlainText(
                    f"Saved:\n  Segmented → {seg_path}\n  Normalized → {norm_path}"
                )
            except Exception as e:
                self.verify_result_box.setPlainText(f"Segmentation/normalization failed: {e}")

    def _on_run_verification(self):
        subject = self.verify_subject_input.text().strip()  # e.g., 000_L_000
        img_path = self.verify_image_path.text().strip()
        codes_dir = "iris_codes"
        if not subject:
            self.verify_result_box.setPlainText("Please enter a subject key (e.g., 000_L_000).")
            return
        if not img_path or not os.path.exists(img_path):
            self.verify_result_box.setPlainText("Please upload a valid image file.")
            return

        try:
            # Load gallery (train) DB
            db_path = os.path.join(codes_dir, "iris_codes_train.json")
            try:
                with open(db_path, "r") as f:
                    db = json.load(f)
            except FileNotFoundError:
                self.verify_result_box.setPlainText("Iris code database not found. Please enroll subjects first.")
                return

            if subject not in db:
                self.verify_result_box.setPlainText(
                    f"Subject '{subject}' not found in '{db_path}'.")
                self.verify_hamming_label.setText("-")
                self.verify_similarity_label.setText("-")
                self.verify_matched_subject_label.setText("Not found")
                return

            # Compute query iris code
            t_query = compute_template(img_path)
            t_ref = np.array(db[subject], dtype=np.uint8)
            hd, similarity = compare_templates(t_query, t_ref)

            threshold = 0.342
            verified = hd <= threshold
            matched_subject = subject if verified else "No Match"

            eer, far, frr, acc = self._compute_eer_from_db(db)

            self.verify_eer_label.setText(f"{eer:.4f}")
            self.verify_far_label.setText(f"{far:.4f}")
            self.verify_frr_label.setText(f"{frr:.4f}")
            self.verify_acc_label.setText(f"{acc:.4f}")

            self.verify_hamming_label.setText(f"{hd:.4f}")
            self.verify_similarity_label.setText(f"{similarity:.2f} %")
            self.verify_matched_subject_label.setText(matched_subject)
            self.verify_result_box.setPlainText(
                f"Verification against subject: {subject}\n"
                f"Hamming distance: {hd:.4f}\n"
                f"Similarity: {similarity:.2f} %\n"
                f"Result: {'Verified' if verified else 'Not Verified'}"
            )
        except Exception as e:
            self.verify_result_box.setPlainText(f"Verification failed: {str(e)}")
            self.verify_hamming_label.setText("-")
            self.verify_similarity_label.setText("-")
            self.verify_matched_subject_label.setText("Error")

    def _on_ident_upload(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select iris image to identify", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.ident_image_path.setText(path)
            try:
                seg_path, norm_path, _ = run_full_pipeline(path, out_root=self._output_dir)
                self._display_pixmap(self.ident_query_image, seg_path)
                self.ident_result_box.setPlainText(
                    f"Saved:\n  Segmented → {seg_path}\n  Normalized → {norm_path}"
                )
            except Exception as e:
                self.ident_result_box.setPlainText(f"Segmentation/normalization failed: {e}")

    def _on_run_identification(self):
        img_path = self.ident_image_path.text().strip()
        if not img_path or not os.path.exists(img_path):
            self.ident_result_box.setPlainText("Please upload a valid image file.")
            return

        try:
            db_path = os.path.join("iris_codes", "iris_codes_train.json")
            try:
                with open(db_path, "r") as f:
                    db = json.load(f)
            except FileNotFoundError:
                self.ident_result_box.setPlainText(
                    "Iris code database not found. Please enroll subjects first."
                )
                self._clear_ident_results()
                return

            query_t = compute_template(img_path)
            best = search_dataset_for_best_match(query_t, codes_file=db_path)
            if best is None:
                self.ident_result_box.setPlainText("No match found in saved iris codes.")
                self._clear_ident_results()
                return

            subject, side, hd, similarity = best
            threshold = 0.342
            confidence = "High" if hd <= threshold else "Low"
            matched_subject = subject if hd <= threshold else "Unknown"

            eer, far, frr, acc = self._compute_eer_from_db(db)
            self.ident_eer_label.setText(f"{eer:.4f}")
            self.ident_far_label.setText(f"{far:.4f}")
            self.ident_frr_label.setText(f"{frr:.4f}")
            self.ident_acc_label.setText(f"{acc:.4f}")

            self.ident_match_subject_label.setText(matched_subject)
            self.ident_match_side_label.setText(side)
            self.ident_match_hamming_label.setText(f"{hd:.4f}")
            self.ident_match_similarity_label.setText(f"{similarity:.2f} %")
            self.ident_result_box.setPlainText(
                f"Best match found:\n"
                f"Subject: {matched_subject}\n"
                f"Hamming distance: {hd:.4f}\n"
                f"Similarity: {similarity:.2f} %\n"
                f"Confidence: {confidence}"
            )
        except Exception as e:
            self.ident_result_box.setPlainText(f"Identification failed: {str(e)}")
            self._clear_ident_results()

    # ---------- Metrics ----------
    def _compute_eer_from_db(self, db_dict: dict) -> Tuple[float, float, float, float]:
        """
        Returns (EER, FAR_at_EER, FRR_at_EER, Accuracy) as fractions in [0, 1]
        Assumes compare_templates returns Hamming distance (lower = better)
        """
        genuine = []
        impostor = []
        subjects = list(db_dict.keys())
        codes = {s: np.array(db_dict[s], dtype=np.uint8) for s in subjects}

        for i, s1 in enumerate(subjects):
            c1 = codes[s1]
            for s2 in subjects[i:]:  # avoid double-counting
                c2 = codes[s2]
                hd, _ = compare_templates(c1, c2)
                if s1 == s2:
                    genuine.append(hd)
                else:
                    impostor.append(hd)

        if not genuine or not impostor:
            return 0.0, 0.0, 0.0, 0.0

        genuine = np.array(genuine)
        impostor = np.array(impostor)
        n_gen = len(genuine)
        n_imp = len(impostor)

        thresholds = np.linspace(0.0, 0.5, 2000)
        far = np.mean(impostor[None, :] <= thresholds[:, None], axis=1)  # False Accept Rate
        frr = np.mean(genuine[None, :] > thresholds[:, None], axis=1)    # False Reject Rate

        abs_diff = np.abs(far - frr)
        best_idx = np.argmin(abs_diff)
        eer = (far[best_idx] + frr[best_idx]) / 2.0
        acc = 1.0 - (frr[best_idx] * n_gen + far[best_idx] * n_imp) / (n_gen + n_imp)
        return float(eer), float(far[best_idx]), float(frr[best_idx]), float(acc)

    # ---------- helpers ----------
    def _clear_ident_results(self):
        self.ident_query_image.clear()
        self.ident_match_subject_label.setText("-")
        self.ident_match_side_label.setText("-")
        self.ident_match_hamming_label.setText("-")
        self.ident_match_similarity_label.setText("-")

    def _display_pixmap(self, label: QLabel, image_path: str):
        try:
            pix = QPixmap(image_path)
            if pix.isNull():
                label.setText("Cannot load image")
                return
            w, h = label.width(), label.height()
            scaled = pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled)
        except Exception:
            label.setText("Error loading image")

    def _qss(self) -> str:
        return """
        QWidget {
            background: #f6f7fb;
            font-family: "Segoe UI", Roboto, Arial;
            color: #222;
            font-size: 11pt;
        }
        #headerLabel {
            font-size: 16pt;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 6px;
        }
        QTabWidget::pane {
            border: 0;
            background: transparent.
        }
        QTabBar::tab {
            background: #e6e9f2;
            border-radius: 8px;
            min-width: 140px;
            padding: 10px 14px;
            margin: 4px;
            font-weight: 600;
            color: #374151;
        }
        QTabBar::tab:selected {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6366f1, stop:1 #4f46e5);
            color: white;
            box-shadow: 0 6px 18px rgba(79,70,229,0.18);
        }
        QGroupBox {
            border: 1px solid rgba(99,102,241,0.06);
            border-radius: 10px;
            padding: 12px;
            background: white;
            font-weight: 600;
        }
        QPushButton {
            background: #4f46e5;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 8px;
            font-weight: 600;
            min-width: 90px;
        }
        QPushButton:hover {
            background: #5b4df0;
        }
        QPushButton:pressed {
            background: #4236c9;
        }
        QLineEdit, QTextEdit {
            border: 1px solid #e6e9f2;
            padding: 8px;
            border-radius: 8px;
            background: #fbfbff;
        }
        QTextEdit {
            font-family: "Consolas", monospace;
            font-size: 10pt;
        }
        QLabel#mutedLabel {
            color: #6b7280;
            font-size: 9pt;
        }
        QLabel {
            font-size: 11pt;
        }
        #footerLabel {
            color: #6b7280;
            font-size: 9pt;
            margin-top: 6px;
        }
        """


def main():
   
    # run_dataset_segmentation_only(
    #     dataset_dir="REAL",
    #     out_root="outputs",
    #     radial_res=64,
    #     angular_res=360
    # )
   
    app = QApplication(sys.argv)
    window = IrisApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
