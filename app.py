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
    QSizePolicy, QFrame, QScrollArea, QGridLayout, QSpacerItem
)
from PyQt5.QtGui import QPixmap, QFont, QIcon, QColor, QPalette
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal

from iris.feature_extraction import encode_iris


# Add to app.py after compute_template function
_preprocessing_cache = {}

def compute_template(image_path: str) -> np.ndarray:
    """Compute iris code with caching."""
    if image_path in _preprocessing_cache:
        return _preprocessing_cache[image_path]
    
    try:
        _, normalized_path, _ = run_full_pipeline(image_path, out_root="outputs")
        norm = cv2.imread(normalized_path, cv2.IMREAD_GRAYSCALE)
        if norm is None:
            raise RuntimeError("Cannot read normalized image.")
        iris_code = encode_iris(norm)
        _preprocessing_cache[image_path] = iris_code
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
    """Optimized search with early exit for exact matches."""
    if not os.path.exists(codes_file):
        print("No saved iris codes found. Run enrollment first.")
        return None

    with open(codes_file, "r") as f:
        saved = json.load(f)

    best_subject = None
    best_hamming = float('inf')
    best_similarity = 0.0
    
    # Precompute query for faster comparisons
    query_flat = query_template.ravel()
    query_len = query_flat.size

    for subject_id, template_list in saved.items():
        stored_template = np.array(template_list, dtype=np.uint8).ravel()[:query_len]
        
        # Early exit for perfect match
        if np.array_equal(query_flat[:stored_template.size], stored_template):
            best_subject = subject_id
            best_hamming = 0.0
            best_similarity = 100.0
            break
            
        hd, sim = compute_hamming_distance(query_template, stored_template)
        if hd < best_hamming:
            best_hamming = hd
            best_similarity = sim
            best_subject = subject_id

    if best_subject is None:
        return None

    # Side label not tracked here
    return best_subject, "Unknown", best_hamming, best_similarity


def compute_hamming_distance(code1: np.ndarray, code2: np.ndarray) -> Tuple[float, float]:
    """Optimized Hamming distance calculation using numpy bit operations."""
    n = min(code1.size, code2.size)
    if n == 0:
        return 1.0, 0.0
    
    # Ensure both arrays are uint8 for bitwise operations
    a = code1.ravel()[:n].astype(np.uint8)
    b = code2.ravel()[:n].astype(np.uint8)
    
    # Use XOR for faster bit comparison
    xor_result = np.bitwise_xor(a, b)
    
    # Count differing bits
    differing_bits = np.count_nonzero(xor_result)
    hd = differing_bits / n
    similarity = max(0.0, 100.0 * (1.0 - hd))
    
    return round(hd, 4), round(similarity, 2)


class ProcessingThread(QThread):
    finished = pyqtSignal(str, str, str)  # seg_path, norm_path, message
    error = pyqtSignal(str)
    
    def __init__(self, image_path, output_dir):
        super().__init__()
        self.image_path = image_path
        self.output_dir = output_dir
    
    def run(self):
        try:
            seg_path, norm_path, _ = run_full_pipeline(
                self.image_path, 
                out_root=self.output_dir
            )
            self.finished.emit(seg_path, norm_path, "Processing complete")
        except Exception as e:
            self.error.emit(str(e))
class MetricCard(QGroupBox):
    """Custom widget for displaying a metric with label and value"""
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setTitle("")
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(4)
        
        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        # FIX: Use #metricTitle selector since we set objectName
        self.title_label.setStyleSheet("""
            #metricTitle {
                color: #5d6d7e;
                font-size: 9pt;
                font-weight: 500;
                padding: 2px;
            }
        """)
        
        self.value_label = QLabel("-")
        self.value_label.setObjectName("metricValue")
        self.value_label.setAlignment(Qt.AlignCenter)
        # FIX: Use #metricValue selector since we set objectName
        self.value_label.setStyleSheet("""
            #metricValue {
                color: #2c3e50;
                font-size: 13pt;
                font-weight: 600;
                padding: 4px;
            }
        """)
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(80)  # Minimum height
        self.setStyleSheet("""
            QGroupBox {
                background: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 6px;
                margin-top: 0px;
            }
        """)
    
    def set_value(self, value: str):
        self.value_label.setText(value)


class ImageDisplayWidget(QLabel):
    """Custom widget for displaying images with proper scaling"""
    def __init__(self, default_text="No image", parent=None):
        super().__init__(parent)
        self.default_text = default_text
        self.original_pixmap = None
        self.setText(default_text)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            QLabel {
                background: #f8f9fa;
                border: 2px dashed #dee2e6;
                border-radius: 12px;
                color: #6c757d;
                font-size: 11pt;
                font-weight: 500;
            }
        """)
    
    def set_image(self, pixmap: QPixmap):
        if pixmap and not pixmap.isNull():
            self.original_pixmap = pixmap
            self.update_display()
            self.setStyleSheet("""
                QLabel {
                    background: white;
                    border: 2px solid #3498db;
                    border-radius: 12px;
                    padding: 4px;
                }
            """)
        else:
            self.clear_image()
    
    def clear_image(self):
        self.original_pixmap = None
        self.setText(self.default_text)
        self.setStyleSheet("""
            QLabel {
                background: #f8f9fa;
                border: 2px dashed #dee2e6;
                border-radius: 12px;
                color: #6c757d;
                font-size: 11pt;
                font-weight: 500;
            }
        """)
    
    def update_display(self):
        if self.original_pixmap:
            scaled = self.original_pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.setPixmap(scaled)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_pixmap:
            self.update_display()


# =========================
# GUI
# =========================
class IrisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Iris Recognition System")
        self.setMinimumSize(1100, 700)
        self.resize(1300, 800)
        
        self._db_cache = None
        self._db_path = None
        self._current_db = None
        self._current_db_path = None
        self._image_cache = {}
        self._verify_last_normalized = None
        self._ident_last_normalized = None
        self._dataset_root = "CASIA-Iris-Thousand"
        self._output_dir = "outputs"

        self.setStyleSheet(self._qss())
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header
        header_layout = QHBoxLayout()
        
        header_text = QLabel("IRIS RECOGNITION SYSTEM")
        header_text.setObjectName("headerLabel")
        header_layout.addWidget(header_text)
        header_layout.addStretch()
        
        version_label = QLabel("Version 2.0")
        version_label.setObjectName("versionLabel")
        header_layout.addWidget(version_label)
        
        main_layout.addLayout(header_layout)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        self.tabs.addTab(self._build_verification_tab(), "Verification")
        self.tabs.addTab(self._build_identification_tab(), "Identification")
        main_layout.addWidget(self.tabs)

        # Footer
        footer = QLabel(f"Dataset: {self._dataset_root} | Output Directory: {self._output_dir}")
        footer.setObjectName("footerLabel")
        footer.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(footer)

        self.setLayout(main_layout)

    def _build_verification_tab(self):
        tab = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left Panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(16)
        
        # Input Section
        input_group = QGroupBox("Input Configuration")
        input_layout = QFormLayout()
        input_layout.setSpacing(12)
        input_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        self.verify_subject_input = QLineEdit()
        self.verify_subject_input.setPlaceholderText("e.g., 000_L_000")
        self.verify_subject_input.setMinimumHeight(36)
        input_layout.addRow("Subject Key:", self.verify_subject_input)
        
        self.verify_image_path = QLineEdit()
        self.verify_image_path.setReadOnly(True)
        self.verify_image_path.setPlaceholderText("No file selected")
        self.verify_image_path.setMinimumHeight(36)
        input_layout.addRow("Image Path:", self.verify_image_path)
        
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_upload = QPushButton("Upload Image")
        btn_upload.setMinimumHeight(40)
        btn_upload.clicked.connect(self._on_verify_upload)
        btn_verify = QPushButton("Verify")
        btn_verify.setMinimumHeight(40)
        btn_verify.setObjectName("primaryButton")
        btn_verify.clicked.connect(self._on_run_verification)
        btn_row.addWidget(btn_upload)
        btn_row.addWidget(btn_verify)
        input_layout.addRow(btn_row)
        
        input_group.setLayout(input_layout)
        left_layout.addWidget(input_group)
        
        # Results Section
        results_group = QGroupBox("Verification Results")
        results_layout = QVBoxLayout()
        results_layout.setSpacing(10)
        
        self.verify_result_box = QTextEdit()
        self.verify_result_box.setReadOnly(True)
        self.verify_result_box.setMinimumHeight(150)
        self.verify_result_box.setPlaceholderText("Results will appear here...")
        results_layout.addWidget(self.verify_result_box)
        
        results_group.setLayout(results_layout)
        left_layout.addWidget(results_group)
        
        left_panel.setLayout(left_layout)
        left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        main_layout.addWidget(left_panel, 2)

        # Right Panel - Visualization & Metrics
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(16)
        
        # Image Display
        image_group = QGroupBox("Iris Segmentation")
        image_layout = QVBoxLayout()
        self.verify_image_label = ImageDisplayWidget("Upload an image to begin")
        image_layout.addWidget(self.verify_image_label)
        image_group.setLayout(image_layout)
        right_layout.addWidget(image_group, 3)
        
        # Metrics Display
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(10)
        
        self.verify_hamming_card = MetricCard("Hamming Distance")
        self.verify_similarity_card = MetricCard("Similarity")
        self.verify_matched_card = MetricCard("Match Status")
        self.verify_eer_card = MetricCard("EER")
        self.verify_far_card = MetricCard("FAR")
        self.verify_frr_card = MetricCard("FRR")
        self.verify_acc_card = MetricCard("Accuracy")
        
        metrics_layout.addWidget(self.verify_hamming_card, 0, 0)
        metrics_layout.addWidget(self.verify_similarity_card, 0, 1)
        metrics_layout.addWidget(self.verify_matched_card, 0, 2)
        metrics_layout.addWidget(self.verify_eer_card, 1, 0)
        metrics_layout.addWidget(self.verify_far_card, 1, 1)
        metrics_layout.addWidget(self.verify_frr_card, 1, 2)
        metrics_layout.addWidget(self.verify_acc_card, 2, 0, 1, 3)
        
        metrics_group.setLayout(metrics_layout)
        right_layout.addWidget(metrics_group, 2)
        
        right_panel.setLayout(right_layout)
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(right_panel, 3)

        tab.setLayout(main_layout)
        return tab

    def _build_identification_tab(self):
        tab = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left Panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(16)
        
        # Input Section
        input_group = QGroupBox("Query Image")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(12)
        
        path_layout = QHBoxLayout()
        self.ident_image_path = QLineEdit()
        self.ident_image_path.setReadOnly(True)
        self.ident_image_path.setPlaceholderText("No file selected")
        self.ident_image_path.setMinimumHeight(36)
        path_layout.addWidget(self.ident_image_path)
        
        btn_upload = QPushButton("Upload")
        btn_upload.setMinimumHeight(36)
        btn_upload.clicked.connect(self._on_ident_upload)
        path_layout.addWidget(btn_upload)
        input_layout.addLayout(path_layout)
        
        btn_identify = QPushButton("Identify Subject")
        btn_identify.setMinimumHeight(44)
        btn_identify.setObjectName("primaryButton")
        btn_identify.clicked.connect(self._on_run_identification)
        input_layout.addWidget(btn_identify)
        
        input_group.setLayout(input_layout)
        left_layout.addWidget(input_group)
        
        # Results Section
        results_group = QGroupBox("Identification Results")
        results_layout = QVBoxLayout()
        
        self.ident_result_box = QTextEdit()
        self.ident_result_box.setReadOnly(True)
        self.ident_result_box.setMinimumHeight(150)
        self.ident_result_box.setPlaceholderText("Results will appear here...")
        results_layout.addWidget(self.ident_result_box)
        
        results_group.setLayout(results_layout)
        left_layout.addWidget(results_group)
        
        left_panel.setLayout(left_layout)
        left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        main_layout.addWidget(left_panel, 2)

        # Right Panel - Match Info
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(16)
        
        # Query Image Display
        query_group = QGroupBox("Query Iris")
        query_layout = QVBoxLayout()
        self.ident_query_image = ImageDisplayWidget("Upload query image")
        query_layout.addWidget(self.ident_query_image)
        query_group.setLayout(query_layout)
        right_layout.addWidget(query_group, 2)
        
        # Match Metrics
        match_group = QGroupBox("Best Match Information")
        match_layout = QGridLayout()
        match_layout.setSpacing(10)
        
        self.ident_subject_card = MetricCard("Subject ID")
        self.ident_hamming_card = MetricCard("Hamming Distance")
        self.ident_similarity_card = MetricCard("Similarity")
        self.ident_eer_card = MetricCard("EER")
        self.ident_far_card = MetricCard("FAR")
        self.ident_frr_card = MetricCard("FRR")
        self.ident_acc_card = MetricCard("Accuracy")
        
        match_layout.addWidget(self.ident_subject_card, 0, 0, 1, 2)
        match_layout.addWidget(self.ident_hamming_card, 1, 0)
        match_layout.addWidget(self.ident_similarity_card, 1, 1)
        match_layout.addWidget(self.ident_eer_card, 2, 0)
        match_layout.addWidget(self.ident_far_card, 2, 1)
        match_layout.addWidget(self.ident_frr_card, 3, 0)
        match_layout.addWidget(self.ident_acc_card, 3, 1)
        
        match_group.setLayout(match_layout)
        right_layout.addWidget(match_group, 3)
        
        right_panel.setLayout(right_layout)
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(right_panel, 3)

        tab.setLayout(main_layout)
        return tab

    def _load_database(self, db_path: str):
        """Lazy load and cache database."""
        if self._current_db is not None and self._current_db_path == db_path:
            return self._current_db
        
        try:
            with open(db_path, "r") as f:
                self._current_db = json.load(f)
                self._current_db_path = db_path
            return self._current_db
        except FileNotFoundError:
            self._current_db = None
            self._current_db_path = None
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {db_path}")
            self._current_db = None
            self._current_db_path = None
            return None

    def _search_dataset_for_best_match_optimized(self, query_template: np.ndarray, 
                                            db: dict, codes_file: str):
        """Optimized search within cached database."""
        if not db:
            return None

        best_subject = None
        best_hamming = float('inf')
        best_similarity = 0.0
        
        query_flat = query_template.ravel()
        query_len = query_flat.size

        for subject_id, template_list in db.items():
            stored_template = np.array(template_list, dtype=np.uint8).ravel()[:query_len]
            
            if np.array_equal(query_flat[:stored_template.size], stored_template):
                best_subject = subject_id
                best_hamming = 0.0
                best_similarity = 100.0
                break
                
            hd, sim = compute_hamming_distance(query_template, stored_template)
            if hd < best_hamming:
                best_hamming = hd
                best_similarity = sim
                best_subject = subject_id

        if best_subject is None:
            return None

        side = "Unknown"
        parts = best_subject.split('_')
        if len(parts) >= 2 and parts[1] in ['L', 'R']:
            side = parts[1]

        return best_subject, side, best_hamming, best_similarity
        
    # ---------- Handlers ----------
    def _on_verify_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Iris Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.verify_image_path.setText(path)
            self.verify_result_box.setPlainText("Processing image, please wait...")
            
            btn = self.sender()
            btn.setEnabled(False)
            btn.setText("Processing...")
            
            self.verify_processing_thread = ProcessingThread(path, self._output_dir)
            self.verify_processing_thread.finished.connect(
                lambda seg, norm, msg: self._on_verify_processing_done(seg, norm, msg, btn)
            )
            self.verify_processing_thread.error.connect(
                lambda err: self._on_processing_error(err, btn, self.verify_result_box)
            )
            self.verify_processing_thread.start()

    def _on_verify_processing_done(self, seg_path, norm_path, message, button):
        if seg_path and os.path.exists(seg_path):
            pixmap = QPixmap(seg_path)
            self.verify_image_label.set_image(pixmap)
        
        self._verify_last_normalized = norm_path
        self.verify_result_box.setPlainText(
            f"Processing Complete\n\n"
            f"Segmented: {os.path.basename(seg_path)}\n"
            f"Normalized: {os.path.basename(norm_path)}\n\n"
            f"Ready for verification."
        )
        button.setEnabled(True)
        button.setText("Upload Image")

    def _on_run_verification(self):
        subject = self.verify_subject_input.text().strip()
        img_path = self.verify_image_path.text().strip()
        codes_dir = "iris_codes"
        
        if not subject:
            self.verify_result_box.setPlainText("Please enter a subject key (e.g., 000_L_000).")
            return
        if not img_path or not os.path.exists(img_path):
            self.verify_result_box.setPlainText("Please upload a valid image file.")
            return

        try:
            db_path = os.path.join(codes_dir, "iris_codes_train.json")
            db = self._load_database(db_path)
            if db is None:
                self.verify_result_box.setPlainText("Iris code database not found. Please enroll subjects first.")
                return

            if subject not in db:
                self.verify_result_box.setPlainText(f"Subject '{subject}' not found in database.")
                self.verify_hamming_card.set_value("-")
                self.verify_similarity_card.set_value("-")
                self.verify_matched_card.set_value("Not Found")
                return

            t_query = compute_template(img_path)
            t_ref = np.array(db[subject], dtype=np.uint8)
            
            hd, similarity = compute_hamming_distance(t_query, t_ref)

            threshold = 0.0049
            verified = hd <= threshold
            matched_subject = subject if verified else "No Match"

            eer, far, frr, acc = self._compute_eer_from_db(db)

            # Update metric cards
            self.verify_hamming_card.set_value(f"{hd:.4f}")
            self.verify_similarity_card.set_value(f"{similarity:.1f}%")
            if verified:
                self.verify_matched_card.set_value("Verified")
                self.verify_matched_card.value_label.setStyleSheet("""
                    QLabel {
                        color: #27ae60;
                        font-size: 14pt;
                        font-weight: 600;
                        padding: 4px;
                    }
                """)
            else:
                self.verify_matched_card.set_value("Rejected")
                self.verify_matched_card.value_label.setStyleSheet("""
                    QLabel {
                        color: #e74c3c;
                        font-size: 14pt;
                        font-weight: 600;
                        padding: 4px;
                    }
                """)
            self.verify_eer_card.set_value(f"{eer:.4f}")
            self.verify_far_card.set_value(f"{far:.4f}")
            self.verify_frr_card.set_value(f"{frr:.4f}")
            self.verify_acc_card.set_value(f"{acc*100:.2f}%")
            
            status_text = "VERIFIED" if verified else "REJECTED"
            status_color = "#27ae60" if verified else "#e74c3c"
            self.verify_result_box.setPlainText(
                f"Verification Results\n\n"
                f"Subject: {subject}\n"
                f"Hamming Distance: {hd:.4f}\n"
                f"Similarity: {similarity:.2f}%\n"
                f"Threshold: {threshold:.4f}\n\n"
                f"Result: {status_text}"
            )
                    
        except Exception as e:
            self.verify_result_box.setPlainText(f"Verification failed:\n{str(e)}")
            self._clear_verify_metrics()

    def _on_ident_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Iris Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.ident_image_path.setText(path)
            self.ident_result_box.setPlainText("Processing image, please wait...")
            
            btn = self.sender()
            btn.setEnabled(False)
            btn.setText("Processing...")
            
            self.ident_processing_thread = ProcessingThread(path, self._output_dir)
            self.ident_processing_thread.finished.connect(
                lambda seg, norm, msg: self._on_ident_processing_done(seg, norm, msg, btn)
            )
            self.ident_processing_thread.error.connect(
                lambda err: self._on_processing_error(err, btn, self.ident_result_box)
            )
            self.ident_processing_thread.start()

    def _on_ident_processing_done(self, seg_path, norm_path, message, button):
        if seg_path and os.path.exists(seg_path):
            pixmap = QPixmap(seg_path)
            self.ident_query_image.set_image(pixmap)
        
        self._ident_last_normalized = norm_path
        self.ident_result_box.setPlainText(
            f"Processing Complete\n\n"
            f"Segmented: {os.path.basename(seg_path)}\n"
            f"Normalized: {os.path.basename(norm_path)}\n\n"
            f"Ready for identification."
        )
        button.setEnabled(True)
        button.setText("Upload")

    def _on_processing_error(self, error_msg, button, result_box):
        result_box.setPlainText(f"Processing failed:\n{error_msg}")
        button.setEnabled(True)
        if "Upload" in button.text() or "Processing" in button.text():
            button.setText("Upload Image" if "Verification" in self.tabs.tabText(self.tabs.currentIndex()) else "Upload")

    def _on_run_identification(self):
        img_path = self.ident_image_path.text().strip()
        if not img_path or not os.path.exists(img_path):
            self.ident_result_box.setPlainText("Please upload a valid image file.")
            return

        try:
            # Load database with caching
            db_path = os.path.join("iris_codes", "iris_codes_train.json")
            db = self._load_database(db_path)
            if db is None:
                self.ident_result_box.setPlainText(
                    "Iris code database not found. Please enroll subjects first."
                )
                self._clear_ident_results()
                return

            # Compute query iris code with caching
            query_t = compute_template(img_path)
            
            # Use optimized search
            best = self._search_dataset_for_best_match_optimized(query_t, db, db_path)
            if best is None:
                self.ident_result_box.setPlainText("No match found in saved iris codes.")
                self._clear_ident_results()
                return

            subject, side, hd, similarity = best
            threshold = 0.342
            confidence = "High" if hd <= threshold else "Low"
            matched_subject = subject if hd <= threshold else "Unknown"

            # Compute metrics with cached database
            eer, far, frr, acc = self._compute_eer_from_db(db)
            
            # Update metric cards
            self.ident_subject_card.set_value(matched_subject)
            self.ident_hamming_card.set_value(f"{hd:.4f}")
            self.ident_similarity_card.set_value(f"{similarity:.1f}%")
            self.ident_eer_card.set_value(f"{eer:.4f}")
            self.ident_far_card.set_value(f"{far:.4f}")
            self.ident_frr_card.set_value(f"{frr:.4f}")
            self.ident_acc_card.set_value(f"{acc*100:.2f}%")
            
            # Color code subject ID based on match confidence
            if hd <= threshold:
                self.ident_subject_card.value_label.setStyleSheet("""
                    QLabel {
                        color: #27ae60;
                        font-size: 14pt;
                        font-weight: 600;
                        padding: 4px;
                    }
                """)
            else:
                self.ident_subject_card.value_label.setStyleSheet("""
                    QLabel {
                        color: #f39c12;
                        font-size: 14pt;
                        font-weight: 600;
                        padding: 4px;
                    }
                """)
            
            self.ident_result_box.setPlainText(
                f"Best match found:\n"
                f"Subject: {matched_subject}\n"
                f"Hamming distance: {hd:.4f}\n"
                f"Similarity: {similarity:.2f}%\n"
                f"Confidence: {confidence}"
            )
            
            # Display the processed image if available
            if hasattr(self, '_ident_last_normalized'):
                seg_path = self._ident_last_normalized.replace('_normalized.png', '_segmented.png')
                if os.path.exists(seg_path):
                    self._display_pixmap(self.ident_query_image, seg_path)
                    
        except Exception as e:
            self.ident_result_box.setPlainText(f"Identification failed: {str(e)}")
            self._clear_ident_results()

    # ---------- Metrics ----------
    def _compute_eer_from_db(self, db_dict: dict) -> Tuple[float, float, float, float]:
        """Compute EER with cached database."""
        if self._db_cache is None or self._db_path != "iris_codes/iris_codes_train.json":
            # Cache the parsed database
            self._db_cache = {}
            for key, template_list in db_dict.items():
                self._db_cache[key] = np.array(template_list, dtype=np.uint8)
            self._db_path = "iris_codes/iris_codes_train.json"
        
        genuine = []
        impostor = []
        subjects = list(self._db_cache.keys())
        
        for i, s1 in enumerate(subjects):
            c1 = self._db_cache[s1]
            for s2 in subjects[i:]:  # avoid double-counting
                c2 = self._db_cache[s2]
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
    def _clear_verify_metrics(self):
        """Clear verification metrics to default state"""
        self.verify_hamming_card.set_value("-")
        self.verify_similarity_card.set_value("-")
        self.verify_matched_card.set_value("-")
        self.verify_eer_card.set_value("-")
        self.verify_far_card.set_value("-")
        self.verify_frr_card.set_value("-")
        self.verify_acc_card.set_value("-")
        
        # Reset match status color
        self.verify_matched_card.value_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 14pt;
                font-weight: 600;
                padding: 4px;
            }
        """)

    def _clear_ident_results(self):
        """Clear identification results to default state"""
        self.ident_query_image.clear()
        self.ident_subject_card.set_value("-")
        self.ident_hamming_card.set_value("-")
        self.ident_similarity_card.set_value("-")
        self.ident_eer_card.set_value("-")
        self.ident_far_card.set_value("-")
        self.ident_frr_card.set_value("-")
        self.ident_acc_card.set_value("-")
        
        # Reset subject ID color
        self.ident_subject_card.value_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 14pt;
                font-weight: 600;
                padding: 4px;
            }
        """)

    def _display_pixmap(self, label: QLabel, image_path: str):
        """Optimized image display with caching."""
        try:
            if image_path not in self._image_cache:
                pix = QPixmap(image_path)
                if pix.isNull():
                    label.setText("Cannot load image")
                    return
                self._image_cache[image_path] = pix
            else:
                pix = self._image_cache[image_path]
                
            w, h = label.width(), label.height()
            scaled = pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled)
        except Exception:
            label.setText("Error loading image")

    def _qss(self) -> str:
        return """
        QWidget {
            background: #f5f7fa;
            font-family: "Segoe UI", "Roboto", "Arial", sans-serif;
            color: #2c3e50;
            font-size: 11pt;
        }
        
        #headerLabel {
            font-size: 20pt;
            font-weight: 700;
            color: #2c3e50;
            letter-spacing: 0.5px;
            padding: 4px 0;
        }
        
        #versionLabel {
            font-size: 10pt;
            color: #7f8c8d;
            font-weight: 500;
            padding: 6px 12px;
            background: #ecf0f1;
            border-radius: 12px;
        }
        
        QTabWidget::pane {
            border: 1px solid #dcdcdc;
            border-radius: 10px;
            background: white;
            margin-top: 4px;
        }
        
        QTabBar::tab {
            background: #ecf0f1;
            border: 1px solid #dcdcdc;
            border-bottom: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            padding: 12px 24px;
            margin-right: 4px;
            font-weight: 600;
            color: #5d6d7e;
            font-size: 11pt;
            min-width: 120px;  /* Increased from default */
        }
        
        QTabBar::tab:selected {
            background: white;
            color: #3498db;
            border-bottom: 2px solid #3498db;
            font-weight: 700;
        }
        
        QTabBar::tab:hover {
            background: #dfe6e9;
            color: #2980b9;
        }
        

        
        QGroupBox {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 16px;
            background: white;
            font-weight: 600;
            font-size: 11pt;
            color: #2c3e50;
            margin-top: 12px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 8px 0 8px;
            color: #3498db;
        }
        
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #3498db, stop:1 #2980b9);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 600;
            min-width: 100px;
            font-size: 11pt;
        }
        
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #3cb0fd, stop:1 #3498db);
        }
        
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #2980b9, stop:1 #2472a4);
        }
        
        QPushButton:disabled {
            background: #bdc3c7;
            color: #7f8c8d;
        }
        
        QPushButton#primaryButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #2ecc71, stop:1 #27ae60);
            font-weight: 700;
        }
        
        QPushButton#primaryButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #52d67e, stop:1 #2ecc71);
        }
        
        QPushButton#primaryButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #27ae60, stop:1 #219955);
        }
        
        QLineEdit {
            border: 1px solid #dcdcdc;
            padding: 10px 12px;
            border-radius: 8px;
            background: #fdfdfd;
            font-size: 11pt;
            selection-background-color: #3498db;
        }
        
        QLineEdit:focus {
            border: 2px solid #3498db;
            padding: 9px 11px;
        }
        
        QLineEdit:disabled {
            background: #f8f9fa;
            color: #95a5a6;
        }
        
        QTextEdit {
            border: 1px solid #dcdcdc;
            padding: 12px;
            border-radius: 8px;
            background: #fdfdfd;
            font-family: "Consolas", "Monaco", monospace;
            font-size: 10pt;
            selection-background-color: #3498db;
        }
        
        QTextEdit:focus {
            border: 2px solid #3498db;
            padding: 11px;
        }
        
        QTextEdit:disabled {
            background: #f8f9fa;
            color: #95a5a6;
        }
        
        QScrollBar:vertical {
            background: #f1f1f1;
            width: 10px;
            border-radius: 5px;
        }
        
        QScrollBar::handle:vertical {
            background: #c1c1c1;
            border-radius: 5px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: #a8a8a8;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        #footerLabel {
            color: #7f8c8d;
            font-size: 10pt;
            font-weight: 500;
            margin-top: 8px;
            padding: 8px;
            background: #ecf0f1;
            border-radius: 8px;
        }
        """


def main():
   
    # run_dataset_segmentation_only(x
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