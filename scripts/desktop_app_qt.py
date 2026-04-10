from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

import pandas as pd
from PySide6.QtCore import QProcess, Qt, QPoint
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.nomogram_report import generate_nomogram_report
from src.inference.predictors import (
    FTTransformerPredictor,
    XGBoostPredictor,
    build_test_scenarios,
    build_single_row_frame,
    find_exact_match,
    find_nearest_reference_rows,
    load_reference_dataset,
)
from src.utils.config import DataConfig


def report_paths(model_key: str) -> dict[str, Path]:
    data_config = DataConfig()
    base = data_config.xgboost_artifact_dir if model_key == "xgboost" else data_config.ft_transformer_artifact_dir
    report_dir = base / "reports"
    return {
        "row_level": report_dir / f"{model_key}_row_level_comparison.csv",
        "slice_summary": report_dir / f"{model_key}_slice_summary.csv",
        "overall_summary": report_dir / f"{model_key}_overall_summary.csv",
        "slice_plot": report_dir / f"{model_key}_slice_predictions.png",
        "summary_plot": report_dir / f"{model_key}_slice_summary.png",
        "excel_report": report_dir / f"{model_key}_table_report.xlsx",
    }


class ResponsiveImageLabel(QLabel):
    def __init__(self, empty_text: str) -> None:
        super().__init__(empty_text)
        self._original_pixmap: QPixmap | None = None
        self._empty_text = empty_text
        self._click_handler: Callable[[], None] | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumHeight(320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.ArrowCursor)

    def set_plot_path(self, path: Path) -> None:
        if not path.exists():
            self._original_pixmap = None
            self.setPixmap(QPixmap())
            self.setText("Grafik bulunamadi.")
            return

        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._original_pixmap = None
            self.setPixmap(QPixmap())
            self.setText(f"Grafik acilamadi:\n{path}")
            return

        self._original_pixmap = pixmap
        self._rescale_pixmap()

    def clear_plot(self) -> None:
        self._original_pixmap = None
        self.setPixmap(QPixmap())
        self.setText(self._empty_text)
        self.setCursor(Qt.ArrowCursor)

    def set_click_handler(self, handler: Callable[[], None] | None) -> None:
        self._click_handler = handler
        self.setCursor(Qt.PointingHandCursor if handler is not None else Qt.ArrowCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._click_handler is not None:
            self._click_handler()
            return
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale_pixmap()

    def _rescale_pixmap(self) -> None:
        if self._original_pixmap is None:
            return

        viewport_width = max(self.width() - 24, 320)
        viewport_height = max(self.height() - 24, 240)
        scaled = self._original_pixmap.scaled(
            viewport_width,
            viewport_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setText("")
        self.setPixmap(scaled)


class ClickableImageLabel(QLabel):
    def __init__(self, empty_text: str) -> None:
        super().__init__(empty_text)
        self._click_handler: Callable[[], None] | None = None
        self._scroll_area: QScrollArea | None = None
        self._panning = False
        self._pan_start = QPoint()
        self._start_h_value = 0
        self._start_v_value = 0
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def set_click_handler(self, handler: Callable[[], None] | None) -> None:
        self._click_handler = handler

    def set_scroll_area(self, scroll_area: QScrollArea) -> None:
        self._scroll_area = scroll_area

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MiddleButton and self._scroll_area is not None:
            self._panning = True
            self._pan_start = event.globalPosition().toPoint()
            self._start_h_value = self._scroll_area.horizontalScrollBar().value()
            self._start_v_value = self._scroll_area.verticalScrollBar().value()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._click_handler is not None:
            self._click_handler()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._panning and self._scroll_area is not None:
            delta = event.globalPosition().toPoint() - self._pan_start
            self._scroll_area.horizontalScrollBar().setValue(self._start_h_value - delta.x())
            self._scroll_area.verticalScrollBar().setValue(self._start_v_value - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.PointingHandCursor if self._click_handler is not None else Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ImageViewerDialog(QDialog):
    def __init__(self, title: str, image_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1200, 900)
        self._base_pixmap = QPixmap(str(image_path))
        self._zoom_factor = 1.0
        self._fit_zoom_factor = 1.0

        layout = QVBoxLayout(self)
        caption = QLabel(str(image_path))
        caption.setWordWrap(True)
        layout.addWidget(caption)

        toolbar = QHBoxLayout()
        zoom_in_btn = QPushButton("Zoom +")
        zoom_out_btn = QPushButton("Zoom -")
        fit_btn = QPushButton("Sigdır")
        fullscreen_btn = QPushButton("Tam ekran")
        for button in (zoom_in_btn, zoom_out_btn, fit_btn, fullscreen_btn):
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        self.image_label = ClickableImageLabel("Gorsel yuklenemedi.")
        self.image_label.set_click_handler(self._toggle_click_zoom)
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setWidget(self.image_label)
        self.scroll = scroll
        self.image_label.set_scroll_area(scroll)
        layout.addWidget(scroll, stretch=1)

        zoom_in_btn.clicked.connect(lambda: self._apply_zoom(1.2))
        zoom_out_btn.clicked.connect(lambda: self._apply_zoom(1 / 1.2))
        fit_btn.clicked.connect(self._fit_to_window)
        fullscreen_btn.clicked.connect(self._toggle_fullscreen)

        self._fit_to_window()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not self.isFullScreen():
            self._fit_to_window()

    def _render_pixmap(self) -> None:
        if self._base_pixmap.isNull():
            self.image_label.setText("Gorsel yuklenemedi.")
            self.image_label.setPixmap(QPixmap())
            return
        width = max(int(self._base_pixmap.width() * self._zoom_factor), 200)
        height = max(int(self._base_pixmap.height() * self._zoom_factor), 200)
        scaled = self._base_pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())

    def _apply_zoom(self, factor: float) -> None:
        self._zoom_factor = min(max(self._zoom_factor * factor, 0.2), 8.0)
        self._render_pixmap()

    def _fit_to_window(self) -> None:
        if self._base_pixmap.isNull():
            return
        viewport = self.scroll.viewport().size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return
        width_ratio = viewport.width() / self._base_pixmap.width()
        height_ratio = viewport.height() / self._base_pixmap.height()
        self._fit_zoom_factor = max(min(width_ratio, height_ratio, 1.0), 0.2)
        self._zoom_factor = self._fit_zoom_factor
        self._render_pixmap()

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            self.resize(1200, 900)
        else:
            self.showFullScreen()
        self._fit_to_window()

    def _toggle_click_zoom(self) -> None:
        if abs(self._zoom_factor - self._fit_zoom_factor) < 0.05:
            self._zoom_factor = min(self._fit_zoom_factor * 2.0, 8.0)
        else:
            self._zoom_factor = self._fit_zoom_factor
        self._render_pixmap()


class ModernDesktopApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Specific Range Studio")
        self.resize(1500, 950)

        self.reference_df = load_reference_dataset()
        self.test_scenarios = build_test_scenarios(self.reference_df)
        self.xgb_predictor: XGBoostPredictor | None = None
        self.ft_predictor: FTTransformerPredictor | None = None
        self.active_process: QProcess | None = None
        self.current_slice_plot_path: Path | None = None
        self.current_summary_plot_path: Path | None = None
        self.current_nomogram_plot_path: Path | None = None
        self.report_row_df = pd.DataFrame()
        self.report_filtered_df = pd.DataFrame()

        self._setup_ui()
        self._apply_selected_scenario()
        self._apply_styles()
        self._load_report()

    def _setup_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        header = QLabel("Specific Range Studio")
        header.setObjectName("titleLabel")
        subtitle = QLabel(
            "FT-Transformer ve XGBoost icin hazir rapor goruntuleme, tekil tahmin ve taslak nomogram uretimi."
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subtitleLabel")
        root_layout.addWidget(header)
        root_layout.addWidget(subtitle)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._wrap_tab_scroll(self._build_setup_tab()), "Setup")
        self.tabs.addTab(self._wrap_tab_scroll(self._build_report_tab()), "Hazir Rapor")
        self.tabs.addTab(self._wrap_tab_scroll(self._build_prediction_tab()), "Tekil Tahmin")
        self.tabs.addTab(self._wrap_tab_scroll(self._build_nomogram_tab()), "Taslak Nomogram")
        root_layout.addWidget(self.tabs)

        self.setCentralWidget(root)

        refresh_action = QAction("Raporu Yenile", self)
        refresh_action.triggered.connect(self._load_report)
        self.menuBar().addAction(refresh_action)

    def _wrap_tab_scroll(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(widget)
        return scroll

    def _build_setup_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        top_row = QHBoxLayout()

        device_group = QGroupBox("Cihaz Secimi")
        device_layout = QFormLayout(device_group)
        self.setup_xgb_device_combo = QComboBox()
        self.setup_xgb_device_combo.addItems(["cpu", "cuda"])
        self.setup_xgb_device_combo.setCurrentText("cuda")
        self.setup_ft_device_combo = QComboBox()
        self.setup_ft_device_combo.addItems(["cpu", "cuda"])
        self.setup_ft_device_combo.setCurrentText("cuda")
        device_layout.addRow("XGBoost", self.setup_xgb_device_combo)
        device_layout.addRow("FT-Transformer", self.setup_ft_device_combo)
        top_row.addWidget(device_group, stretch=1)

        workflow_group = QGroupBox("Quickstart Akisi")
        workflow_layout = QVBoxLayout(workflow_group)
        self.setup_command_preview = QTextEdit()
        self.setup_command_preview.setReadOnly(True)
        self.setup_command_preview.setMinimumHeight(180)
        workflow_layout.addWidget(self.setup_command_preview)
        self.setup_artifact_info = QTextEdit()
        self.setup_artifact_info.setReadOnly(True)
        self.setup_artifact_info.setMinimumHeight(130)
        workflow_layout.addWidget(self.setup_artifact_info)

        button_row = QHBoxLayout()
        quickstart_btn = QPushButton("Quickstart Setup")
        quickstart_btn.clicked.connect(self._run_full_setup_sequence)
        pipeline_btn = QPushButton("Sadece Veri Pipeline")
        pipeline_btn.clicked.connect(lambda: self._run_named_command("Veri pipeline"))
        report_btn = QPushButton("Sadece Toplu Rapor")
        report_btn.clicked.connect(lambda: self._run_named_command("Toplu rapor uret"))
        for button in (quickstart_btn, pipeline_btn, report_btn):
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        button_row.addWidget(quickstart_btn)
        button_row.addWidget(pipeline_btn)
        button_row.addWidget(report_btn)
        button_row.addStretch(1)
        workflow_layout.addLayout(button_row)
        top_row.addWidget(workflow_group, stretch=2)
        layout.addLayout(top_row)

        actions_group = QGroupBox("Tekil Komutlar")
        actions_layout = QHBoxLayout(actions_group)
        for label in [
            "XGBoost egit + rapor",
            "FT-Transformer egit + rapor",
            "PSO calistir",
            "Model karsilastir",
        ]:
            button = QPushButton(label)
            button.clicked.connect(lambda _, current_label=label: self._run_named_command(current_label))
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            actions_layout.addWidget(button)
        actions_layout.addStretch(1)
        layout.addWidget(actions_group)

        self.setup_status = QLabel("Hazir.")
        self.setup_status.setWordWrap(True)
        layout.addWidget(self.setup_status)
        self.setup_progress = QProgressBar()
        self.setup_progress.setRange(0, 100)
        self.setup_progress.setValue(0)
        layout.addWidget(self.setup_progress)

        self.setup_log = QTextEdit()
        self.setup_log.setReadOnly(True)
        layout.addWidget(self.setup_log, stretch=1)

        self._refresh_setup_preview()
        self.setup_xgb_device_combo.currentTextChanged.connect(self._refresh_setup_preview)
        self.setup_ft_device_combo.currentTextChanged.connect(self._refresh_setup_preview)
        return tab

    def _build_report_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        layout.setContentsMargins(8, 8, 8, 16)

        controls_group = QGroupBox("Rapor Kontrolleri")
        controls_layout = QHBoxLayout(controls_group)
        self.report_model_combo = QComboBox()
        self.report_model_combo.addItems(["xgboost", "ft_transformer"])
        self.report_model_combo.currentTextChanged.connect(self._load_report)
        self.ft_device_combo = QComboBox()
        self.ft_device_combo.addItems(["cpu", "cuda"])
        load_btn = QPushButton("Raporu Yukle")
        load_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        load_btn.clicked.connect(self._load_report)
        self.report_model_combo.setMinimumWidth(180)
        self.ft_device_combo.setMinimumWidth(120)
        controls_layout.addWidget(QLabel("Rapor modeli"))
        controls_layout.addWidget(self.report_model_combo)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(QLabel("FT cihaz"))
        controls_layout.addWidget(self.ft_device_combo)
        controls_layout.addStretch(1)
        controls_layout.addWidget(load_btn)
        layout.addWidget(controls_group)

        suggested_flow = QGroupBox("Onerilen Akis")
        suggested_layout = QVBoxLayout(suggested_flow)
        suggested_text = QLabel(
            "1. Setup ile veri ve egitim adimlarini tamamla.\n"
            "2. Hazir raporda genel metrikleri incele.\n"
            "3. Filtrelerle belirli engine_type / altitude kesitlerini daralt.\n"
            "4. Secili satirin detayina bak.\n"
            "5. Gerekirse Tekil Tahmin ile ara deger test et."
        )
        suggested_text.setWordWrap(True)
        suggested_layout.addWidget(suggested_text)
        layout.addWidget(suggested_flow)

        filter_group = QGroupBox("Rapor Filtreleri")
        filter_layout = QHBoxLayout(filter_group)
        self.report_engine_filter = QComboBox()
        self.report_altitude_filter = QComboBox()
        self.report_sort_filter = QComboBox()
        self.report_sort_filter.addItems(["En buyuk hata ustte", "En kucuk hata ustte", "row_id"])
        self.report_engine_filter.currentTextChanged.connect(self._apply_report_filters)
        self.report_altitude_filter.currentTextChanged.connect(self._apply_report_filters)
        self.report_sort_filter.currentTextChanged.connect(self._apply_report_filters)
        filter_layout.addWidget(QLabel("engine_type"))
        filter_layout.addWidget(self.report_engine_filter)
        filter_layout.addWidget(QLabel("altitude"))
        filter_layout.addWidget(self.report_altitude_filter)
        filter_layout.addWidget(QLabel("siralama"))
        filter_layout.addWidget(self.report_sort_filter)
        layout.addWidget(filter_group)

        summary_group = QGroupBox("Rapor Ozeti")
        summary_layout = QVBoxLayout(summary_group)
        self.report_summary = QTextEdit()
        self.report_summary.setReadOnly(True)
        self.report_summary.setMinimumHeight(120)
        self.report_summary.setMaximumHeight(180)
        summary_layout.addWidget(self.report_summary)
        layout.addWidget(summary_group)

        table_group = QGroupBox("Tum Satirlarin Karsilastirmasi")
        table_layout = QVBoxLayout(table_group)
        self.row_table = QTableWidget()
        self.row_table.setAlternatingRowColors(True)
        self.row_table.setSortingEnabled(True)
        self.row_table.setWordWrap(False)
        self.row_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.row_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.row_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.row_table.setSizeAdjustPolicy(QAbstractItemView.AdjustIgnored)
        self.row_table.setMinimumHeight(420)
        self.row_table.itemSelectionChanged.connect(self._update_report_detail_from_selection)
        table_layout.addWidget(self.row_table)
        layout.addWidget(table_group)

        detail_group = QGroupBox("Secili Satir Detayi")
        detail_layout = QVBoxLayout(detail_group)
        self.report_detail = QTextEdit()
        self.report_detail.setReadOnly(True)
        self.report_detail.setMinimumHeight(180)
        detail_layout.addWidget(self.report_detail)
        layout.addWidget(detail_group)

        charts_group = QGroupBox("Grafikler")
        charts_layout = QVBoxLayout(charts_group)
        self.slice_plot_label = ResponsiveImageLabel("Gercek vs tahmin grafigi henuz yuklenmedi.")
        self.summary_plot_label = ResponsiveImageLabel("Slice ozet hata grafigi henuz yuklenmedi.")
        self.slice_plot_label.setMinimumHeight(460)
        self.summary_plot_label.setMinimumHeight(460)

        self.slice_plot_scroll = QScrollArea()
        self.slice_plot_scroll.setWidgetResizable(True)
        self.slice_plot_scroll.setWidget(self.slice_plot_label)
        self.slice_plot_scroll.setMinimumHeight(500)

        self.summary_plot_scroll = QScrollArea()
        self.summary_plot_scroll.setWidgetResizable(True)
        self.summary_plot_scroll.setWidget(self.summary_plot_label)
        self.summary_plot_scroll.setMinimumHeight(500)

        slice_chart_group = QGroupBox("Gercek vs Tahmin")
        slice_chart_layout = QVBoxLayout(slice_chart_group)
        self.slice_open_btn = QPushButton("Ayrı pencerede ac")
        self.slice_open_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.slice_open_btn.clicked.connect(lambda: self._open_image_viewer(self.current_slice_plot_path, "Gercek vs Tahmin Grafigi"))
        self.slice_file_btn = QPushButton("Dosyayi ac")
        self.slice_file_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.slice_file_btn.clicked.connect(lambda: self._open_path(self.current_slice_plot_path))
        self.slice_folder_btn = QPushButton("Klasoru ac")
        self.slice_folder_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.slice_folder_btn.clicked.connect(lambda: self._open_folder(self.current_slice_plot_path))
        self.summary_open_btn = QPushButton("Ayrı pencerede ac")
        self.summary_open_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.summary_open_btn.clicked.connect(lambda: self._open_image_viewer(self.current_summary_plot_path, "Slice Ozet Hata Grafigi"))
        self.summary_file_btn = QPushButton("Dosyayi ac")
        self.summary_file_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.summary_file_btn.clicked.connect(lambda: self._open_path(self.current_summary_plot_path))
        self.summary_folder_btn = QPushButton("Klasoru ac")
        self.summary_folder_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.summary_folder_btn.clicked.connect(lambda: self._open_folder(self.current_summary_plot_path))
        self.slice_plot_label.set_click_handler(
            lambda: self._open_image_viewer(self.current_slice_plot_path, "Gercek vs Tahmin Grafigi")
        )
        self.summary_plot_label.set_click_handler(
            lambda: self._open_image_viewer(self.current_summary_plot_path, "Slice Ozet Hata Grafigi")
        )
        slice_chart_layout.addWidget(self.slice_plot_scroll, stretch=1)
        left_buttons = QHBoxLayout()
        left_buttons.addWidget(self.slice_open_btn)
        left_buttons.addWidget(self.slice_file_btn)
        left_buttons.addWidget(self.slice_folder_btn)
        left_buttons.addStretch(1)
        slice_chart_layout.addLayout(left_buttons)

        summary_chart_group = QGroupBox("Slice Ozet Hata Grafigi")
        summary_chart_layout = QVBoxLayout(summary_chart_group)
        summary_chart_layout.addWidget(self.summary_plot_scroll, stretch=1)
        right_buttons = QHBoxLayout()
        right_buttons.addWidget(self.summary_open_btn)
        right_buttons.addWidget(self.summary_file_btn)
        right_buttons.addWidget(self.summary_folder_btn)
        right_buttons.addStretch(1)
        summary_chart_layout.addLayout(right_buttons)
        charts_layout.addWidget(slice_chart_group)
        charts_layout.addWidget(summary_chart_group)
        layout.addWidget(charts_group)

        layout.addStretch(1)

        return tab

    def _build_prediction_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        scenario_group = QGroupBox("Hazir Test Senaryosu")
        scenario_layout = QHBoxLayout(scenario_group)
        self.prediction_scenario_combo = QComboBox()
        for scenario in self.test_scenarios:
            self.prediction_scenario_combo.addItem(str(scenario["name"]))
        scenario_apply_btn = QPushButton("Senaryoyu Yukle")
        scenario_apply_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        scenario_apply_btn.clicked.connect(self._apply_selected_scenario)
        scenario_layout.addWidget(self.prediction_scenario_combo, stretch=1)
        scenario_layout.addWidget(scenario_apply_btn)
        layout.addWidget(scenario_group)

        form_group = QGroupBox("Tekil Tahmin Girisleri")
        form_layout = QFormLayout(form_group)
        self.altitude_input = QLineEdit("11000")
        self.gross_weight_input = QLineEdit("30000")
        self.drag_index_input = QLineEdit("0")
        self.mach_input = QLineEdit("0.22")
        self.fuel_flow_input = QLineEdit("5000")
        self.engine_type_combo = QComboBox()
        self.engine_type_combo.addItems(["one_engine", "two_engine"])
        form_layout.addRow("Altitude (ft)", self.altitude_input)
        form_layout.addRow("Gross Weight (lb)", self.gross_weight_input)
        form_layout.addRow("Drag Index", self.drag_index_input)
        form_layout.addRow("Mach", self.mach_input)
        form_layout.addRow("Fuel Flow (lb / h)", self.fuel_flow_input)
        form_layout.addRow("Engine Type", self.engine_type_combo)
        layout.addWidget(form_group)

        button_row = QHBoxLayout()
        predict_btn = QPushButton("Tahmini Hesapla")
        predict_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        predict_btn.clicked.connect(self._predict_single)
        button_row.addWidget(predict_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.prediction_output = QTextEdit()
        self.prediction_output.setReadOnly(True)
        self.prediction_output.setMinimumHeight(170)
        layout.addWidget(self.prediction_output)

        comparison_group = QGroupBox("Gercek Veriyle Kiyas")
        comparison_layout = QVBoxLayout(comparison_group)
        self.prediction_comparison = QTextEdit()
        self.prediction_comparison.setReadOnly(True)
        self.prediction_comparison.setMinimumHeight(180)
        comparison_layout.addWidget(self.prediction_comparison)
        self.prediction_nearest_table = QTableWidget()
        self.prediction_nearest_table.setMinimumHeight(220)
        self.prediction_nearest_table.setAlternatingRowColors(True)
        comparison_layout.addWidget(self.prediction_nearest_table)
        layout.addWidget(comparison_group, stretch=1)
        return tab

    def _build_nomogram_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls_group = QGroupBox("Nomogram Parametreleri")
        controls = QFormLayout(controls_group)
        self.nomo_model_combo = QComboBox()
        self.nomo_model_combo.addItems(["xgboost", "ft_transformer"])
        self.nomo_engine_combo = QComboBox()
        self.nomo_engine_combo.addItems(["one_engine", "two_engine"])
        self.nomo_altitude_input = QLineEdit("5000")
        self.nomo_weight_input = QLineEdit("50000")
        controls.addRow("Model", self.nomo_model_combo)
        controls.addRow("engine_type", self.nomo_engine_combo)
        controls.addRow("altitude", self.nomo_altitude_input)
        controls.addRow("gross_weight", self.nomo_weight_input)
        layout.addWidget(controls_group)

        nomo_btn = QPushButton("Taslak Nomogram Uret")
        nomo_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        nomo_btn.clicked.connect(self._generate_nomogram)
        layout.addWidget(nomo_btn)

        self.nomogram_info = QTextEdit()
        self.nomogram_info.setReadOnly(True)
        self.nomogram_info.setFixedHeight(110)
        layout.addWidget(self.nomogram_info)

        self.nomogram_image = ResponsiveImageLabel("Nomogram henuz uretilmedi.")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.nomogram_image)
        layout.addWidget(scroll, stretch=1)

        self.nomogram_open_btn = QPushButton("Ayrı pencerede ac")
        self.nomogram_open_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.nomogram_open_btn.clicked.connect(
            lambda: self._open_image_viewer(self.current_nomogram_plot_path, "Taslak Nomogram")
        )
        self.nomogram_file_btn = QPushButton("Dosyayi ac")
        self.nomogram_file_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.nomogram_file_btn.clicked.connect(lambda: self._open_path(self.current_nomogram_plot_path))
        self.nomogram_folder_btn = QPushButton("Klasoru ac")
        self.nomogram_folder_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.nomogram_folder_btn.clicked.connect(lambda: self._open_folder(self.current_nomogram_plot_path))
        self.nomogram_image.set_click_handler(lambda: self._open_image_viewer(self.current_nomogram_plot_path, "Taslak Nomogram"))
        nomo_buttons = QHBoxLayout()
        nomo_buttons.addWidget(self.nomogram_open_btn)
        nomo_buttons.addWidget(self.nomogram_file_btn)
        nomo_buttons.addWidget(self.nomogram_folder_btn)
        nomo_buttons.addStretch(1)
        layout.addLayout(nomo_buttons)
        return tab

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0f172a;
                color: #e2e8f0;
                font-family: Segoe UI;
                font-size: 13px;
            }
            #titleLabel {
                font-size: 28px;
                font-weight: 700;
                color: #f8fafc;
            }
            #subtitleLabel {
                color: #94a3b8;
                font-size: 14px;
            }
            QTabWidget::pane {
                border: 1px solid #1e293b;
                border-radius: 12px;
                background: #111827;
            }
            QTabBar::tab {
                background: #172033;
                color: #cbd5e1;
                padding: 10px 18px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #2563eb;
                color: white;
            }
            QGroupBox {
                border: 1px solid #243244;
                border-radius: 12px;
                margin-top: 12px;
                padding: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QLineEdit, QComboBox, QTextEdit, QTableWidget {
                background: #0b1220;
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 8px;
                color: #e5e7eb;
            }
            QPushButton {
                background: #ef4444;
                border: none;
                border-radius: 10px;
                padding: 8px 14px;
                color: white;
                font-weight: 600;
                min-width: 0px;
            }
            QPushButton:hover {
                background: #dc2626;
            }
            QHeaderView::section {
                background: #1e293b;
                color: #f8fafc;
                padding: 8px;
                border: none;
            }
            QScrollArea {
                border: 1px solid #243244;
                border-radius: 12px;
                background: #0b1220;
            }
            """
        )

    def _dataset_path(self) -> str:
        return str(DataConfig().processed_path)

    def _setup_command_specs(self) -> dict[str, dict[str, object]]:
        python_exec = sys.executable
        dataset_path = self._dataset_path()
        return {
            "Veri pipeline": {
                "command": [python_exec, "scripts/run_data_pipeline.py"],
                "eta": "yaklasik 10-30 sn",
                "artifacts": ["data/processed/combined_specific_range.csv"],
            },
            "XGBoost egit + rapor": {
                "command": [
                    python_exec,
                    "scripts/train_xgboost.py",
                    "--dataset",
                    dataset_path,
                    "--device",
                    self.setup_xgb_device_combo.currentText(),
                    "--run-table-report",
                ],
                "eta": "yaklasik 30 sn - 3 dk",
                "artifacts": [
                    "artifacts/xgboost/model.json",
                    "artifacts/xgboost/preprocessor.joblib",
                    "artifacts/xgboost/reports/*.csv|*.xlsx|*.png",
                ],
            },
            "FT-Transformer egit + rapor": {
                "command": [
                    python_exec,
                    "scripts/train_ft_transformer.py",
                    "--dataset",
                    dataset_path,
                    "--device",
                    self.setup_ft_device_combo.currentText(),
                    "--run-table-report",
                ],
                "eta": "yaklasik 2-15 dk",
                "artifacts": [
                    "artifacts/ft_transformer/model.pt",
                    "artifacts/ft_transformer/preprocessor.joblib",
                    "artifacts/ft_transformer/reports/*.csv|*.xlsx|*.png",
                ],
            },
            "Toplu rapor uret": {
                "command": [
                    python_exec,
                    "scripts/run_table_report.py",
                    "--dataset",
                    dataset_path,
                    "--model",
                    "both",
                    "--ft-device",
                    self.setup_ft_device_combo.currentText(),
                ],
                "eta": "yaklasik 20 sn - 3 dk",
                "artifacts": [
                    "artifacts/xgboost/reports/*.csv|*.xlsx|*.png",
                    "artifacts/ft_transformer/reports/*.csv|*.xlsx|*.png",
                ],
            },
            "PSO calistir": {
                "command": [
                    python_exec,
                    "scripts/run_pso.py",
                    "--dataset",
                    dataset_path,
                ],
                "eta": "yaklasik 5-30 dk",
                "artifacts": ["optimization outputs / metrics"],
            },
            "Model karsilastir": {
                "command": [
                    python_exec,
                    "scripts/compare_models.py",
                    "--dataset",
                    dataset_path,
                ],
                "eta": "yaklasik 1-5 dk",
                "artifacts": ["terminal comparison summary"],
            },
        }

    def _setup_commands(self) -> dict[str, list[str]]:
        return {label: spec["command"] for label, spec in self._setup_command_specs().items()}

    def _refresh_setup_preview(self) -> None:
        specs = self._setup_command_specs()
        sequence = [
            "Veri pipeline",
            "XGBoost egit + rapor",
            "FT-Transformer egit + rapor",
            "Toplu rapor uret",
        ]
        lines = ["Quickstart setup akisi:\n"]
        for label in sequence:
            lines.append(f"[{label}]")
            lines.append(" ".join(specs[label]["command"]))
            lines.append(f"Tahmini sure: {specs[label]['eta']}")
            lines.append("")
        self.setup_command_preview.setPlainText("\n".join(lines).strip())
        artifact_lines = ["Artefact Ozetleri:\n"]
        for label, spec in specs.items():
            artifact_lines.append(f"[{label}]")
            for artifact in spec["artifacts"]:
                artifact_lines.append(f"- {artifact}")
            artifact_lines.append("")
        self.setup_artifact_info.setPlainText("\n".join(artifact_lines).strip())

    def _open_image_viewer(self, image_path: Path | None, title: str) -> None:
        if image_path is None or not image_path.exists():
            QMessageBox.information(self, "Gorsel yok", "Acilabilecek bir grafik bulunamadi.")
            return
        dialog = ImageViewerDialog(title, image_path, self)
        dialog.exec()

    def _open_path(self, file_path: Path | None) -> None:
        if file_path is None or not file_path.exists():
            QMessageBox.information(self, "Dosya yok", "Acilabilecek bir dosya bulunamadi.")
            return
        os.startfile(str(file_path))

    def _open_folder(self, file_path: Path | None) -> None:
        if file_path is None or not file_path.exists():
            QMessageBox.information(self, "Klasor yok", "Acilabilecek bir klasor bulunamadi.")
            return
        os.startfile(str(file_path.parent))

    def _append_setup_log(self, text: str) -> None:
        if not text:
            return
        current = self.setup_log.toPlainText()
        self.setup_log.setPlainText(f"{current}{text}")
        cursor = self.setup_log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.setup_log.setTextCursor(cursor)

    def _run_process(self, command_parts: list[str], label: str, on_finish: Callable[[bool], None] | None = None) -> None:
        if self.active_process is not None:
            QMessageBox.information(self, "Calisan islem var", "Once aktif setup komutunun bitmesini bekleyelim.")
            return

        process = QProcess(self)
        self.active_process = process
        process.setProgram(command_parts[0])
        process.setArguments(command_parts[1:])
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(
            lambda: self._append_setup_log(bytes(process.readAllStandardOutput()).decode(errors="replace"))
        )

        def _finished(exit_code: int, _status) -> None:
            success = exit_code == 0
            self._append_setup_log(f"\n[{label}] tamamlandi. exit_code={exit_code}\n\n")
            self.setup_status.setText(f"{label} {'tamamlandi' if success else 'basarisiz oldu'}.")
            self.setup_progress.setValue(100 if success else 0)
            self.active_process = None
            if on_finish is not None:
                on_finish(success)

        process.finished.connect(_finished)
        self.setup_status.setText(f"{label} calisiyor...")
        self.setup_progress.setValue(10)
        self._append_setup_log(f"\n$ {' '.join(command_parts)}\n\n")
        process.start()

    def _run_named_command(self, label: str) -> None:
        self._run_process(self._setup_commands()[label], label)

    def _run_full_setup_sequence(self) -> None:
        sequence = [
            "Veri pipeline",
            "XGBoost egit + rapor",
            "FT-Transformer egit + rapor",
            "Toplu rapor uret",
        ]

        def _run_next(index: int) -> None:
            if index >= len(sequence):
                self.setup_status.setText("Quickstart setup akisi tamamlandi.")
                self.setup_progress.setValue(100)
                return

            label = sequence[index]
            self.setup_progress.setValue(int((index / max(len(sequence), 1)) * 100))

            def _after(success: bool) -> None:
                if success:
                    _run_next(index + 1)
                else:
                    self.setup_status.setText(f"Quickstart setup akisi {label} adiminda durdu.")
                    self.setup_progress.setValue(0)

            self._run_process(self._setup_commands()[label], label, on_finish=_after)

        _run_next(0)

    def _load_predictor(self, model_key: str):
        if model_key == "xgboost":
            if self.xgb_predictor is None:
                self.xgb_predictor = XGBoostPredictor.from_artifacts()
            return self.xgb_predictor
        self.ft_predictor = FTTransformerPredictor.from_artifacts(device=self.ft_device_combo.currentText())
        return self.ft_predictor

    def _report_error_column(self) -> str:
        return "xgboost_absolute_error" if self.report_model_combo.currentText() == "xgboost" else "ft_transformer_absolute_error"

    def _set_table(self, table: QTableWidget, df: pd.DataFrame) -> None:
        table.clear()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(list(df.columns))
        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, value in enumerate(row):
                table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        table.resizeColumnsToContents()

    def _populate_prediction_comparison(self, frame: pd.DataFrame, xgb_value: float, ft_value: float) -> None:
        exact_matches = find_exact_match(self.reference_df, frame)
        if not exact_matches.empty:
            actual_value = float(exact_matches.iloc[0]["specific_range"])
            self.prediction_comparison.setPlainText(
                "Birebir eslesen gercek satir bulundu.\n\n"
                f"actual_specific_range = {actual_value:.6f}\n"
                f"XGBoost absolute error = {abs(xgb_value - actual_value):.6f}\n"
                f"FT absolute error = {abs(ft_value - actual_value):.6f}"
            )
            self._set_table(self.prediction_nearest_table, exact_matches)
            return

        nearest = find_nearest_reference_rows(self.reference_df, frame, top_k=5)
        if nearest.empty:
            self.prediction_comparison.setPlainText("Gercek veriyle kiyas icin uygun satir bulunamadi.")
            self.prediction_nearest_table.clear()
            return

        nearest_actual = float(nearest.iloc[0]["specific_range"])
        self.prediction_comparison.setPlainText(
            "Birebir eslesen satir yok. En yakin gercek satirlar gosteriliyor.\n\n"
            f"nearest_actual = {nearest_actual:.6f}\n"
            f"XGBoost abs diff to nearest = {abs(xgb_value - nearest_actual):.6f}\n"
            f"FT abs diff to nearest = {abs(ft_value - nearest_actual):.6f}"
        )
        visible_columns = [
            column
            for column in [
                "engine_type",
                "altitude",
                "gross_weight",
                "drag_index",
                "mach",
                "fuel_flow",
                "specific_range",
                "distance",
            ]
            if column in nearest.columns
        ]
        self._set_table(self.prediction_nearest_table, nearest[visible_columns])

    def _apply_selected_scenario(self) -> None:
        if not self.test_scenarios:
            return
        scenario_name = self.prediction_scenario_combo.currentText()
        scenario = next((item for item in self.test_scenarios if str(item["name"]) == scenario_name), None)
        if scenario is None:
            return
        self.altitude_input.setText(str(float(scenario["altitude"])))
        self.gross_weight_input.setText(str(float(scenario["gross_weight"])))
        self.drag_index_input.setText(str(float(scenario["drag_index"])))
        self.mach_input.setText(str(float(scenario["mach"])))
        self.fuel_flow_input.setText(str(float(scenario["fuel_flow"])))
        self.engine_type_combo.setCurrentText(str(scenario["engine_type"]))

    def _populate_report_filters(self) -> None:
        if self.report_row_df.empty:
            self.report_engine_filter.blockSignals(True)
            self.report_altitude_filter.blockSignals(True)
            self.report_engine_filter.clear()
            self.report_altitude_filter.clear()
            self.report_engine_filter.addItem("All")
            self.report_altitude_filter.addItem("All")
            self.report_engine_filter.blockSignals(False)
            self.report_altitude_filter.blockSignals(False)
            return

        engine_values = ["All"] + sorted(self.report_row_df["engine_type"].dropna().astype(str).unique().tolist())
        altitude_values = ["All"] + [f"{int(value)} ft" for value in sorted(self.report_row_df["altitude"].dropna().astype(float).unique().tolist())]
        current_engine = self.report_engine_filter.currentText()
        current_altitude = self.report_altitude_filter.currentText()
        self.report_engine_filter.blockSignals(True)
        self.report_altitude_filter.blockSignals(True)
        self.report_engine_filter.clear()
        self.report_altitude_filter.clear()
        self.report_engine_filter.addItems(engine_values)
        self.report_altitude_filter.addItems(altitude_values)
        if current_engine in engine_values:
            self.report_engine_filter.setCurrentText(current_engine)
        if current_altitude in altitude_values:
            self.report_altitude_filter.setCurrentText(current_altitude)
        self.report_engine_filter.blockSignals(False)
        self.report_altitude_filter.blockSignals(False)

    def _apply_report_filters(self) -> None:
        if self.report_row_df.empty:
            return

        filtered = self.report_row_df.copy()
        engine_value = self.report_engine_filter.currentText()
        altitude_value = self.report_altitude_filter.currentText()
        error_column = self._report_error_column()

        if engine_value and engine_value != "All":
            filtered = filtered[filtered["engine_type"] == engine_value]
        if altitude_value and altitude_value != "All":
            altitude = float(altitude_value.replace(" ft", ""))
            filtered = filtered[filtered["altitude"].astype(float) == altitude]
        sort_mode = self.report_sort_filter.currentText()
        if error_column in filtered.columns:
            if sort_mode == "En buyuk hata ustte":
                filtered = filtered.sort_values(error_column, ascending=False)
            elif sort_mode == "En kucuk hata ustte":
                filtered = filtered.sort_values(error_column, ascending=True)
            else:
                filtered = filtered.sort_values("row_id", ascending=True)
        self.report_filtered_df = filtered.reset_index(drop=True)
        self._set_table(self.row_table, self.report_filtered_df)
        if not self.report_filtered_df.empty:
            self.row_table.selectRow(0)
            self._update_report_detail_from_selection()
        else:
            self.report_detail.setPlainText("Filtre sonucunda satir kalmadi.")

    def _update_report_detail_from_selection(self) -> None:
        if self.report_filtered_df.empty:
            return
        selected_items = self.row_table.selectedItems()
        if not selected_items:
            return
        row_index = selected_items[0].row()
        if row_index >= len(self.report_filtered_df):
            return
        row = self.report_filtered_df.iloc[row_index]
        lines = [f"{column}: {row[column]}" for column in self.report_filtered_df.columns]
        self.report_detail.setPlainText("\n".join(lines))

    def _set_plot(self, label: ResponsiveImageLabel, path: Path) -> None:
        label.set_plot_path(path)

    def _load_report(self) -> None:
        model_key = self.report_model_combo.currentText()
        paths = report_paths(model_key)
        if not paths["row_level"].exists():
            self.report_summary.setPlainText(
                "Hazir rapor bulunamadi.\n"
                "Su komutu calistir:\n"
                f"python scripts/run_table_report.py --dataset data/processed/combined_specific_range.csv --model {model_key}"
            )
            self.row_table.clear()
            self.current_slice_plot_path = None
            self.current_summary_plot_path = None
            self.slice_plot_label.clear_plot()
            self.summary_plot_label.clear_plot()
            self.report_row_df = pd.DataFrame()
            self.report_filtered_df = pd.DataFrame()
            self._populate_report_filters()
            return

        row_df = pd.read_csv(paths["row_level"])
        overall_df = pd.read_csv(paths["overall_summary"])
        slice_df = pd.read_csv(paths["slice_summary"])
        row = overall_df.iloc[0]
        self.report_summary.setPlainText(
            f"Model: {model_key}\n"
            f"rows={int(row['rows'])} | mae={row['mae']:.6f} | rmse={row['rmse']:.6f} | "
            f"mape={row['mape']:.4f} | r2={row['r2']:.6f}\n"
            f"Excel raporu: {paths['excel_report']}\n"
            f"Slice summary satir sayisi: {len(slice_df)}"
        )
        self.report_row_df = row_df.copy()
        self._populate_report_filters()
        self._apply_report_filters()
        self.current_slice_plot_path = paths["slice_plot"]
        self.current_summary_plot_path = paths["summary_plot"]
        self._set_plot(self.slice_plot_label, paths["slice_plot"])
        self._set_plot(self.summary_plot_label, paths["summary_plot"])

    def _predict_single(self) -> None:
        try:
            frame = build_single_row_frame(
                altitude=float(self.altitude_input.text()),
                gross_weight=float(self.gross_weight_input.text()),
                drag_index=float(self.drag_index_input.text()),
                mach=float(self.mach_input.text()),
                fuel_flow=float(self.fuel_flow_input.text()),
                engine_type=self.engine_type_combo.currentText(),
            )
            xgb_value = self._load_predictor("xgboost").predict_from_frame(frame)
            ft_value = self._load_predictor("ft_transformer").predict_from_frame(frame)
            self.prediction_output.setPlainText(
                "Tekil Tahmin Sonucu\n\n"
                f"XGBoost: {xgb_value:.6f}\n"
                f"FT-Transformer: {ft_value:.6f}\n"
                f"Fark (FT - XGB): {ft_value - xgb_value:.6f}"
            )
            self._populate_prediction_comparison(frame, xgb_value, ft_value)
        except Exception as exc:
            QMessageBox.critical(self, "Hata", str(exc))

    def _generate_nomogram(self) -> None:
        try:
            model_key = self.nomo_model_combo.currentText()
            predictor = self._load_predictor(model_key)
            base = DataConfig().xgboost_artifact_dir if model_key == "xgboost" else DataConfig().ft_transformer_artifact_dir
            result = generate_nomogram_report(
                self.reference_df,
                model_name=model_key,
                batch_predict_fn=predictor.predict_many_from_frame,
                output_dir=base / "nomogram_reports",
                engine_type=self.nomo_engine_combo.currentText(),
                altitude=float(self.nomo_altitude_input.text()),
                gross_weight=float(self.nomo_weight_input.text()),
            )
            self.nomogram_info.setPlainText(
                f"Nomogram olusturuldu.\n"
                f"row_level: {result.row_level_csv}\n"
                f"slice_summary: {result.slice_summary_csv}\n"
                f"plot: {result.nomogram_png}"
            )
            self.current_nomogram_plot_path = result.nomogram_png
            self._set_plot(self.nomogram_image, result.nomogram_png)
        except Exception as exc:
            QMessageBox.critical(self, "Hata", str(exc))


def main() -> None:
    app = QApplication(sys.argv)
    window = ModernDesktopApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
