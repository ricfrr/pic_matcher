
import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QSlider, QScrollArea,
    QMessageBox, QCheckBox, QGroupBox, QComboBox, QTabWidget, QDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PIL import Image

import io
import image_scanner

# --- Custom Widgets ---
class ClickableLabel(QLabel):
    clicked = pyqtSignal(str)

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        self.clicked.emit(self.image_path)
        super().mousePressEvent(event)

# --- Worker Threads ---

class DuplicateScannerThread(QThread):
    progress_updated = pyqtSignal(int)
    scan_complete = pyqtSignal(list)
    status_update = pyqtSignal(str)

    def __init__(self, folder_path, method, threshold):
        super().__init__()
        self.folder_path = folder_path
        self.method = method
        self.threshold = threshold

    def run(self):
        duplicate_groups = image_scanner.scan_for_duplicates(
            self.folder_path, self.method, self.threshold, self.progress_updated, self.status_update
        )
        self.scan_complete.emit(duplicate_groups)

class ScreenshotScannerThread(QThread):
    progress_updated = pyqtSignal(int)
    scan_complete = pyqtSignal(list)
    status_update = pyqtSignal(str)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        screenshots = image_scanner.find_screenshots(
            self.folder_path, self.progress_updated, self.status_update
        )
        self.scan_complete.emit(screenshots)


# --- Main Application Window ---

class PicMatcher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pic Matcher - Advanced Photo Tools")
        self.setGeometry(100, 100, 1300, 800)
        self.folder_path = None

        # Main Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create Tabs
        self.duplicate_finder_tab = QWidget()
        self.screenshot_finder_tab = QWidget()

        self.tabs.addTab(self.duplicate_finder_tab, "Duplicate Finder")
        self.tabs.addTab(self.screenshot_finder_tab, "Screenshot Finder")

        # Setup UI for each tab
        self.setup_duplicate_finder_ui()
        self.setup_screenshot_finder_ui()

    # --- Duplicate Finder UI Setup ---
    def setup_duplicate_finder_ui(self):
        self.duplicate_groups = []
        self.displayed_duplicate_groups_count = 0
        self.DISPLAY_CHUNK_SIZE = 50 # Number of groups to display at a time

        layout = QVBoxLayout(self.duplicate_finder_tab)

        # Controls
        scan_controls_layout = QHBoxLayout()
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Perceptual Hash (pHash)", "Difference Hash (dHash)", "Wavelet Hash (wHash)", "Deep Learning (ResNet)"])
        self.lbl_threshold = QLabel()
        self.slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_threshold.setTickInterval(1)
        self.slider_threshold.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.btn_scan_duplicates = QPushButton("Scan for Duplicates")
        self.btn_scan_duplicates.setEnabled(False)

        scan_controls_layout.addWidget(QLabel("Method:"))
        scan_controls_layout.addWidget(self.method_combo)
        scan_controls_layout.addWidget(self.lbl_threshold)
        scan_controls_layout.addWidget(self.slider_threshold, 1)
        scan_controls_layout.addWidget(self.btn_scan_duplicates)
        
        # Progress Bar & Status
        progress_layout = QHBoxLayout()
        self.dup_progress_bar = QProgressBar()
        self.dup_progress_bar.setVisible(False)
        self.dup_scan_status = QLabel("")
        self.dup_scan_status.setVisible(False)
        progress_layout.addWidget(self.dup_progress_bar, 1)
        progress_layout.addWidget(self.dup_scan_status)

        # Results Area
        self.dup_scroll_area = QScrollArea()
        self.dup_scroll_area.setWidgetResizable(True)
        self.dup_results_widget = QWidget()
        self.dup_results_layout = QVBoxLayout(self.dup_results_widget)
        self.dup_results_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.dup_scroll_area.setWidget(self.dup_results_widget)
        
        # Bottom Actions
        actions_layout = QHBoxLayout()
        self.dup_status = QLabel("Select a folder to begin.")
        self.btn_delete_duplicates = QPushButton("Delete Marked Photos")
        self.btn_delete_duplicates.setEnabled(False)
        self.btn_delete_duplicates.setStyleSheet("background-color: #ff6b6b; font-weight: bold;")
        
        self.btn_load_more_duplicates = QPushButton("Load More Duplicates")
        self.btn_load_more_duplicates.setVisible(False) # Initially hidden

        actions_layout.addWidget(self.dup_status, 1)
        actions_layout.addStretch()
        actions_layout.addWidget(self.btn_load_more_duplicates)
        actions_layout.addWidget(self.btn_delete_duplicates)

        # Folder selection (common for all tabs)
        folder_layout = QHBoxLayout()
        self.btn_select_folder = QPushButton("Select Image Folder")
        self.lbl_folder_path = QLabel("No folder selected.")
        self.lbl_folder_path.setStyleSheet("font-style: italic;")
        folder_layout.addWidget(self.btn_select_folder)
        folder_layout.addWidget(self.lbl_folder_path, 1)

        layout.addLayout(folder_layout)
        layout.addLayout(scan_controls_layout)
        layout.addLayout(progress_layout)
        layout.addWidget(self.dup_scroll_area, 1)
        layout.addLayout(actions_layout)

        # --- Connect Signals ---
        self.btn_select_folder.clicked.connect(self.select_folder)
        self.method_combo.currentIndexChanged.connect(self.update_threshold_slider)
        self.slider_threshold.valueChanged.connect(self.update_threshold_label)
        self.btn_scan_duplicates.clicked.connect(self.start_duplicate_scan)
        self.btn_delete_duplicates.clicked.connect(self.delete_marked_duplicates)
        self.btn_load_more_duplicates.clicked.connect(self.load_more_duplicates)

        self.update_threshold_slider(0)

    # --- Screenshot Finder UI Setup ---
    def setup_screenshot_finder_ui(self):
        self.screenshots = []
        layout = QVBoxLayout(self.screenshot_finder_tab)

        # Controls
        controls_layout = QHBoxLayout()
        self.btn_scan_screenshots = QPushButton("Find All Screenshots")
        self.btn_scan_screenshots.setEnabled(False)
        controls_layout.addWidget(self.btn_scan_screenshots)
        controls_layout.addStretch()

        # Progress
        progress_layout = QHBoxLayout()
        self.ss_progress_bar = QProgressBar()
        self.ss_progress_bar.setVisible(False)
        self.ss_scan_status = QLabel("")
        self.ss_scan_status.setVisible(False)
        progress_layout.addWidget(self.ss_progress_bar, 1)
        progress_layout.addWidget(self.ss_scan_status)

        # Results
        self.ss_scroll_area = QScrollArea()
        self.ss_scroll_area.setWidgetResizable(True)
        self.ss_results_widget = QWidget()
        self.ss_results_layout = QVBoxLayout(self.ss_results_widget)
        self.ss_results_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.ss_scroll_area.setWidget(self.ss_results_widget)

        # Actions
        actions_layout = QHBoxLayout()
        self.ss_status = QLabel("Ready.")
        self.btn_delete_screenshots = QPushButton("Delete Marked Screenshots")
        self.btn_delete_screenshots.setEnabled(False)
        self.btn_delete_screenshots.setStyleSheet("background-color: #ff6b6b; font-weight: bold;")
        actions_layout.addWidget(self.ss_status, 1)
        actions_layout.addStretch()
        actions_layout.addWidget(self.btn_delete_screenshots)

        layout.addLayout(controls_layout)
        layout.addLayout(progress_layout)
        layout.addWidget(self.ss_scroll_area)
        layout.addLayout(actions_layout)

        # --- Connect Signals ---
        self.btn_scan_screenshots.clicked.connect(self.start_screenshot_scan)
        self.btn_delete_screenshots.clicked.connect(self.delete_marked_screenshots)


    # --- Common Functions ---
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folder_path = folder_path
            self.lbl_folder_path.setText(self.folder_path)
            self.btn_scan_duplicates.setEnabled(True)
            self.btn_scan_screenshots.setEnabled(True)
            self.dup_status.setText("Ready to scan for duplicates.")
            self.ss_status.setText("Ready to scan for screenshots.")

    def create_thumbnail(self, image_path, size=(150, 150)):
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                img.thumbnail(size)
                buffer = io.BytesIO()
                img.save(buffer, "PNG")
                pixmap = QPixmap()
                pixmap.loadFromData(buffer.getvalue())
                return pixmap, width, height
        except Exception as e:
            print(f"Error creating thumbnail for {image_path}: {e}")
            return QPixmap(), 0, 0

    # --- Duplicate Finder Logic ---
    def update_threshold_slider(self, index):
        method = self.method_combo.itemText(index)
        if "Deep Learning" in method:
            self.slider_threshold.setMinimum(80)
            self.slider_threshold.setMaximum(100)
            self.slider_threshold.setValue(95)
        else:
            self.slider_threshold.setMinimum(0)
            self.slider_threshold.setMaximum(20)
            self.slider_threshold.setValue(5)
        self.update_threshold_label(self.slider_threshold.value())

    def update_threshold_label(self, value):
        method = self.method_combo.currentText()
        if "Deep Learning" in method:
            self.lbl_threshold.setText(f"Similarity >= {value}%")
        else:
            self.lbl_threshold.setText(f"Distance <= {value}")

    def start_duplicate_scan(self):
        self.clear_duplicate_results()
        self.dup_progress_bar.setVisible(True)
        self.dup_scan_status.setVisible(True)
        self.set_controls_enabled(False)

        method_text = self.method_combo.currentText()
        threshold = self.slider_threshold.value()
        
        self.dup_scanner_thread = DuplicateScannerThread(self.folder_path, method_text, threshold)
        self.dup_scanner_thread.progress_updated.connect(self.dup_progress_bar.setValue)
        self.dup_scanner_thread.status_update.connect(self.dup_scan_status.setText)
        self.dup_scanner_thread.scan_complete.connect(self.on_duplicate_scan_complete)
        self.dup_scanner_thread.start()

    def on_duplicate_scan_complete(self, groups):
        self.dup_progress_bar.setVisible(False)
        self.dup_scan_status.setVisible(False)
        self.set_controls_enabled(True)
        self.duplicate_groups = groups
        self.displayed_duplicate_groups_count = 0 # Reset count for new scan
        
        if not groups:
            self.dup_status.setText("Scan complete. No duplicates found.")
            self.btn_load_more_duplicates.setVisible(False)
            return

        self.dup_status.setText(f"Scan complete. Found {len(groups)} duplicate groups.")
        self.load_more_duplicates() # Load the first chunk
        self.btn_delete_duplicates.setEnabled(True)

    def mark_group_for_deletion(self, group_box):
        checkboxes = group_box.findChildren(QCheckBox)
        for checkbox in checkboxes:
            checkbox.setChecked(True)

    def load_more_duplicates(self):
        start_index = self.displayed_duplicate_groups_count
        end_index = min(start_index + self.DISPLAY_CHUNK_SIZE, len(self.duplicate_groups))

        for i in range(start_index, end_index):
            group = self.duplicate_groups[i]
            group_box = QGroupBox(f"Duplicate Group ({len(group)} images)")
            
            # Layout for images within the group
            images_layout = QHBoxLayout()
            for image_path in group:
                item_layout, checkbox = self.create_image_display_item(image_path)
                images_layout.addLayout(item_layout)
            
            # Scroll area for images within the group
            images_scroll_area = QScrollArea()
            images_scroll_area.setWidgetResizable(True)
            images_scroll_area.setFixedHeight(260)
            images_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            images_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            
            images_widget = QWidget()
            images_widget.setLayout(images_layout)
            images_scroll_area.setWidget(images_widget)
            images_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # Align items to top to prevent vertical stretching

            # Layout for the group box (includes images scroll area and mark all button)
            group_box_layout = QVBoxLayout()
            group_box_layout.addWidget(images_scroll_area)

            btn_mark_all = QPushButton("Mark All in Group for Deletion")
            btn_mark_all.clicked.connect(lambda checked, gb=group_box: self.mark_group_for_deletion(gb))
            group_box_layout.addWidget(btn_mark_all)

            group_box.setLayout(group_box_layout)
            self.dup_results_layout.addWidget(group_box)
        
        self.displayed_duplicate_groups_count = end_index

        # Show/hide "Load More" button
        if self.displayed_duplicate_groups_count < len(self.duplicate_groups):
            self.btn_load_more_duplicates.setVisible(True)
        else:
            self.btn_load_more_duplicates.setVisible(False)

    def delete_marked_duplicates(self):
        files_to_delete = []
        for i in range(self.dup_results_layout.count()):
            group_box = self.dup_results_layout.itemAt(i).widget()
            if isinstance(group_box, QGroupBox):
                checkboxes = group_box.findChildren(QCheckBox)
                for checkbox in checkboxes:
                    if checkbox.isChecked():
                        files_to_delete.append(checkbox.property("image_path"))
        self.confirm_and_delete_files(files_to_delete, self.start_duplicate_scan)

    def clear_duplicate_results(self):
        while self.dup_results_layout.count():
            child = self.dup_results_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        self.btn_delete_duplicates.setEnabled(False)
        self.btn_load_more_duplicates.setVisible(False) # Hide load more button
        self.dup_status.setText("Ready.")

    # --- Screenshot Finder Logic ---
    def start_screenshot_scan(self):
        self.clear_screenshot_results()
        self.ss_progress_bar.setVisible(True)
        self.ss_scan_status.setVisible(True)
        self.set_controls_enabled(False)

        self.ss_scanner_thread = ScreenshotScannerThread(self.folder_path)
        self.ss_scanner_thread.progress_updated.connect(self.ss_progress_bar.setValue)
        self.ss_scanner_thread.status_update.connect(self.ss_scan_status.setText)
        self.ss_scanner_thread.scan_complete.connect(self.on_screenshot_scan_complete)
        self.ss_scanner_thread.start()

    def on_screenshot_scan_complete(self, screenshots):
        self.ss_progress_bar.setVisible(False)
        self.ss_scan_status.setVisible(False)
        self.set_controls_enabled(True)
        self.screenshots = screenshots

        if not screenshots:
            self.ss_status.setText("Scan complete. No screenshots found.")
            return
        
        self.ss_status.setText(f"Scan complete. Found {len(screenshots)} potential screenshots.")
        self.display_screenshot_results()
        self.btn_delete_screenshots.setEnabled(True)

    def display_screenshot_results(self):
        # For screenshots, we can use a more compact grid layout
        grid_layout = QHBoxLayout() # This will be replaced with a proper grid later
        for image_path in self.screenshots:
            item_layout, checkbox = self.create_image_display_item(image_path)
            self.ss_results_layout.addLayout(item_layout)

    def delete_marked_screenshots(self):
        files_to_delete = []
        for i in range(self.ss_results_layout.count()):
            item_layout = self.ss_results_layout.itemAt(i)
            # This needs to be smarter to find the checkbox
            checkbox = item_layout.itemAt(2).widget() # Assuming checkbox is the 3rd item
            if checkbox and checkbox.isChecked():
                files_to_delete.append(checkbox.property("image_path"))
        self.confirm_and_delete_files(files_to_delete, self.start_screenshot_scan)

    def clear_screenshot_results(self):
        while self.ss_results_layout.count():
            child = self.ss_results_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        self.btn_delete_screenshots.setEnabled(False)
        self.ss_status.setText("Ready.")

    # --- Helper Functions ---
    def create_image_display_item(self, image_path):
        item_layout = QVBoxLayout()
        pixmap_label = ClickableLabel(image_path)
        pixmap, width, height = self.create_thumbnail(image_path)
        if pixmap:
            pixmap_label.setPixmap(pixmap)
        pixmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap_label.clicked.connect(self.show_full_image_preview)

        file_size = os.path.getsize(image_path) / 1024
        info_text = f"{os.path.basename(image_path)}\n{width}x{height} - {file_size:.1f} KB"
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        checkbox = QCheckBox("Mark for deletion")
        checkbox.setProperty("image_path", image_path)

        item_layout.addWidget(pixmap_label)
        item_layout.addWidget(info_label)
        item_layout.addWidget(checkbox)
        return item_layout, checkbox

    def show_full_image_preview(self, image_path):
        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle(os.path.basename(image_path))
        preview_dialog.setWindowFlags(preview_dialog.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowCloseButtonHint)
        
        dialog_layout = QVBoxLayout(preview_dialog)
        
        scroll_area = QScrollArea(preview_dialog)
        scroll_area.setWidgetResizable(True)
        
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        try:
            full_pixmap = QPixmap(image_path)
            if full_pixmap.isNull():
                raise Exception("Could not load image")
            
            # Scale pixmap to fit the dialog, maintaining aspect ratio
            # Get screen size to set a reasonable maximum for the dialog
            screen_geo = QApplication.primaryScreen().availableGeometry()
            max_width = screen_geo.width() * 0.9
            max_height = screen_geo.height() * 0.9

            # Calculate scaled size
            scaled_pixmap = full_pixmap.scaled(
                int(max_width), int(max_height),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            image_label.setPixmap(scaled_pixmap)
            image_label.adjustSize() # Adjust label size to fit the scaled pixmap

        except Exception as e:
            image_label.setText(f"Error loading image: {e}")
            image_label.setStyleSheet("color: red;")

        scroll_area.setWidget(image_label)
        dialog_layout.addWidget(scroll_area)
        
        # Resize dialog to fit the scaled image, plus some padding
        preview_dialog.resize(image_label.width() + 50, image_label.height() + 50)
        preview_dialog.exec()

    def confirm_and_delete_files(self, files_to_delete, rescan_function):
        if not files_to_delete:
            QMessageBox.information(self, "No Files Selected", "No files were marked for deletion.")
            return

        reply = QMessageBox.warning(self, "Confirm Deletion", 
            f"You are about to permanently delete {len(files_to_delete)} files.\nThis action cannot be undone.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel, 
            QMessageBox.StandardButton.Cancel)

        if reply == QMessageBox.StandardButton.Ok:
            deleted_count = 0
            failed_deletions = []
            for fpath in files_to_delete:
                try:
                    os.remove(fpath)
                    deleted_count += 1
                except OSError as e:
                    failed_deletions.append(f"{os.path.basename(fpath)}: {e}")
            
            message = f"Successfully deleted {deleted_count} files."
            if failed_deletions:
                message += f"\n\nFailed to delete {len(failed_deletions)} files:\n" + "\n".join(failed_deletions)
                QMessageBox.warning(self, "Deletion with Errors", message)
            else:
                QMessageBox.information(self, "Deletion Complete", message)
            rescan_function()

    def set_controls_enabled(self, enabled):
        self.btn_select_folder.setEnabled(enabled)
        self.btn_scan_duplicates.setEnabled(enabled)
        self.btn_scan_screenshots.setEnabled(enabled)
        self.method_combo.setEnabled(enabled)
        self.slider_threshold.setEnabled(enabled)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PicMatcher()
    window.show()
    sys.exit(app.exec())
