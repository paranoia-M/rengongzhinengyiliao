import os
import cv2
import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSlider, QGroupBox, QSplitter, QFrame, 
                             QProgressBar, QCheckBox, QListWidget, QMessageBox)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont

class ProcessEngine(QThread):
    """
    【高性能处理引擎】
    修复点：增强了算法效果，使处理前后的视觉差异更具冲击力
    """
    finished = Signal(object, object)

    def __init__(self):
        super().__init__()
        self.raw_img = None
        self.params = {}

    def set_data(self, img, params):
        self.raw_img = img
        self.params = params

    def run(self):
        if self.raw_img is None: return
        
        # 1. 转换为灰度（医疗影像 AI 的标准流转格式）
        img = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2GRAY)
        
        # 2. 增强型 CLAHE (对比度受限自适应直方图均衡化)
        # 增加强度，使结节内部纹理（如钙化、血流空腔）极其明显
        if self.params['clahe']:
            # 强化 clipLimit
            clahe = cv2.createCLAHE(clipLimit=self.params['clahe_val'] * 2.5, tileGridSize=(8, 8))
            img = clahe.apply(img)

        # 3. 影像去噪
        if self.params['denoise']:
            img = cv2.bilateralFilter(img, 9, 75, 75)

        # 4. 增强型边界显化 (Canny)
        if self.params['edge']:
            # 使用较小的阈值抓取更多边缘细节
            edges = cv2.Canny(img, self.params['canny_t'], self.params['canny_t'] * 2)
            
            # 【视觉增强】：对边缘进行膨胀处理，让线条变粗，方便观察
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # 将亮白色的边缘以 50% 的权重叠加，效果非常显著
            img = cv2.addWeighted(img, 0.5, edges, 0.5, 0)

        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        # 发送处理结果
        self.finished.emit(img, hist)

class PreprocessPage(QWidget):
    def __init__(self):
        super().__init__()
        self.raw_cv_img = None      # 原始 BGR 图像
        self.processed_cv_img = None # 处理后的灰度特征图
        self.engine = ProcessEngine()
        self.engine.finished.connect(self.on_process_done)
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # --- 1. 左侧：病例文件库 ---
        self.file_panel = QFrame()
        self.file_panel.setFixedWidth(240)
        self.file_panel.setStyleSheet("background: white; border-radius: 8px; border: 1px solid #ddd;")
        fp_layout = QVBoxLayout(self.file_panel)
        fp_layout.addWidget(QLabel("📂 待处理病例库"))
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.load_selected_image)
        fp_layout.addWidget(self.file_list)
        
        btn_refresh = QPushButton("刷新图集清单")
        btn_refresh.clicked.connect(self.refresh_file_list)
        fp_layout.addWidget(btn_refresh)
        main_layout.addWidget(self.file_panel)

        # --- 2. 中间：算法控制区 ---
        ctrl_panel = QFrame()
        ctrl_panel.setFixedWidth(300)
        ctrl_layout = QVBoxLayout(ctrl_panel)
        
        g1 = QGroupBox("✨ 特征增强 (CLAHE)")
        v1 = QVBoxLayout(g1)
        self.cb_clahe = QCheckBox("启用对比度增强")
        self.cb_clahe.setChecked(True)
        self.sl_clahe = QSlider(Qt.Horizontal)
        self.sl_clahe.setRange(1, 100)
        self.sl_clahe.setValue(30)
        v1.addWidget(self.cb_clahe)
        v1.addWidget(QLabel("增强系数 (Clip Limit):"))
        v1.addWidget(self.sl_clahe)
        ctrl_layout.addWidget(g1)

        g2 = QGroupBox("🌊 影像平滑 (Denoise)")
        v2 = QVBoxLayout(g2)
        self.cb_denoise = QCheckBox("启用双边滤波去噪")
        self.cb_denoise.setChecked(True)
        v2.addWidget(self.cb_denoise)
        ctrl_layout.addWidget(g2)

        g3 = QGroupBox("🔍 边界显化 (Edge)")
        v3 = QVBoxLayout(g3)
        self.cb_edge = QCheckBox("提取结节轮廓线")
        self.cb_edge.setChecked(True) # 默认开启看效果
        self.sl_canny = QSlider(Qt.Horizontal)
        self.sl_canny.setRange(10, 200)
        self.sl_canny.setValue(50)
        v3.addWidget(self.cb_edge)
        v3.addWidget(QLabel("边缘敏感度 (越小越强):"))
        v3.addWidget(self.sl_canny)
        ctrl_layout.addWidget(g3)

        ctrl_layout.addStretch()
        
        self.btn_apply = QPushButton("⚡ 执行实时处理")
        self.btn_apply.setFixedHeight(45)
        self.btn_apply.setStyleSheet("background: #3498db; color: white; font-weight: bold;")
        self.btn_apply.clicked.connect(self.run_process)
        
        self.btn_save = QPushButton("💾 保存到预处理训练库")
        self.btn_save.setFixedHeight(45)
        self.btn_save.setStyleSheet("background: #27ae60; color: white; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_to_processed)
        
        ctrl_layout.addWidget(self.btn_apply)
        ctrl_layout.addWidget(self.btn_save)
        main_layout.addWidget(ctrl_panel)

        # --- 3. 右侧：全屏对比视图 ---
        view_panel = QFrame()
        view_layout = QVBoxLayout(view_panel)
        self.splitter = QSplitter(Qt.Vertical)
        
        # 原始图
        self.v_raw = QLabel("原始超声影像 [RAW DATA]")
        self.v_raw.setAlignment(Qt.AlignCenter)
        self.v_raw.setStyleSheet("background: #000; border: 2px solid #333; color: #7f8c8d;")
        
        # 特征图
        self.v_proc = QLabel("特征提取结果 [PROCESSED FEATURE]")
        self.v_proc.setAlignment(Qt.AlignCenter)
        self.v_proc.setStyleSheet("background: #000; border: 3px solid #27ae60; color: #27ae60;")
        
        self.splitter.addWidget(self.v_raw)
        self.splitter.addWidget(self.v_proc)
        view_layout.addWidget(self.splitter)
        main_layout.addWidget(view_panel)

        self.refresh_file_list()

    # --- 逻辑实现 ---

    def refresh_file_list(self):
        self.file_list.clear()
        base = "data/raw"
        if not os.path.exists(base): return
        for cat in ["exing", "liangxing", "quezhenexing"]:
            p = os.path.join(base, cat)
            if os.path.exists(p):
                for f in os.listdir(p):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.file_list.addItem(f"{cat}/{f}")

    def load_selected_image(self, item):
        path = os.path.join("data/raw", item.text())
        self.raw_cv_img = cv2.imread(path)
        if self.raw_cv_img is not None:
            # 渲染左侧原始图
            self.display_image(self.raw_cv_img, self.v_raw, is_gray=False)
            # 自动运行一次处理
            self.run_process()

    def run_process(self):
        """【核心交互】：获取 UI 参数并启动后台算法线程"""
        if self.raw_cv_img is None: return
        
        params = {
            'clahe': self.cb_clahe.isChecked(),
            'clahe_val': self.sl_clahe.value() / 10.0,
            'denoise': self.cb_denoise.isChecked(),
            'edge': self.cb_edge.isChecked(),
            'canny_t': self.sl_canny.value()
        }
        self.engine.set_data(self.raw_cv_img, params)
        self.btn_apply.setText("⏳ 处理中...")
        self.engine.start()

    def on_process_done(self, processed_img, hist):
        """处理完成后，将 numpy 数组渲染到界面上"""
        self.processed_cv_img = processed_img
        self.display_image(processed_img, self.v_proc, is_gray=True)
        self.btn_apply.setText("⚡ 执行实时处理")

    def display_image(self, cv_img, label, is_gray=False):
        """
        【关键修复点】：
        使用 .copy() 解决 macOS 下 UI 不刷新的内存引用问题
        """
        if cv_img is None: return
        h, w = cv_img.shape[:2]
        
        if is_gray:
            # 灰度图流转
            qimg = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            # BGR 转 RGB 流转
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)

        # 重点：.copy() 会申请独立内存，确保 QPixmap 能够正确感知并重绘界面
        pix = QPixmap.fromImage(qimg.copy()).scaled(
            label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        label.setPixmap(pix)

    def save_to_processed(self):
        """数据物理流转：将处理好的特征图存入训练文件夹"""
        if self.processed_cv_img is None: return
        item = self.file_list.currentItem()
        if not item: return
        
        cat, name = item.text().split('/')
        save_dir = os.path.join("data/processed", cat)
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, name)
        cv2.imwrite(save_path, self.processed_cv_img)
        QMessageBox.information(self, "数据流转成功", f"特征图已成功存入：\n{save_path}")