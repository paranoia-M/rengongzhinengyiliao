import os
import cv2
import random
import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QFrame, QProgressBar, 
                             QSplitter, QGroupBox, QTextEdit, QGridLayout)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QBrush, QPen

class AIInferenceThread(QThread):
    """
    AI 推理引擎线程：模拟深度学习模型前向传播过程
    """
    progress_sig = Signal(int, str)  # 进度, 当前层描述
    result_sig = Signal(dict)        # 最终诊断结果字典

    def run(self):
        steps = [
            (20, "📥 影像数据标准化 (Normalization)..."),
            (40, "🧠 卷积层特征提取 (Feature Extraction)..."),
            (60, "🔍 结节形态与钙化点对比 (Pattern Matching)..."),
            (80, "⚖️ 全连接层分类决策 (Final Classification)..."),
            (100, "✅ 诊断报告生成中...")
        ]
        
        for p, desc in steps:
            self.msleep(600) # 模拟推理耗时
            self.progress_sig.emit(p, desc)
            
        # 模拟模型输出概率
        # 随机生成结果，但体现医学逻辑的严密性
        probs = np.random.dirichlet(np.ones(3), size=1)[0]
        idx = np.argmax(probs)
        
        classes = [
            {"label": "良性大结节", "folder": "liangxing", "advice": "结节边缘清晰，回声均匀。存在极低恶变可能性，建议每 6-12 个月复查超声，暂无需手术。"},
            {"label": "恶性 4C 级", "folder": "exing", "advice": "结节形态不规则，纵横比 > 1。高度怀疑恶性（4C级），建议进行穿刺活检或择期手术治疗。"},
            {"label": "确诊恶性 (Urgent)", "folder": "quezhenexing", "advice": "病灶可见明显微钙化且血流信号异常。确诊为恶性肿瘤，必须立即安排住院并进行手术切除！"}
        ]
        
        result = {
            "class": classes[idx]["label"],
            "probs": probs,
            "advice": classes[idx]["advice"],
            "score": float(probs[idx]),
            "folder": classes[idx]["folder"]
        }
        self.result_sig.emit(result)

class DiagnosisPage(QWidget):
    """
    智能诊断页面：实现从影像输入到手术决策的完整流转
    """
    def __init__(self):
        super().__init__()
        self.current_img_path = None
        self.init_ui()

    def init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        # 使用 QSplitter 分割视图
        self.splitter = QSplitter(Qt.Horizontal)
        
        # --- 左侧：影像上传与 AI 可视化 ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.view_box = QGroupBox("📸 病例影像输入 (Input View)")
        view_vbox = QVBoxLayout(self.view_box)
        
        self.img_display = QLabel("请上传甲状腺超声影像\n(支持 JPG/PNG/DICOM)")
        self.img_display.setFixedSize(500, 400)
        self.img_display.setAlignment(Qt.AlignCenter)
        self.img_display.setStyleSheet("""
            background-color: #2c3e50; 
            color: #ecf0f1; 
            border: 3px dashed #34495e;
            border-radius: 10px;
            font-size: 16px;
        """)
        view_vbox.addWidget(self.img_display)
        
        # 按钮区
        btn_layout = QHBoxLayout()
        self.btn_upload = QPushButton("📂 上传病例图集")
        self.btn_upload.clicked.connect(self.upload_image)
        self.btn_upload.setStyleSheet("height: 40px; background: #34495e; color: white;")
        
        self.btn_diagnose = QPushButton("🚀 启动 AI 智能诊断")
        self.btn_diagnose.setEnabled(False)
        self.btn_diagnose.clicked.connect(self.start_diagnosis)
        self.btn_diagnose.setStyleSheet("height: 40px; background: #27ae60; color: white; font-weight: bold;")
        
        btn_layout.addWidget(self.btn_upload)
        btn_layout.addWidget(self.btn_diagnose)
        view_vbox.addLayout(btn_layout)
        
        left_layout.addWidget(self.view_box)
        
        # 推理进度
        self.p_bar = QProgressBar()
        self.p_bar.setStyleSheet("QProgressBar::chunk { background-color: #2ecc71; }")
        self.p_bar.setVisible(False)
        self.status_lbl = QLabel("准备就绪")
        left_layout.addWidget(self.p_bar)
        left_layout.addWidget(self.status_lbl)
        
        self.splitter.addWidget(left_widget)
        
        # --- 右侧：诊断结果报告 ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        self.res_group = QGroupBox("🩺 AI 临床诊断报告 (Diagnostic Report)")
        res_layout = QVBoxLayout(self.res_group)
        
        # 结果看板
        self.res_frame = QFrame()
        self.res_frame.setStyleSheet("background: #fff; border: 1px solid #dee2e6; border-radius: 8px;")
        rf_layout = QVBoxLayout(self.res_frame)
        
        self.lbl_class = QLabel("诊断结果: --")
        self.lbl_class.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        self.lbl_score = QLabel("置信度: --")
        self.lbl_advice = QTextEdit("临床建议将在此处生成...")
        self.lbl_advice.setReadOnly(True)
        self.lbl_advice.setStyleSheet("background: #f8f9fa; border: none; font-size: 14px; color: #34495e;")
        
        rf_layout.addWidget(self.lbl_class)
        rf_layout.addWidget(self.lbl_score)
        rf_layout.addWidget(QLabel("-" * 40))
        rf_layout.addWidget(QLabel("<b>临床手术决策建议:</b>"))
        rf_layout.addWidget(self.lbl_advice)
        
        res_layout.addWidget(self.res_frame)
        
        # 概率分布图模拟 (使用简单的进度条模拟)
        prob_group = QGroupBox("📊 各类别预测概率 (Probability)")
        prob_layout = QGridLayout(prob_group)
        self.p_bars = {
            "liangxing": QProgressBar(),
            "exing": QProgressBar(),
            "quezhenexing": QProgressBar()
        }
        labels = ["良性", "恶性4C", "确诊恶性"]
        for i, (k, pb) in enumerate(self.p_bars.items()):
            prob_layout.addWidget(QLabel(labels[i]), i, 0)
            prob_layout.addWidget(pb, i, 1)
            pb.setTextVisible(True)
            
        res_layout.addWidget(prob_group)
        
        # 导出按钮
        self.btn_export = QPushButton("📄 打印并导出诊断报告")
        self.btn_export.setEnabled(False)
        res_layout.addWidget(self.btn_export)
        
        right_layout.addWidget(self.res_group)
        self.splitter.addWidget(right_widget)
        
        self.main_layout.addWidget(self.splitter)

    # --- 核心交互函数 ---

    def upload_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择影像", "", "Images (*.jpg *.png *.jpeg)")
        if path:
            self.current_img_path = path
            pix = QPixmap(path).scaled(500, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_display.setPixmap(pix)
            self.btn_diagnose.setEnabled(True)
            self.status_lbl.setText(f"影像载入成功: {os.path.basename(path)}")
            # 重置结果界面
            self.lbl_class.setText("诊断结果: 等待分析...")
            self.lbl_advice.setText("")

    def start_diagnosis(self):
        self.btn_diagnose.setEnabled(False)
        self.p_bar.setVisible(True)
        self.p_bar.setValue(0)
        
        # 启动模拟推理线程
        self.thread = AIInferenceThread()
        self.thread.progress_sig.connect(self.update_progress)
        self.thread.result_sig.connect(self.show_results)
        self.thread.start()

    def update_progress(self, val, text):
        self.p_bar.setValue(val)
        self.status_lbl.setText(text)

    def show_results(self, res):
        self.p_bar.setVisible(False)
        self.btn_diagnose.setEnabled(True)
        self.btn_export.setEnabled(True)
        
        # 更新文字
        self.lbl_class.setText(f"诊断结果: {res['class']}")
        self.lbl_score.setText(f"AI 置信度: {res['score']:.2%}")
        self.lbl_advice.setText(res['advice'])
        
        # 设置结果颜色
        colors = {"liangxing": "#27ae60", "exing": "#e67e22", "quezhenexing": "#c0392b"}
        self.lbl_class.setStyleSheet(f"color: {colors[res['folder']]}; font-weight: bold; font-size: 22px;")
        
        # 更新概率条
        self.p_bars["liangxing"].setValue(int(res['probs'][0]*100))
        self.p_bars["exing"].setValue(int(res['probs'][1]*100))
        self.p_bars["quezhenexing"].setValue(int(res['probs'][2]*100))
        
        # 关键技术体现：模拟 Grad-CAM 热力图生成
        self.generate_mock_heatmap(res['folder'])

    def generate_mock_heatmap(self, folder):
        """
        核心独创：模拟 AI 热力图显示，体现数据流转中‘特征关注点’的可视化
        """
        if not self.current_img_path: return
        
        # 读取原图并用 OpenCV 模拟热力叠加
        img = cv2.imread(self.current_img_path)
        img = cv2.resize(img, (500, 400))
        
        # 创建一个随机的热力图点（模拟 AI 关注的结节区域）
        heatmap_overlay = np.zeros(img.shape[:2], dtype=np.uint8)
        center = (random.randint(150, 350), random.randint(150, 250))
        cv2.circle(heatmap_overlay, center, random.randint(40, 80), 255, -1)
        heatmap_overlay = cv2.GaussianBlur(heatmap_overlay, (51, 51), 0)
        
        # 应用伪彩色
        heatmap_color = cv2.applyColorMap(heatmap_overlay, cv2.COLORMAP_JET)
        
        # 叠加到原图
        alpha = 0.4
        out = cv2.addWeighted(img, 1.0 - alpha, heatmap_color, alpha, 0)
        
        # 转换回 QPixmap 并显示
        h, w, ch = out.shape
        qt_img = QImage(out.data, w, h, ch * w, QImage.Format_BGR888)
        self.img_display.setPixmap(QPixmap.fromImage(qt_img))
        self.status_lbl.setText("✅ 诊断完成：热力图已标出 AI 重点关注的病灶特征区域")