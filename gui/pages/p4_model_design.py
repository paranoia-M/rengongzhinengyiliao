import json
import random
import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QFormLayout, QLineEdit, QComboBox, QPushButton, 
                             QLabel, QSlider, QFrame, QGraphicsView, QGraphicsScene, 
                             QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem,
                             QSplitter, QTabWidget, QCheckBox, QSpinBox, QDoubleSpinBox,
                             QTextEdit, QMessageBox)
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QFont, QColor, QPen, QBrush, QPainter

class LayerNode(QGraphicsEllipseItem):
    """
    可视化神经网络层节点：
    体现数据流转中 Tensor 的维度(Shape)变化
    """
    def __init__(self, x, y, name, shape, color="#3498db"):
        super().__init__(-35, -35, 70, 70)
        self.setPos(x, y)
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor("#ffffff"), 2))
        
        # 层名称
        txt_name = QGraphicsTextItem(name, self)
        txt_name.setDefaultTextColor(QColor("white"))
        txt_name.setFont(QFont("Segoe UI", 9, QFont.Bold))
        txt_name.setPos(-30, -25)
        
        # Shape 信息 (核心流转数据)
        txt_shape = QGraphicsTextItem(shape, self)
        txt_shape.setDefaultTextColor(QColor("#ecf0f1"))
        txt_shape.setFont(QFont("Consolas", 8))
        txt_shape.setPos(-32, 5)

class AdvancedGraphView(QGraphicsView):
    """
    计算图画布：
    可视化甲状腺 AI 模型从输入到三分类输出的完整逻辑
    """
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(0, 0, 800, 400)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setStyleSheet("background-color: #1e272e; border-radius: 15px;")

    def update_graph(self, backbone_name, res):
        self.scene.clear()
        
        # 模拟深度学习中的降维流转 (Downsampling Flow)
        # 对应：输入 -> 浅层特征 -> 深层特征 -> 池化 -> 全连接 -> 三分类输出
        layers = [
            ("Input", f"3x{res}x{res}"),
            ("Stem", f"64x{res//2}x{res//2}"),
            (backbone_name.split('-')[0], f"512x{res//32}x{res//32}"),
            ("GlobalAvg", "512x1x1"),
            ("FC-Drop", "256"),
            ("Output", "3 (Prob)")
        ]
        
        nodes = []
        for i, (name, shape) in enumerate(layers):
            x = 80 + i * 135
            y = 200
            # 颜色区分：输入(绿)、处理(蓝)、输出(红)
            color = "#27ae60" if i == 0 else ("#c0392b" if i == len(layers)-1 else "#2980b9")
            node = LayerNode(x, y, name, shape, color)
            self.scene.addItem(node)
            nodes.append(node)
            
            if i > 0:
                # 绘制带箭头的连接线，体现数据流向
                line = QGraphicsLineItem(nodes[i-1].x() + 35, y, x - 35, y)
                line.setPen(QPen(QColor("#7f8c8d"), 2, Qt.SolidLine))
                self.scene.addItem(line)

class ModelDesignPage(QWidget):
    """
    模型构建菜单：
    实现复杂的参数配置、架构验证及计算图可视化
    """
    def __init__(self):
        super().__init__()
        self.init_ui()
        # 初始化时执行一次刷新，防止空白
        self.refresh_viz()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        self.main_splitter = QSplitter(Qt.Horizontal)

        # --- 左侧：复杂参数配置面板 (使用选项卡) ---
        self.left_panel = QFrame()
        self.left_panel.setFixedWidth(450)
        left_vbox = QVBoxLayout(self.left_panel)
        
        self.tabs = QTabWidget()
        
        # Tab 1: 核心架构 (Architecture)
        self.tab_arch = QWidget()
        arch_form = QFormLayout(self.tab_arch)
        
        self.cb_backbone = QComboBox()
        self.cb_backbone.addItems(["ResNet-50", "ResNet-101", "EfficientNet-B4", "DenseNet-121", "Swin-Transformer"])
        self.cb_backbone.currentIndexChanged.connect(self.refresh_viz) # 关键：刷新可视化
        
        self.cb_resolution = QComboBox()
        self.cb_resolution.addItems(["224", "256", "384", "512"])
        self.cb_resolution.currentIndexChanged.connect(self.refresh_viz)
        
        self.spin_freeze = QSpinBox()
        self.spin_freeze.setRange(0, 150)
        self.spin_freeze.setValue(50)
        self.spin_freeze.setSuffix(" 层")
        
        self.cb_weights = QComboBox()
        self.cb_weights.addItems(["Medical-Pretrained (推荐)", "ImageNet-V1", "Random Initialize"])

        arch_form.addRow("骨干网络结构:", self.cb_backbone)
        arch_form.addRow("输入图像尺寸:", self.cb_resolution)
        arch_form.addRow("迁移学习冻结层:", self.spin_freeze)
        arch_form.addRow("预训练权重加载:", self.cb_weights)
        self.tabs.addTab(self.tab_arch, "🏗️ 架构设计")

        # Tab 2: 训练优化 (Optimization)
        self.tab_opt = QWidget()
        opt_form = QFormLayout(self.tab_opt)
        
        self.ds_lr = QDoubleSpinBox()
        self.ds_lr.setRange(0.00001, 0.1)
        self.ds_lr.setDecimals(5)
        self.ds_lr.setValue(0.001)
        self.ds_lr.setSingleStep(0.001)
        
        self.cb_opt = QComboBox()
        self.cb_opt.addItems(["AdamW (推荐)", "SGD (带动量)", "Adagrad", "RMSprop"])
        
        self.cb_scheduler = QComboBox()
        self.cb_scheduler.addItems(["CosineAnnealingLR", "StepLR", "ReduceLROnPlateau"])

        opt_form.addRow("基础学习率 (LR):", self.ds_lr)
        opt_form.addRow("优化器类型:", self.cb_opt)
        opt_form.addRow("学习率调度策略:", self.cb_scheduler)
        opt_form.addRow("权重衰减 (L2):", QDoubleSpinBox())
        self.tabs.addTab(self.tab_opt, "⚙️ 训练策略")

        # Tab 3: 数据增强 (Data Augmentation)
        self.tab_aug = QWidget()
        aug_layout = QVBoxLayout(self.tab_aug)
        aug_layout.addWidget(QLabel("<b>实时数据增强流配置：</b>"))
        self.aug_flip = QCheckBox("随机水平/垂直翻转")
        self.aug_rotate = QCheckBox("随机旋转 (-45°, 45°)")
        self.aug_noise = QCheckBox("高斯噪声 (Gaussian Noise)")
        self.aug_blur = QCheckBox("中心随机裁切 (Center Crop)")
        
        for cb in [self.aug_flip, self.aug_rotate, self.aug_noise, self.aug_blur]:
            cb.setChecked(True)
            aug_layout.addWidget(cb)
        aug_layout.addStretch()
        self.tabs.addTab(self.tab_aug, "🪄 增强算子")

        left_vbox.addWidget(self.tabs)

        # 实时状态摘要
        self.status_box = QFrame()
        self.status_box.setStyleSheet("background: #2c3e50; border-radius: 10px; color: #1abc9c; padding: 15px;")
        sv = QVBoxLayout(self.status_box)
        self.lbl_stats = QLabel("模型状态: 待校验\n估计参数量: 计算中...\nFLOPs: 计算中...")
        self.lbl_stats.setFont(QFont("Consolas", 10))
        sv.addWidget(self.lbl_stats)
        left_vbox.addWidget(self.status_box)

        # 编译与操作按钮
        self.btn_compile = QPushButton("🔨 验证并编译 AI 架构")
        self.btn_compile.setFixedHeight(50)
        self.btn_compile.setStyleSheet("background: #3498db; color: white; font-weight: bold; font-size: 14px;")
        self.btn_compile.clicked.connect(self.compile_model)
        
        self.btn_export = QPushButton("📤 导出模型定义清单 (JSON)")
        self.btn_export.clicked.connect(self.export_config)
        
        left_vbox.addWidget(self.btn_compile)
        left_vbox.addWidget(self.btn_export)
        
        self.main_splitter.addWidget(self.left_panel)

        # --- 右侧：可视化与日志 ---
        self.right_panel = QWidget()
        right_vbox = QVBoxLayout(self.right_panel)
        
        right_vbox.addWidget(QLabel("<h2>🧬 计算图与数据张量流 (Tensor Flow)</h2>"))
        
        self.graph_view = AdvancedGraphView()
        right_vbox.addWidget(self.graph_view)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("background: #000; color: #bdc3c7; font-family: 'Courier New';")
        self.log_area.setFixedHeight(150)
        right_vbox.addWidget(self.log_area)
        
        self.main_splitter.addWidget(self.right_panel)
        self.main_splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(self.main_splitter)

    # --- 核心业务方法 (修复 AttributeError) ---

    def refresh_viz(self):
        """
        核心方法：当 UI 参数改变时，重新计算模型规模并更新图形
        """
        backbone = self.cb_backbone.currentText()
        res = int(self.cb_resolution.currentText())
        
        # 1. 更新图形
        self.graph_view.update_graph(backbone, res)
        
        # 2. 模拟计算参数量 (体现数据流转深度)
        if "ResNet-50" in backbone:
            params = 23.5 + random.uniform(0, 2)
            flops = 4.1 * (res / 224)**2
        elif "101" in backbone:
            params = 44.5 + random.uniform(0, 3)
            flops = 7.8 * (res / 224)**2
        else: # Transformer
            params = 86.7
            flops = 15.4 * (res / 224)**2
            
        self.lbl_stats.setText(
            f"架构校验: 匹配三分类任务\n"
            f"估计参数量: {params:.2f} Million\n"
            f"估计计算量: {flops:.2f} GFLOPs"
        )
        self.log_area.append(f">>> 架构变更: {backbone}, 输入分辨率已调整为 {res}x{res}")

    def compile_model(self):
        """
        执行模型编译：校验参数合法性，模拟 AI 引擎加载
        """
        self.log_area.append(">>> 启动模型架构合法性静态分析...")
        self.log_area.append(f">>> 正在应用迁移学习策略：冻结前 {self.spin_freeze.value()} 层.")
        
        # 模拟数据流转校验
        self.log_area.append(">>> 校验分类输出头: [3 类] 匹配成功.")
        self.log_area.append(f">>> 正在构建数据增强 Pipeline: " + 
                            ("Flip " if self.aug_flip.isChecked() else "") + 
                            ("Noise " if self.aug_noise.isChecked() else ""))
        
        self.btn_compile.setText("⌛ 正在同步显存...")
        self.btn_compile.setEnabled(False)
        
        # 延迟模拟成功效果
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1500, self._on_compile_success)

    def _on_compile_success(self):
        self.btn_compile.setText("✅ 架构已编译并锁定")
        self.btn_compile.setStyleSheet("background: #27ae60; color: white; font-weight: bold;")
        self.log_area.append(">>> [SUCCESS] 模型引擎初始化完成。可以开始训练任务。")
        QMessageBox.information(self, "编译成功", "甲状腺诊断模型架构已成功锁定并编译。")

    def export_config(self):
        """
        将配置导出为 JSON，体现全平台数据同步
        """
        config = {
            "version": "1.5.0",
            "backbone": self.cb_backbone.currentText(),
            "hyperparameters": {
                "lr": self.ds_lr.value(),
                "optimizer": self.cb_opt.currentText(),
                "resolution": self.cb_resolution.currentText()
            },
            "classes": ["liangxing", "exing", "quezhenexing"]
        }
        
        os.makedirs("models", exist_ok=True)
        with open("models/design_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
            
        self.log_area.append(">>> [FILE] 配置文件已导出至 models/design_config.json")
        QMessageBox.information(self, "导出成功", "模型定义文件已保存。")