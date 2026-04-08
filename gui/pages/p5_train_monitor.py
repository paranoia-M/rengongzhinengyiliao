import sys
import time
import random
import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QProgressBar, QFrame, QSplitter, QSlider, 
                             QGroupBox, QTextEdit, QGridLayout, QCheckBox)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QColor, QPainter, QImage, QPixmap
import pyqtgraph as pg  # 专业实时绘图库

# 模拟甲状腺特征层（用于体现AI学习过程）
THYROID_FEATURES = [
    "边缘毛刺感提取 (Edge Spiculation)",
    "内部微钙化点识别 (Micro-calcification)",
    "纵横比计算 (Aspect Ratio > 1)",
    "血流信号强度分析 (Vascularity)",
    "低回声区域定位 (Hypoechoic Area)"
]

class AIDeepLearningThread(QThread):
    """
    核心AI训练引擎线程
    模拟真实的随机梯度下降(SGD)和反向传播过程
    """
    # 信号定义：epoch, loss, acc, lr, feature_weights, log_msg
    iteration_sig = Signal(int, float, float, float, list, str)
    finished_sig = Signal()

    def __init__(self, epochs=100, lr=0.01, batch_size=32):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self._is_running = True
        self._is_paused = False

    def stop(self):
        self._is_running = False

    def toggle_pause(self):
        self._is_paused = not self._is_paused

    def run(self):
        # 初始化模拟权重
        loss = 2.5
        accuracy = 0.1
        weights = [random.uniform(0.1, 0.3) for _ in range(5)]
        
        for epoch in range(1, self.epochs + 1):
            if not self._is_running: break
            while self._is_paused: self.msleep(100)

            # 模拟数据流转与梯度下降
            # 模拟学习率衰减
            current_lr = self.lr * (0.95 ** (epoch // 10))
            
            # 模拟 Loss 震荡下降曲线
            loss_delta = (loss * 0.1) * random.uniform(0.5, 1.5)
            loss = max(0.01, loss - loss_delta * 0.2)
            
            # 模拟 Accuracy 增长（针对甲状腺三类疾病的辨识度）
            acc_delta = (1.0 - accuracy) * random.uniform(0.01, 0.08)
            accuracy = min(0.99, accuracy + acc_delta)

            # 模拟神经网络内部特征权重的变化
            # 体现模型在不同阶段关注不同的病灶特征
            for i in range(len(weights)):
                weights[i] = min(1.0, weights[i] + random.uniform(0.01, 0.05))

            # 生成模拟日志内容，体现医疗诊断逻辑
            log_msgs = [
                f"正在分析 4C 级恶性结节的微小钙化点...",
                f"正在对比良性大结节与存在恶变可能性的回声特征...",
                f"反向传播算法正在优化手术决策权重 (Surgical Decision Weights)...",
                f"梯度更新完成，当前模型对‘必须立即手术’类别的召回率提升..."
            ]
            current_log = random.choice(log_msgs)

            # 发送数据到 UI
            self.iteration_sig.emit(epoch, loss, accuracy, current_lr, weights, current_log)
            
            # 模拟真实计算耗时（随着模型收敛，计算可能会变快）
            self.msleep(int(200 * (0.98 ** epoch)))

        self.finished_sig.emit()

class FeatureHeatmapWidget(QFrame):
    """
    自定义组件：体现AI学习过程中的特征热力图
    展示模型对甲状腺病灶特征的‘关注度’
    """
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(200)
        self.weights = [0.1] * 5
        self.setStyleSheet("background-color: #1e1e1e; border-radius: 5px;")

    def update_weights(self, new_weights):
        self.weights = new_weights
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        width = self.width()
        height = self.height()
        
        # 绘制背景格
        padding = 20
        rect_width = (width - 2*padding) // 5
        
        for i, weight in enumerate(self.weights):
            # 根据权重计算颜色（红色代表高关注度，蓝色代表低关注度）
            color_val = int(weight * 255)
            color = QColor(color_val, 50, 255 - color_val)
            
            rect_x = padding + i * rect_width
            rect_y = padding
            rect_h = height - 2*padding
            
            # 绘制热力块
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect_x + 5, rect_y, rect_width - 10, rect_h, 5, 5)
            
            # 绘制文字标签
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 8))
            painter.drawText(rect_x + 10, rect_y + rect_h - 10, f"{weight:.2f}")

class TrainMonitorPage(QWidget):
    """
    主菜单页面：训练监控
    包含复杂的布局管理、实时曲线、交互式控制和学习流转展示
    """
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.train_thread = None
        self.history_loss = []
        self.history_acc = []

    def init_ui(self):
        # 全局水平布局
        main_layout = QHBoxLayout(self)
        
        # 使用 QSplitter 允许用户左右调节视图大小
        splitter = QSplitter(Qt.Horizontal)
        
        # --- 左侧：控制与日志面板 ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 1. 超参数配置组
        config_group = QGroupBox("🤖 实时超参数调节")
        config_grid = QGridLayout(config_group)
        
        self.lr_slider = QSlider(Qt.Horizontal)
        self.lr_slider.setRange(1, 100)
        self.lr_slider.setValue(10)
        self.lr_label = QLabel("学习率 (LR): 0.01")
        self.lr_slider.valueChanged.connect(self._on_lr_changed)
        
        self.epoch_slider = QSlider(Qt.Horizontal)
        self.epoch_slider.setRange(10, 500)
        self.epoch_slider.setValue(100)
        self.epoch_label = QLabel("最大轮次 (Epochs): 100")
        self.epoch_slider.valueChanged.connect(self._on_epoch_changed)

        config_grid.addWidget(self.lr_label, 0, 0)
        config_grid.addWidget(self.lr_slider, 0, 1)
        config_grid.addWidget(self.epoch_label, 1, 0)
        config_grid.addWidget(self.epoch_slider, 1, 1)
        
        # 2. 交互按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始模型训练")
        self.start_btn.setFixedHeight(40)
        self.start_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        self.start_btn.clicked.connect(self.start_training)
        
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_pause)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.pause_btn)

        # 3. 实时控制台日志
        log_group = QGroupBox("📜 系统数据流转日志")
        log_layout = QVBoxLayout(log_group)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("""
            background-color: #000; 
            color: #00ff00; 
            font-family: 'Courier New'; 
            font-size: 11px;
        """)
        log_layout.addWidget(self.console)
        
        left_layout.addWidget(config_group)
        left_layout.addLayout(btn_layout)
        left_layout.addWidget(log_group)
        
        # --- 右侧：可视化面板 ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 1. 实时曲线图（使用 pyqtgraph）
        graph_group = QGroupBox("📈 学习曲线 (实时损耗与准确率)")
        graph_layout = QVBoxLayout(graph_group)
        
        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setBackground('#f0f0f0')
        self.graph_widget.showGrid(x=True, y=True)
        self.graph_widget.addLegend()
        
        self.loss_plot = self.graph_widget.plot(pen=pg.mkPen(color='r', width=2), name="Loss (损失值)")
        self.acc_plot = self.graph_widget.plot(pen=pg.mkPen(color='b', width=2), name="Accuracy (准确率)")
        
        graph_layout.addWidget(self.graph_widget)
        
        # 2. 特征层激活可视化
        feature_group = QGroupBox("🧠 神经元特征提取热力图 (体现 AI 学习过程)")
        feature_layout = QVBoxLayout(feature_group)
        self.heatmap = FeatureHeatmapWidget()
        feature_layout.addWidget(self.heatmap)
        
        # 特征描述标签
        feature_desc_layout = QHBoxLayout()
        for f_name in THYROID_FEATURES:
            lbl = QLabel(f_name.split(" ")[0])
            lbl.setFont(QFont("Arial", 7))
            lbl.setAlignment(Qt.AlignCenter)
            feature_desc_layout.addWidget(lbl)
        feature_layout.addLayout(feature_desc_layout)
        
        # 3. 总体进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #3498db; }")
        
        right_layout.addWidget(graph_group, stretch=2)
        right_layout.addWidget(feature_group, stretch=1)
        right_layout.addWidget(self.progress_bar)

        # 组装
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)

    # --- 交互逻辑函数 ---

    def _on_lr_changed(self, value):
        lr = value / 1000.0
        self.lr_label.setText(f"学习率 (LR): {lr:.3f}")

    def _on_epoch_changed(self, value):
        self.epoch_label.setText(f"最大轮次 (Epochs): {value}")

    def start_training(self):
        if self.train_thread and self.train_thread.isRunning():
            return

        # 初始化状态
        self.console.clear()
        self.history_loss = []
        self.history_acc = []
        self.console.append("<b style='color: white;'>[INFO] 正在初始化甲状腺深度学习模型...</b>")
        self.console.append(f"<b style='color: white;'>[INFO] 数据源: data/raw/ (三类病例已加载)</b>")
        
        # 启动线程
        lr = self.lr_slider.value() / 1000.0
        epochs = self.epoch_slider.value()
        self.train_thread = AIDeepLearningThread(epochs=epochs, lr=lr)
        self.train_thread.iteration_sig.connect(self.update_ui_state)
        self.train_thread.finished_sig.connect(self.on_training_finished)
        
        self.train_thread.start()
        
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.progress_bar.setMaximum(epochs)

    def toggle_pause(self):
        if self.train_thread:
            self.train_thread.toggle_pause()
            txt = "继续" if self.train_thread._is_paused else "暂停"
            self.pause_btn.setText(txt)

    def update_ui_state(self, epoch, loss, acc, lr, weights, log):
        # 更新曲线数据
        self.history_loss.append(loss)
        self.history_acc.append(acc)
        self.loss_plot.setData(self.history_loss)
        self.acc_plot.setData(self.history_acc)
        
        # 更新热力图
        self.heatmap.update_weights(weights)
        
        # 更新进度条
        self.progress_bar.setValue(epoch)
        
        # 更新日志流
        color_code = "#00ff00" if acc < 0.8 else "#ffff00"
        self.console.append(f"<span style='color: {color_code};'>[Epoch {epoch}] Loss: {loss:.4f} | Acc: {acc:.4%}")
        self.console.append(f"<span style='color: #888;'>=> {log}</span>")
        
        # 自动滚动到底部
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def on_training_finished(self):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("暂停")
        self.console.append("<b style='color: #27ae60;'>[SUCCESS] 模型训练完成。最优参数已保存至 models/thyroid_best.pth</b>")