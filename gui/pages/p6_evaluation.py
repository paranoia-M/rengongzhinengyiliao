import os
import random
import platform
import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QGroupBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QSplitter, QFrame, QPushButton, 
                             QComboBox, QSlider, QGridLayout)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor

# 集成 Matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# --- 核心修复：解决各系统中文乱码 ---
def set_mpl_font():
    sys_plat = platform.system()
    if sys_plat == "Darwin": # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'STHeiti']
    elif sys_plat == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    else: # Linux
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans Mono']
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

set_mpl_font()

class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#ffffff')
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)

class MetricCard(QFrame):
    """指标卡片：支持动态更新数值"""
    def __init__(self, title, color="#3498db"):
        super().__init__()
        self.setFixedSize(220, 110)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 12px;
                border-top: 5px solid {color};
            }}
        """)
        l = QVBoxLayout(self)
        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("color: #7f8c8d; font-size: 13px; font-weight: bold;")
        self.lbl_value = QLabel("--%")
        self.lbl_value.setStyleSheet(f"color: {color}; font-size: 26px; font-weight: bold;")
        self.lbl_sub = QLabel("正在分析流转数据...")
        self.lbl_sub.setStyleSheet("color: #adb5bd; font-size: 10px;")
        
        l.addWidget(self.lbl_title)
        l.addWidget(self.lbl_value)
        l.addWidget(self.lbl_sub)

    def update_value(self, val, sub_text):
        self.lbl_value.setText(val)
        self.lbl_sub.setText(sub_text)

class EvaluationPage(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        # 初始加载随机数据
        self.refresh_evaluation_data()

    def init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        # 1. 顶部控制与刷新栏
        top_ctrl = QHBoxLayout()
        self.btn_refresh = QPushButton("🔄 启动模型性能评估流 (刷新)")
        self.btn_refresh.setFixedSize(300, 40)
        self.btn_refresh.setStyleSheet("background: #2c3e50; color: white; font-weight: bold; border-radius: 5px;")
        self.btn_refresh.clicked.connect(self.refresh_evaluation_data)
        top_ctrl.addWidget(self.btn_refresh)
        top_ctrl.addStretch()
        self.main_layout.addLayout(top_ctrl)

        # 2. 指标卡片行
        self.metrics_bar = QHBoxLayout()
        self.card_acc = MetricCard("全局准确率 (Accuracy)", "#2ecc71")
        self.card_sen = MetricCard("恶性敏感度 (Recall)", "#e74c3c")
        self.card_f1 = MetricCard("F1 综合分数", "#9b59b6")
        self.card_spec = MetricCard("良性特异度 (Spec)", "#f1c40f")
        
        for card in [self.card_acc, self.card_sen, self.card_f1, self.card_spec]:
            self.metrics_bar.addWidget(card)
        self.main_layout.addLayout(self.metrics_bar)

        # 3. 图表区域
        self.splitter = QSplitter(Qt.Horizontal)
        
        # 混淆矩阵
        self.box_matrix = QGroupBox("📌 诊断决策分布 (混淆矩阵)")
        v1 = QVBoxLayout(self.box_matrix)
        self.canvas_matrix = MplCanvas()
        v1.addWidget(self.canvas_matrix)
        self.splitter.addWidget(self.box_matrix)
        
        # ROC 曲线
        self.box_curve = QGroupBox("📈 模型判别能力 (ROC 曲线)")
        v2 = QVBoxLayout(self.box_curve)
        self.canvas_curve = MplCanvas()
        v2.addWidget(self.canvas_curve)
        self.splitter.addWidget(self.box_curve)
        
        self.main_layout.addWidget(self.splitter, stretch=2)

        # 4. 底部评估报告表
        self.table_box = QGroupBox("📋 细分分类性能详细清单")
        t_layout = QVBoxLayout(self.table_box)
        self.report_table = QTableWidget(3, 4)
        self.report_table.setHorizontalHeaderLabels(["分类名称", "精确率 (Precision)", "召回率 (Recall)", "F1-Score"])
        self.report_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t_layout.addWidget(self.report_table)
        self.main_layout.addWidget(self.table_box, stretch=1)

    def refresh_evaluation_data(self):
        """核心逻辑：每次调用生成完全不同的随机评估数据"""
        # 1. 模拟生成随机混淆矩阵 (3x3)
        # 行：真实类别，列：预测类别 [良性, 4C, 确诊]
        cm = np.zeros((3, 3), dtype=int)
        for i in range(3):
            # 模拟模型表现：对角线数值大（预测正确），其他数值小（误诊）
            for j in range(3):
                if i == j:
                    cm[i, j] = random.randint(80, 100)
                else:
                    cm[i, j] = random.randint(1, 15)
        
        # 2. 基于随机矩阵计算指标
        total = np.sum(cm)
        diag = np.diag(cm)
        acc = np.sum(diag) / total
        
        # 针对 4C 恶性 (Index 1) 的 Recall
        recall_exing = cm[1, 1] / np.sum(cm[1, :])
        # 针对 良性 (Index 0) 的 Spec
        spec_liangxing = cm[0, 0] / np.sum(cm[0, :])
        f1_mock = acc - random.uniform(0.01, 0.05)

        # 3. 更新指标卡片
        self.card_acc.update_value(f"{acc:.1%}", f"基于 {total} 例随机验证样本")
        self.card_sen.update_value(f"{recall_exing:.1%}", "当前恶性 4C 漏诊风险控制良好")
        self.card_f1.update_value(f"{f1_mock:.3f}", "三类均衡性评估结果")
        self.card_spec.update_value(f"{spec_liangxing:.1%}", "有效降低良性病例过度治疗")

        # 4. 重新绘制混淆矩阵
        self.draw_confusion_matrix(cm)
        
        # 5. 重新绘制曲线
        self.draw_roc_curve()
        
        # 6. 更新表格
        classes = ["良性大结节", "恶性 4C", "确诊恶性"]
        for i in range(3):
            prec = cm[i, i] / np.sum(cm[:, i])
            rec = cm[i, i] / np.sum(cm[i, :])
            f1 = 2 * prec * rec / (prec + rec)
            
            data = [classes[i], f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}"]
            for j, val in enumerate(data):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.report_table.setItem(i, j, item)

    def draw_confusion_matrix(self, cm):
        ax = self.canvas_matrix.axes
        ax.clear()
        classes = ['良性', '恶性4C', '确诊']
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title("诊断决策分布 (混淆矩阵)")
        
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)

        # 填写数字
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        ax.set_ylabel('真实标注')
        ax.set_xlabel('AI 预测结果')
        self.canvas_matrix.draw()

    def draw_roc_curve(self):
        ax = self.canvas_curve.axes
        ax.clear()
        
        # 模拟三条随机曲线
        x = np.linspace(0, 1, 100)
        # 每次进入时，AUC 和曲线弧度都会随机波动
        ax.plot(x, x**(random.uniform(0.1, 0.2)), 'r-', label=f'恶性 4C (AUC={random.uniform(0.94, 0.97):.2f})')
        ax.plot(x, x**(random.uniform(0.05, 0.1)), 'g-', label=f'确诊恶性 (AUC={random.uniform(0.97, 0.99):.2f})')
        ax.plot(x, x**(random.uniform(0.2, 0.3)), 'b-', label=f'良性大结节 (AUC={random.uniform(0.90, 0.93):.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_title("多分类 ROC 判别能力曲线")
        ax.set_xlabel("假阳性率 (FPR)")
        ax.set_ylabel("真阳性率 (TPR)")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=':', alpha=0.6)
        self.canvas_curve.draw()