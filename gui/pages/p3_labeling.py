import os
import shutil
import platform
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
                             QProgressBar, QFrame, QSplitter, QGroupBox, QMessageBox,
                             QAbstractItemView, QGridLayout)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor, QPixmap

# 集成 Matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# --- 解决 macOS 字体乱码核心修复 ---
if platform.system() == "Darwin": # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else: # Windows/Linux
    plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class LabelingPage(QWidget):
    """
    标注管理页面：实现病例影像的分类标注、物理流转与状态统计
    """
    def __init__(self):
        super().__init__()
        self.root_data_path = "data/raw"
        self.current_img_path = None # 当前正在标注的图片路径
        self.current_filename = None
        self.init_ui()
        self.refresh_data()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # 使用 QSplitter 将页面分为【预览标注区】和【列表统计区】
        self.splitter = QSplitter(Qt.Vertical)

        # --- 1. 上部：交互式标注控制台 ---
        self.labeling_console = QGroupBox("🏗️ 实时影像标注控制台")
        console_layout = QHBoxLayout(self.labeling_console)
        
        # 左侧：影像预览
        self.img_preview = QLabel("请在下方列表中选择病例进行标注")
        self.img_preview.setFixedSize(320, 240)
        self.img_preview.setStyleSheet("background: #000; border: 2px solid #34495e; border-radius: 5px;")
        self.img_preview.setAlignment(Qt.AlignCenter)
        console_layout.addWidget(self.img_preview)

        # 右侧：标注动作按钮
        actions_frame = QFrame()
        actions_layout = QVBoxLayout(actions_frame)
        actions_layout.addWidget(QLabel("<b>请选择该病例的最终诊断标签：</b>"))
        
        self.btn_to_lx = QPushButton("✅ 标注为：良性大结节")
        self.btn_to_ex = QPushButton("⚠️ 标注为：恶性 4C")
        self.btn_to_qz = QPushButton("🚨 标注为：确诊恶性")
        
        # 设置按钮样式（体现医学优先级）
        self.btn_to_lx.setStyleSheet("background: #27ae60; color: white; height: 40px; font-weight: bold;")
        self.btn_to_ex.setStyleSheet("background: #e67e22; color: white; height: 40px; font-weight: bold;")
        self.btn_to_qz.setStyleSheet("background: #c0392b; color: white; height: 40px; font-weight: bold;")
        
        # 绑定标注逻辑
        self.btn_to_lx.clicked.connect(lambda: self.execute_labeling("liangxing"))
        self.btn_to_ex.clicked.connect(lambda: self.execute_labeling("exing"))
        self.btn_to_qz.clicked.connect(lambda: self.execute_labeling("quezhenexing"))

        actions_layout.addWidget(self.btn_to_lx)
        actions_layout.addWidget(self.btn_to_ex)
        actions_layout.addWidget(self.btn_to_qz)
        actions_layout.addStretch()
        console_layout.addWidget(actions_frame)
        
        # 最右侧：饼图统计（缩小一点）
        self.canvas = FigureCanvas(Figure(figsize=(4, 3)))
        console_layout.addWidget(self.canvas)
        
        self.splitter.addWidget(self.labeling_console)

        # --- 2. 下部：详细病例列表 ---
        self.list_box = QGroupBox("📋 病例档案列表 (点击项进行标注)")
        list_layout = QVBoxLayout(self.list_box)
        
        # 工具栏
        tool_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("🔄 刷新病例库")
        self.btn_refresh.clicked.connect(self.refresh_data)
        tool_layout.addWidget(self.btn_refresh)
        tool_layout.addStretch()
        list_layout.addLayout(tool_layout)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["文件名", "当前所属目录", "当前标签状态", "文件大小"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.cellClicked.connect(self.on_table_click) # 点击行触发预览
        
        list_layout.addWidget(self.table)
        self.splitter.addWidget(self.list_box)

        main_layout.addWidget(self.splitter)

    # --- 核心标注逻辑 ---

    def refresh_data(self):
        """扫描磁盘并更新 UI"""
        self.table.setRowCount(0)
        stats = {"liangxing": 0, "exing": 0, "quezhenexing": 0}
        
        categories = ["liangxing", "exing", "quezhenexing"]
        mapping = {
            "liangxing": "良性大结节·存在恶变可能性",
            "exing": "恶性4C·建议手术治疗",
            "quezhenexing": "确诊恶性·必须立即手术"
        }

        row = 0
        for cat in categories:
            path = os.path.join(self.root_data_path, cat)
            if not os.path.exists(path): os.makedirs(path, exist_ok=True)
            
            for f in os.listdir(path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(path, f)
                    self.table.insertRow(row)
                    self.table.setItem(row, 0, QTableWidgetItem(f))
                    self.table.setItem(row, 1, QTableWidgetItem(cat))
                    
                    label_item = QTableWidgetItem(mapping[cat])
                    # 着色
                    if cat == "exing": label_item.setForeground(QColor("#e67e22"))
                    elif cat == "quezhenexing": label_item.setForeground(QColor("#c0392b"))
                    else: label_item.setForeground(QColor("#27ae60"))
                    
                    self.table.setItem(row, 2, label_item)
                    size = f"{os.path.getsize(full_path)/1024:.1f} KB"
                    self.table.setItem(row, 3, QTableWidgetItem(size))
                    
                    stats[cat] += 1
                    row += 1
        
        self.update_pie_chart(stats)

    def on_table_click(self, row, col):
        """点击表格行，将数据加载到标注预览区"""
        filename = self.table.item(row, 0).text()
        category = self.table.item(row, 1).text()
        full_path = os.path.join(self.root_data_path, category, filename)
        
        self.current_img_path = full_path
        self.current_filename = filename
        
        # 更新预览图
        pix = QPixmap(full_path).scaled(self.img_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_preview.setPixmap(pix)

    def execute_labeling(self, target_cat):
        """执行标注：物理移动文件到对应的标签文件夹"""
        if not self.current_img_path:
            QMessageBox.warning(self, "提示", "请先在下方列表中选择一个病例！")
            return
            
        # 确定目标路径
        target_dir = os.path.join(self.root_data_path, target_cat)
        target_path = os.path.join(target_dir, self.current_filename)
        
        # 如果路径没变，不需要操作
        if os.path.normpath(self.current_img_path) == os.path.normpath(target_path):
            QMessageBox.information(self, "提示", "该病例已经属于此标注类别。")
            return

        try:
            # 执行物理流转
            shutil.move(self.current_img_path, target_path)
            
            # 更新状态
            self.current_img_path = target_path
            QMessageBox.information(self, "标注成功", f"病例 {self.current_filename} 已重新标注并流转至 {target_cat} 目录。")
            
            # 刷新列表和统计
            self.refresh_data()
        except Exception as e:
            QMessageBox.critical(self, "标注失败", f"文件流转出错：{str(e)}")

    def update_pie_chart(self, stats):
        """更新统计图表 (修复 macOS 乱码)"""
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        
        labels = ['良性', '恶性4C', '确诊恶性']
        sizes = [stats['liangxing'], stats['exing'], stats['quezhenexing']]
        colors = ['#2ecc71', '#f1c40f', '#e74c3c']
        
        if sum(sizes) > 0:
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
            ax.set_title("病例标注库实时分布", fontsize=10)
        else:
            ax.text(0.5, 0.5, '暂无标注数据', ha='center')
            
        self.canvas.draw()