import sys
import os
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QListWidget, QStackedWidget, QLabel, QFrame, 
                             QPushButton, QStatusBar, QListWidgetItem, QApplication)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QIcon, QColor

# --- 正式引入所有已实现的页面 ---
from gui.pages.p1_data_center import DataCenterPage
from gui.pages.p2_preprocess import PreprocessPage
from gui.pages.p3_labeling import LabelingPage
from gui.pages.p4_model_design import ModelDesignPage
from gui.pages.p5_train_monitor import TrainMonitorPage
from gui.pages.p6_evaluation import EvaluationPage
from gui.pages.p7_diagnosis import DiagnosisPage
from gui.pages.p8_history_report import HistoryReportPage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" 人工智能数据分析与预测平台")
        self.resize(1400, 950)
        
        # 初始化界面
        self.init_ui()
        self.apply_styles()
        
        # 默认选中第一个
        self.menu_list.setCurrentRow(0)

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # --- 1. 左侧侧边栏 ---
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(280)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # 侧边栏顶部 LOGO
        logo_frame = QFrame()
        logo_frame.setFixedHeight(150)
        logo_frame.setObjectName("logo_frame")
        logo_layout = QVBoxLayout(logo_frame)
        
        logo_text = QLabel("")
        logo_text.setStyleSheet("font-size: 28px; color: #3498db; font-weight: bold; font-family: 'Segoe UI';")
        logo_sub = QLabel("")
        logo_sub.setStyleSheet("font-size: 14px; color: #bdc3c7;")
        logo_layout.addWidget(logo_text, alignment=Qt.AlignCenter)
        logo_layout.addWidget(logo_sub, alignment=Qt.AlignCenter)
        sidebar_layout.addWidget(logo_frame)

        # 侧边栏菜单列表 (核心修改：增加间距和高度)
        self.menu_list = QListWidget()
        self.menu_list.setObjectName("menu_list")
        self.menu_list.setIconSize(QSize(24, 24))
        
        menus = [
            "📁 数据中心", "🪄 预处理模块", "🏷️ 标注管理", 
            "🧠 模型构建", "🚀 训练监控", "📊 效果评估", 
            "🩺 智能诊断", "📑 历史报告"
        ]
        
        for menu_name in menus:
            item = QListWidgetItem(menu_name)
            item.setSizeHint(QSize(0, 70)) # 每个菜单项高度设为 70 像素
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.menu_list.addItem(item)
            
        sidebar_layout.addWidget(self.menu_list)
        
        # 侧边栏底部
        version_lbl = QLabel("V1.5.0 企业版\n数据流转状态: 正常")
        version_lbl.setStyleSheet("color: #7f8c8d; font-size: 11px; padding: 20px;")
        sidebar_layout.addWidget(version_lbl)

        # --- 2. 右侧内容区 ---
        self.content_area = QFrame()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(30, 20, 30, 20)

        # 标题栏
        header_layout = QHBoxLayout()
        self.title_label = QLabel("数据中心")
        self.title_label.setObjectName("page_title")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        self.content_layout.addLayout(header_layout)

        # 核心堆栈容器
        self.stack = QStackedWidget()
        self.init_pages()
        self.content_layout.addWidget(self.stack)

        self.main_layout.addWidget(self.sidebar)
        self.main_layout.addWidget(self.content_area)

        # 信号连接
        self.menu_list.currentRowChanged.connect(self.switch_page)

    def init_pages(self):
        """将 8 个页面按顺序加入 stack"""
        self.stack.addWidget(DataCenterPage())     # 0
        self.stack.addWidget(PreprocessPage())     # 1
        self.stack.addWidget(LabelingPage())       # 2
        self.stack.addWidget(ModelDesignPage())    # 3
        self.stack.addWidget(TrainMonitorPage())   # 4
        self.stack.addWidget(EvaluationPage())     # 5
        self.stack.addWidget(DiagnosisPage())      # 6
        self.stack.addWidget(HistoryReportPage())  # 7

    def switch_page(self, index):
        self.stack.setCurrentIndex(index)
        # 获取当前菜单文字
        menu_text = self.menu_list.item(index).text()
        self.title_label.setText(menu_text)

    def apply_styles(self):
        """全面美化界面的 QSS"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f4f7f9;
            }
            #sidebar {
                background-color: #1c2833;
                border: none;
            }
            #logo_frame {
                background-color: #17202a;
                border-bottom: 1px solid #2c3e50;
            }
            #menu_list {
                background-color: transparent;
                border: none;
                outline: none;
                padding-top: 10px;
            }
            #menu_list::item {
                color: #d5dbdb;
                padding-left: 30px;
                border-left: 5px solid transparent;
                font-size: 15px;
                font-weight: 500;
            }
            #menu_list::item:hover {
                background-color: #2c3e50;
                color: white;
            }
            #menu_list::item:selected {
                background-color: #2471a3;
                color: white;
                border-left: 5px solid #3498db;
            }
            #page_title {
                font-size: 26px;
                font-weight: bold;
                color: #2e4053;
                margin-bottom: 15px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d5dbdb;
                border-radius: 8px;
                margin-top: 20px;
                padding-top: 20px;
                font-size: 14px;
                color: #34495e;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #d5dbdb;
                gridline-color: #f2f4f4;
                border-radius: 5px;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 10px;
                border: none;
                font-weight: bold;
                color: #7f8c8d;
            }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())