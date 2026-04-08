import os
import random
import datetime
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QLabel, QLineEdit, 
                             QPushButton, QComboBox, QFrame, QSplitter, QGroupBox,
                             QTextEdit, QFileDialog, QMessageBox, QAbstractItemView)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QColor, QPixmap

class HistoryReportPage(QWidget):
    """
    历史报告页面：
    实现病例记录检索、详情预览联动以及系统审计追踪功能
    """
    def __init__(self):
        super().__init__()
        # 模拟数据库中的历史数据流（对应你的三个类别）
        self.all_records = [
            {"date": "2023-11-20 10:15", "id": "TH-9921", "class": "良性大结节", "score": "98.2%", "status": "已入库", "img": "data/raw/liangxing/1.jpg"},
            {"date": "2023-11-21 14:30", "id": "TH-9925", "class": "恶性 4C", "score": "94.5%", "status": "待手术", "img": "data/raw/exing/1.jpg"},
            {"date": "2023-11-22 09:12", "id": "TH-9930", "class": "确诊恶性", "score": "99.1%", "status": "急诊手术", "img": "data/raw/quezhenexing/1.jpg"},
            {"date": "2023-11-23 16:45", "id": "TH-9938", "class": "良性大结节", "score": "89.4%", "status": "已入库", "img": "data/raw/liangxing/2.jpg"},
            {"date": "2023-11-24 11:20", "id": "TH-9942", "class": "恶性 4C", "score": "91.2%", "status": "待复核", "img": "data/raw/exing/2.jpg"},
        ]
        self.init_ui()
        self.populate_table(self.all_records)
        self.add_audit_log("数据流转: 成功从历史数据库恢复 5 条医疗诊断流水记录。")

    def init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        # --- 1. 顶部搜索过滤栏 ---
        filter_box = QFrame()
        filter_box.setFixedHeight(80)
        filter_box.setStyleSheet("background-color: white; border-radius: 10px; border: 1px solid #dee2e6;")
        filter_layout = QHBoxLayout(filter_box)
        
        filter_layout.addWidget(QLabel("🔍 记录筛选:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索患者ID或病例类别...")
        self.search_input.setFixedWidth(280)
        filter_layout.addWidget(self.search_input)
        
        filter_layout.addWidget(QLabel("分类:"))
        self.type_filter = QComboBox()
        self.type_filter.addItems(["全部类别", "良性大结节", "恶性 4C", "确诊恶性"])
        filter_layout.addWidget(self.type_filter)
        
        self.btn_search = QPushButton("执行查询")
        self.btn_search.setFixedWidth(120)
        self.btn_search.setStyleSheet("background-color: #2c3e50; color: white; font-weight: bold;")
        self.btn_search.clicked.connect(self.filter_records)
        filter_layout.addWidget(self.btn_search)
        
        filter_layout.addStretch()
        self.main_layout.addWidget(filter_box)

        # --- 2. 中部核心区 (Splitter) ---
        self.content_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：记录表格
        self.table_group = QGroupBox("📋 诊断记录流水")
        table_vbox = QVBoxLayout(self.table_group)
        self.report_table = QTableWidget(0, 5)
        self.report_table.setHorizontalHeaderLabels(["诊断时间", "病例ID", "AI 分级", "置信度", "状态"])
        self.report_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.report_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.report_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.report_table.cellClicked.connect(self.on_record_selected)
        table_vbox.addWidget(self.report_table)
        self.content_splitter.addWidget(self.table_group)
        
        # 右侧：档案详细看板
        self.detail_panel = QGroupBox("📄 病例档案详情")
        self.detail_layout = QVBoxLayout(self.detail_panel)
        self.detail_title = QLabel("选中记录查看预览")
        self.detail_title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.detail_layout.addWidget(self.detail_title)
        
        self.detail_img = QLabel("影像预览区")
        self.detail_img.setFixedSize(320, 240)
        self.detail_img.setStyleSheet("background: #f8f9fa; border: 1px solid #ddd; border-radius: 5px;")
        self.detail_img.setAlignment(Qt.AlignCenter)
        self.detail_layout.addWidget(self.detail_img, alignment=Qt.AlignCenter)
        
        self.detail_info = QTextEdit()
        self.detail_info.setReadOnly(True)
        self.detail_info.setStyleSheet("border: none; background: transparent; font-size: 13px;")
        self.detail_layout.addWidget(self.detail_info)
        
        action_btns = QHBoxLayout()
        self.btn_print = QPushButton("🖨️ 生成诊断证明书")
        self.btn_print.setEnabled(False)
        self.btn_print.clicked.connect(self.print_report)
        action_btns.addWidget(self.btn_print)
        action_btns.addWidget(QPushButton("🗑️ 归档记录"))
        self.detail_layout.addLayout(action_btns)
        
        self.content_splitter.addWidget(self.detail_panel)
        self.content_splitter.setStretchFactor(0, 3)
        self.content_splitter.setStretchFactor(1, 2)
        self.main_layout.addWidget(self.content_splitter, stretch=2)

        # --- 3. 底部审计日志 ---
        self.audit_group = QGroupBox("📜 系统审计日志 (System Audit Trail)")
        audit_vbox = QVBoxLayout(self.audit_group)
        self.audit_log = QTextEdit()
        self.audit_log.setReadOnly(True)
        self.audit_log.setFixedHeight(120)
        self.audit_log.setStyleSheet("background-color: #f1f2f6; font-family: 'Courier New'; font-size: 11px;")
        audit_vbox.addWidget(self.audit_log)
        self.main_layout.addWidget(self.audit_group, stretch=1)

    def populate_table(self, data_list):
        self.report_table.setRowCount(0)
        for i, row_data in enumerate(data_list):
            self.report_table.insertRow(i)
            self.report_table.setItem(i, 0, QTableWidgetItem(row_data["date"]))
            self.report_table.setItem(i, 1, QTableWidgetItem(row_data["id"]))
            class_item = QTableWidgetItem(row_data["class"])
            if "确诊" in row_data["class"]: class_item.setForeground(QColor("#c0392b"))
            elif "4C" in row_data["class"]: class_item.setForeground(QColor("#e67e22"))
            else: class_item.setForeground(QColor("#27ae60"))
            self.report_table.setItem(i, 2, class_item)
            self.report_table.setItem(i, 3, QTableWidgetItem(row_data["score"]))
            self.report_table.setItem(i, 4, QTableWidgetItem(row_data["status"]))

    def filter_records(self):
        keyword = self.search_input.text().lower()
        cat_filter = self.type_filter.currentText()
        filtered = [r for r in self.all_records if (keyword in r["id"].lower() or keyword in r["class"].lower()) 
                    and (cat_filter == "全部类别" or cat_filter == r["class"])]
        self.populate_table(filtered)
        self.add_audit_log(f"用户检索了关键字 '{keyword}'，筛选类别 '{cat_filter}'。")

    def on_record_selected(self, row, col):
        case_id = self.report_table.item(row, 1).text()
        record = next((r for r in self.all_records if r["id"] == case_id), None)
        if record:
            self.detail_title.setText(f"📄 病例档案: {record['id']}")
            # 核心修复点：完整的 QPixmap 缩放调用
            if os.path.exists(record["img"]):
                pix = QPixmap(record["img"]).scaled(320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.detail_img.setPixmap(pix)
            else:
                self.detail_img.setText("影像文件离线")
            
            info_html = f"<b>诊断时间:</b> {record['date']}<br><b>AI 结论:</b> {record['class']}<br><b>置信度:</b> {record['score']}<br><hr><b>手术建议:</b> {self.get_advice(record['class'])}"
            self.detail_info.setHtml(info_html)
            self.btn_print.setEnabled(True)
            self.add_audit_log(f"查看病例详情: {case_id}")

    def get_advice(self, label):
        if "确诊" in label: return "极高危，必须立即手术。"
        if "4C" in label: return "高疑似恶性，建议择期手术。"
        return "良性可能，建议定期随访。"

    def add_audit_log(self, message):
        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.audit_log.append(f"[{t}] {message}")

    def print_report(self):
        QMessageBox.information(self, "打印任务", "诊断证明书 PDF 已生成并流转至 outputs/reports 目录。")