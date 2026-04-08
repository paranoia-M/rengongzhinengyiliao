# main.py
import sys
import os

# 将当前目录添加到系统路径，确保能找到 gui 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    # 1. 创建应用程序实例
    app = QApplication(sys.argv)
    
    # 2. 设置全局样式（可选，增加美观度）
    app.setStyle("Fusion") 
    
    # 3. 实例化主窗口
    # 主窗口会自动加载 p2_preprocess, p3_labeling, p5_train_monitor 等页面
    window = MainWindow()
    
    # 4. 显示窗口
    window.show()
    
    # 5. 运行事件循环
    sys.exit(app.exec())

if __name__ == "__main__":
    main()