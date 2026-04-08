import time
import random
from PySide6.QtCore import QThread, Signal

class TrainingSimulator(QThread):
    # 信号：发送 (epoch, loss, accuracy, 学习状态描述)
    update_signal = Signal(int, float, float, str)

    def run(self):
        # 模拟 100 轮训练
        for epoch in range(1, 101):
            time.sleep(0.2)  # 模拟计算耗时
            
            # 模拟 Loss 下降和 Acc 上升
            loss = 1.5 * (0.9 ** epoch) + random.uniform(0, 0.05)
            acc = 0.3 + (0.65 * (epoch / 100)) + random.uniform(0, 0.02)
            
            # 模拟学习过程中的数据描述
            status = "正在提取甲状腺结节边缘特征..."
            if epoch > 30: status = "识别 4C 级恶性钙化点..."
            if epoch > 70: status = "优化手术决策权重算法..."
            
            self.update_signal.emit(epoch, loss, acc, status)
            