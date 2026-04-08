import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ThyroidDataset(Dataset):
    """甲状腺疾病影像数据集管理"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 严格对应你要求的三个类别
        self.classes = [
            "benign_large",      # 良性大结节·存在恶变可能性
            "malignant_4c",     # 恶性4C·建议手术治疗
            "malignant_urgent"  # 确诊恶性·必须立即进行手术治疗
        ]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for class_name in self.classes:
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_path):
                # 自动创建目录结构
                os.makedirs(class_path, exist_ok=True)
                continue
            
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    samples.append((
                        os.path.join(class_path, img_name),
                        self.class_to_idx[class_name]
                    ))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"读取图片失败: {path}, 错误: {e}")
            return torch.zeros((3, 224, 224)), label