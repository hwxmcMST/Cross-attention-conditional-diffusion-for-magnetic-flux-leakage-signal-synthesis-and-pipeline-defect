from config import *
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torch



class MagneticImageDataset(Dataset):
    def __init__(self, image_folder):
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_paths) * 10

    def __getitem__(self, idx):
        path = self.image_paths[idx % len(self.image_paths)]
        img = Image.open(path).convert("I").resize((IMG_SIZE,IMG_SIZE))
        img = np.array(img, dtype=np.uint16).astype(np.float32) / 65535.0
        img = (img - 0.5) * 2  # 归一化到 [-1, 1]
        img = torch.tensor(img).unsqueeze(0)  # [1, 100, 100]

        # 提取前四个条件：长、宽、深、角度
        name = os.path.basename(path).split('_')
        condition = torch.tensor([float(name[0]), float(name[1]), float(name[2]), float(name[3])], dtype=torch.float32)
        return img, condition

