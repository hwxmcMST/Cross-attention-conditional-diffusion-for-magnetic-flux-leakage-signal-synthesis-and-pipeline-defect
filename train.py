
from stable_unet import UNet
from config import *
from torch.utils.data import DataLoader
from dataset import MagneticImageDataset
from diffusion import forward_diffusion as forward_add_noise
import torch
from torch import nn
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备

parser = argparse.ArgumentParser(description="Train DiT diffusion model on magnetic field channel")

parser.add_argument('--channel', type=str, required=True,

                    help='Name of the magnetic field channel (e.g., Bx, By, Bz)')

args = parser.parse_args()

channel_name = args.channel

train_path = './images/train/images/' + channel_name
train_dataset = MagneticImageDataset(train_path)  # 数据集

val_path = './images/val/images/' + channel_name
val_dataset = MagneticImageDataset(val_path)


model=UNet(1).to(DEVICE)   # 噪音预测模型
optimzer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 优化器
loss_fn = nn.L1Loss()  # 损失函数(绝对值误差均值)

'''
    训练模型
'''

EPOCH = 500
BATCH_SIZE = 16

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                              persistent_workers=True)  # 数据加载器
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, persistent_workers=True)

train_losses = []
val_losses = []
best_val_loss = 1000

for epoch in range(EPOCH):
    model.train()
    train_loss = 0
    for imgs, labels in tqdm(train_dataloader):
        x = imgs
        t = torch.randint(0, T, (imgs.size(0),))  # 为每张图片生成随机t时刻
        y = labels

        x, noise = forward_add_noise(x.to(DEVICE), t.to(DEVICE))  # x:加噪图 noise:噪音
        pred_noise = model(x.to(DEVICE), t.to(DEVICE), y.to(DEVICE))

        loss = loss_fn(pred_noise, noise.to(DEVICE))

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        train_loss += loss.item()

    print('epoch:{},train_loss:{}'.format(epoch, train_loss / len(train_dataloader)))
    train_losses.append(train_loss / len(train_dataloader))

    model.eval()
    val_loss = 0
    for imgs, labels in tqdm(val_dataloader):
        x = imgs 
        t = torch.randint(0, T, (imgs.size(0),))  # 为每张图片生成随机t时刻
        y = labels

        x, noise = forward_add_noise(x.to(DEVICE), t.to(DEVICE))  # x:加噪图 noise:噪音
        pred_noise = model(x.to(DEVICE), t.to(DEVICE), y.to(DEVICE))

        loss = loss_fn(pred_noise, noise.to(DEVICE))
        val_loss += loss.item()

    print('epoch:{},val_loss:{}'.format(epoch, val_loss / len(val_dataloader)))
    val_losses.append(val_loss / len(val_dataloader))

    if val_loss / len(val_dataloader) < best_val_loss:
        best_val_loss = val_loss / len(val_dataloader)
        torch.save(model.state_dict(), f'./weight/model_{channel_name}.pth')

val_losses = np.array(val_losses)
train_losses = np.array(train_losses)

save_path = './logs/' + channel_name
os.makedirs(save_path, exist_ok=True)

np.save(os.path.join(save_path, 'train_losses.npy'), train_losses)
np.save(os.path.join(save_path, 'val_losses.npy'), val_losses)

# 创建横坐标：epoch 数
epochs = list(range(EPOCH))
# 绘制训练和验证损失曲线

plt.figure()
plt.plot(epochs, train_losses, label='train_loss', color='blue')
plt.plot(epochs, val_losses, label='val_loss', color='red')

# 添加标题和标签

plt.title(f'{channel_name}_LOSS')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# 保存图像（建议保存为高分辨率）

plt.savefig(os.path.join(save_path, f'{channel_name}.png'))