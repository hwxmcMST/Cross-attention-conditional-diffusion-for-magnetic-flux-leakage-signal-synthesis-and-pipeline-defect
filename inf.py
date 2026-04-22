import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pandas as pd
from diffusion import forward_diffusion as forward_add_noise
from diffusion import *
from stable_unet import UNet
DEVICE='cuda' if torch.cuda.is_available() else 'cpu' # 设备

# ====== 保存图像函数 ======
def save_image(tensor_img, path):
    img_np = tensor_img.squeeze().cpu().detach().numpy()
    img_np = ((img_np + 1) / 2 * 65535).clip(0, 65535).astype(np.uint16)
    Image.fromarray(img_np).resize((100,100)).save(path)

# ====== 单图生成函数 ======
def backward_denoise(model, y , output_path):
    model.eval()
    x = (torch.randn(size=(1,1,128,128))-0.5) * 2
    y = torch.tensor(y,dtype = torch.float32)
    steps = [x.clone(), ]

    global alphas, alphas_cumprod, variance

    x = x.to(DEVICE)
    alphas = alphas.to(DEVICE)
    alphas_cumprod = alphas_cumprod.to(DEVICE)
    variance = variance.to(DEVICE)
    y = y.to(DEVICE)

    model.eval()
    with torch.no_grad():
        for time in range(T - 1, -1, -1):
            t = torch.full((x.size(0),), time).to(DEVICE)

            # 预测x_t时刻的噪音
            noise = model(x, t, y)

            # 生成t-1时刻的图像
            shape = (x.size(0), 1, 1, 1)
            mean = 1 / torch.sqrt(alphas[t].view(*shape)) * \
                   (
                           x -
                           (1 - alphas[t].view(*shape)) / torch.sqrt(1 - alphas_cumprod[t].view(*shape)) * noise
                   )
            if time != 0:
                x = mean + \
                    torch.randn_like(x) * \
                    torch.sqrt(variance[t].view(*shape))
            else:
                x = mean
            x = torch.clamp(x, -1.0, 1.0).detach()
            steps.append(x)
    save_image(steps[-1][0],output_path)

# ====== 主流程：读取CSV批量生成 ======
def generate_from_csv(model_path, csv_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model=UNet(1).to(DEVICE)  # 模型
    model.load_state_dict(torch.load(model_path,map_location=device))

    # 读取CSV（必须包含：length,width,height,angle）
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    for idx, row in df.iterrows():
        length = float(row['length'])
        width = float(row['width'])
        height = float(row['height'])
        angle = float(row['angle'])
        cond = np.array([[length, width, height, angle]])

        # 修改后的文件名生成逻辑：不补零
        def format_val(v):
            return str(int(v)) if v == int(v) else str(v).rstrip('0').rstrip('.') if '.' in str(v) else str(v)
        filename = f"{format_val(length)}_{format_val(width)}_{format_val(height)}_{format_val(angle)}.png"
        output_path = os.path.join(output_dir, filename)

        backward_denoise(model, cond, output_path)
        print(f"✅ 生成图像: {output_path}")


# ====== 示例调用 ======
if __name__ == "__main__":
    model_path = r"./weight/model_Bz.pth"      # 模型路径
    csv_path = r"./csv/test.csv"                               # 输入CSV路径，必须包含4列
    output_dir = r"./gen/Bz"                  # 输出目录
    generate_from_csv(model_path, csv_path, output_dir)