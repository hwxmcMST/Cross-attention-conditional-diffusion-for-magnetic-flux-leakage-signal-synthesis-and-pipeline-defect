import os
import numpy as np
from PIL import Image

import sys
sys.stdout.reconfigure(encoding='utf-8')


def combine_channels_to_txt(base_dir, output_dir):
    """
    合并三个通道(Bx,By,Bz)的图像数据到一个TXT文件
    参数:
        base_dir: 包含Bx,By,Bz子目录的根目录
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    channels = ['Bx', 'By', 'Bz']
    
    # 获取所有文件名(假设三个目录中的文件名相同)
    sample_names = [f for f in os.listdir(os.path.join(base_dir, 'Bx')) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    
    for name in sample_names:
        try:
            # 读取三个通道的数据
            channel_data = []
            for ch in channels:
                img_path = os.path.join(base_dir, ch, name)
                img = Image.open(img_path)
                img_array = np.array(img).flatten()
                channel_data.append(img_array)
            
            # 合并为Nx3格式
            combined = np.column_stack(channel_data)
            
            # 保存为TXT
            output_name = os.path.splitext(name)[0] + '.txt'
            output_path = os.path.join(output_dir, output_name)
            np.savetxt(output_path, combined, fmt='%d')
            print(f" 已保存合并数据: {output_path}")
            
        except Exception as e:
            print(f"❌ 处理 {name} 出错: {e}")

# 示例用法
if __name__ == "__main__":
    base_dir = r"D:\Diffusion\Stable-Diffusion\gen"  # 包含Bx,By,Bz子目录
    output_dir = r"D:\Diffusion\Stable-Diffusion\gen\combined_txt"
    combine_channels_to_txt(base_dir, output_dir)