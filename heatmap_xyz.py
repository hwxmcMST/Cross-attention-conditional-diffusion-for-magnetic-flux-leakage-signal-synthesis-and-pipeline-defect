import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
sys.stdout.reconfigure(encoding='utf-8')

def plot_txt_heatmaps(txt_dir, output_dir=None, levels=50):
    """
    可视化TXT文件为heatmap
    参数:
        txt_dir: 包含TXT文件的目录
        output_dir: 图片输出目录(可选)
    """
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    
    # 自定义colormap
    cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'white', 'red'])
    
    # 修改colormap为viridis
    cmap = 'viridis'  # 改用更平滑的colormap
    
    for filename in os.listdir(txt_dir):
        if filename.endswith('.txt'):
            try:
                # 读取TXT文件
                filepath = os.path.join(txt_dir, filename)
                data = np.loadtxt(filepath)
                
                # 计算图像尺寸 (100x100)
                img_size = 100  # 固定为100x100
                
                # 创建包含三个子图的图像
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                channels = ['Bx', 'By', 'Bz']
                
                for i, ax in enumerate(axes):
                    channel_data = data[:, i].reshape((img_size, img_size))
                    
                    # 使用新的colormap
                    im = ax.contourf(channel_data, levels=levels, cmap=cmap)
                    ax.set_title(f"{channels[i]} Channel")
                    fig.colorbar(im, ax=ax)
                
                plt.suptitle(f"Heatmap of {os.path.splitext(filename)[0]}")
                
                # 保存或显示
                if output_dir:
                    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
                    plt.savefig(output_path)
                    plt.close()
                    print(f"✅ 已保存: {output_path}")
                else:
                    plt.show()
                    
            except Exception as e:
                print(f"❌ 处理 {filename} 出错: {e}")

# 示例用法
if __name__ == "__main__":
    txt_dir = r"D:\Diffusion\Stable-Diffusion\gen\combined_txt_T"  # TXT文件目录
    output_dir = r"D:\Diffusion\Stable-Diffusion\gen\heatmaps_XYZ"    # 图片输出目录(可选)
    
    plot_txt_heatmaps(txt_dir, output_dir)  # 带输出目录
    # plot_txt_heatmaps(txt_dir)  # 不带输出目录，直接显示