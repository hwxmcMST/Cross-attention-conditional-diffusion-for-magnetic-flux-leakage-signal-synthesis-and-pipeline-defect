import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
sys.stdout.reconfigure(encoding='utf-8')

def plot_txt_heatmaps(txt_dir, output_dir=None, levels=50):
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    
    # 自定义colormap
    cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'white', 'red'])
    
    # 修改colormap为viridis
    cmap = 'viridis'  # 改用更平滑的colormap
    
    # 定义各通道的范围
    CHANNEL_RANGES = {
        'Bx': (0.012983683832, 0.082275489925),
        'By': (-0.046791405976, 0.046332394822),
        'Bz': (-0.034353275374, 0.033069276353)
    }
    
    for filename in os.listdir(txt_dir):
        if filename.endswith('.txt'):
            try:
                # 读取TXT文件
                filepath = os.path.join(txt_dir, filename)
                data = np.loadtxt(filepath)
                
                # 计算图像尺寸 (100x100)
                img_size = 100  # 固定为100x100
                
                channels = ['Bx', 'By', 'Bz']
                
                for i, channel in enumerate(channels):
                    plt.figure()
                    channel_data = data[:, i].reshape((img_size, img_size))
                    
                    # 获取当前通道的范围
                    vmin, vmax = CHANNEL_RANGES[channel]
                    
                    # 绘制单个通道的heatmap，并指定范围
                    plt.contourf(channel_data, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
                    plt.axis('off')
                    
                    # 保存单个通道图片
                    if output_dir:
                        base_name = os.path.splitext(filename)[0]
                        output_path = os.path.join(output_dir, f"{base_name}_{channel}.png")
                        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        print(f"已保存: {output_path}")
                    else:
                        plt.show()
                        
            except Exception as e:
                print(f" 处理 {filename} 出错: {e}")

# 示例用法
if __name__ == "__main__":
    # txt_dir = r"D:\DZ\CGAN_0705\TXT"  # TXT文件目录
    # output_dir = r"D:\DZ\CGAN_0705\gen\heatmaps_part_ori"    # 图片输出目录(可选)

    txt_dir = r"D:\research\GAN\DZ\CGAN_0705\gen\WGANGP_50\combined_txt_T"  # TXT文件目录
    output_dir = r"D:\research\GAN\DZ\CGAN_0705\gen\WGANGP_50\heatmaps_part"    # 图片输出目录(可选)

    plot_txt_heatmaps(txt_dir, output_dir)  # 带输出目录
    # plot_txt_heatmaps(txt_dir)  # 不带输出目录，直接显示