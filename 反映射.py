import os
import glob
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ====== 原始物理范围（来自训练映射）======
BX_MIN, BX_MAX = 0.012983683832, 0.082275489925
BY_MIN, BY_MAX = -0.046791405976, 0.046332394822
BZ_MIN, BZ_MAX = -0.034353275374, 0.033069276353

def unmap_from_uint16(x, x_min, x_max):
    """将整数值（0~65535）映射回物理值"""
    return x / 65535.0 * (x_max - x_min) + x_min

def process_file(input_path, output_path):
    try:
        data = np.loadtxt(input_path, dtype=np.uint16)
        if data.ndim == 1 and data.shape[0] == 3:
            data = data[np.newaxis, :]  # 单行情况修正
        elif data.ndim != 2 or data.shape[1] != 3:
            print(f"文件格式错误: {input_path}")
            return

        bx = unmap_from_uint16(data[:, 0], BX_MIN, BX_MAX)
        by = unmap_from_uint16(data[:, 1], BY_MIN, BY_MAX)
        bz = unmap_from_uint16(data[:, 2], BZ_MIN, BZ_MAX)

        restored = np.stack([bx, by, bz], axis=1)
        np.savetxt(output_path, restored, fmt='%.32f', delimiter='\t')
        print(f" 已保存: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"处理失败 {input_path}: {e}")

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files = glob.glob(os.path.join(input_folder, "*.txt"))
    print(f"找到 {len(files)} 个 .txt 文件")

    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_folder, base + ".txt")
        process_file(path, out_path)

    print("全部反映射完成。输出路径：", output_folder)

# ====== 示例调用 ======
if __name__ == "__main__":
    input_folder = r"D:\Diffusion\Stable-Diffusion\gen\combined_txt"     # 输入文件夹（映射后）   
    output_folder = r"D:\Diffusion\Stable-Diffusion\gen\combined_txt_T"   # 输出文件夹（反映射结果）
    process_folder(input_folder, output_folder)
