import os
import json
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

name = "Bz"


class ImageFolderDataset(Dataset):
    def __init__(self, folder_gt, folder_pred):
        self.folder_gt = folder_gt
        self.folder_pred = folder_pred

        self.pred_files = [f for f in os.listdir(folder_pred) if f.endswith('.png')]
        self.gt_files = [f[:-4] + f'_{name}.png' for f in self.pred_files]

    def __len__(self):
        return len(self.pred_files)

    def __getitem__(self, idx):
        gt_path = os.path.join(self.folder_gt, self.gt_files[idx])
        pred_path = os.path.join(self.folder_pred, self.pred_files[idx])

        img_gt = cv2.imread(gt_path)
        img_pred = cv2.imread(pred_path)

        if img_gt is None or img_pred is None:
            raise FileNotFoundError(f"Missing image: {gt_path} or {pred_path}")

        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)

        h, w = img_gt.shape[:2]
        if img_pred.shape[:2] != (h, w):
            img_pred = cv2.resize(img_pred, (w, h))

        img_gt = torch.from_numpy(img_gt).permute(2, 0, 1).to(torch.uint8)
        img_pred = torch.from_numpy(img_pred).permute(2, 0, 1).to(torch.uint8)

        return img_gt, img_pred


def calculate_metrics(folder1, folder2, use_ssim_rgb=False, device="cuda"):
    files2 = [f for f in os.listdir(folder2) if f.endswith('.png')]

    ssim_vals, psnr_vals, cc_vals, rmse_vals = [], [], [], []

    print("🔍 计算 SSIM、PSNR、CC 和 RMSE...")
    for filename in tqdm(files2, desc="SSIM/PSNR/CC/RMSE"):
        path1 = os.path.join(folder1, filename[:-4] + f'_{name}.png')
        path2 = os.path.join(folder2, filename)

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        if img1 is None or img2 is None:
            print(f"⚠️ 跳过无法读取的图像 {filename}")
            continue

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

        psnr_vals.append(psnr(img1, img2, data_range=1.0))

        if use_ssim_rgb and img1.ndim == 3 and img1.shape[2] == 3:
            ssim_val = ssim(img1, img2, channel_axis=2, data_range=1.0)
            gray1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            gray2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            gray2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            ssim_val = ssim(gray1, gray2, data_range=1.0)
        ssim_vals.append(ssim_val)

        gray1_flat = gray1.flatten()
        gray2_flat = gray2.flatten()
        cc_vals.append(np.corrcoef(gray1_flat, gray2_flat)[0, 1])
        rmse_vals.append(np.sqrt(np.mean((gray1_flat - gray2_flat) ** 2)))

    avg_ssim = np.mean(ssim_vals)
    avg_psnr = np.mean(psnr_vals)
    avg_cc   = np.mean(cc_vals)
    avg_rmse = np.mean(rmse_vals)

    # ===== FID 和 KID =====
    print("\n🧠 计算 FID 和 KID...")
    dataset = ImageFolderDataset(folder1, folder2)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)

    fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(device)
    kid_metric = KernelInceptionDistance(subset_size=min(50, len(dataset))).to(device)

    print("📌 更新真实图像...")
    for imgs_gt, _ in tqdm(dataloader, desc="Real"):
        imgs_gt = imgs_gt.to(device)
        fid_metric.update(imgs_gt, real=True)
        kid_metric.update(imgs_gt, real=True)

    print("📌 更新生成图像...")
    for _, imgs_pred in tqdm(dataloader, desc="Fake"):
        imgs_pred = imgs_pred.to(device)
        fid_metric.update(imgs_pred, real=False)
        kid_metric.update(imgs_pred, real=False)

    fid_score = fid_metric.compute().item()
    kid_mean, kid_std = kid_metric.compute()
    kid_mean = kid_mean.item()
    kid_std = kid_std.item()

    print(f"\n✅ 平均 SSIM: {avg_ssim:.4f}")
    print(f"✅ 平均 PSNR: {avg_psnr:.4f} dB")
    print(f"✅ 平均 CC:   {avg_cc:.4f}")
    print(f"✅ 平均 RMSE: {avg_rmse:.4f}")
    print(f"✅ FID: {fid_score:.4f}")
    print(f"✅ KID: {kid_mean:.6f} ± {kid_std:.6f}")

    return {
        "ssim":     avg_ssim,
        "psnr":     avg_psnr,
        "cc":       avg_cc,
        "rmse":     avg_rmse,
        "fid":      fid_score,
        "kid_mean": kid_mean,
        "kid_std":  kid_std,
    }


if __name__ == "__main__":
    directions = ["Bx", "By", "Bz"]
    all_metrics = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name in directions:
        print(f"\n{'='*50}")
        print(f"📊 评估方向: {name}")
        print(f"{'='*50}")

        folder_gt   = f"/root/autodl-tmp/myproject/Transformer_Diffusion/images/test/images/{name}"
        folder_pred = f"/root/autodl-tmp/myproject/Transformer_Diffusion/gen/{name}"

        globals()['name'] = name

        try:
            metrics = calculate_metrics(folder_gt, folder_pred, use_ssim_rgb=False, device=device)
            metrics["direction"] = name
            all_metrics.append(metrics)
        except Exception as e:
            print(f"❌ 跳过 {name} 因错误: {e}")
            continue

    if all_metrics:
        keys = ["ssim", "psnr", "cc", "rmse", "fid", "kid_mean"]
        avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in keys}

        print(f"\n{'='*60}")
        print("📈 三方向平均指标 (Bx, By, Bz):")
        print(f"{'='*60}")
        print(f"✅ 平均 SSIM: {avg_metrics['ssim']:.4f}")
        print(f"✅ 平均 PSNR: {avg_metrics['psnr']:.4f} dB")
        print(f"✅ 平均 CC:   {avg_metrics['cc']:.4f}")
        print(f"✅ 平均 RMSE: {avg_metrics['rmse']:.4f}")
        print(f"✅ 平均 FID:  {avg_metrics['fid']:.4f}")
        print(f"✅ 平均 KID:  {avg_metrics['kid_mean']:.6f}")

        with open("average_metrics.json", "w") as f:
            json.dump({k: float(v) for k, v in avg_metrics.items()}, f, indent=4)
        print("\n💾 结果已保存至 average_metrics.json")
    else:
        print("❌ 未成功计算任何方向的指标。")