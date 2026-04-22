import math
import torch
from torch import nn

from config import DEVICE, T
from time_position_emb import TimePositionEmbedding
from diffusion import forward_diffusion
from dataset import MagneticImageDataset


def _group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    for g in (32, 16, 8, 4, 2, 1):
        if g <= max_groups and num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class FiLMResBlock(nn.Module):
    # Pure conv residual block + FiLM conditioning (time + geometry). No attention.
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.act = nn.SiLU()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = _group_norm(out_ch)

        emb_dim = time_dim + cond_dim
        self.film1 = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * out_ch))
        self.film2 = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * out_ch))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    @staticmethod
    def _apply_film(h: torch.Tensor, film: nn.Module, emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = film(emb).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return h * (1.0 + gamma) + beta

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([t_emb, cond_emb], dim=1)

        h = self.conv1(x)
        h = self.norm1(h)
        h = self._apply_film(h, self.film1, emb)
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self._apply_film(h, self.film2, emb)
        h = self.act(self.dropout(h))

        return h + self.skip(x)


class BottleneckGeometryCrossAttn(nn.Module):
    # Geometry-conditioned cross-attention ONLY for bottleneck.
    def __init__(self, feat_dim: int, cond_dim: int, heads: int = 4, qsize: int = 16, vsize: int = 16, ctx_len: int = 8):
        super().__init__()
        self.heads = heads
        self.qsize = qsize
        self.vsize = vsize
        self.ctx_len = ctx_len

        self.to_q = nn.Linear(feat_dim, heads * qsize, bias=False)
        self.to_k = nn.Linear(cond_dim, heads * qsize, bias=False)
        self.to_v = nn.Linear(cond_dim, heads * vsize, bias=False)
        self.to_out = nn.Linear(heads * vsize, feat_dim, bias=False)

        # cond_emb -> L context tokens
        self.ctx_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, ctx_len * cond_dim))
        self.scale = 1.0 / math.sqrt(qsize)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        hw = H * W

        xt = x.permute(0, 2, 3, 1).reshape(B, hw, C)  # [B, HW, C]
        q = self.to_q(xt).view(B, hw, self.heads, self.qsize).permute(0, 2, 1, 3)  # [B, h, HW, q]

        ctx = self.ctx_proj(cond_emb).view(B, self.ctx_len, -1)  # [B, L, cond_dim]
        k = self.to_k(ctx).view(B, self.ctx_len, self.heads, self.qsize).permute(0, 2, 1, 3)  # [B, h, L, q]
        v = self.to_v(ctx).view(B, self.ctx_len, self.heads, self.vsize).permute(0, 2, 1, 3)  # [B, h, L, v]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B,h,HW,v]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, hw, self.heads * self.vsize)  # [B,HW,h*v]
        out = self.to_out(out).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return x + out


class UNet_BottleneckCA(nn.Module):
    # XCDiff (Only Bottleneck CA): only bottleneck has geometry-conditioned cross-attn.
    def __init__(
        self,
        img_channel: int,
        channels=[64, 128, 256, 512, 1024],
        time_emb_size: int = 256,
        qsize: int = 16,
        vsize: int = 16,
        fsize: int = 32,      # kept for signature compatibility (unused)
        cls_emb_size: int = 32,
        heads: int = 4,
        ctx_len: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        channels = [img_channel] + channels

        self.time_emb = nn.Sequential(
            TimePositionEmbedding(time_emb_size),
            nn.Linear(time_emb_size, time_emb_size),
            nn.ReLU(),
        )

        # cls is geometry: [l, w, d, theta], float tensor of shape [B,4]
        self.cls_emb = nn.Sequential(
            nn.Linear(4, cls_emb_size),
            nn.ReLU(),
            nn.Linear(cls_emb_size, cls_emb_size),
        )

        self.enc_convs = nn.ModuleList([
            FiLMResBlock(channels[i], channels[i + 1], time_dim=time_emb_size, cond_dim=cls_emb_size, dropout=dropout)
            for i in range(len(channels) - 1)
        ])

        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2)
            for _ in range(len(channels) - 2)
        ])

        self.bottleneck_ca = BottleneckGeometryCrossAttn(
            feat_dim=channels[-1],
            cond_dim=cls_emb_size,
            heads=heads,
            qsize=qsize,
            vsize=vsize,
            ctx_len=ctx_len,
        )

        self.deconvs = nn.ModuleList([
            nn.ConvTranspose2d(channels[-i - 1], channels[-i - 2], kernel_size=2, stride=2)
            for i in range(len(channels) - 2)
        ])

        self.dec_convs = nn.ModuleList([
            FiLMResBlock(channels[-i - 1], channels[-i - 2], time_dim=time_emb_size, cond_dim=cls_emb_size, dropout=dropout)
            for i in range(len(channels) - 2)
        ])

        self.output = nn.Conv2d(channels[1], img_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        cls_emb = self.cls_emb(cls)

        residual = []
        for i, block in enumerate(self.enc_convs):
            x = block(x, t_emb, cls_emb)
            if i != len(self.enc_convs) - 1:
                residual.append(x)
                x = self.maxpools[i](x)

        # ONLY bottleneck cross-attn
        x = self.bottleneck_ca(x, cls_emb)

        for i, up in enumerate(self.deconvs):
            x = up(x)
            skip = residual.pop(-1)
            x = self.dec_convs[i](torch.cat([skip, x], dim=1), t_emb, cls_emb)

        return self.output(x)


if __name__ == "__main__":
    channel_name = "Bx"
    train_path = f"/root/autodl-tmp/myproject/B_Stable-Diffusion/images/train/images/{channel_name}"
    train_dataset = MagneticImageDataset(train_path)

    x0, c0 = train_dataset[0]
    print("single sample shape:", x0.shape)

    batch_x = torch.stack([train_dataset[0][0], train_dataset[1][0]], dim=0).to(DEVICE)

    # IMPORTANT: cls must be float tensor of shape [B,4]
    batch_cls = torch.stack([train_dataset[0][1], train_dataset[1][1]], dim=0).float().to(DEVICE)

    batch_t = torch.randint(0, T, size=(batch_x.size(0),), device=DEVICE)
    batch_x_t, batch_noise_t = forward_diffusion(batch_x, batch_t)

    print("batch_x_t:", batch_x_t.size())
    print("batch_noise_t:", batch_noise_t.size())

    unet = UNet_BottleneckCA(img_channel=1).to(DEVICE)
    batch_predict_noise_t = unet(batch_x_t, batch_t, batch_cls)
    print("batch_predict_noise_t:", batch_predict_noise_t.size())
