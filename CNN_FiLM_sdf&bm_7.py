# =============================================================
# CNN.py  ——  边界条件 → 3D 温度场
# 3D UNet + Optuna + Mask + 梯度损失 + 小批量训练（避免 OOM）
# 输入（7通道体素）来自：cnn_input_channels_no_normals.csv 的后 7 列
#   列名：x y z C0 C1 C2 C3 C4 C5 sdf
#   作为体素输入通道使用：[C0,C1,C2,C3,C4,C5,sdf]
# 监督 mask 使用 C0（inside_mask）
# 温度标签来自：Temp_all.csv（每列长度=仅 C0==1 的点数）
# =============================================================

import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------
# 全局监督 Mask（1,1,nx,ny,nz）：来自 C0（inside_mask）
# 固定 3D 体素输入（1,7,nx,ny,nz）：来自 cnn_input_channels_no_normals.csv
# -------------------------------------------------------------
GEOM_MASK = None
VOXEL_INPUT = None

# （可选）坐标离散轴（用于调试/可视化；模型不直接使用）
X_UNIQUE_T = None  # (nx,)
Y_UNIQUE_T = None  # (ny,)
Z_UNIQUE_T = None  # (nz,)

# --------------------- 设备 ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 提前初始化 CUDA 上下文，避免第一次使用 cuBLAS 时出现 "no current CUDA context" warning
if device.type == "cuda":
    torch.cuda.init()
    print(f"CUDA devices: {torch.cuda.device_count()} visible.")


# =============================================================
# 3D UNet（FiLM 方案A）：固定 7 通道体素输入 → UNet；bc 仅用于 FiLM 调制
# =============================================================

class ConvBlock(nn.Module):
    """3D Residual Block：Conv3d → BN → GELU → Conv3d → BN → GELU → Dropout3d + residual"""

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            nn.Dropout3d(dropout_p),
        )
        self.residual_proj = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out = self.conv(x)
        return out + residual


def create_cnn3d_from_trial(trial: optuna.trial.Trial, input_dim: int, nx: int, ny: int, nz: int) -> nn.Module:
    """
    方案A（FiLM 条件调制）：
    - 输入为固定 7 通道体素场（来自 cnn_input_channels_no_normals.csv 的后 7 列）：
        [C0 inside_mask,
         C1 wall_mask,
         C2 inlet_mask,
         C3 outlet_mask,
         C4 heat_source_mask,
         C5 (extra mask/channel),
         sdf]
      其中 C0(inside_mask) == 1 表示有效监督区域。
    - 边界条件 bc (R^input_dim) 仅用于生成各 block 的 FiLM 参数 gamma/beta。
    """

    # 结构超参
    depth = 2
    base_ch = trial.suggest_categorical("base_ch", [12, 16, 24])
    dropout_p = trial.suggest_float("dropout_p", 0.0, 0.3)

    # FiLM 的隐藏层宽度
    film_mult = trial.suggest_categorical("film_mult", [4, 8])  # 4x or 8x
    film_hidden = int(base_ch * film_mult)
    class FiLMGen(nn.Module):
        """bc -> {gamma,beta} for each block (channel-wise), shapes: (B,C,1,1,1)."""

        def __init__(self, input_dim: int, ch_list: list[int], hidden: int):
            super().__init__()
            self.ch_list = ch_list
            out_dim = 2 * sum(ch_list)  # gamma + beta
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, bc: torch.Tensor):
            B = bc.size(0)
            v = self.net(bc)  # (B, 2*sumC)
            gammas = []
            betas = []
            offset = 0
            for C in self.ch_list:
                g = v[:, offset : offset + C]
                offset += C
                b = v[:, offset : offset + C]
                offset += C
                gammas.append(g.view(B, C, 1, 1, 1))
                betas.append(b.view(B, C, 1, 1, 1))
            return gammas, betas

    def apply_film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
        # 让 gamma 初始更接近 1：用 (1 + gamma) 稳定训练
        return x * (1.0 + gamma) + beta

    class FiLMConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, dropout_p: float):
            super().__init__()
            self.block = ConvBlock(in_ch, out_ch, dropout_p=dropout_p)

        def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
            y = self.block(x)
            y = apply_film(y, gamma, beta)
            return y

    class CNN3D_FiLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.nx, self.ny, self.nz = nx, ny, nz
            self.depth = depth
            self.base_ch = base_ch
            self.film_hidden = film_hidden

            # 固定输入：7 通道体素输入
            assert VOXEL_INPUT is not None, (
                "VOXEL_INPUT 尚未初始化：请先读取 cnn_input_channels_no_normals.csv 并设置 VOXEL_INPUT"
            )
            self.register_buffer("voxel_input", VOXEL_INPUT)  # (1,7,nx,ny,nz)

            # 监督/门控 mask：使用 C0（inside_mask）
            assert GEOM_MASK is not None, "GEOM_MASK 尚未初始化：请先从 C0 构造并设置 GEOM_MASK"
            self.register_buffer("geom_mask", GEOM_MASK)  # (1,1,nx,ny,nz)

            # Stem 下采样：UNet 从低分辨率开始算，降低显存
            self.stem_stride = 2
            self.stem = nn.Sequential(
                nn.Conv3d(7, base_ch, kernel_size=3, padding=1, stride=self.stem_stride),
                nn.BatchNorm3d(base_ch),
                nn.GELU(),
            )

            # Encoder
            self.enc0 = FiLMConvBlock(base_ch, base_ch, dropout_p=dropout_p)
            self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
            self.enc1 = FiLMConvBlock(base_ch, base_ch * 2, dropout_p=dropout_p)
            if depth >= 3:
                self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
                self.enc2 = FiLMConvBlock(base_ch * 2, base_ch * 4, dropout_p=dropout_p)

            # Bottleneck
            bottleneck_ch = base_ch * 2 if depth == 2 else base_ch * 4
            self.bottleneck = FiLMConvBlock(bottleneck_ch, bottleneck_ch, dropout_p=dropout_p)

            # Decoder
            if depth == 2:
                self.up1_conv = FiLMConvBlock(bottleneck_ch + base_ch, base_ch, dropout_p=dropout_p)
                dec_out_ch = base_ch
            else:
                self.up2_conv = FiLMConvBlock(bottleneck_ch + base_ch * 2, base_ch * 2, dropout_p=dropout_p)
                self.up1_conv = FiLMConvBlock(base_ch * 2 + base_ch, base_ch, dropout_p=dropout_p)
                dec_out_ch = base_ch

            self.out_proj = nn.Conv3d(dec_out_ch, base_ch, kernel_size=1)

            # 全分辨率融合：decoder_feat + 7通道体素输入
            self.geom_block = FiLMConvBlock(base_ch + 7, base_ch, dropout_p=dropout_p)
            self.final_conv = nn.Conv3d(base_ch, 1, kernel_size=1)

            # FiLM block 输出通道列表（顺序必须与 forward 使用一致）
            ch_list = [base_ch, base_ch * 2]
            if depth >= 3:
                ch_list.append(base_ch * 4)
            ch_list.append(bottleneck_ch)
            if depth == 2:
                ch_list.append(base_ch)      # up1
            else:
                ch_list.append(base_ch * 2)  # up2
                ch_list.append(base_ch)      # up1
            ch_list.append(base_ch)          # geom_block
            self.film = FiLMGen(input_dim=input_dim, ch_list=ch_list, hidden=film_hidden)

        def forward(self, bc: torch.Tensor) -> torch.Tensor:
            B = bc.size(0)

            vox = self.voxel_input.expand(B, -1, -1, -1, -1)          # (B,7,nx,ny,nz)
            mask_ch = self.geom_mask.expand(B, -1, -1, -1, -1)        # (B,1,nx,ny,nz)

            x = self.stem(vox)  # (B,base_ch,nx/2,ny/2,nz/2)

            gammas, betas = self.film(bc)
            gi = 0

            # Encoder
            x0 = self.enc0(x, gammas[gi], betas[gi]); gi += 1
            x1 = self.pool1(x0)
            x1 = self.enc1(x1, gammas[gi], betas[gi]); gi += 1

            if self.depth >= 3:
                x2 = self.pool2(x1)
                x2 = self.enc2(x2, gammas[gi], betas[gi]); gi += 1
                xb = self.bottleneck(x2, gammas[gi], betas[gi]); gi += 1

                x_up2 = F.interpolate(xb, size=x1.shape[2:], mode="trilinear", align_corners=False)
                x_cat2 = torch.cat([x_up2, x1], dim=1)
                x_dec2 = self.up2_conv(x_cat2, gammas[gi], betas[gi]); gi += 1

                x_up1 = F.interpolate(x_dec2, size=x0.shape[2:], mode="trilinear", align_corners=False)
                x_cat1 = torch.cat([x_up1, x0], dim=1)
                x_dec = self.up1_conv(x_cat1, gammas[gi], betas[gi]); gi += 1
            else:
                xb = self.bottleneck(x1, gammas[gi], betas[gi]); gi += 1

                x_up = F.interpolate(xb, size=x0.shape[2:], mode="trilinear", align_corners=False)
                x_cat = torch.cat([x_up, x0], dim=1)
                x_dec = self.up1_conv(x_cat, gammas[gi], betas[gi]); gi += 1

            x_dec = self.out_proj(x_dec)  # (B,base_ch,nx/2,ny/2,nz/2)

            # 回到全分辨率
            x_dec_full = F.interpolate(
                x_dec, size=(self.nx, self.ny, self.nz), mode="trilinear", align_corners=False
            )  # (B,base_ch,nx,ny,nz)

            # 拼接 decoder 特征 + 7 通道体素输入
            vox_full = self.voxel_input.expand(B, -1, -1, -1, -1)     # (B,7,nx,ny,nz)
            x_full = torch.cat([x_dec_full, vox_full], dim=1)         # (B,base_ch+7,nx,ny,nz)
            x_full = self.geom_block(x_full, gammas[gi], betas[gi]); gi += 1
            x_full = self.final_conv(x_full)  # (B,1,nx,ny,nz)

            out = x_full.squeeze(1)
            out = out * mask_ch.squeeze(1)  # 输出门控：仅 inside 区域
            return out

    return CNN3D_FiLM().to(device)


# =============================================================
# 数据读取：体素输入 + 温度标签（仅有效点） + mask
# =============================================================

datapath_bc = "data/boundary_condition.csv"
datapath_temp = "data/Temp_all.csv"
datapath_voxel = "data/cnn_input_channels_no_normals.csv"

# -------------------------------------------------------------
# 读取 7 通道体素输入（长方体笛卡尔网格）
# -------------------------------------------------------------
df_vox = pd.read_csv(datapath_voxel)

required_cols = [
    "x", "y", "z",
    "C0", "C1", "C2", "C3", "C4", "C5",
    "sdf",
]
missing = [c for c in required_cols if c not in df_vox.columns]
if missing:
    raise KeyError(f"cnn_input_channels_no_normals.csv 缺少列: {missing}")

xv = df_vox["x"].to_numpy(dtype=np.float32)
yv = df_vox["y"].to_numpy(dtype=np.float32)
zv = df_vox["z"].to_numpy(dtype=np.float32)

x_unique = np.sort(np.unique(xv))
y_unique = np.sort(np.unique(yv))
z_unique = np.sort(np.unique(zv))
(nx, ny, nz) = (len(x_unique), len(y_unique), len(z_unique))
print(f"体素输入网格尺寸: nx={nx}, ny={ny}, nz={nz}")

# 坐标到索引
x_index = {float(v): i for i, v in enumerate(x_unique)}
y_index = {float(v): i for i, v in enumerate(y_unique)}
z_index = {float(v): i for i, v in enumerate(z_unique)}

# 组装 VOXEL_INPUT: (7, nx, ny, nz)
voxel_grid = np.zeros((7, nx, ny, nz), dtype=np.float32)
col_order = [
    "C0", "C1", "C2", "C3", "C4", "C5",
    "sdf",
]
cols_np = [df_vox[c].to_numpy(dtype=np.float32) for c in col_order]

for i in range(df_vox.shape[0]):
    ix = x_index[float(xv[i])]
    iy = y_index[float(yv[i])]
    iz = z_index[float(zv[i])]
    voxel_grid[:, ix, iy, iz] = np.array([c[i] for c in cols_np], dtype=np.float32)

# 监督/门控 mask：C0
geom_mask_np = (voxel_grid[0] > 0.5).astype(np.float32)  # (nx,ny,nz)
GEOM_MASK = torch.tensor(geom_mask_np[None, None, ...], dtype=torch.float32, device=device)
VOXEL_INPUT = torch.tensor(voxel_grid[None, ...], dtype=torch.float32, device=device)  # (1,7,nx,ny,nz)
print(f"全局 C0(inside_mask) 占比: {geom_mask_np.mean() * 100:.3f}%")

# （可选）坐标离散轴：模型不使用，仅用于调试/可视化
X_UNIQUE_T = torch.tensor(x_unique, dtype=torch.float32, device=device)
Y_UNIQUE_T = torch.tensor(y_unique, dtype=torch.float32, device=device)
Z_UNIQUE_T = torch.tensor(z_unique, dtype=torch.float32, device=device)

# -------------------------------------------------------------
# 读取温度：Temp_all.csv 包含【全网格所有点】
# - 行顺序与 cnn_input_channels_no_normals.csv 的行顺序一致
# - 无效区域（通常 C0==0）温度可能已被置为 0，但监督/统计仍以 C0 为准
# 最终希望 Y_raw.shape == (num_samples, num_points)
# -------------------------------------------------------------
T_np = pd.read_csv(datapath_temp).to_numpy(dtype=np.float32)
num_points = int(df_vox.shape[0])

# 兼容两种排布：
#   (num_points, num_samples) 或 (num_samples, num_points)
if T_np.shape[0] == num_points:
    Y_raw = T_np.T
elif T_np.shape[1] == num_points:
    Y_raw = T_np
else:
    raise ValueError(
        f"Temp_all.csv 维度 {T_np.shape} 与点数 num_points={num_points} 不匹配。"
        "请确认 Temp_all.csv 的行/列是否为点维度。"
    )

num_samples = int(Y_raw.shape[0])
print(f"温度样本数: {num_samples}, 点数(全网格): {num_points}")

# -------------------------------------------------------------
# 将一维温度向量（全网格点）填入 (nx,ny,nz) 网格
# 关键：Temp_all 的点顺序必须与 cnn_input_channels_no_normals.csv 行顺序一致
# -------------------------------------------------------------
# 为所有点预先计算 (ix,iy,iz) 与线性索引（按 df_vox 行顺序）
ix_all = np.array([x_index[float(v)] for v in xv], dtype=np.int64)
iy_all = np.array([y_index[float(v)] for v in yv], dtype=np.int64)
iz_all = np.array([z_index[float(v)] for v in zv], dtype=np.int64)
lin_all = ix_all * (ny * nz) + iy_all * nz + iz_all

Y_grid = np.zeros((num_samples, nx * ny * nz), dtype=np.float32)
Y_grid[:, lin_all] = Y_raw  # scatter 全点
Y_grid = Y_grid.reshape((num_samples, nx, ny, nz))

# 样本级 mask：直接使用 C0（所有样本相同）
mask_valid = np.broadcast_to(geom_mask_np[None, ...], (num_samples, nx, ny, nz)).astype(np.float32)
valid_ratio = float(mask_valid.mean())
print(f"真实点占比(由C0给定): {valid_ratio * 100:.3f}%")

# 预计算：有效点在“扁平网格”上的索引（用于快速统计/赋值）
lin_valid = np.where(geom_mask_np.reshape(-1) > 0.5)[0]

# =============================================================
# X: 边界条件；split: train/val/test
# =============================================================

df_bc = pd.read_csv(datapath_bc)
X_data = df_bc.iloc[:, :6].to_numpy(dtype=np.float32)
split_raw = df_bc.iloc[:, 6].to_numpy()

if split_raw.dtype.kind in "OUS":
    split = np.array([str(s).strip().lower() for s in split_raw])
    train_idx = np.where(split == "train")[0]
    val_idx = np.where((split == "val") | (split == "valid") | (split == "validation"))[0]
    test_idx_final = np.where(split == "test")[0]
else:
    train_idx = np.where(split_raw == 0)[0]
    val_idx = np.where(split_raw == 1)[0]
    test_idx_final = np.where(split_raw == 2)[0]

print(f"训练集数量: {len(train_idx)}, 验证集数量: {len(val_idx)}, 测试集数量: {len(test_idx_final)}")


# =============================================================
# 标准化：X 用 StandardScaler；Y 做【样本级】Z-score（仅在 C0==1 的点上）
# =============================================================

scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X_data)

Y_scaled = np.zeros_like(Y_grid, dtype=np.float32)
Y_means = np.zeros((num_samples,), dtype=np.float32)
Y_stds = np.zeros((num_samples,), dtype=np.float32)

# 只在有效点(C0==1)上统计均值方差，并只对有效点做标准化
for i in range(num_samples):
    valid_i = Y_grid[i].reshape(-1)[lin_valid]
    m = float(valid_i.mean())
    s = float(valid_i.std()) + 1e-8
    Y_means[i] = m
    Y_stds[i] = s

    Y_scaled_flat = Y_scaled[i].reshape(-1)
    Y_scaled_flat[lin_valid] = (valid_i - m) / s


# =============================================================
# 数据集切分
# =============================================================

x_train = X_scaled[train_idx]
y_train = Y_scaled[train_idx]
mask_train = mask_valid[train_idx]

x_val = X_scaled[val_idx]
y_val = Y_scaled[val_idx]
mask_val = mask_valid[val_idx]

x_test = X_scaled[test_idx_final]
y_test = Y_scaled[test_idx_final]
mask_test = mask_valid[test_idx_final]

input_dim = x_train.shape[1]
print(f"训练样本数: {x_train.shape[0]}, 验证样本数: {x_val.shape[0]}, 测试样本数: {x_test.shape[0]}")
print(f"输入维度: {input_dim}")

# DataLoader
BATCH_SIZE = 16


# =============================================================
# Masked Loss：点值损失 + 梯度损失（只在真实点上计算）
# =============================================================

def masked_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "huber",
    grad_weight: float = 0.1,
) -> torch.Tensor:
    """pred/target/mask: (B, nx, ny, nz), mask=1 表示真实点"""

    if loss_type == "huber":
        per_elem = F.smooth_l1_loss(pred, target, reduction="none")
        masked = per_elem * mask
    else:
        diff2 = (pred - target) ** 2
        masked = diff2 * mask

    base_loss = masked.sum() / (mask.sum() + 1e-8)

    def _spatial_grads(t: torch.Tensor):
        gx = t[:, 1:, :, :] - t[:, :-1, :, :]
        gy = t[:, :, 1:, :] - t[:, :, :-1, :]
        gz = t[:, :, :, 1:] - t[:, :, :, :-1]
        return gx, gy, gz

    pred_dx, pred_dy, pred_dz = _spatial_grads(pred)
    true_dx, true_dy, true_dz = _spatial_grads(target)

    mask_dx = mask[:, 1:, :, :] * mask[:, :-1, :, :]
    mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    mask_dz = mask[:, :, :, 1:] * mask[:, :, :, :-1]

    grad_loss = pred.new_tensor(0.0)
    for pd, td, md in [(pred_dx, true_dx, mask_dx), (pred_dy, true_dy, mask_dy), (pred_dz, true_dz, mask_dz)]:
        diff2 = (pd - td) ** 2
        grad_loss += (diff2 * md).sum() / (md.sum() + 1e-8)
    grad_loss = grad_loss / 3.0
    return base_loss + grad_weight * grad_loss


# =============================================================
# Optuna 目标函数：搜索 FiLM/结构超参 + lr + weight_decay（验证集 EarlyStopping）
# =============================================================

def objective(trial: optuna.trial.Trial) -> float:
    model = create_cnn3d_from_trial(trial, input_dim, nx, ny, nz)

    # 多 GPU
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 张 GPU 进行数据并行训练 (Optuna trial)...")
        model = nn.DataParallel(model)

    lr = trial.suggest_float("lr", 1e-4, 1.5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_type = trial.suggest_categorical("loss_type", ["huber", "mse"])
    grad_weight = trial.suggest_float("grad_weight", 0.015, 0.08, log=True)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(mask_train, dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
            torch.tensor(mask_val, dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    best_val = float("inf")
    patience = 30
    no_improve = 0

    for epoch in range(80):
        model.train()
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = masked_loss(pred, yb, mb, loss_type=loss_type, grad_weight=grad_weight)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                pred = model(xb)
                vloss = masked_loss(pred, yb, mb, loss_type=loss_type, grad_weight=grad_weight)
                val_loss_total += vloss.item() * xb.size(0)
        val_loss = val_loss_total / len(val_loader.dataset)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[Optuna] Early stopping at epoch {epoch}, best_val={best_val:.6f}")
                break

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return best_val


# =============================================================
# 运行 Optuna 搜索 + 用最佳超参数重新训练（train+val）
# =============================================================

print("开始 Optuna 超参搜索（FiLM/结构 + lr + weight_decay）...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

best_params = study.best_params
print("最佳参数:", best_params)


# =============================================================
# Two-stage training
# Stage A: train on train, early stop on val
# Stage B: train+val re-train to best_epoch
# =============================================================

best_trial = optuna.trial.FixedTrial(best_params)

# --------------------- Stage A ---------------------
model_a = create_cnn3d_from_trial(best_trial, input_dim, nx, ny, nz)
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 张 GPU 进行数据并行训练 (Stage A)...")
    model_a = nn.DataParallel(model_a)

best_lr = best_params.get("lr", 1e-3)
best_weight_decay = best_params.get("weight_decay", 0.0)
best_loss_type = best_params.get("loss_type", "mse")
best_grad_weight = float(best_params.get("grad_weight", 0.1))

optimizer_a = optim.Adam(model_a.parameters(), lr=best_lr, weight_decay=best_weight_decay)

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(mask_train, dtype=torch.float32),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=(device.type == "cuda"),
)

val_loader = DataLoader(
    TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(mask_val, dtype=torch.float32),
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=(device.type == "cuda"),
)

# warmup + cosine（Stage A）
total_epochs_a = 500
warmup_epochs_a = 20
initial_lr_a = best_lr * 0.1
for pg in optimizer_a.param_groups:
    pg["lr"] = initial_lr_a

scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_a,
    T_max=max(1, total_epochs_a - warmup_epochs_a),
)

patience = 40
min_delta = 1e-6
best_val = float("inf")
best_epoch = -1
no_improve = 0

model_a_core = model_a.module if isinstance(model_a, nn.DataParallel) else model_a
best_state = None

print("===== Stage A: train on train, early stop on val =====")
for epoch in range(total_epochs_a):
    model_a.train()
    running_loss = 0.0
    for xb, yb, mb in train_loader:
        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
        optimizer_a.zero_grad(set_to_none=True)
        pred = model_a(xb)
        loss = masked_loss(pred, yb, mb, loss_type=best_loss_type, grad_weight=best_grad_weight)
        loss.backward()
        optimizer_a.step()
        running_loss += loss.item() * xb.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    # lr schedule
    if epoch < warmup_epochs_a:
        lr = initial_lr_a + (best_lr - initial_lr_a) * float(epoch + 1) / float(warmup_epochs_a)
        for pg in optimizer_a.param_groups:
            pg["lr"] = lr
    else:
        scheduler_a.step()
        lr = optimizer_a.param_groups[0]["lr"]

    # val
    model_a.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for xb, yb, mb in val_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            pred = model_a(xb)
            vloss = masked_loss(pred, yb, mb, loss_type=best_loss_type, grad_weight=best_grad_weight)
            val_loss_total += vloss.item() * xb.size(0)
    val_loss = val_loss_total / len(val_loader.dataset)

    if epoch % 10 == 0:
        print(f"[Stage A] Epoch {epoch}, lr={lr:.3e}, train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    # early stopping
    if val_loss < best_val - min_delta:
        best_val = val_loss
        best_epoch = epoch
        no_improve = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model_a_core.state_dict().items()}
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"[Stage A] Early stopping at epoch {epoch}. best_epoch={best_epoch}, best_val={best_val:.6f}")
            break

if best_state is None:
    raise RuntimeError("Stage A 没有记录到 best_state（不应发生）。")

print(f"[Stage A] Selected best_epoch={best_epoch} (0-indexed), best_val={best_val:.6f}")

# --------------------- Stage B ---------------------
model_b = create_cnn3d_from_trial(best_trial, input_dim, nx, ny, nz)
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 张 GPU 进行数据并行训练 (Stage B: train+val)...")
    model_b = nn.DataParallel(model_b)

optimizer_b = optim.Adam(model_b.parameters(), lr=best_lr, weight_decay=best_weight_decay)

x_trainval = np.concatenate([x_train, x_val], axis=0)
y_trainval = np.concatenate([y_train, y_val], axis=0)
mask_trainval = np.concatenate([mask_train, mask_val], axis=0)

trainval_loader = DataLoader(
    TensorDataset(
        torch.tensor(x_trainval, dtype=torch.float32),
        torch.tensor(y_trainval, dtype=torch.float32),
        torch.tensor(mask_trainval, dtype=torch.float32),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=(device.type == "cuda"),
)

num_epochs_b = max(1, int(best_epoch + 1))

warmup_epochs_b = min(20, max(1, num_epochs_b // 5))
initial_lr_b = best_lr * 0.1
for pg in optimizer_b.param_groups:
    pg["lr"] = initial_lr_b

scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_b,
    T_max=max(1, num_epochs_b - warmup_epochs_b),
)

print("===== Stage B: re-train on train+val to selected epoch =====")
model_b_core = model_b.module if isinstance(model_b, nn.DataParallel) else model_b

for epoch in range(num_epochs_b):
    model_b.train()
    running_loss = 0.0

    for xb, yb, mb in trainval_loader:
        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
        optimizer_b.zero_grad(set_to_none=True)
        pred = model_b(xb)
        loss = masked_loss(pred, yb, mb, loss_type=best_loss_type, grad_weight=best_grad_weight)
        loss.backward()
        optimizer_b.step()
        running_loss += loss.item() * xb.size(0)

    train_loss = running_loss / len(trainval_loader.dataset)

    if epoch < warmup_epochs_b:
        lr = initial_lr_b + (best_lr - initial_lr_b) * float(epoch + 1) / float(warmup_epochs_b)
        for pg in optimizer_b.param_groups:
            pg["lr"] = lr
    else:
        scheduler_b.step()
        lr = optimizer_b.param_groups[0]["lr"]

    if epoch % 10 == 0 or epoch == num_epochs_b - 1:
        print(f"[Stage B] Epoch {epoch}, lr={lr:.3e}, train_loss={train_loss:.6f}")

print("模型训练完成（Stage B: train+val）。")

model_core = model_b_core


# =============================================================
# 保存训练好的 3D CNN 模型
# =============================================================

torch.save(
    {
        "state_dict": model_core.state_dict(),
        "input_dim": input_dim,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "depth": 2,
        "base_ch": best_params.get("base_ch", 24),
        "dropout_p": best_params.get("dropout_p", 0.1),
        "model_type": "FiLM_A_voxel7",
        "stem_stride": 2,
        "film_mult": best_params.get("film_mult", 4),
        "lr": best_lr,
        "weight_decay": best_weight_decay,
        "loss_type": best_loss_type,
        "grad_weight": best_grad_weight,
        "x_mean": scaler_x.mean_,
        "x_scale": scaler_x.scale_,
        "Y_means": Y_means,
        "Y_stds": Y_stds,
        "geom_mask_np": geom_mask_np,
        "lin_valid": lin_valid,
        "voxel_cols": col_order,
    },
    "best_3dcnn_film_voxel7.pth",
)
print("模型已保存为 best_3dcnn_film_voxel7.pth")
# CUDA_VISIBLE_DEVICES=1,2,3,4,5 python "CNN_FiLM_sdf&bm_7.py"