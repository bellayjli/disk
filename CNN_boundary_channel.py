# =============================================================
# CNN.py  ——  边界条件 → 3D 温度场
# 3D UNet + Optuna + Mask + 梯度损失 + 小批量训练（避免 OOM）
# 输入（15通道体素）：
#   - 9通道体素来自：cnn_input_channels_with_sdf.csv（去掉 x_norm/y_norm/z_norm）
#   - 6个边界条件通道来自 boundary_condition.csv（每个样本复制到整个体素网格）
# 监督 mask 使用 C0（inside_mask）
# 温度标签来自：Temp_all.csv（全网格点）
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
# 固定 3D 体素输入（1,9,nx,ny,nz）：来自 cnn_input_channels_with_sdf.csv
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

if device.type == "cuda":
    torch.cuda.init()
    print(f"CUDA devices: {torch.cuda.device_count()} visible.")


# =============================================================
# 3D UNet（无 FiLM）：输入=固定9通道体素 + 复制的6通道边界条件 => 15通道
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
    无 FiLM：
    - 固定 9 通道体素场（来自 cnn_input_channels_with_sdf.csv）：
        [C0 inside_mask,
         C1 wall_mask,
         C2 inlet_mask,
         C3 outlet_mask,
         C4 heat_source_mask,
         nx, ny, nz,
         sdf]
    - 边界条件 bc (R^6) 直接复制成 6 个通道，拼到体素上：
        bc_grid: (B,6,nx,ny,nz)
    - 最终输入 CNN 的通道数：9 + 6 = 15
    """

    # 结构超参
    depth = 2
    base_ch = trial.suggest_categorical("base_ch", [12, 16, 24])
    dropout_p = trial.suggest_float("dropout_p", 0.0, 0.3)

    class CNN3D_NoFiLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.nx, self.ny, self.nz = nx, ny, nz
            self.depth = depth
            self.base_ch = base_ch

            # 固定输入：9 通道体素输入
            assert VOXEL_INPUT is not None, (
                "VOXEL_INPUT 尚未初始化：请先读取 cnn_input_channels_with_sdf.csv 并设置 VOXEL_INPUT"
            )
            self.register_buffer("voxel_input", VOXEL_INPUT)  # (1,9,nx,ny,nz)

            # 监督/门控 mask：使用 C0（inside_mask）
            assert GEOM_MASK is not None, "GEOM_MASK 尚未初始化：请先从 C0 构造并设置 GEOM_MASK"
            self.register_buffer("geom_mask", GEOM_MASK)  # (1,1,nx,ny,nz)

            # Stem 下采样：UNet 从低分辨率开始算，降低显存
            self.stem_stride = 2

            in_ch_stem = 13  # 7(voxel) + 6(bc)
            self.stem = nn.Sequential(
                nn.Conv3d(in_ch_stem, base_ch, kernel_size=3, padding=1, stride=self.stem_stride),
                nn.BatchNorm3d(base_ch),
                nn.GELU(),
            )

            # Encoder
            self.enc0 = ConvBlock(base_ch, base_ch, dropout_p=dropout_p)
            self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
            self.enc1 = ConvBlock(base_ch, base_ch * 2, dropout_p=dropout_p)
            if depth >= 3:
                self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
                self.enc2 = ConvBlock(base_ch * 2, base_ch * 4, dropout_p=dropout_p)

            # Bottleneck
            bottleneck_ch = base_ch * 2 if depth == 2 else base_ch * 4
            self.bottleneck = ConvBlock(bottleneck_ch, bottleneck_ch, dropout_p=dropout_p)

            # Decoder
            if depth == 2:
                self.up1_conv = ConvBlock(bottleneck_ch + base_ch, base_ch, dropout_p=dropout_p)
                dec_out_ch = base_ch
            else:
                self.up2_conv = ConvBlock(bottleneck_ch + base_ch * 2, base_ch * 2, dropout_p=dropout_p)
                self.up1_conv = ConvBlock(base_ch * 2 + base_ch, base_ch, dropout_p=dropout_p)
                dec_out_ch = base_ch

            self.out_proj = nn.Conv3d(dec_out_ch, base_ch, kernel_size=1)

            # 全分辨率融合：decoder_feat + 7通道体素 + 6通道bc
            in_ch_full = base_ch + 13
            self.geom_block = ConvBlock(in_ch_full, base_ch, dropout_p=dropout_p)
            self.final_conv = nn.Conv3d(base_ch, 1, kernel_size=1)

        def forward(self, bc: torch.Tensor) -> torch.Tensor:
            """
            bc: (B,6) —— 标准化后的边界条件
            return: (B,nx,ny,nz) —— 温度场（仅inside区域有效）
            """
            B = bc.size(0)

            vox = self.voxel_input.expand(B, -1, -1, -1, -1)          # (B,7,nx,ny,nz)
            mask_ch = self.geom_mask.expand(B, -1, -1, -1, -1)        # (B,1,nx,ny,nz)

            # bc -> 体素通道 (B,6,nx,ny,nz)
            bc_grid = bc.view(B, 6, 1, 1, 1).expand(B, 6, self.nx, self.ny, self.nz)

            # 拼成 15 通道输入
            x_in = torch.cat([vox, bc_grid], dim=1)  # (B,13,nx,ny,nz)

            x = self.stem(x_in)  # (B,base_ch,nx/2,ny/2,nz/2)

            # Encoder
            x0 = self.enc0(x)
            x1 = self.pool1(x0)
            x1 = self.enc1(x1)

            if self.depth >= 3:
                x2 = self.pool2(x1)
                x2 = self.enc2(x2)
                xb = self.bottleneck(x2)

                x_up2 = F.interpolate(xb, size=x1.shape[2:], mode="trilinear", align_corners=False)
                x_cat2 = torch.cat([x_up2, x1], dim=1)
                x_dec2 = self.up2_conv(x_cat2)

                x_up1 = F.interpolate(x_dec2, size=x0.shape[2:], mode="trilinear", align_corners=False)
                x_cat1 = torch.cat([x_up1, x0], dim=1)
                x_dec = self.up1_conv(x_cat1)
            else:
                xb = self.bottleneck(x1)

                x_up = F.interpolate(xb, size=x0.shape[2:], mode="trilinear", align_corners=False)
                x_cat = torch.cat([x_up, x0], dim=1)
                x_dec = self.up1_conv(x_cat)

            x_dec = self.out_proj(x_dec)  # (B,base_ch,nx/2,ny/2,nz/2)

            # 回到全分辨率
            x_dec_full = F.interpolate(
                x_dec, size=(self.nx, self.ny, self.nz), mode="trilinear", align_corners=False
            )  # (B,base_ch,nx,ny,nz)

            # 全分辨率拼接：decoder特征 + voxel(9) + bc(6)
            vox_full = self.voxel_input.expand(B, -1, -1, -1, -1)  # (B,7,nx,ny,nz)
            bc_full = bc_grid                                    # (B,6,nx,ny,nz)
            x_full = torch.cat([x_dec_full, vox_full, bc_full], dim=1)  # (B,base_ch+13,nx,ny,nz)

            x_full = self.geom_block(x_full)
            x_full = self.final_conv(x_full)  # (B,1,nx,ny,nz)

            out = x_full.squeeze(1)
            out = out * mask_ch.squeeze(1)  # 输出门控：仅 inside 区域
            return out

    return CNN3D_NoFiLM().to(device)


# =============================================================
# 数据读取：体素输入 + 温度标签（仅有效点） + mask
# =============================================================

datapath_bc = "data/boundary_condition.csv"
datapath_temp = "data/Temp_all.csv"
datapath_voxel = "data/cnn_input_channels_no_normals.csv"

df_vox = pd.read_csv(datapath_voxel)

# 必须有坐标 + C0(inside_mask)
for c in ["x", "y", "z", "C0"]:
    if c not in df_vox.columns:
        raise KeyError(f"{datapath_voxel} 缺少列: {c}")

# ✅ 体素输入：取“后7列”
voxel_cols = list(df_vox.columns[-7:])
print(f"[voxel] using last 7 columns: {voxel_cols}")

# 如果你希望强制要求 C0 必须在后7列里（更“对齐”），就打开这句：
# if "C0" not in voxel_cols:
#     raise ValueError(f"C0 不在后7列里（{voxel_cols}）。请检查 csv 列顺序是否符合预期。")

xv = df_vox["x"].to_numpy(dtype=np.float32)
yv = df_vox["y"].to_numpy(dtype=np.float32)
zv = df_vox["z"].to_numpy(dtype=np.float32)

x_unique = np.sort(np.unique(xv))
y_unique = np.sort(np.unique(yv))
z_unique = np.sort(np.unique(zv))
(nx, ny, nz) = (len(x_unique), len(y_unique), len(z_unique))
print(f"体素输入网格尺寸: nx={nx}, ny={ny}, nz={nz}")

x_index = {float(v): i for i, v in enumerate(x_unique)}
y_index = {float(v): i for i, v in enumerate(y_unique)}
z_index = {float(v): i for i, v in enumerate(z_unique)}

# ✅ 7通道体素网格
voxel_grid = np.zeros((7, nx, ny, nz), dtype=np.float32)
cols_np = [df_vox[c].to_numpy(dtype=np.float32) for c in voxel_cols]

for i in range(df_vox.shape[0]):
    ix = x_index[float(xv[i])]
    iy = y_index[float(yv[i])]
    iz = z_index[float(zv[i])]
    voxel_grid[:, ix, iy, iz] = np.array([c[i] for c in cols_np], dtype=np.float32)

# ✅ 监督/门控 mask：仍然来自 df_vox["C0"]（不依赖后7列里是否包含C0）
geom_mask_np = np.zeros((nx, ny, nz), dtype=np.float32)
c0_np = df_vox["C0"].to_numpy(dtype=np.float32)
for i in range(df_vox.shape[0]):
    ix = x_index[float(xv[i])]
    iy = y_index[float(yv[i])]
    iz = z_index[float(zv[i])]
    geom_mask_np[ix, iy, iz] = 1.0 if c0_np[i] > 0.5 else 0.0

GEOM_MASK = torch.tensor(geom_mask_np[None, None, ...], dtype=torch.float32, device=device)
VOXEL_INPUT = torch.tensor(voxel_grid[None, ...], dtype=torch.float32, device=device)  # (1,7,nx,ny,nz)
print(f"全局 C0(inside_mask) 占比: {geom_mask_np.mean() * 100:.3f}%")

cols_np = [df_vox[c].to_numpy(dtype=np.float32) for c in voxel_cols]

for i in range(df_vox.shape[0]):
    ix = x_index[float(xv[i])]
    iy = y_index[float(yv[i])]
    iz = z_index[float(zv[i])]
    voxel_grid[:, ix, iy, iz] = np.array([c[i] for c in cols_np], dtype=np.float32)

geom_mask_np = (voxel_grid[0] > 0.5).astype(np.float32)  # (nx,ny,nz)
GEOM_MASK = torch.tensor(geom_mask_np[None, None, ...], dtype=torch.float32, device=device)
VOXEL_INPUT = torch.tensor(voxel_grid[None, ...], dtype=torch.float32, device=device)  # (1,9,nx,ny,nz)
print(f"全局 C0(inside_mask) 占比: {geom_mask_np.mean() * 100:.3f}%")

X_UNIQUE_T = torch.tensor(x_unique, dtype=torch.float32, device=device)
Y_UNIQUE_T = torch.tensor(y_unique, dtype=torch.float32, device=device)
Z_UNIQUE_T = torch.tensor(z_unique, dtype=torch.float32, device=device)

# -------------------------------------------------------------
# 读取温度：Temp_all.csv（全网格所有点）
# -------------------------------------------------------------
T_np = pd.read_csv(datapath_temp).to_numpy(dtype=np.float32)
num_points = int(df_vox.shape[0])

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

ix_all = np.array([x_index[float(v)] for v in xv], dtype=np.int64)
iy_all = np.array([y_index[float(v)] for v in yv], dtype=np.int64)
iz_all = np.array([z_index[float(v)] for v in zv], dtype=np.int64)
lin_all = ix_all * (ny * nz) + iy_all * nz + iz_all

Y_grid = np.zeros((num_samples, nx * ny * nz), dtype=np.float32)
Y_grid[:, lin_all] = Y_raw
Y_grid = Y_grid.reshape((num_samples, nx, ny, nz))

mask_valid = np.broadcast_to(geom_mask_np[None, ...], (num_samples, nx, ny, nz)).astype(np.float32)
valid_ratio = float(mask_valid.mean())
print(f"真实点占比(由C0给定): {valid_ratio * 100:.3f}%")

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
# Optuna 目标函数：搜索结构超参 + lr + weight_decay（验证集 EarlyStopping）
# =============================================================

def objective(trial: optuna.trial.Trial) -> float:
    model = create_cnn3d_from_trial(trial, input_dim, nx, ny, nz)

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
            pred = model(xb)  # xb 是 (B,6)，模型内部复制成 6 个体素通道
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

print("开始 Optuna 超参搜索（结构 + lr + weight_decay）...")
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

    if epoch < warmup_epochs_a:
        lr = initial_lr_a + (best_lr - initial_lr_a) * float(epoch + 1) / float(warmup_epochs_a)
        for pg in optimizer_a.param_groups:
            pg["lr"] = lr
    else:
        scheduler_a.step()
        lr = optimizer_a.param_groups[0]["lr"]

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
        "model_type": "NoFiLM_BCAsVoxels_voxel7plus6",
        "stem_stride": 2,
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
        "voxel_cols": voxel_cols,  # ✅记下来这次用的是哪7列
        "note": "input channels = last7(voxel) + 6 bc replicated => 13",
    },
    "best_3dcnn_nofilm_bc15_voxel9.pth",
)
print("模型已保存为 best_3dcnn_nofilm_bc15_voxel9.pth")
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python "CNN_boundary_channel.py"
# nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | xargs ps -fp
# nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits