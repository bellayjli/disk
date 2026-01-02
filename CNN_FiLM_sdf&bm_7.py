# =============================================================
# CNN_FiLM_sdf&bm_7.py
# 边界条件(bc, 6维) → 3D 温度场
#
# 3D UNet (depth=2) + Optuna + Mask + 梯度损失
#
# 输入（7 通道体素）来自：cnn_input_channels_no_normals.csv 的后 7 列
#   列：x y z C0 C1 C2 C3 C4 C5 sdf
#   体素输入通道：[C0,C1,C2,C3,C4,C5,sdf]
# 监督 mask 使用 C0（inside_mask）
# 温度标签来自：Temp_all.csv（包含全网格点，顺序与体素 CSV 行顺序一致）
#
# FiLM 注入策略（关键点）：
# 1) Residual block 内：BN -> FiLM -> GELU（让条件参与特征生成，而非输出后上色）
# 2) Stem 也做 FiLM：Conv -> BN -> FiLM -> GELU（让条件从第一层就参与特征提取）
#
# FiLM 超参搜索（关键点）：
# - film_mult ∈ [1,2,4,8,16]，hidden = base_ch * film_mult
# - film_scale s ∈ [0.1, 1.0] (log)，gamma = s*gamma_raw, beta = s*beta_raw
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

# --------------------- 设备 ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cuda":
    torch.cuda.init()
    print(f"CUDA devices: {torch.cuda.device_count()} visible.")


# =============================================================
# FiLM Residual Block（FiLM 放在 BN 后、激活前）
# =============================================================

class FiLMResidualBlock(nn.Module):
    """(Conv -> BN -> FiLM -> GELU) x2 -> Dropout + residual"""

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout3d(dropout_p)

        self.residual_proj = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = out * (1.0 + gamma) + beta
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out * (1.0 + gamma) + beta
        out = self.act(out)

        out = self.drop(out)
        return out + residual


# =============================================================
# 模型构建（Optuna trial）
# =============================================================

def create_cnn3d_from_trial(trial: optuna.trial.Trial, input_dim: int, nx: int, ny: int, nz: int) -> nn.Module:
    """
    depth=2 的 FiLM-UNet(voxel7)：
    - 体素输入固定 7 通道
    - bc(6维) 只用于生成各层 gamma/beta
    - Stem 也做 FiLM：Conv -> BN -> FiLM -> GELU
    """

    depth = 2
    base_ch = trial.suggest_categorical("base_ch", [12, 16, 24])
    dropout_p = trial.suggest_float("dropout_p", 0.0, 0.3)

    # ===== FiLM 搜索范围（扩展）=====
    film_mult = trial.suggest_categorical("film_mult", [1, 2, 4, 8, 16])
    film_hidden = int(base_ch * film_mult)

    # 输出缩放，避免早期 gamma/beta 幅度过大导致训练不稳
    film_scale = trial.suggest_float("film_scale", 0.1, 1.0, log=True)

    class FiLMGen(nn.Module):
        """bc -> {gamma,beta} for each injection point; each is (B,C,1,1,1)."""

        def __init__(self, input_dim: int, ch_list: list[int], hidden: int, scale: float):
            super().__init__()
            self.ch_list = ch_list
            self.scale = float(scale)
            out_dim = 2 * sum(ch_list)
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
            gammas, betas = [], []
            offset = 0
            s = self.scale
            for C in self.ch_list:
                g_raw = v[:, offset: offset + C]; offset += C
                b_raw = v[:, offset: offset + C]; offset += C
                # ===== 关键：输出缩放 =====
                g = s * g_raw
                b = s * b_raw
                gammas.append(g.view(B, C, 1, 1, 1))
                betas.append(b.view(B, C, 1, 1, 1))
            return gammas, betas

    class CNN3D_FiLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.nx, self.ny, self.nz = nx, ny, nz
            self.depth = depth
            self.base_ch = base_ch
            self.film_hidden = film_hidden
            self.film_mult = film_mult
            self.film_scale = film_scale

            # 固定输入：7 通道体素输入
            assert VOXEL_INPUT is not None, "VOXEL_INPUT 未初始化"
            self.register_buffer("voxel_input", VOXEL_INPUT)  # (1,7,nx,ny,nz)

            # 监督/门控 mask：C0
            assert GEOM_MASK is not None, "GEOM_MASK 未初始化"
            self.register_buffer("geom_mask", GEOM_MASK)  # (1,1,nx,ny,nz)

            # Stem：Conv -> BN -> (FiLM) -> GELU
            self.stem_stride = 2
            self.stem_conv = nn.Conv3d(7, base_ch, kernel_size=3, padding=1, stride=self.stem_stride)
            self.stem_bn = nn.BatchNorm3d(base_ch)
            self.stem_act = nn.GELU()

            # Encoder
            self.enc0 = FiLMResidualBlock(base_ch, base_ch, dropout_p=dropout_p)
            self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
            self.enc1 = FiLMResidualBlock(base_ch, base_ch * 2, dropout_p=dropout_p)

            # Bottleneck
            bottleneck_ch = base_ch * 2
            self.bottleneck = FiLMResidualBlock(bottleneck_ch, bottleneck_ch, dropout_p=dropout_p)

            # Decoder（depth=2 只有 up1）
            self.up1_conv = FiLMResidualBlock(bottleneck_ch + base_ch, base_ch, dropout_p=dropout_p)
            self.out_proj = nn.Conv3d(base_ch, base_ch, kernel_size=1)

            # 全分辨率融合：decoder_feat + 7通道体素输入
            self.geom_block = FiLMResidualBlock(base_ch + 7, base_ch, dropout_p=dropout_p)
            self.final_conv = nn.Conv3d(base_ch, 1, kernel_size=1)

            # ===== FiLM 注入点通道列表（顺序必须与 forward 的 gi 使用一致）=====
            # 0) stem (base_ch)
            # 1) enc0 (base_ch)
            # 2) enc1 (base_ch*2)
            # 3) bottleneck (base_ch*2)
            # 4) up1 (base_ch)
            # 5) geom_block (base_ch)
            ch_list = [
                base_ch,        # stem
                base_ch,        # enc0
                base_ch * 2,    # enc1
                base_ch * 2,    # bottleneck
                base_ch,        # up1
                base_ch,        # geom_block
            ]
            self.film = FiLMGen(input_dim=input_dim, ch_list=ch_list, hidden=film_hidden, scale=film_scale)

        def forward(self, bc: torch.Tensor) -> torch.Tensor:
            B = bc.size(0)
            vox = self.voxel_input.expand(B, -1, -1, -1, -1)      # (B,7,nx,ny,nz)
            mask = self.geom_mask.expand(B, -1, -1, -1, -1)       # (B,1,nx,ny,nz)

            gammas, betas = self.film(bc)
            gi = 0

            # ---- Stem: Conv -> BN -> FiLM -> GELU ----
            x = self.stem_conv(vox)
            x = self.stem_bn(x)
            x = x * (1.0 + gammas[gi]) + betas[gi]
            x = self.stem_act(x)
            gi += 1

            # ---- Encoder ----
            x0 = self.enc0(x, gammas[gi], betas[gi]); gi += 1
            x1 = self.pool1(x0)
            x1 = self.enc1(x1, gammas[gi], betas[gi]); gi += 1

            # ---- Bottleneck ----
            xb = self.bottleneck(x1, gammas[gi], betas[gi]); gi += 1

            # ---- Decoder (up1) ----
            x_up = F.interpolate(xb, size=x0.shape[2:], mode="trilinear", align_corners=False)
            x_cat = torch.cat([x_up, x0], dim=1)
            x_dec = self.up1_conv(x_cat, gammas[gi], betas[gi]); gi += 1

            x_dec = self.out_proj(x_dec)  # (B,base_ch,nx/2,ny/2,nz/2)

            # ---- Back to full resolution ----
            x_dec_full = F.interpolate(x_dec, size=(self.nx, self.ny, self.nz), mode="trilinear", align_corners=False)

            # ---- Fuse with voxel input at full res ----
            vox_full = self.voxel_input.expand(B, -1, -1, -1, -1)   # (B,7,nx,ny,nz)
            x_full = torch.cat([x_dec_full, vox_full], dim=1)       # (B,base_ch+7,nx,ny,nz)
            x_full = self.geom_block(x_full, gammas[gi], betas[gi]); gi += 1

            x_full = self.final_conv(x_full)  # (B,1,nx,ny,nz)

            out = x_full.squeeze(1)
            out = out * mask.squeeze(1)
            return out

    return CNN3D_FiLM().to(device)


# =============================================================
# 数据读取：体素输入 + 温度标签 + mask
# =============================================================

datapath_bc = "data/boundary_condition.csv"
datapath_temp = "data/Temp_all.csv"
datapath_voxel = "data/cnn_input_channels_no_normals.csv"

df_vox = pd.read_csv(datapath_voxel)

required_cols = ["x", "y", "z", "C0", "C1", "C2", "C3", "C4", "C5", "sdf"]
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

x_index = {float(v): i for i, v in enumerate(x_unique)}
y_index = {float(v): i for i, v in enumerate(y_unique)}
z_index = {float(v): i for i, v in enumerate(z_unique)}

# VOXEL_INPUT: (1,7,nx,ny,nz)
voxel_grid = np.zeros((7, nx, ny, nz), dtype=np.float32)
col_order = ["C0", "C1", "C2", "C3", "C4", "C5", "sdf"]
cols_np = [df_vox[c].to_numpy(dtype=np.float32) for c in col_order]

for i in range(df_vox.shape[0]):
    ix = x_index[float(xv[i])]
    iy = y_index[float(yv[i])]
    iz = z_index[float(zv[i])]
    voxel_grid[:, ix, iy, iz] = np.array([c[i] for c in cols_np], dtype=np.float32)

geom_mask_np = (voxel_grid[0] > 0.5).astype(np.float32)
GEOM_MASK = torch.tensor(geom_mask_np[None, None, ...], dtype=torch.float32, device=device)
VOXEL_INPUT = torch.tensor(voxel_grid[None, ...], dtype=torch.float32, device=device)

print(f"全局 C0(inside_mask) 占比: {geom_mask_np.mean() * 100:.3f}%")

# -------------------------------------------------------------
# 温度：Temp_all.csv (num_points x num_samples) 或 (num_samples x num_points)
# 点顺序必须与 df_vox 行顺序一致
# -------------------------------------------------------------
T_np = pd.read_csv(datapath_temp).to_numpy(dtype=np.float32)
num_points = int(df_vox.shape[0])

if T_np.shape[0] == num_points:
    Y_raw = T_np.T
elif T_np.shape[1] == num_points:
    Y_raw = T_np
else:
    raise ValueError(f"Temp_all.csv 维度 {T_np.shape} 与点数 num_points={num_points} 不匹配。")

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
lin_valid = np.where(geom_mask_np.reshape(-1) > 0.5)[0]
print(f"真实点占比(由C0给定): {float(mask_valid.mean()) * 100:.3f}%")

# =============================================================
# 边界条件与 split
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
# 标准化：X 用 StandardScaler；Y 做样本级 Z-score（仅在 mask==1 点上）
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

# 切分
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
# Masked Loss：点值损失 + 梯度损失（只在真实点上）
# =============================================================

def masked_loss(pred, target, mask, loss_type="huber", grad_weight=0.1):
    if loss_type == "huber":
        per_elem = F.smooth_l1_loss(pred, target, reduction="none")
        masked = per_elem * mask
    else:
        masked = ((pred - target) ** 2) * mask
    base_loss = masked.sum() / (mask.sum() + 1e-8)

    def _grads(t):
        gx = t[:, 1:, :, :] - t[:, :-1, :, :]
        gy = t[:, :, 1:, :] - t[:, :, :-1, :]
        gz = t[:, :, :, 1:] - t[:, :, :, :-1]
        return gx, gy, gz

    pdx, pdy, pdz = _grads(pred)
    tdx, tdy, tdz = _grads(target)

    mdx = mask[:, 1:, :, :] * mask[:, :-1, :, :]
    mdy = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    mdz = mask[:, :, :, 1:] * mask[:, :, :, :-1]

    grad_loss = pred.new_tensor(0.0)
    for pd, td, md in [(pdx, tdx, mdx), (pdy, tdy, mdy), (pdz, tdz, mdz)]:
        grad_loss += (((pd - td) ** 2) * md).sum() / (md.sum() + 1e-8)
    grad_loss = grad_loss / 3.0

    return base_loss + grad_weight * grad_loss


# =============================================================
# Optuna 目标函数
# =============================================================

def objective(trial):
    model = create_cnn3d_from_trial(trial, input_dim, nx, ny, nz)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
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
                break

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return best_val


# =============================================================
# Optuna 搜索 + Two-stage training
# =============================================================

print("开始 Optuna 超参搜索（FiLM/结构 + lr + weight_decay）...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
best_params = study.best_params
print("最佳参数:", best_params)

best_trial = optuna.trial.FixedTrial(best_params)

# --------------------- Stage A ---------------------
model_a = create_cnn3d_from_trial(best_trial, input_dim, nx, ny, nz)
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
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
    run = 0.0
    for xb, yb, mb in train_loader:
        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
        optimizer_a.zero_grad(set_to_none=True)
        pred = model_a(xb)
        loss = masked_loss(pred, yb, mb, loss_type=best_loss_type, grad_weight=best_grad_weight)
        loss.backward()
        optimizer_a.step()
        run += loss.item() * xb.size(0)

    if epoch < warmup_epochs_a:
        lr = initial_lr_a + (best_lr - initial_lr_a) * float(epoch + 1) / float(warmup_epochs_a)
        for pg in optimizer_a.param_groups:
            pg["lr"] = lr
    else:
        scheduler_a.step()
        lr = optimizer_a.param_groups[0]["lr"]

    model_a.eval()
    vtot = 0.0
    with torch.no_grad():
        for xb, yb, mb in val_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            pred = model_a(xb)
            vloss = masked_loss(pred, yb, mb, loss_type=best_loss_type, grad_weight=best_grad_weight)
            vtot += vloss.item() * xb.size(0)
    val_loss = vtot / len(val_loader.dataset)

    if epoch % 10 == 0:
        print(f"[Stage A] Epoch {epoch}, lr={lr:.3e}, val_loss={val_loss:.6f}")

    if val_loss < best_val - min_delta:
        best_val = val_loss
        best_epoch = epoch
        no_improve = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model_a_core.state_dict().items()}
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"[Stage A] Early stopping. best_epoch={best_epoch}, best_val={best_val:.6f}")
            break

if best_state is None:
    raise RuntimeError("Stage A 没有记录到 best_state（不应发生）。")

print(f"[Stage A] Selected best_epoch={best_epoch}, best_val={best_val:.6f}")

# --------------------- Stage B ---------------------
model_b = create_cnn3d_from_trial(best_trial, input_dim, nx, ny, nz)
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
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
for epoch in range(num_epochs_b):
    model_b.train()
    for xb, yb, mb in trainval_loader:
        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
        optimizer_b.zero_grad(set_to_none=True)
        pred = model_b(xb)
        loss = masked_loss(pred, yb, mb, loss_type=best_loss_type, grad_weight=best_grad_weight)
        loss.backward()
        optimizer_b.step()

    if epoch < warmup_epochs_b:
        lr = initial_lr_b + (best_lr - initial_lr_b) * float(epoch + 1) / float(warmup_epochs_b)
        for pg in optimizer_b.param_groups:
            pg["lr"] = lr
    else:
        scheduler_b.step()
        lr = optimizer_b.param_groups[0]["lr"]

    if epoch % 10 == 0 or epoch == num_epochs_b - 1:
        print(f"[Stage B] Epoch {epoch}, lr={lr:.3e}")

model_core = model_b.module if isinstance(model_b, nn.DataParallel) else model_b

# =============================================================
# 保存 ckpt（注意：把 film_scale 也保存进去，便于 test 脚本对齐）
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
        "model_type": "FiLM_A_voxel7_stem_film_scaled",
        "stem_stride": 2,
        "film_mult": best_params.get("film_mult", 4),
        "film_scale": float(best_params.get("film_scale", 1.0)),
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
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python "CNN_FiLM_sdf&bm_7.py"
