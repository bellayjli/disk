#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================
# CNN.py —— 边界条件 → 3D 温度场（Concat BC as voxel channels）
#
# 改造点（同 FiLM 版本）：
# 1) trial -> params: create_cnn3d_from_params(params, ...)
# 2) best_params 落盘 JSON
# 3) RUN_MODE: search vs train-only
# 4) 多 seed 只影响训练阶段：search 用 SEARCH_SEED 固定；train 用 SEEDS
# 5) 同一份超参跑一组 seeds，并输出汇总 seed_summary_cnn_concat.csv
# =============================================================

import os
import json
import random
from typing import Dict, Any, List

import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


# =============================================================
# /dev/shm (optional)
# =============================================================
os.environ.setdefault("TMPDIR", "/dev/shm")
os.environ.setdefault("TEMP", "/dev/shm")
os.environ.setdefault("TMP", "/dev/shm")


# =============================================================
# 运行模式 / 配置（环境变量）
# =============================================================
RUN_MODE = os.environ.get("RUN_MODE", "search").strip().lower()  # "search" or "train"
TRAIN_AFTER_SEARCH = os.environ.get("TRAIN_AFTER_SEARCH", "1").strip() not in ("0", "false", "no")
BEST_PARAMS_PATH = os.environ.get("BEST_PARAMS_PATH", "best_params_cnn_concat.json")

# Search 阶段固定 seed（保证 trial 可比；换训练 seed 不会影响搜索）
SEARCH_SEED = int(os.environ.get("SEARCH_SEED", "42"))

# 训练阶段 seed 列表（只影响训练）
# 用法：SEEDS="42,43,44,45,46"
SEEDS_ENV = os.environ.get("SEEDS", "").strip()
if SEEDS_ENV:
    SEEDS: List[int] = [int(s) for s in SEEDS_ENV.split(",") if s.strip() != ""]
else:
    # 兼容：如果你只想跑一个 seed，可以用 SEED=43
    SEEDS = [int(os.environ.get("SEED", "43"))]

# Optuna trials
N_TRIALS = int(os.environ.get("N_TRIALS", "20"))

# 固定训练预算（你的单阶段 fixed-budget 逻辑）
TOTAL_EPOCHS = int(os.environ.get("TOTAL_EPOCHS", "300"))
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "20"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))

# EMA / early stop
VAL_EMA_DECAY = float(os.environ.get("VAL_EMA_DECAY", "0.9"))          # e.g. 0.9 / 0.95
MIN_DELTA = float(os.environ.get("MIN_DELTA", "1e-6"))
PATIENCE_SEARCH = int(os.environ.get("PATIENCE_SEARCH", "30"))         # objective 内的 early stop
PATIENCE_TRAIN = int(os.environ.get("PATIENCE_TRAIN", "30"))           # train_one_seed 的 early stop
MIN_EPOCHS_BEFORE_STOP = int(os.environ.get("MIN_EPOCHS_BEFORE_STOP", "0"))  # e.g. 0 / WARMUP_EPOCHS

# lr_min hold（可选）：warmup + cosine 到 eta_min 之后，再保持 eta_min 若干 epoch
LR_MIN_HOLD_EPOCHS = int(os.environ.get("LR_MIN_HOLD_EPOCHS", "0"))    # 默认 0（纯 warmup+cosine）

# 输出路径
SUMMARY_CSV_PATH = os.environ.get("SUMMARY_CSV", "seed_summary.csv")

# ckpt 命名前缀
CKPT_PREFIX = os.environ.get("CKPT_PREFIX", "best_CNN_Concat_sdf+bm_GN_RemoveStem")


# =============================================================
# 随机种子工具（训练阶段会反复调用；search 阶段也会固定调用）
# =============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =============================================================
# best_params JSON 落盘/读盘
# =============================================================
def save_best_params(best_params: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_best_params(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def make_gn(C: int, max_groups: int = 8):
    G = min(max_groups, C)
    while C % G != 0:
        G -= 1
    return nn.GroupNorm(G, C)


# =============================================================
# 3D UNet（无 FiLM）：输入=固定 7 通道体素 + 复制的 6 通道边界条件 => 13 通道
# =============================================================
class ConvBlock(nn.Module):
    """3D Residual Block: (Conv3d->GN->GELU)x2 -> Dropout3d + residual"""

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            make_gn(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            make_gn(out_ch),
            nn.GELU(),
            nn.Dropout3d(dropout_p),
        )
        self.residual_proj = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out = self.conv(x)
        return out + residual


# =============================================================
# ✅ 1) 模型构建：from params（不依赖 optuna trial）
# =============================================================
def create_cnn3d_from_params(params: Dict[str, Any], input_dim: int, nx: int, ny: int, nz: int) -> nn.Module:
    """
    NoFiLM (depth=3), BC-as-voxels concat:
    - 固定 7 通道体素：[C0,C1,C2,C3,C4,C5,sdf]
    - bc (B,6) 复制成 bc_grid (B,6,nx,ny,nz)，与 vox 拼成 13 通道输入
    """
    depth = 3
    base_ch = 24
    dropout_p = float(params.get("dropout_p", 0.1))

    class CNN3D_NoFiLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.nx, self.ny, self.nz = nx, ny, nz
            self.depth = depth
            self.base_ch = base_ch

            assert VOXEL_INPUT is not None, "VOXEL_INPUT 未初始化"
            self.register_buffer("voxel_input", VOXEL_INPUT)  # (1,7,nx,ny,nz)

            assert GEOM_MASK is not None, "GEOM_MASK 未初始化"
            self.register_buffer("geom_mask", GEOM_MASK)  # (1,1,nx,ny,nz)

            in_ch_in = 13  # 7(voxel) + 6(bc)

            c0 = max(8, base_ch // 2)

            self.enc0_full = ConvBlock(in_ch_in, c0, dropout_p=dropout_p)  # /1
            self.pool0 = nn.MaxPool3d(kernel_size=2, stride=2)  # /2

            self.enc0 = ConvBlock(c0, base_ch, dropout_p=dropout_p)  # /2
            self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # /4
            self.enc1 = ConvBlock(base_ch, base_ch * 2, dropout_p=dropout_p)  # /4
            self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # /8
            self.enc2 = ConvBlock(base_ch * 2, base_ch * 4, dropout_p=dropout_p)  # /8

            bottleneck_ch = base_ch * 4
            self.bottleneck = ConvBlock(bottleneck_ch, bottleneck_ch, dropout_p=dropout_p)

            self.up2_conv = ConvBlock(bottleneck_ch + base_ch * 2, base_ch * 2, dropout_p=dropout_p)  # /4
            self.up1_conv = ConvBlock(base_ch * 2 + base_ch, base_ch, dropout_p=dropout_p)  # /2

            self.out_proj = nn.Conv3d(base_ch, base_ch, kernel_size=1)

            self.up0_conv = ConvBlock(base_ch + c0, c0, dropout_p=dropout_p)  # /1
            self.final_conv = nn.Conv3d(c0, 1, kernel_size=1)

        def forward(self, bc: torch.Tensor) -> torch.Tensor:
            B = bc.size(0)
            vox = self.voxel_input.expand(B, -1, -1, -1, -1)  # (B,7,nx,ny,nz)
            mask_ch = self.geom_mask.expand(B, -1, -1, -1, -1)  # (B,1,nx,ny,nz)

            bc_grid = bc.view(B, 6, 1, 1, 1).expand(B, 6, self.nx, self.ny, self.nz)
            x_in = torch.cat([vox, bc_grid], dim=1)  # (B,13,nx,ny,nz)

            x_full_skip = self.enc0_full(x_in)  # /1
            x = self.pool0(x_full_skip)  # /2

            x0 = self.enc0(x)  # /2
            x1 = self.pool1(x0)
            x1 = self.enc1(x1)  # /4
            x2 = self.pool2(x1)
            x2 = self.enc2(x2)  # /8

            xb = self.bottleneck(x2)  # /8

            x_up2 = F.interpolate(xb, size=x1.shape[2:], mode="trilinear", align_corners=False)
            x_dec2 = self.up2_conv(torch.cat([x_up2, x1], dim=1))  # /4

            x_up1 = F.interpolate(x_dec2, size=x0.shape[2:], mode="trilinear", align_corners=False)
            x_dec1 = self.up1_conv(torch.cat([x_up1, x0], dim=1))  # /2

            x_dec = self.out_proj(x_dec1)  # /2

            x_up0 = F.interpolate(x_dec, size=x_full_skip.shape[2:], mode="trilinear", align_corners=False)
            x0_full = self.up0_conv(torch.cat([x_up0, x_full_skip], dim=1))  # /1

            x_full = self.final_conv(x0_full)  # (B,1,nx,ny,nz)
            out = x_full.squeeze(1)
            out = out * mask_ch.squeeze(1)
            return out

    return CNN3D_NoFiLM().to(device)


# =============================================================
# 保留 trial 版本（search 用）：采样 -> params -> from_params
# =============================================================
def create_cnn3d_from_trial(trial: optuna.trial.Trial, input_dim: int, nx: int, ny: int, nz: int) -> nn.Module:
    params = {
        "dropout_p": trial.suggest_float("dropout_p", 0.0, 0.3),
    }
    return create_cnn3d_from_params(params, input_dim, nx, ny, nz)


# =============================================================
# 数据读取：体素输入 + 温度标签（全网格） + mask
# =============================================================
datapath_bc = "data/boundary_condition.csv"
datapath_temp = "data/Temp_all.csv"
datapath_voxel = "data/cnn_input_channels_no_normals.csv"

df_vox = pd.read_csv(datapath_voxel)

for c in ["x", "y", "z", "C0"]:
    if c not in df_vox.columns:
        raise KeyError(f"{datapath_voxel} 缺少列: {c}")

# 体素通道固定为：[C0,C1,C2,C3,C4,C5,sdf]
voxel_cols = ["C0", "C1", "C2", "C3", "C4", "C5", "sdf"]
missing_cols = [c for c in voxel_cols if c not in df_vox.columns]
if missing_cols:
    raise KeyError(f"{datapath_voxel} 缺少体素通道列: {missing_cols}（需要 {voxel_cols}）")

last7 = list(df_vox.columns[-7:])
if last7 != voxel_cols:
    print(f"[voxel][warn] CSV 后7列为 {last7}，但本脚本将按显式顺序使用 {voxel_cols}。")
print(f"[voxel] using voxel_cols: {voxel_cols}")

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

voxel_grid = np.zeros((len(voxel_cols), nx, ny, nz), dtype=np.float32)
cols_np = [df_vox[c].to_numpy(dtype=np.float32) for c in voxel_cols]

geom_mask_np = np.zeros((nx, ny, nz), dtype=np.float32)
c0_np = df_vox["C0"].to_numpy(dtype=np.float32)

for i in range(df_vox.shape[0]):
    ix = x_index[float(xv[i])]
    iy = y_index[float(yv[i])]
    iz = z_index[float(zv[i])]
    voxel_grid[:, ix, iy, iz] = np.array([c[i] for c in cols_np], dtype=np.float32)
    geom_mask_np[ix, iy, iz] = 1.0 if c0_np[i] > 0.5 else 0.0

GEOM_MASK = torch.tensor(geom_mask_np[None, None, ...], dtype=torch.float32, device=device)
VOXEL_INPUT = torch.tensor(voxel_grid[None, ...], dtype=torch.float32, device=device)
print(f"全局 C0(inside_mask) 占比: {geom_mask_np.mean() * 100:.3f}%")

# 温度
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

# 边界条件与 split
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

# 标准化：X 用 StandardScaler；Y 做样本级 Z-score（仅在 mask==1 点上）
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

# Build dataset tensors ONCE (CPU)
x_train_t = torch.from_numpy(x_train).to(torch.float32)
y_train_t = torch.from_numpy(y_train).to(torch.float32)
m_train_t = torch.from_numpy(mask_train).to(torch.float32)

x_val_t = torch.from_numpy(x_val).to(torch.float32)
y_val_t = torch.from_numpy(y_val).to(torch.float32)
m_val_t = torch.from_numpy(mask_val).to(torch.float32)

x_test_t = torch.from_numpy(x_test).to(torch.float32)
y_test_t = torch.from_numpy(y_test).to(torch.float32)
m_test_t = torch.from_numpy(mask_test).to(torch.float32)

PIN_MEMORY = (device.type == "cuda")


# =============================================================
# Masked Loss（Huber / smooth_l1）
# =============================================================
def masked_loss(pred, target, mask):
    per_elem = F.smooth_l1_loss(pred, target, reduction="none")
    masked = per_elem * mask
    return masked.sum() / (mask.sum() + 1e-8)


# =============================================================
# val EMA helpers
# =============================================================
def ema_update(prev: Optional[float], x: float, decay: float) -> float:
    if prev is None or (isinstance(prev, float) and (not math.isfinite(prev))):
        return float(x)
    return float(decay) * float(prev) + (1.0 - float(decay)) * float(x)


# =============================================================
# LR schedule: warmup -> cosine to eta_min -> (optional) hold eta_min
# =============================================================
def lr_warmup_cosine_eta_min(
    epoch: int,
    *,
    total_epochs: int,
    warmup_epochs: int,
    lr_max: float,
    lr_init: float,
    eta_min: float,
    hold_epochs: int = 0,
) -> float:
    """
    - warmup: linear lr_init -> lr_max
    - cosine: lr_max -> eta_min (with eta_min floor)
      lr = eta_min + (lr_max - eta_min) * 0.5 * (1 + cos(pi * t))
    - optional: last hold_epochs keep eta_min
    """
    total_epochs = int(total_epochs)
    warmup_epochs = int(warmup_epochs)
    hold_epochs = int(max(0, hold_epochs))

    if total_epochs <= 0:
        return float(eta_min)

    # clamp hold so we still have at least 1 cosine step if possible
    hold_epochs = min(hold_epochs, max(0, total_epochs - warmup_epochs - 1))

    if epoch < warmup_epochs:
        t = float(epoch + 1) / float(max(1, warmup_epochs))
        return float(lr_init + (lr_max - lr_init) * t)

    # epochs remaining after warmup
    cosine_total = max(1, total_epochs - warmup_epochs - hold_epochs)

    # if in hold region
    if epoch >= warmup_epochs + cosine_total:
        return float(eta_min)

    # cosine region index
    e = epoch - warmup_epochs  # 0..cosine_total-1
    t = float(e) / float(cosine_total)  # [0,1)
    return float(eta_min + (lr_max - eta_min) * 0.5 * (1.0 + math.cos(math.pi * t)))


def set_optimizer_lr(optimizer: optim.Optimizer, lr_value: float) -> None:
    lr_value = float(lr_value)
    for pg in optimizer.param_groups:
        pg["lr"] = lr_value


# =============================================================
# Optuna objective（search 阶段固定 SEARCH_SEED）
# - best selection: val_ema
# - early stop: val_ema
# =============================================================
def objective(trial: optuna.trial.Trial) -> float:
    # ✅ search 阶段固定住一切随机性（只让 trial 的 params 变化）
    set_seed(SEARCH_SEED)

    g_trial = torch.Generator()
    g_trial.manual_seed(SEARCH_SEED)

    model = create_cnn3d_from_trial(trial, input_dim, nx, ny, nz)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    lr_max = float(trial.suggest_float("lr", 1e-5, 1e-2, log=True))
    optimizer = optim.Adam(model.parameters(), lr=lr_max)

    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t, m_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g_trial,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        TensorDataset(x_val_t, y_val_t, m_val_t),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=PIN_MEMORY,
    )

    # objective 训练预算（保持你原来的 80）
    total_epochs = 80
    warmup_epochs = min(10, total_epochs // 4)  # objective 内给个小 warmup（不影响 train-one-seed）
    lr_init = lr_max * 0.1
    eta_min = lr_max * 1e-2  # ✅ 经验值
    hold_epochs = 0

    best_ema = float("inf")
    best_epoch = -1
    val_ema: Optional[float] = None
    no_improve = 0

    for epoch in range(total_epochs):
        # --- train ---
        model.train()
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = masked_loss(pred, yb, mb)
            loss.backward()
            optimizer.step()

        # --- lr step ---
        lr = lr_warmup_cosine_eta_min(
            epoch=epoch,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            lr_max=lr_max,
            lr_init=lr_init,
            eta_min=eta_min,
            hold_epochs=hold_epochs,
        )
        set_optimizer_lr(optimizer, lr)

        # --- val ---
        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                pred = model(xb)
                vloss = masked_loss(pred, yb, mb)
                vtot += vloss.item() * xb.size(0)

        val_loss = float(vtot / len(val_loader.dataset))
        val_ema = ema_update(val_ema, val_loss, decay=VAL_EMA_DECAY)

        # --- best/early stop based on EMA ---
        if val_ema < best_ema - MIN_DELTA:
            best_ema = float(val_ema)
            best_epoch = int(epoch)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE_SEARCH:
                break

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return float(best_ema)


# =============================================================
# 单次训练（固定超参 + 单 seed）：
# - fixed budget (TOTAL_EPOCHS)，但允许 EMA-based early stop（PATIENCE_TRAIN）
# - best selection: val_ema
# - 同时保存 last & best
# =============================================================
def train_one_seed(best_params: Dict[str, Any], seed: int) -> Dict[str, Any]:
    set_seed(seed)

    model = create_cnn3d_from_params(best_params, input_dim, nx, ny, nz)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    lr_max = float(best_params.get("lr", 1e-3))
    dropout_p = float(best_params.get("dropout_p", 0.1))

    optimizer = optim.Adam(model.parameters(), lr=lr_max)

    # dataloaders (shuffle depends on seed)
    g_train = torch.Generator()
    g_train.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t, m_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g_train,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        TensorDataset(x_val_t, y_val_t, m_val_t),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=PIN_MEMORY,
    )

    total_epochs = int(TOTAL_EPOCHS)
    warmup_epochs = int(WARMUP_EPOCHS)

    lr_init = lr_max * 0.1
    eta_min = lr_max * 1e-2  # ✅ 经验值
    hold_epochs = int(LR_MIN_HOLD_EPOCHS)

    set_optimizer_lr(optimizer, lr_init)

    # best checkpoint tracking (based on val_ema)
    best_ema = float("inf")
    best_epoch = -1
    best_state = None

    val_ema: Optional[float] = None
    no_improve = 0

    model_core = model.module if isinstance(model, nn.DataParallel) else model

    last_epoch = -1
    last_val = float("nan")
    last_val_ema = float("nan")

    print(f"\n===== Train seed={seed} | budget={total_epochs} | warmup={warmup_epochs} | ema_decay={VAL_EMA_DECAY} =====")
    print(f"[seed={seed}] lr_max={lr_max:.3e} | eta_min={eta_min:.3e} | lr_init={lr_init:.3e} | hold_epochs={hold_epochs}")

    for epoch in range(total_epochs):
        last_epoch = int(epoch)

        # ---- train ----
        model.train()
        running = 0.0
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = masked_loss(pred, yb, mb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)

        train_loss = float(running / len(train_loader.dataset))

        # ---- lr: warmup -> cosine to eta_min -> (optional) hold ----
        lr = lr_warmup_cosine_eta_min(
            epoch=epoch,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            lr_max=lr_max,
            lr_init=lr_init,
            eta_min=eta_min,
            hold_epochs=hold_epochs,
        )
        set_optimizer_lr(optimizer, lr)

        # ---- val ----
        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                pred = model(xb)
                vloss = masked_loss(pred, yb, mb)
                vtot += vloss.item() * xb.size(0)

        val_loss = float(vtot / len(val_loader.dataset))
        val_ema = ema_update(val_ema, val_loss, decay=VAL_EMA_DECAY)

        last_val = float(val_loss)
        last_val_ema = float(val_ema)

        # ---- best / early stop based on val_ema ----
        improved = val_ema < best_ema - MIN_DELTA
        if improved:
            best_ema = float(val_ema)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model_core.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == total_epochs - 1:
            print(
                f"[seed={seed}] Epoch {epoch:03d}, lr={lr:.3e}, "
                f"train={train_loss:.6f}, val={val_loss:.6f}, val_ema={val_ema:.6f}, "
                f"best_epoch={best_epoch}, best_ema={best_ema:.6f}, "
                f"no_improve={no_improve}/{PATIENCE_TRAIN}"
            )

        # early stop gate
        if (epoch + 1) >= int(MIN_EPOCHS_BEFORE_STOP) and no_improve >= int(PATIENCE_TRAIN):
            print(
                f"[seed={seed}] Early stopping at epoch={epoch} "
                f"(best_epoch={best_epoch}, best_ema={best_ema:.6f})."
            )
            break

    if best_state is None:
        raise RuntimeError("训练结束但 best_state 为空（不应发生）。")

    # -------------------- Save LAST checkpoint --------------------
    last_ckpt_path = f"{CKPT_PREFIX}_seed{seed}_last.pth"
    torch.save(
        {
            "state_dict": model_core.state_dict(),
            "input_dim": input_dim,
            "nx": nx, "ny": ny, "nz": nz,
            "depth": 3,
            "base_ch": 24,
            "dropout_p": float(dropout_p),
            "model_type": "NoFiLM_BCAsVoxels_no_stem_pooldown_fullres_light",
            "lr": float(lr_max),
            "eta_min": float(eta_min),
            "warmup_epochs": int(warmup_epochs),
            "total_epochs_budget": int(total_epochs),
            "early_stop_patience": int(PATIENCE_TRAIN),
            "val_ema_decay": float(VAL_EMA_DECAY),
            "x_mean": scaler_x.mean_,
            "x_scale": scaler_x.scale_,
            "Y_means": Y_means,
            "Y_stds": Y_stds,
            "geom_mask_np": geom_mask_np,
            "lin_valid": lin_valid,
            "voxel_cols": voxel_cols,
            "note": "input channels = 7(voxel) + 6 bc replicated => 13",
            "train_seed": int(seed),
            "ckpt_kind": "last",
            "last_epoch": int(last_epoch),
            "last_val": float(last_val),
            "last_val_ema": float(last_val_ema),
            "best_epoch": int(best_epoch),
            "best_val_ema": float(best_ema),
        },
        last_ckpt_path,
    )
    print(f"[seed={seed}] Saved LAST ckpt: {last_ckpt_path}")

    # -------------------- Save BEST checkpoint --------------------
    model_core.load_state_dict(best_state, strict=True)
    best_ckpt_path = f"{CKPT_PREFIX}_seed{seed}_best.pth"
    torch.save(
        {
            "state_dict": model_core.state_dict(),
            "input_dim": input_dim,
            "nx": nx, "ny": ny, "nz": nz,
            "depth": 3,
            "base_ch": 24,
            "dropout_p": float(dropout_p),
            "model_type": "NoFiLM_BCAsVoxels_no_stem_pooldown_fullres_light",
            "lr": float(lr_max),
            "eta_min": float(eta_min),
            "warmup_epochs": int(warmup_epochs),
            "total_epochs_budget": int(total_epochs),
            "early_stop_patience": int(PATIENCE_TRAIN),
            "val_ema_decay": float(VAL_EMA_DECAY),
            "x_mean": scaler_x.mean_,
            "x_scale": scaler_x.scale_,
            "Y_means": Y_means,
            "Y_stds": Y_stds,
            "geom_mask_np": geom_mask_np,
            "lin_valid": lin_valid,
            "voxel_cols": voxel_cols,
            "note": "input channels = 7(voxel) + 6 bc replicated => 13",
            "train_seed": int(seed),
            "ckpt_kind": "best",
            "best_epoch": int(best_epoch),
            "best_val_ema": float(best_ema),
            "last_epoch": int(last_epoch),
            "last_val": float(last_val),
            "last_val_ema": float(last_val_ema),
        },
        best_ckpt_path,
    )
    print(f"[seed={seed}] Saved BEST ckpt: {best_ckpt_path}")
    print(
        f"[seed={seed}] best_epoch={best_epoch} | best_val_ema={best_ema:.6f} | "
        f"last_epoch={last_epoch} | last_val={last_val:.6f} | last_val_ema={last_val_ema:.6f}"
    )

    # cleanup
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_ema": float(best_ema),
        "last_epoch": int(last_epoch),
        "last_val": float(last_val),
        "last_val_ema": float(last_val_ema),
        "ckpt_best_path": best_ckpt_path,
        "ckpt_last_path": last_ckpt_path,
        "lr_max": float(lr_max),
        "eta_min": float(eta_min),
        "dropout_p": float(dropout_p),
        "total_epochs_budget": int(total_epochs),
        "warmup_epochs": int(warmup_epochs),
        "patience_train": int(PATIENCE_TRAIN),
        "ema_decay": float(VAL_EMA_DECAY),
    }


# =============================================================
# 主流程：search or train-only + multi-seed training + summary
# =============================================================
print("\n=============================================================")
print(f"RUN_MODE={RUN_MODE} | BEST_PARAMS_PATH={BEST_PARAMS_PATH}")
print(f"SEARCH_SEED={SEARCH_SEED} | SEEDS={SEEDS}")
print(f"N_TRIALS={N_TRIALS} | TOTAL_EPOCHS={TOTAL_EPOCHS} | WARMUP_EPOCHS={WARMUP_EPOCHS} | BATCH_SIZE={BATCH_SIZE}")
print(f"VAL_EMA_DECAY={VAL_EMA_DECAY} | MIN_DELTA={MIN_DELTA} | PATIENCE_SEARCH={PATIENCE_SEARCH} | PATIENCE_TRAIN={PATIENCE_TRAIN}")
print(f"LR_MIN_HOLD_EPOCHS={LR_MIN_HOLD_EPOCHS} | MIN_EPOCHS_BEFORE_STOP={MIN_EPOCHS_BEFORE_STOP}")
print(f"CKPT_PREFIX={CKPT_PREFIX} | SUMMARY_CSV={SUMMARY_CSV_PATH}")
print("=============================================================\n")

if RUN_MODE not in ("search", "train"):
    raise ValueError('RUN_MODE must be "search" or "train"')

if RUN_MODE == "search":
    set_seed(SEARCH_SEED)
    print("开始 Optuna 超参搜索（结构 + lr；目标=best_val_ema）...")

    sampler = optuna.samplers.TPESampler(seed=SEARCH_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)

    best_params = dict(study.best_params)

    print("最佳参数:", best_params)
    save_best_params(best_params, BEST_PARAMS_PATH)
    print("best_params saved to:", BEST_PARAMS_PATH)
else:
    best_params = load_best_params(BEST_PARAMS_PATH)
    print("Loaded best_params:", best_params)

# ✅ 多 seed 训练 + 汇总
results = []
for s in SEEDS:
    r = train_one_seed(best_params, seed=int(s))
    results.append(r)

df_sum = pd.DataFrame(results)
df_sum.to_csv(SUMMARY_CSV_PATH, index=False, encoding="utf-8")

print("\n===== Seed summary =====")
print(df_sum)
print(f"Saved summary CSV: {SUMMARY_CSV_PATH}")

# =============================================================
# 命令行用法（示例）
# =============================================================
# 1) 先跑一次 Optuna 搜索 + 自动保存 best_params_cnn_concat.json，然后默认 SEED=43 训练
# RUN_MODE=search N_TRIALS=20 SEARCH_SEED=42 SEED=43 python CNN.py
#
# 2) 只训练（train-only），不做任何超参搜索：读取 best_params_cnn_concat.json
# RUN_MODE=train BEST_PARAMS_PATH=best_params_cnn_concat.json SEED=43 python CNN.py
#
# 3) 同一份超参，跑一组 seeds，并输出 seed_summary_cnn_concat.csv（推荐做鲁棒性）
# RUN_MODE=train BEST_PARAMS_PATH=best_params_cnn_concat.json SEEDS="42,43,44,45,46" python CNN.py
#
# 4) 你想固定训练预算/批量大小（只影响训练，不影响 search）
# RUN_MODE=train BEST_PARAMS_PATH=best_params_cnn_concat.json SEEDS="42,43,44" TOTAL_EPOCHS=300 WARMUP_EPOCHS=20 BATCH_SIZE=16 python CNN.py
#
# 5) 自定义 ckpt 前缀与汇总表路径
# RUN_MODE=train BEST_PARAMS_PATH=best_params_cnn_concat.json SEEDS="42,43" CKPT_PREFIX="ckpt/CNN_Concat" SUMMARY_CSV="ckpt/seed_summary_cnn_concat.csv" python CNN.py

# 6)（如果你加了 TRAIN_AFTER_SEARCH 开关）只搜索不训练
# RUN_MODE=search TRAIN_AFTER_SEARCH=0 SEARCH_SEED=42 N_TRIALS=20 BEST_PARAMS_PATH=best_params_film_allBlock.json python CNN_FiLM_sdf+bm_GN_RemoveStem.py