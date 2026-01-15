#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================
# CNN_Concat_sdf+bm_GN_RemoveStem_Bottleneck.py —— 边界条件 → 3D 温度场（BC 拼在 bottleneck）
# 3D UNet (depth=3) + Optuna + Mask + 小批量训练（避免 OOM）
#
# 改造点：
# 1) trial -> params: create_cnn3d_from_params(params, ...)
# 2) best_params 落盘 JSON
# 3) RUN_MODE: search vs train-only
# 4) 多 seed 只影响训练阶段：search 用 SEARCH_SEED 固定；train 用 SEEDS
# 5) 同一份超参跑一组 seeds，并输出汇总 seed_summary.csv
# 7) 命令行用法见文件底部注释
# =============================================================

import os
import json
import math
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
BEST_PARAMS_PATH = os.environ.get("BEST_PARAMS_PATH", "best_params_cnn_Concat_bottleneck.json")

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

# 输出路径
SUMMARY_CSV_PATH = os.environ.get("SUMMARY_CSV", "seed_summary.csv")

# ckpt 命名前缀
CKPT_PREFIX = os.environ.get("CKPT_PREFIX", "best_CNN_Concat_sdf+bm_GN_RemoveStem_Bottleneck")


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
# 固定 3D 体素输入（1,7,nx,ny,nz）：来自 cnn_input_channels_no_normals.csv 的后 7 列
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
# 3D UNet（无 FiLM）：入口仅 voxel(7)，在 bottleneck(/8) 拼接 bc(6)
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
    NoFiLM (depth=3), BC concatenated at /8 bottleneck:
    - 固定 7 通道体素：[C0,C1,C2,C3,C4,C5,sdf]
    - bc (B,6) -> bc_grid_8 (B,6,nx/8,ny/8,nz/8)，与 x2(/8) 拼接
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

            # input: ONLY voxel
            in_ch_in = 7

            c0 = max(8, base_ch // 2)

            self.enc0_full = ConvBlock(in_ch_in, c0, dropout_p=dropout_p)  # /1
            self.pool0 = nn.MaxPool3d(kernel_size=2, stride=2)  # /2

            self.enc0 = ConvBlock(c0, base_ch, dropout_p=dropout_p)  # /2
            self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # /4
            self.enc1 = ConvBlock(base_ch, base_ch * 2, dropout_p=dropout_p)  # /4
            self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # /8
            self.enc2 = ConvBlock(base_ch * 2, base_ch * 4, dropout_p=dropout_p)  # /8

            bottleneck_ch = base_ch * 4
            # ✅ bottleneck 拼 bc(6)
            self.bottleneck = ConvBlock(bottleneck_ch + 6, bottleneck_ch, dropout_p=dropout_p)

            self.up2_conv = ConvBlock(bottleneck_ch + base_ch * 2, base_ch * 2, dropout_p=dropout_p)  # /4
            self.up1_conv = ConvBlock(base_ch * 2 + base_ch, base_ch, dropout_p=dropout_p)            # /2

            self.out_proj = nn.Conv3d(base_ch, base_ch, kernel_size=1)

            self.up0_conv = ConvBlock(base_ch + c0, c0, dropout_p=dropout_p)  # /1
            self.final_conv = nn.Conv3d(c0, 1, kernel_size=1)

        def forward(self, bc: torch.Tensor) -> torch.Tensor:
            B = bc.size(0)

            vox = self.voxel_input.expand(B, -1, -1, -1, -1)     # (B,7,nx,ny,nz)
            mask_ch = self.geom_mask.expand(B, -1, -1, -1, -1)   # (B,1,nx,ny,nz)

            # /1
            x_full_skip = self.enc0_full(vox)

            # /2
            x = self.pool0(x_full_skip)

            # /2 -> /4 -> /8
            x0 = self.enc0(x)
            x1 = self.pool1(x0)
            x1 = self.enc1(x1)
            x2 = self.pool2(x1)
            x2 = self.enc2(x2)  # (B,4C,/8)

            # bottleneck: concat bc at /8
            bc_grid_8 = bc.view(B, 6, 1, 1, 1).expand(B, 6, x2.shape[2], x2.shape[3], x2.shape[4])
            xb = self.bottleneck(torch.cat([x2, bc_grid_8], dim=1))  # (B,4C,/8)

            # decoder
            x_up2 = F.interpolate(xb, size=x1.shape[2:], mode="trilinear", align_corners=False)
            x_dec2 = self.up2_conv(torch.cat([x_up2, x1], dim=1))  # /4

            x_up1 = F.interpolate(x_dec2, size=x0.shape[2:], mode="trilinear", align_corners=False)
            x_dec1 = self.up1_conv(torch.cat([x_up1, x0], dim=1))  # /2

            x_dec = self.out_proj(x_dec1)

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

# voxel 通道固定顺序
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

# BC + split
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

# 标准化：X 用 StandardScaler；Y 样本级 Z-score（仅 mask==1 点）
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

# split tensors
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
# Masked Loss
# =============================================================
def masked_loss(pred, target, mask):
    per_elem = F.smooth_l1_loss(pred, target, reduction="none")
    masked = per_elem * mask
    return masked.sum() / (mask.sum() + 1e-8)

def cosine_lr_with_eta_min(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    lr_max: float,
    lr_init: float,
    eta_min: float,
) -> float:
    """
    LR schedule: warmup -> cosine decay to eta_min (NOT to 0)

    warmup: lr_init -> lr_max (linear)
    cosine: lr = eta_min + (lr_max - eta_min)*0.5*(1+cos(pi*t)), t in [0,1]
    """
    if epoch < warmup_epochs:
        t = float(epoch + 1) / float(max(1, warmup_epochs))
        return float(lr_init + (lr_max - lr_init) * t)

    e = epoch - warmup_epochs
    T = max(1, total_epochs - warmup_epochs)
    t = float(e) / float(T)
    return float(eta_min) + (float(lr_max) - float(eta_min)) * 0.5 * (1.0 + math.cos(math.pi * t))

# =============================================================
# Optuna objective（search 阶段固定 SEARCH_SEED）
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

    # ---------- EMA-based selection & early stopping ----------
    alpha = float(os.environ.get("VAL_EMA_ALPHA", "0.30"))  # 0.2~0.4 常用
    min_delta = 1e-6
    patience = 30
    no_improve = 0

    best_ema = float("inf")
    val_ema = None

    # ---------- LR schedule (search: 80 epochs) ----------
    total_epochs = 80
    warmup_epochs = min(int(WARMUP_EPOCHS), 10)  # search 阶段别 warmup 太久
    lr_init = lr_max * 0.1
    eta_min = lr_max * 1e-2  # ✅ 经验值：lr_max * 1e-2

    # init lr
    for pg in optimizer.param_groups:
        pg["lr"] = lr_init

    for epoch in range(total_epochs):
        # ---- train ----
        model.train()
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = masked_loss(pred, yb, mb)
            loss.backward()
            optimizer.step()

        # ---- lr step: warmup -> cosine to eta_min ----
        lr_now = cosine_lr_with_eta_min(
            epoch=epoch,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            lr_max=lr_max,
            lr_init=lr_init,
            eta_min=eta_min,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # ---- val ----
        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                pred = model(xb)
                vloss = masked_loss(pred, yb, mb)
                vtot += vloss.item() * xb.size(0)
        val_loss = vtot / len(val_loader.dataset)

        # ---- EMA update ----
        if val_ema is None:
            val_ema = float(val_loss)
        else:
            val_ema = alpha * float(val_loss) + (1.0 - alpha) * float(val_ema)

        # ---- best selection & early stop based on EMA ----
        if val_ema < best_ema - min_delta:
            best_ema = float(val_ema)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return float(best_ema)

# =============================================================
# 单次训练（固定超参 + 单 seed）：
# - fixed budget
# - 同时保存 last & best(val)
# =============================================================
def train_one_seed(best_params: Dict[str, Any], seed: int) -> Dict[str, Any]:
    # ✅ 训练阶段随机性只来自这个 seed
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

    # fixed-budget schedule params
    warmup_epochs = int(WARMUP_EPOCHS)
    total_epochs = int(TOTAL_EPOCHS)

    lr_init = lr_max * 0.1
    eta_min = lr_max * 1e-2  # ✅ 经验值：lr_max * 1e-2

    for pg in optimizer.param_groups:
        pg["lr"] = lr_init

    # ---- EMA tracking ----
    alpha = float(os.environ.get("VAL_EMA_ALPHA", "0.30"))
    min_delta = 1e-6

    best_ema = float("inf")
    best_epoch = -1
    best_val_at_best = float("inf")
    val_ema = None
    best_state = None

    model_core = model.module if isinstance(model, nn.DataParallel) else model

    last_val = float("nan")
    last_ema = float("nan")

    print(f"\n===== Train seed={seed} | fixed-budget={total_epochs} | eta_min={eta_min:.3e} =====")

    for epoch in range(total_epochs):
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
        train_loss = running / len(train_loader.dataset)

        # ---- lr step: warmup -> cosine to eta_min ----
        lr_now = cosine_lr_with_eta_min(
            epoch=epoch,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            lr_max=lr_max,
            lr_init=lr_init,
            eta_min=eta_min,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # ---- val ----
        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                pred = model(xb)
                vloss = masked_loss(pred, yb, mb)
                vtot += vloss.item() * xb.size(0)
        val_loss = vtot / len(val_loader.dataset)
        last_val = float(val_loss)

        # ---- EMA update ----
        if val_ema is None:
            val_ema = float(val_loss)
        else:
            val_ema = alpha * float(val_loss) + (1.0 - alpha) * float(val_ema)
        last_ema = float(val_ema)

        # ---- best selection based on EMA ----
        if val_ema < best_ema - min_delta:
            best_ema = float(val_ema)
            best_epoch = int(epoch)
            best_val_at_best = float(val_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model_core.state_dict().items()}

        if epoch % 10 == 0 or epoch == total_epochs - 1:
            print(
                f"[seed={seed}] Epoch {epoch:03d}, lr={lr_now:.3e}, "
                f"train={train_loss:.6f}, val={val_loss:.6f}, val_ema={val_ema:.6f}, "
                f"best_epoch={best_epoch}, best_ema={best_ema:.6f}"
            )

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
            "model_type": "NoFiLM_BCConcatAtBottleneck_no_stem_pooldown_fullres_light",
            "lr": float(lr_max),
            "eta_min": float(eta_min),
            "eta_min_ratio": 1e-2,
            "val_ema_alpha": float(alpha),
            "x_mean": scaler_x.mean_,
            "x_scale": scaler_x.scale_,
            "Y_means": Y_means,
            "Y_stds": Y_stds,
            "geom_mask_np": geom_mask_np,
            "lin_valid": lin_valid,
            "voxel_cols": voxel_cols,
            "note": "voxel=7; bc(6) concatenated at /8 bottleneck",
            "train_seed": int(seed),
            "ckpt_kind": "last",
            "last_epoch": int(total_epochs - 1),
            "last_val": float(last_val),
            "last_val_ema": float(last_ema),
            "best_epoch_ema": int(best_epoch),
            "best_val_ema": float(best_ema),
            "best_val_at_best_epoch": float(best_val_at_best),
        },
        last_ckpt_path,
    )
    print(f"[seed={seed}] Saved LAST ckpt: {last_ckpt_path}")

    # -------------------- Save BEST checkpoint (EMA-best) --------------------
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
            "model_type": "NoFiLM_BCConcatAtBottleneck_no_stem_pooldown_fullres_light",
            "lr": float(lr_max),
            "eta_min": float(eta_min),
            "eta_min_ratio": 1e-2,
            "val_ema_alpha": float(alpha),
            "x_mean": scaler_x.mean_,
            "x_scale": scaler_x.scale_,
            "Y_means": Y_means,
            "Y_stds": Y_stds,
            "geom_mask_np": geom_mask_np,
            "lin_valid": lin_valid,
            "voxel_cols": voxel_cols,
            "note": "voxel=7; bc(6) concatenated at /8 bottleneck",
            "train_seed": int(seed),
            "ckpt_kind": "best_ema",
            "best_epoch_ema": int(best_epoch),
            "best_val_ema": float(best_ema),
            "best_val_at_best_epoch": float(best_val_at_best),
            "last_epoch": int(total_epochs - 1),
            "last_val": float(last_val),
            "last_val_ema": float(last_ema),
        },
        best_ckpt_path,
    )
    print(f"[seed={seed}] Saved BEST ckpt: {best_ckpt_path}")
    print(f"[seed={seed}] best_epoch(EMA)={best_epoch} | best_ema={best_ema:.6f} | val@best={best_val_at_best:.6f}")
    print(f"[seed={seed}] last_epoch={total_epochs - 1} | last_val={last_val:.6f} | last_ema={last_ema:.6f}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "seed": int(seed),
        "best_epoch_ema": int(best_epoch),
        "best_val_ema": float(best_ema),
        "best_val_at_best_epoch": float(best_val_at_best),
        "last_epoch": int(total_epochs - 1),
        "last_val": float(last_val),
        "last_val_ema": float(last_ema),
        "ckpt_best_path": best_ckpt_path,
        "ckpt_last_path": last_ckpt_path,
        "lr": float(lr_max),
        "eta_min": float(eta_min),
        "dropout_p": float(dropout_p),
        "total_epochs": int(total_epochs),
        "warmup_epochs": int(warmup_epochs),
        "val_ema_alpha": float(alpha),
    }

# =============================================================
# 主流程：search or train-only + multi-seed training + summary
# =============================================================
print("\n=============================================================")
print(f"RUN_MODE={RUN_MODE} | BEST_PARAMS_PATH={BEST_PARAMS_PATH}")
print(f"SEARCH_SEED={SEARCH_SEED} | SEEDS={SEEDS}")
print(f"N_TRIALS={N_TRIALS} | TOTAL_EPOCHS={TOTAL_EPOCHS} | WARMUP_EPOCHS={WARMUP_EPOCHS} | BATCH_SIZE={BATCH_SIZE}")
print(f"CKPT_PREFIX={CKPT_PREFIX} | SUMMARY_CSV={SUMMARY_CSV_PATH}")
print("=============================================================\n")

if RUN_MODE not in ("search", "train"):
    raise ValueError('RUN_MODE must be "search" or "train"')

if RUN_MODE == "search":
    # ✅ search 使用固定 SEARCH_SEED，确保可比
    set_seed(SEARCH_SEED)
    print("开始 Optuna 超参搜索（dropout + lr）...")

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
    results.append(train_one_seed(best_params, seed=int(s)))

df_sum = pd.DataFrame(results)
df_sum.to_csv(SUMMARY_CSV_PATH, index=False, encoding="utf-8")

print("\n===== Seed summary =====")
print(df_sum)
print(f"Saved summary CSV: {SUMMARY_CSV_PATH}")

# =============================================================
# 命令行用法（示例）
# =============================================================
# 1) 先跑 Optuna 搜索 + 自动保存 best_params_cnn_Concat_bottleneck.json，然后训练一个 seed（默认 SEED=43）
# RUN_MODE=search N_TRIALS=20 SEARCH_SEED=42 SEED=43 python CNN_Concat_sdf+bm_GN_RemoveStem_Bottleneck.py
#
# 2) 只训练（train-only），不做任何超参搜索：读取 best_params_cnn_Concat_bottleneck.json
# RUN_MODE=train BEST_PARAMS_PATH=best_params_cnn_Concat_bottleneck.json SEED=43 python CNN_Concat_sdf+bm_GN_RemoveStem_Bottleneck.py
#
# 3) 同一份超参，跑一组 seeds，并输出 seed_summary.csv（推荐做鲁棒性）
# RUN_MODE=train BEST_PARAMS_PATH=best_params_cnn_Concat_bottleneck.json SEEDS="42,43,44,45,46" python CNN_Concat_sdf+bm_GN_RemoveStem_Bottleneck.py
#
# 4) 固定训练预算/批量大小（只影响训练，不影响 search）
# RUN_MODE=train BEST_PARAMS_PATH=best_params_cnn_Concat_bottleneck.json SEEDS="42,43,44" TOTAL_EPOCHS=300 WARMUP_EPOCHS=20 BATCH_SIZE=16 python CNN_Concat_sdf+bm_GN_RemoveStem_Bottleneck.py
#
# 5) 自定义 ckpt 前缀与汇总表路径
# RUN_MODE=train BEST_PARAMS_PATH=best_params_cnn_Concat_bottleneck.json SEEDS="42,43" CKPT_PREFIX="ckpt/CNN_BN" SUMMARY_CSV="ckpt/seed_summary.csv" python CNN_Concat_sdf+bm_GN_RemoveStem_Bottleneck.py

# 6)（如果你加了 TRAIN_AFTER_SEARCH 开关）只搜索不训练
# RUN_MODE=search TRAIN_AFTER_SEARCH=0 SEARCH_SEED=42 N_TRIALS=20 BEST_PARAMS_PATH=best_params_cnn_Concat_bottleneck.json python CNN_Concat_sdf+bm_GN_RemoveStem_Bottleneck.py