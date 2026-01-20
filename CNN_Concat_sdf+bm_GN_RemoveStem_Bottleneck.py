# =============================================================
# CNN_Concat_sdf+bm_GN_RemoveStem_Bottleneck.py  ——  边界条件 → 3D 温度场
# 3D UNet (depth=3) + Optuna + Mask + 小批量训练（避免 OOM）
# 输入：
#   - 7 通道体素来自：cnn_input_channels_no_normals.csv 的后 7 列
#   - 6 个边界条件通道来自：boundary_condition.csv（每个样本复制到bottleneck）
# 监督 mask 使用 C0（inside_mask）
# 温度标签来自：Temp_all.csv（全网格点）
# 【两段式改版要点】
# 1) Optuna 搜索阶段：使用“轻量代理训练（proxy training）”
#    - 仍用原始 train/val 划分评估（objective = val loss）
#    - 训练 budget 很小：少量 epoch + 每个 epoch 少量 step
# 2) 最终训练阶段：用 train+val 合并后的大训练集
#    - 固定 total_epochs = 300
#    - 保存 last + bestTrain（训练集最优）
# 用法：
#   - 搜索一次并保存超参：--run_mode search
#   - 用保存的超参训练：  --run_mode train --seed 43
# =============================================================

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

os.environ["TMPDIR"] = "/dev/shm"
os.environ["TEMP"] = "/dev/shm"
os.environ["TMP"] = "/dev/shm"

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

# =============================================================
# Utilities
# =============================================================
def set_seed(seed: int, *, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # 这会略慢，但更可复现；你也可以关掉
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_gn(C: int, max_groups: int = 8):
    G = min(max_groups, C)
    while C % G != 0:
        G -= 1
    return nn.GroupNorm(G, C)

def masked_loss(pred, target, mask):
    per_elem = F.smooth_l1_loss(pred, target, reduction="none")
    masked = per_elem * mask
    return masked.sum() / (mask.sum() + 1e-8)

def set_all_lrs(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = float(lr)

def backbone_cosine_lr(epoch: int, total_epochs: int, warmup_epochs: int, lr_max: float, lr_init: float) -> float:
    if epoch < warmup_epochs:
        t = float(epoch + 1) / float(max(1, warmup_epochs))
        return lr_init + (lr_max - lr_init) * t
    e = epoch - warmup_epochs
    T = max(1, total_epochs - warmup_epochs)
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * (e / T)))

# =============================================================
# model
# =============================================================

class ConvBlock(nn.Module):
    """3D Residual Block：Conv3d → BN → GELU → Conv3d → BN → GELU → Dropout3d + residual"""

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


def create_cnn3d_from_params(params: Dict[str, Any], input_dim: int, nx: int, ny: int, nz: int) -> nn.Module:

    depth = 3
    base_ch = 24
    dropout_p = float(params.get("dropout_p", 0.1))

    class CNN3D_NoFiLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.nx, self.ny, self.nz = nx, ny, nz
            self.depth = depth
            self.base_ch = base_ch

            # 固定输入：7 通道体素输入
            assert VOXEL_INPUT is not None, (
                "VOXEL_INPUT 尚未初始化：请先读取 cnn_input_channels_no_normals.csv 并设置 VOXEL_INPUT"
            )
            self.register_buffer("voxel_input", VOXEL_INPUT)  # (1,7,nx,ny,nz)

            # 监督/门控 mask：使用 C0（inside_mask）
            assert GEOM_MASK is not None, "GEOM_MASK 尚未初始化：请先从 C0 构造并设置 GEOM_MASK"
            self.register_buffer("geom_mask", GEOM_MASK)  # (1,1,nx,ny,nz)

            # ===== Remove stride-2 stem; use full-res light block + MaxPool =====
            in_ch_in = 7  # 只输入 voxel，不在入口拼 bc

            # full-res channels kept small to avoid OOM (same idea as FiLM version)
            c0 = max(8, base_ch // 2)

            # Full-res "pre-encoder" block: (13 -> c0) at resolution /1
            self.enc0_full = ConvBlock(in_ch_in, c0, dropout_p=dropout_p)  # res: /1

            # Downsample to /2 using pooling (less aliasing than stride-2 conv)
            self.pool0 = nn.MaxPool3d(kernel_size=2, stride=2)  # res: /2

            # Encoder (depth=3), keep the rest aligned
            self.enc0 = ConvBlock(c0, base_ch, dropout_p=dropout_p)  # res: /2
            self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # res: /4
            self.enc1 = ConvBlock(base_ch, base_ch * 2, dropout_p=dropout_p)  # res: /4
            self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # res: /8
            self.enc2 = ConvBlock(base_ch * 2, base_ch * 4, dropout_p=dropout_p)  # res: /8

            # Bottleneck (/8)
            bottleneck_ch = base_ch * 4
            self.bottleneck = ConvBlock(bottleneck_ch + 6, bottleneck_ch, dropout_p=dropout_p)  # ✅ bottleneck 拼 bc

            # Decoder
            self.up2_conv = ConvBlock(bottleneck_ch + base_ch * 2, base_ch * 2, dropout_p=dropout_p)  # /8 -> /4
            self.up1_conv = ConvBlock(base_ch * 2 + base_ch, base_ch, dropout_p=dropout_p)  # /4 -> /2

            self.out_proj = nn.Conv3d(base_ch, base_ch, kernel_size=1)

            # ===== NEW: Decoder stage to full resolution (/2 -> /1) using skip from enc0_full =====
            self.up0_conv = ConvBlock(base_ch + c0, c0, dropout_p=dropout_p)  # /2 -> /1

            # Final prediction at full resolution
            self.final_conv = nn.Conv3d(c0, 1, kernel_size=1)

        def forward(self, bc: torch.Tensor) -> torch.Tensor:
            B = bc.size(0)

            vox = self.voxel_input.expand(B, -1, -1, -1, -1)  # (B,7,nx,ny,nz)
            mask_ch = self.geom_mask.expand(B, -1, -1, -1, -1)  # (B,1,nx,ny,nz)

            # ---- Full-res pre-encoder ---- (入口不拼 bc)
            x_full_skip = self.enc0_full(vox)  # (B,c0,nx,ny,nz)

            # ---- Downsample to /2 ----
            x = self.pool0(x_full_skip)  # (B,c0,nx/2,ny/2,nz/2)

            # ---- Encoder ----
            x0 = self.enc0(x)  # (B,base_ch, /2)
            x1 = self.pool1(x0)
            x1 = self.enc1(x1)  # (B,2C, /4)
            x2 = self.pool2(x1)
            x2 = self.enc2(x2)  # (B,4C, /8)

            # Bottleneck: 在 /8 分辨率拼接 bc (6 通道)
            bc_grid_8 = bc.view(B, 6, 1, 1, 1).expand(B, 6, x2.shape[2], x2.shape[3], x2.shape[4])
            x2_cat = torch.cat([x2, bc_grid_8], dim=1)  # (B, 4C+6, /8)
            xb = self.bottleneck(x2_cat)  # (B, 4C, /8)

            # Decoder
            x_up2 = F.interpolate(xb, size=x1.shape[2:], mode="trilinear", align_corners=False)
            x_cat2 = torch.cat([x_up2, x1], dim=1)
            x_dec2 = self.up2_conv(x_cat2)  # /4

            x_up1 = F.interpolate(x_dec2, size=x0.shape[2:], mode="trilinear", align_corners=False)
            x_cat1 = torch.cat([x_up1, x0], dim=1)
            x_dec1 = self.up1_conv(x_cat1)  # /2

            x_dec = self.out_proj(x_dec1)  # (B,base_ch,nx/2,ny/2,nz/2)

            # ---- Decoder to full resolution with skip from enc0_full ----
            x_up0 = F.interpolate(x_dec, size=x_full_skip.shape[2:], mode="trilinear", align_corners=False)  # /2 -> /1
            x_cat0 = torch.cat([x_up0, x_full_skip], dim=1)  # (B, base_ch + c0, nx, ny, nz)
            x0_full = self.up0_conv(x_cat0)  # (B, c0, nx, ny, nz)

            x_full = self.final_conv(x0_full)  # (B,1,nx,ny,nz)

            out = x_full.squeeze(1)
            out = out * mask_ch.squeeze(1)  # 输出门控：仅 inside 区域
            return out

    return CNN3D_NoFiLM().to(device)


# =============================================================
# Data
# =============================================================
def load_data():
    global GEOM_MASK, VOXEL_INPUT

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

    # Temp
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
        train_idx_raw = np.where(split == "train")[0]
        val_idx_raw = np.where((split == "val") | (split == "valid") | (split == "validation"))[0]
        test_idx_final = np.where(split == "test")[0]
    else:
        train_idx_raw = np.where(split_raw == 0)[0]
        val_idx_raw = np.where(split_raw == 1)[0]
        test_idx_final = np.where(split_raw == 2)[0]

    trainval_idx = np.sort(np.concatenate([train_idx_raw, val_idx_raw], axis=0))
    print(f"原始划分: train={len(train_idx_raw)}, val={len(val_idx_raw)}, test={len(test_idx_final)}")
    print(f"合并训练集(train+val)={len(trainval_idx)}")

    # Scale X
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_data)

    # Z-score Y per-sample on valid points
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

    # Slices
    x_train_raw = X_scaled[train_idx_raw]
    y_train_raw = Y_scaled[train_idx_raw]
    m_train_raw = mask_valid[train_idx_raw]

    x_val_raw = X_scaled[val_idx_raw]
    y_val_raw = Y_scaled[val_idx_raw]
    m_val_raw = mask_valid[val_idx_raw]

    x_trainval = X_scaled[trainval_idx]
    y_trainval = Y_scaled[trainval_idx]
    m_trainval = mask_valid[trainval_idx]

    x_test = X_scaled[test_idx_final]
    y_test = Y_scaled[test_idx_final]
    m_test = mask_valid[test_idx_final]

    input_dim = x_trainval.shape[1]
    print(f"输入维度: {input_dim}")
    print(f"train_raw={x_train_raw.shape[0]}, val_raw={x_val_raw.shape[0]}, trainval={x_trainval.shape[0]}, test={x_test.shape[0]}")

    # Torch tensors (CPU)
    t = dict(
        x_train_raw_t=torch.from_numpy(x_train_raw).float(),
        y_train_raw_t=torch.from_numpy(y_train_raw).float(),
        m_train_raw_t=torch.from_numpy(m_train_raw).float(),
        x_val_raw_t=torch.from_numpy(x_val_raw).float(),
        y_val_raw_t=torch.from_numpy(y_val_raw).float(),
        m_val_raw_t=torch.from_numpy(m_val_raw).float(),
        x_trainval_t=torch.from_numpy(x_trainval).float(),
        y_trainval_t=torch.from_numpy(y_trainval).float(),
        m_trainval_t=torch.from_numpy(m_trainval).float(),
        x_test_t=torch.from_numpy(x_test).float(),
        y_test_t=torch.from_numpy(y_test).float(),
        m_test_t=torch.from_numpy(m_test).float(),
    )

    return dict(
        nx=nx, ny=ny, nz=nz,
        input_dim=input_dim,
        scaler_x=scaler_x,
        Y_means=Y_means,
        Y_stds=Y_stds,
        geom_mask_np=geom_mask_np,
        lin_valid=lin_valid,
        voxel_cols=col_order,
        **t,
    )
# =============================================================
# Proxy config（和 FiLM 一致）
# =============================================================
PROXY_EPOCHS = 30
PROXY_MAX_STEPS_PER_EPOCH = 16
PROXY_PATIENCE = 6
PROXY_MIN_DELTA = 1e-6


# =============================================================
# Optuna search (proxy training + val eval)  —— aligned with FiLM
# =============================================================
def run_optuna_search(data: Dict[str, Any], *, n_trials: int, params_json_path: str):
    print("开始 Optuna 超参搜索（proxy training on train_raw, eval on val_raw）...")

    nx, ny, nz = data["nx"], data["ny"], data["nz"]
    input_dim = data["input_dim"]

    x_train_raw_t = data["x_train_raw_t"]
    y_train_raw_t = data["y_train_raw_t"]
    m_train_raw_t = data["m_train_raw_t"]
    x_val_raw_t = data["x_val_raw_t"]
    y_val_raw_t = data["y_val_raw_t"]
    m_val_raw_t = data["m_val_raw_t"]

    pin_memory = (device.type == "cuda")

    def objective(trial: optuna.trial.Trial):
        # ✅ 搜索阶段“不固定 seed”：不 set_seed，不固定 generator seed
        # 让不同 trial 的 shuffle/初始化噪声自然存在
        g_trial = torch.Generator()
        g_trial.seed()  # random seed from internal entropy

        # ---- model hyperparams to search (Concat usually less; keep same pattern) ----
        params = {
            "dropout_p": trial.suggest_float("dropout_p", 0.0, 0.3),
            # 如果 Concat 版本还有别的结构超参，也放这里，保持 FiLM 风格
            # e.g. "concat_scale": trial.suggest_float(...)
        }

        # lr params are also searched
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        params["lr"] = float(lr)

        model = create_cnn3d_from_params(params, input_dim, nx, ny, nz)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_loader = DataLoader(
            TensorDataset(x_train_raw_t, y_train_raw_t, m_train_raw_t),
            batch_size=16,
            shuffle=True,
            generator=g_trial,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            TensorDataset(x_val_raw_t, y_val_raw_t, m_val_raw_t),
            batch_size=16,
            shuffle=False,
            pin_memory=pin_memory,
        )

        best_val = float("inf")
        no_improve = 0

        for epoch in range(PROXY_EPOCHS):
            model.train()
            for step_i, (xb, yb, mb) in enumerate(train_loader):
                if step_i >= PROXY_MAX_STEPS_PER_EPOCH:
                    break
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)  # Concat: xb(B,6) -> inside model replicate/concat
                loss = masked_loss(pred, yb, mb)
                loss.backward()
                optimizer.step()

            model.eval()
            vtot = 0.0
            with torch.no_grad():
                for xb, yb, mb in val_loader:
                    xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                    pred = model(xb)
                    vloss = masked_loss(pred, yb, mb)
                    vtot += vloss.item() * xb.size(0)
            val_loss = vtot / len(val_loader.dataset)

            # --- early-stop + pruning (FIXED, aligned) ---
            prev_best = best_val  # keep previous best BEFORE update

            # update best/plateau counters
            if val_loss < prev_best - PROXY_MIN_DELTA:
                best_val = val_loss
                no_improve = 0
            else:
                no_improve += 1

            # report a monotonic signal (best so far) for more stable pruning
            trial.report(best_val, step=epoch)
            if trial.should_prune():
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned()

            # early stop after reporting (so Optuna has the last metric)
            if no_improve >= PROXY_PATIENCE:
                break

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return float(best_val)

    # ✅ 搜索阶段不固定 sampler seed（aligned）
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=8, interval_steps=1)

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print("Optuna best_params:", best_params)

    os.makedirs(os.path.dirname(params_json_path) or ".", exist_ok=True)
    with open(params_json_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    print(f"已保存 best_params 到: {params_json_path}")

    return best_params


def load_best_params(params_json_path: str) -> Dict[str, Any]:
    if not os.path.isfile(params_json_path):
        raise FileNotFoundError(f"找不到 params_json_path: {params_json_path}（请先 run_mode=search 生成它）")
    with open(params_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================
# Final training (fixed seed) —— aligned with FiLM
# =============================================================
def train_final(data: Dict[str, Any], best_params: Dict[str, Any], *, seed: int, out_prefix: str):
    # ✅ 训练阶段固定 seed（你要的鲁棒性评估入口）
    set_seed(seed, deterministic=False)

    nx, ny, nz = data["nx"], data["ny"], data["nz"]
    input_dim = data["input_dim"]

    x_trainval_t = data["x_trainval_t"]
    y_trainval_t = data["y_trainval_t"]
    m_trainval_t = data["m_trainval_t"]

    scaler_x = data["scaler_x"]
    Y_means = data["Y_means"]
    Y_stds = data["Y_stds"]
    geom_mask_np = data["geom_mask_np"]
    lin_valid = data["lin_valid"]
    voxel_cols = data["voxel_cols"]

    # ---- model params (aligned structure) ----
    params_model = {
        "dropout_p": float(best_params.get("dropout_p", 0.1)),
    }

    model = create_cnn3d_from_params(params_model, input_dim, nx, ny, nz)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    best_lr = float(best_params.get("lr", 1e-3))
    optimizer = optim.Adam(model.parameters(), lr=best_lr)

    g_train = torch.Generator()
    g_train.manual_seed(seed)

    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(
        TensorDataset(x_trainval_t, y_trainval_t, m_trainval_t),
        batch_size=16,
        shuffle=True,
        generator=g_train,
        pin_memory=pin_memory,
    )

    total_epochs = 300
    warmup_epochs = 20

    initial_lr = best_lr * 0.1
    set_all_lrs(optimizer, initial_lr)

    model_core = model.module if isinstance(model, nn.DataParallel) else model

    print(f"===== Final train: seed={seed}, fixed 300 epochs on (train+val) =====")

    for epoch in range(total_epochs):
        model.train()
        run = 0.0
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = masked_loss(pred, yb, mb)
            loss.backward()
            optimizer.step()
            run += loss.item() * xb.size(0)

        train_loss = run / len(train_loader.dataset)

        lr = backbone_cosine_lr(
            epoch=epoch,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            lr_max=best_lr,
            lr_init=best_lr * 0.1,
        )
        set_all_lrs(optimizer, lr)

        if epoch % 10 == 0 or epoch == total_epochs - 1:
            print(
                f"[Final] Epoch {epoch:03d}, lr={lr:.3e}, "
                f"train_loss={train_loss:.6f}"
            )

    # --- save lastTrain only (as you requested) ---
    save_common = {
        "input_dim": input_dim,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "depth": 3,
        "base_ch": 24,
        "dropout_p": params_model["dropout_p"],
        "model_type": "CNN_Concat_sdf+bm_GN_RemoveStem_Bottleneck",
        "lr": float(best_lr),
        "x_mean": scaler_x.mean_,
        "x_scale": scaler_x.scale_,
        "Y_means": Y_means,
        "Y_stds": Y_stds,
        "geom_mask_np": geom_mask_np,
        "lin_valid": lin_valid,
        "voxel_cols": voxel_cols,
        "train_only": True,
        "total_epochs": int(total_epochs),
        "seed": int(seed),
        "best_params": best_params,
        "optuna_proxy_epochs": int(PROXY_EPOCHS),
        "optuna_proxy_max_steps_per_epoch": int(PROXY_MAX_STEPS_PER_EPOCH),
    }

    last_path = f"{out_prefix}_seed{seed}_lastTrain.pth"
    torch.save({"state_dict": model_core.state_dict(), **save_common}, last_path)
    print(f"模型已保存: {last_path}")


# =============================================================
# Main —— aligned with FiLM
# =============================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_mode", type=str, choices=["search", "train"], required=True,
                    help="search: run optuna once and save best_params json; train: load json and train.")
    ap.add_argument("--seed", type=int, default=43, help="Training seed (only used in run_mode=train).")
    ap.add_argument("--n_trials", type=int, default=20, help="Optuna trials (only used in run_mode=search).")
    ap.add_argument("--params_json", type=str, default="best_params_CNN_Concat_sdf+bm_GN_RemoveStem_Bottleneck.json",
                    help="Path to save/load best_params JSON.")
    ap.add_argument("--out_prefix", type=str, default="CNN_Concat_sdf+bm_GN_RemoveStem_Bottleneck",
                    help="Prefix for output ckpt files (seed will be appended).")
    return ap.parse_args()

def main():
    args = parse_args()

    # 不要在 import 时就固定 seed —— 由 run_mode 决定
    data = load_data()

    if args.run_mode == "search":
        # 搜索阶段不固定 seed：不调用 set_seed()
        run_optuna_search(data, n_trials=args.n_trials, params_json_path=args.params_json)
        print("Search done. You can now run train mode with fixed seed(s).")
        return

    # train mode
    best_params = load_best_params(args.params_json)
    train_final(data, best_params, seed=args.seed, out_prefix=args.out_prefix)

if __name__ == "__main__":
    main()
# CUDA_VISIBLE_DEVICES=2,3,4 python CNN_Concat_sdf+bm_GN_RemoveStem_Bottleneck.py --run_mode search --n_trials 20 --params_json best_params_CNN_Concat_sdf+bm_GN_RemoveStem_Bottleneck.json
# CUDA_VISIBLE_DEVICES=2,4 python CNN_Concat_sdf+bm_GN_RemoveStem_Bottleneck.py --run_mode train --seed 43 --params_json best_params_CNN_Concat_sdf+bm_GN_RemoveStem_Bottleneck.json --out_prefix CNN_Concat_sdf+bm_GN_RemoveStem_Bottleneck
# nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | xargs ps -fp
# nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
