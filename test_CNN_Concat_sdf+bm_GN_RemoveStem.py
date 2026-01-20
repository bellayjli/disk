#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================
# test_3dcnn_concat_voxel7.py
# 加载训练好的 Concat(voxel7) 3D UNet 模型并在测试集上评估
#
# ✅ 与 FiLM 测试脚本逐句对齐的改造版本（方便肉眼 diff）
#
# 与最新版训练脚本（Concat / CNN3D_NoFiLM depth=3 RemoveStem）对齐要点：
# - 模型：Concat-UNet(depth=3)，固定 7 通道体素输入（来自 cnn_input_channels_no_normals.csv）
# - BC 注入：入口 full-res 将 bc(6) broadcast 成 (B,6,nx,ny,nz) 与 vox(7) concat => 13 通道
# - “RemoveStem：full-res light block(enc0_full) + MaxPool 下采样”
# - “末端：/2→/1 上采样并 concat(enc0_full skip)，再 final_conv 输出”
# - 监督/门控 mask：使用 C0(inside_mask)
# - X: StandardScaler（使用训练时保存的 mean/scale）
# - Y: 样本级 Z-score（使用训练时保存的每个样本 mean/std；只在有效点上归一化）
# - 指标：按样本输出 NRMSE / MAE / MSE / R2 / GradMSE（仅在 mask==1 的点上）
# - 可选导出：.npy / 有效点 CSV / ✅整场 full-grid CSV
# =============================================================

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# --------------------- 设备 ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 提前初始化 CUDA 上下文，避免第一次使用 cuBLAS 时出现 "no current CUDA context" warning
if device.type == "cuda":
    torch.cuda.init()
    print(f"CUDA devices: {torch.cuda.device_count()} visible.")

# -------------------------------------------------------------
# 全局：体素输入 + mask（测试脚本运行时从 voxel csv 构造）
# -------------------------------------------------------------
GEOM_MASK = None   # (1,1,nx,ny,nz)
VOXEL_INPUT = None # (1,7,nx,ny,nz)

def make_gn(C: int, max_groups: int = 8) -> nn.GroupNorm:
    G = min(max_groups, C)
    while C % G != 0:
        G -= 1
    return nn.GroupNorm(G, C)

# =============================================================
# 模型定义（对齐 Concat 训练脚本）
# =============================================================

class ConvResidualBlock(nn.Module):
    """(Conv -> GN -> GELU) x2 -> Dropout3d + residual"""

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = make_gn(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = make_gn(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout3d(dropout_p)

        self.residual_proj = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)

        out = self.drop(out)
        return out + residual


def build_model_from_ckpt(ckpt: dict) -> nn.Module:
    """从 CNN_Concat_*ckpt 重建最新版 Concat(voxel7) 网络结构。"""

    model_type = str(ckpt.get("model_type", "NoFiLM_BCAsVoxels_no_stem_pooldown_fullres_light"))

    input_dim = int(ckpt.get("input_dim", 6))
    if input_dim != 6:
        print(f"[warn] ckpt.input_dim={input_dim}，Concat 版本通常应为 6（bc 维度）。")

    nx = int(ckpt["nx"])
    ny = int(ckpt["ny"])
    nz = int(ckpt["nz"])

    depth = int(ckpt.get("depth", 3))
    if depth != 3:
        raise ValueError(f"该测试脚本对齐的是 depth=3，但 ckpt.depth={depth}")

    base_ch = int(ckpt.get("base_ch", 24))
    dropout_p = float(ckpt.get("dropout_p", 0.1))

    class CNN3D_Concat(nn.Module):
        def __init__(self):
            super().__init__()
            self.nx, self.ny, self.nz = nx, ny, nz
            self.depth = 3
            self.base_ch = base_ch

            # 固定输入：7 通道体素输入 + mask
            assert VOXEL_INPUT is not None, "VOXEL_INPUT 未初始化：请先读取 cnn_input_channels_no_normals.csv"
            assert GEOM_MASK is not None, "GEOM_MASK 未初始化：请先从 C0 构造 mask"
            self.register_buffer("voxel_input", VOXEL_INPUT)  # (1,7,nx,ny,nz)
            self.register_buffer("geom_mask", GEOM_MASK)      # (1,1,nx,ny,nz)

            # ===== full-res light block + MaxPool =====
            c0 = max(8, base_ch // 2)  # 与 FiLM 测试脚本一致的 c0 规则
            self.c0 = c0

            # full-res "pre-encoder" block: (13 -> c0) at resolution /1
            # 13 = 7 voxel + 6 bc
            self.enc0_full = ConvResidualBlock(13, c0, dropout_p=dropout_p)  # res: /1

            # Downsample to /2
            self.pool0 = nn.MaxPool3d(kernel_size=2, stride=2)  # res: /2

            # Encoder
            self.enc0 = ConvResidualBlock(c0, base_ch, dropout_p=dropout_p)  # /2
            self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)              # /4
            self.enc1 = ConvResidualBlock(base_ch, base_ch * 2, dropout_p=dropout_p)   # /4
            self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)              # /8
            self.enc2 = ConvResidualBlock(base_ch * 2, base_ch * 4, dropout_p=dropout_p)  # /8

            # Bottleneck
            bottleneck_ch = base_ch * 4
            self.bottleneck = ConvResidualBlock(bottleneck_ch, bottleneck_ch, dropout_p=dropout_p)

            # Decoder
            self.up2_conv = ConvResidualBlock(bottleneck_ch + base_ch * 2, base_ch * 2, dropout_p=dropout_p)  # /8->/4
            self.up1_conv = ConvResidualBlock(base_ch * 2 + base_ch, base_ch, dropout_p=dropout_p)            # /4->/2

            self.out_proj = nn.Conv3d(base_ch, base_ch, kernel_size=1)

            # /2 -> /1 + concat enc0_full skip
            self.up0_conv = ConvResidualBlock(base_ch + c0, c0, dropout_p=dropout_p)
            self.final_conv = nn.Conv3d(c0, 1, kernel_size=1)

        def forward(self, bc: torch.Tensor) -> torch.Tensor:
            B = bc.size(0)
            vox = self.voxel_input.expand(B, -1, -1, -1, -1)    # (B,7,nx,ny,nz)
            mask_ch = self.geom_mask.expand(B, -1, -1, -1, -1)  # (B,1,nx,ny,nz)

            # bc broadcast
            bc_grid = bc.view(B, 6, 1, 1, 1).expand(B, 6, self.nx, self.ny, self.nz)  # (B,6,nx,ny,nz)

            # ---- Full-res pre-encoder ----
            x_in_full = torch.cat([vox, bc_grid], dim=1)  # (B,13,nx,ny,nz)
            x_full_skip = self.enc0_full(x_in_full)       # (B,c0,nx,ny,nz)

            # ---- Downsample to /2 ----
            x = self.pool0(x_full_skip)  # /2

            # ---- Encoder ----
            x0 = self.enc0(x)                           # /2
            x1 = self.pool1(x0)
            x1 = self.enc1(x1)                          # /4
            x2 = self.pool2(x1)
            x2 = self.enc2(x2)                          # /8

            # ---- Bottleneck ----
            xb = self.bottleneck(x2)                    # /8

            # ---- Decoder ----
            x_up2 = F.interpolate(xb, size=x1.shape[2:], mode="trilinear", align_corners=False)  # /8->/4
            x_dec2 = self.up2_conv(torch.cat([x_up2, x1], dim=1))                                 # /4

            x_up1 = F.interpolate(x_dec2, size=x0.shape[2:], mode="trilinear", align_corners=False)  # /4->/2
            x_dec1 = self.up1_conv(torch.cat([x_up1, x0], dim=1))                                     # /2

            x_dec = self.out_proj(x_dec1)  # /2

            # ---- Decoder to full resolution with skip from enc0_full ----
            x_up0 = F.interpolate(x_dec, size=x_full_skip.shape[2:], mode="trilinear", align_corners=False)  # /2->/1
            x0_full = self.up0_conv(torch.cat([x_up0, x_full_skip], dim=1))                                    # /1
            x_full = self.final_conv(x0_full)  # (B,1,nx,ny,nz)

            out = x_full.squeeze(1)
            out = out * mask_ch.squeeze(1)
            return out

    return CNN3D_Concat()

# =============================================================
# Masked Loss（与 FiLM 测试脚本一致；用于在测试集上报告 scaled-space loss）
# =============================================================

def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """与训练脚本一致：masked smooth_l1（scaled space）"""
    per_elem = F.smooth_l1_loss(pred, target, reduction="none")
    masked = per_elem * mask
    return masked.sum() / (mask.sum() + 1e-8)

# =============================================================
# 指标（仅在 mask==1 的真实点上计算）——与 FiLM 测试脚本一致
# =============================================================

def _masked_metrics_per_sample(pred_np: np.ndarray, true_np: np.ndarray, mask_np: np.ndarray):
    """pred/true/mask: (nx,ny,nz) -> nrmse, mae, mse, r2, grad_mse"""
    m = mask_np.astype(bool)
    y = true_np[m]
    p = pred_np[m]

    if y.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    diff = p - y
    mse = float(np.mean(diff * diff))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    denom = float(np.max(y) - np.min(y))
    nrmse = float(rmse / (denom + 1e-8))

    sse = float(np.sum((p - y) ** 2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - sse / (sst + 1e-12))

    # Gradient MSE：只在相邻两点都为有效点的位置计算（forward diff）
    def _grads(a: np.ndarray):
        gx = a[1:, :, :] - a[:-1, :, :]
        gy = a[:, 1:, :] - a[:, :-1, :]
        gz = a[:, :, 1:] - a[:, :, :-1]
        return gx, gy, gz

    pgx, pgy, pgz = _grads(pred_np)
    ygx, ygy, ygz = _grads(true_np)

    mgx = mask_np[1:, :, :] * mask_np[:-1, :, :]
    mgy = mask_np[:, 1:, :] * mask_np[:, :-1, :]
    mgz = mask_np[:, :, 1:] * mask_np[:, :, :-1]

    grad_mse_sum = 0.0
    grad_cnt = 0
    for pd, td, md in [(pgx, ygx, mgx), (pgy, ygy, mgy), (pgz, ygz, mgz)]:
        valid = md.astype(bool)
        if np.any(valid):
            d = pd[valid] - td[valid]
            grad_mse_sum += float(np.mean(d * d))
            grad_cnt += 1

    grad_mse = grad_mse_sum / max(grad_cnt, 1)
    return nrmse, mae, mse, r2, grad_mse

# =============================================================
# 参数统计（与 FiLM 测试脚本一致）
# =============================================================

def _format_int(n: int) -> str:
    return f"{n:,}"

def count_parameters(m: nn.Module):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

def parameter_breakdown(m: nn.Module):
    owner_type = {}
    owner_name = {}
    for mod_name, mod in m.named_modules():
        for _, p in mod.named_parameters(recurse=False):
            owner_type[id(p)] = mod.__class__.__name__
            owner_name[id(p)] = mod_name

    by_type_total = defaultdict(int)
    by_type_train = defaultdict(int)
    by_top_total = defaultdict(int)
    by_top_train = defaultdict(int)

    for _, p in m.named_parameters():
        n = p.numel()
        t = owner_type.get(id(p), "<Unknown>")
        mod_name = owner_name.get(id(p), "")
        top = mod_name.split(".")[0] if mod_name else "<root>"

        by_type_total[t] += n
        by_top_total[top] += n
        if p.requires_grad:
            by_type_train[t] += n
            by_top_train[top] += n

    return by_type_total, by_type_train, by_top_total, by_top_train

# =============================================================
# 结果保存：预测场 + 有效点 CSV + ✅整场 CSV（与 FiLM 测试脚本一致）
# =============================================================

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _lin_to_ijk(lin: np.ndarray, ny: int, nz: int):
    lin = lin.astype(np.int64)
    ix = lin // (ny * nz)
    rem = lin % (ny * nz)
    iy = rem // nz
    iz = rem % nz
    return ix, iy, iz

def _full_grid_xyz_columns(x_unique: np.ndarray, y_unique: np.ndarray, z_unique: np.ndarray):
    """
    生成整场 (x,y,z) 三列（长度 = nx*ny*nz），顺序与 pred_real.ravel(order='C') 对齐：
    - z 最快变化，其次 y，其次 x
    - 对应索引：lin = ix*(ny*nz) + iy*nz + iz
    """
    x_unique = np.asarray(x_unique)
    y_unique = np.asarray(y_unique)
    z_unique = np.asarray(z_unique)

    nx = int(x_unique.size)
    ny = int(y_unique.size)
    nz = int(z_unique.size)
    n = nx * ny * nz

    xs = np.repeat(x_unique, ny * nz)
    ys = np.tile(np.repeat(y_unique, nz), nx)
    zs = np.tile(z_unique, nx * ny)

    assert xs.size == n and ys.size == n and zs.size == n
    return xs, ys, zs

def save_prediction_artifacts(
    out_dir: str,
    sample_id: int,
    pred_real: np.ndarray,
    true_real: np.ndarray,
    mask: np.ndarray,
    lin_valid: np.ndarray,
    x_unique: np.ndarray,
    y_unique: np.ndarray,
    z_unique: np.ndarray,
    save_npy: bool = False,
    save_csv_valid_points: bool = False,
    save_csv_full_grid: bool = False,
    csv_fullgrid_chunksize: int = 1_000_000,
):
    """保存：
    - （可选）全网格预测/真值/误差/mask 为 .npy
    - （可选）只导出有效点的 CSV：x,y,z,true,pred,err
    - （可选）整场 CSV：x,y,z,Temp_pred,Temp_true,Temp_err,mask
      注：脚本外部已把 mask==0 的点置零，所以 Temp_* 在无信息点上为 0。
    """
    _ensure_dir(out_dir)

    if save_npy:
        np.save(os.path.join(out_dir, f"sample_{sample_id:04d}_pred.npy"), pred_real.astype(np.float32))
        np.save(os.path.join(out_dir, f"sample_{sample_id:04d}_true.npy"), true_real.astype(np.float32))
        np.save(os.path.join(out_dir, f"sample_{sample_id:04d}_err.npy"), (pred_real - true_real).astype(np.float32))
        np.save(os.path.join(out_dir, f"sample_{sample_id:04d}_mask.npy"), mask.astype(np.float32))

    if save_csv_valid_points:
        ix, iy, iz = _lin_to_ijk(lin_valid, ny=len(y_unique), nz=len(z_unique))
        xs = x_unique[ix]
        ys = y_unique[iy]
        zs = z_unique[iz]

        p = pred_real[ix, iy, iz]
        y = true_real[ix, iy, iz]
        e = p - y

        df = pd.DataFrame(
            {
                "x": xs.astype(np.float32),
                "y": ys.astype(np.float32),
                "z": zs.astype(np.float32),
                "true": y.astype(np.float32),
                "pred": p.astype(np.float32),
                "err": e.astype(np.float32),
            }
        )
        df.to_csv(os.path.join(out_dir, f"sample_{sample_id:04d}_valid_points.csv"), index=False)

    if save_csv_full_grid:
        xs, ys, zs = _full_grid_xyz_columns(x_unique, y_unique, z_unique)

        pred_flat = pred_real.astype(np.float32, copy=False).ravel(order="C")
        true_flat = true_real.astype(np.float32, copy=False).ravel(order="C")
        mask_flat = mask.astype(np.float32, copy=False).ravel(order="C")
        err_flat = (pred_flat - true_flat).astype(np.float32, copy=False)

        df_full = pd.DataFrame(
            {
                "x": xs.astype(np.float32, copy=False),
                "y": ys.astype(np.float32, copy=False),
                "z": zs.astype(np.float32, copy=False),
                "Temp_pred": pred_flat,
                "Temp_true": true_flat,
                "Temp_err": err_flat,
                "mask": mask_flat,
            }
        )
        out_path = os.path.join(out_dir, f"sample_{sample_id:04d}_fullgrid.csv")
        df_full.to_csv(out_path, index=False, chunksize=csv_fullgrid_chunksize)

# =============================================================
# 主流程：加载 ckpt → 读取 voxel/temp/bc → 构建 test set → 推理评估
# =============================================================

CKPT_PATH = "CNN_Concat_sdf+bm_GN_RemoveStem_seed44_lastTrain.pth"
datapath_bc = "data/boundary_condition.csv"
datapath_temp = "data/Temp_all.csv"
datapath_voxel = "data/cnn_input_channels_no_normals.csv"

# ===================== 预测保存设置 =====================
SAVE_DIR = "test_outputs_concat_voxel7"

SAVE_PRED_NPY = False              # 需要 .npy 就打开
SAVE_VALID_POINTS_CSV = False      # 仅导出有效点（x,y,z,true,pred,err）
SAVE_FULL_GRID_CSV = False         # ✅整场 full-grid CSV（x,y,z,Temp_pred,Temp_true,Temp_err,mask）

CSV_FULLGRID_CHUNKSIZE = 1_000_000
# =======================================================

print(f"Loading checkpoint: {CKPT_PATH}")
# ⚠️ 注意：weights_only=False 可能带来任意代码执行风险，只对可信 ckpt 使用。
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

print("===== ckpt meta =====")
print(
    f"model_type={ckpt.get('model_type','NA')}, depth={ckpt.get('depth','NA')}, base_ch={ckpt.get('base_ch','NA')}, "
    f"dropout_p={ckpt.get('dropout_p','NA')}"
)
print(
    f"x_mean={'OK' if ckpt.get('x_mean', None) is not None else 'NA'}, "
    f"Y_means={'OK' if ckpt.get('Y_means', None) is not None else 'NA'}"
)

# -------------------------------------------------------------
# 读取 7 通道体素输入（与 FiLM 测试脚本一致）
# -------------------------------------------------------------
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

# 与 ckpt 对齐（强约束，避免 silent mismatch）
if (nx, ny, nz) != (int(ckpt["nx"]), int(ckpt["ny"]), int(ckpt["nz"])):
    raise ValueError(
        f"voxel 网格 {(nx, ny, nz)} 与 ckpt 网格 {(int(ckpt['nx']), int(ckpt['ny']), int(ckpt['nz']))} 不一致。"
    )

x_index = {float(v): i for i, v in enumerate(x_unique)}
y_index = {float(v): i for i, v in enumerate(y_unique)}
z_index = {float(v): i for i, v in enumerate(z_unique)}

# 7 通道输入顺序：使用 ckpt 保存的 voxel_cols（训练脚本保存）；否则用默认顺序
col_order = ckpt.get("voxel_cols", ["C0", "C1", "C2", "C3", "C4", "C5", "sdf"])

missing_ch = [c for c in col_order if c not in df_vox.columns]
if missing_ch:
    raise KeyError(f"voxel_cols 中包含不存在的列: {missing_ch}；df_vox.columns={list(df_vox.columns)}")

voxel_grid = np.zeros((len(col_order), nx, ny, nz), dtype=np.float32)
cols_np = [df_vox[c].to_numpy(dtype=np.float32) for c in col_order]

for i in range(df_vox.shape[0]):
    ix = x_index[float(xv[i])]
    iy = y_index[float(yv[i])]
    iz = z_index[float(zv[i])]
    voxel_grid[:, ix, iy, iz] = np.array([c[i] for c in cols_np], dtype=np.float32)

# VOXEL_INPUT 先构建（训练端固定 7 通道体素输入）
VOXEL_INPUT = torch.tensor(voxel_grid[None, ...], dtype=torch.float32, device=device)

# mask：优先使用 ckpt["geom_mask_np"]；否则从 C0 重建
if "geom_mask_np" in ckpt and ckpt["geom_mask_np"] is not None:
    geom_mask_np = np.asarray(ckpt["geom_mask_np"], dtype=np.float32)
    if geom_mask_np.shape != (nx, ny, nz):
        raise ValueError(f"ckpt.geom_mask_np shape={geom_mask_np.shape} != {(nx,ny,nz)}")
else:
    if "C0" not in col_order:
        raise ValueError(f"voxel_cols 必须包含 C0 用于 mask，但现在是: {col_order}")
    geom_mask_np = (voxel_grid[col_order.index("C0")] > 0.5).astype(np.float32)

GEOM_MASK = torch.tensor(geom_mask_np[None, None, ...], dtype=torch.float32, device=device)
print(f"inside_mask 占比: {geom_mask_np.mean() * 100:.3f}%")

# valid 点：优先使用 ckpt["lin_valid"]；否则从 mask 推导
if "lin_valid" in ckpt and ckpt["lin_valid"] is not None:
    lin = np.asarray(ckpt["lin_valid"], dtype=np.int64)
else:
    lin = np.where(geom_mask_np.reshape(-1) > 0.5)[0].astype(np.int64)

# -------------------------------------------------------------
# 读取温度：支持两种格式（与 FiLM 测试脚本一致）
# A) 仅有效点：长度 = valid_points
# B) 全网格点：长度 = total_points（且顺序与 df_vox 行顺序一致）
# -------------------------------------------------------------
T_np = pd.read_csv(datapath_temp).to_numpy(dtype=np.float32)

valid_points = int(lin.shape[0])
total_points = int(df_vox.shape[0])

def _as_samples_first(a: np.ndarray, n_points: int) -> np.ndarray:
    if a.shape[0] == n_points:
        return a.T
    if a.shape[1] == n_points:
        return a
    raise ValueError(f"Temp_all.csv 维度 {a.shape} 与点数 {n_points} 不匹配。")

if (T_np.shape[0] == valid_points) or (T_np.shape[1] == valid_points):
    Y_valid = _as_samples_first(T_np, valid_points)  # (num_samples, valid_points)
    num_samples = int(Y_valid.shape[0])
    print(f"温度样本数: {num_samples}, 格式=仅有效点, 有效点数: {valid_points}")

    Y_grid_flat = np.zeros((num_samples, nx * ny * nz), dtype=np.float32)
    Y_grid_flat[:, lin] = Y_valid

elif (T_np.shape[0] == total_points) or (T_np.shape[1] == total_points):
    Y_all = _as_samples_first(T_np, total_points)  # (num_samples, total_points)
    num_samples = int(Y_all.shape[0])
    print(f"温度样本数: {num_samples}, 格式=全网格点, 总点数: {total_points}, 有效点数: {valid_points}")

    ix_all = np.array([x_index[float(v)] for v in xv], dtype=np.int64)
    iy_all = np.array([y_index[float(v)] for v in yv], dtype=np.int64)
    iz_all = np.array([z_index[float(v)] for v in zv], dtype=np.int64)
    lin_all = ix_all * (ny * nz) + iy_all * nz + iz_all

    Y_grid_flat = np.zeros((num_samples, nx * ny * nz), dtype=np.float32)
    Y_grid_flat[:, lin_all] = Y_all

else:
    raise ValueError(
        f"Temp_all.csv 维度 {T_np.shape} 既不匹配有效点数 {valid_points}，也不匹配全点数 {total_points}。"
    )

Y_grid = Y_grid_flat.reshape((num_samples, nx, ny, nz))
mask_valid = np.broadcast_to(geom_mask_np[None, ...], (num_samples, nx, ny, nz)).astype(np.float32)

# -------------------------------------------------------------
# 边界条件与 split（与 FiLM 测试脚本一致）
# -------------------------------------------------------------
df_bc = pd.read_csv(datapath_bc)
X_data = df_bc.iloc[:, :6].to_numpy(dtype=np.float32)
split_raw = df_bc.iloc[:, 6].to_numpy()

if split_raw.dtype.kind in "OUS":
    split = np.array([str(s).strip().lower() for s in split_raw])
    test_idx = np.where(split == "test")[0]
else:
    test_idx = np.where(split_raw == 2)[0]

print(f"测试集数量: {len(test_idx)}")

# -------------------------------------------------------------
# 恢复标准化参数（与 FiLM 测试脚本保存一致）
# -------------------------------------------------------------
x_mean = ckpt.get("x_mean", None)
x_scale = ckpt.get("x_scale", None)
if x_mean is None or x_scale is None:
    raise ValueError("ckpt 中未找到 x_mean/x_scale")

Y_means_all = ckpt.get("Y_means", None)
Y_stds_all = ckpt.get("Y_stds", None)
if Y_means_all is None or Y_stds_all is None:
    raise ValueError("ckpt 中未找到 Y_means/Y_stds")
Y_means_all = np.asarray(Y_means_all, dtype=np.float32)
Y_stds_all = np.asarray(Y_stds_all, dtype=np.float32)

# X 标准化（使用训练保存参数）
X_scaled = (X_data - x_mean) / (x_scale + 1e-12)

# 构建测试集 y_scaled：只在有效点上归一化（与 FiLM 测试脚本一致）
x_test = X_scaled[test_idx].astype(np.float32)
mask_test = mask_valid[test_idx].astype(np.float32)

y_test_scaled = np.zeros((len(test_idx), nx * ny * nz), dtype=np.float32)
for j, i in enumerate(test_idx):
    m = float(Y_means_all[i])
    s = float(Y_stds_all[i]) + 1e-8
    y_valid_i = Y_grid_flat[i, lin]
    y_test_scaled[j, lin] = (y_valid_i - m) / s

y_test_scaled = y_test_scaled.reshape((len(test_idx), nx, ny, nz))

# -------------------------------------------------------------
# 构建模型并加载权重（与 FiLM 测试脚本一致）
# -------------------------------------------------------------
model = build_model_from_ckpt(ckpt).to(device)

state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
# 兼容保存时带 module. 前缀的情况
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}

missing, unexpected = model.load_state_dict(state_dict, strict=True)
if missing:
    print("[Warning] Missing keys:", missing)
if unexpected:
    print("[Warning] Unexpected keys:", unexpected)

model.eval()

# 参数统计（与 FiLM 测试脚本一致）
core_for_count = model.module if isinstance(model, nn.DataParallel) else model
total_params, trainable_params = count_parameters(core_for_count)
print("===== Model parameter count =====")
print(f"Total parameters     : {_format_int(total_params)}")
print(f"Trainable parameters : {_format_int(trainable_params)}")

by_type_total, by_type_train, by_top_total, by_top_train = parameter_breakdown(core_for_count)

print("===== Parameter breakdown: by layer type =====")
for k in sorted(by_type_total.keys(), key=lambda x: (-by_type_total[x], x)):
    print(f"{k:>14s} | total={_format_int(by_type_total[k]):>12s} | trainable={_format_int(by_type_train.get(k, 0)):>12s}")
print("")

print("===== Parameter breakdown: by top-level block =====")
for k in sorted(by_top_total.keys(), key=lambda x: (-by_top_total[x], x)):
    print(f"{k:>14s} | total={_format_int(by_top_total[k]):>12s} | trainable={_format_int(by_top_train.get(k, 0)):>12s}")
print("")

# -------------------------------------------------------------
# DataLoader（与 FiLM 测试脚本一致）
# -------------------------------------------------------------
BATCH_SIZE = 16

test_loader = DataLoader(
    TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test_scaled, dtype=torch.float32),
        torch.tensor(mask_test, dtype=torch.float32),
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=(device.type == "cuda"),
)

# =============================================================
# 推理 + 评估（与 FiLM 测试脚本一致）
# =============================================================
NRMSE_list, MAE_list, MSE_list, R2_list, GradMSE_list = [], [], [], [], []

loss_sum = 0.0
count_sum = 0

pred_time_sum = 0.0
pred_batches = 0
pred_samples = 0

if SAVE_PRED_NPY or SAVE_VALID_POINTS_CSV or SAVE_FULL_GRID_CSV:
    _ensure_dir(SAVE_DIR)

with torch.no_grad():
    base = 0
    for xb, yb_scaled, mb in test_loader:
        B = xb.size(0)
        xb = xb.to(device)
        yb_scaled = yb_scaled.to(device)
        mb = mb.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred_scaled = model(xb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        pred_time_sum += (t1 - t0)
        pred_batches += 1
        pred_samples += int(B)

        batch_loss = masked_loss(pred_scaled, yb_scaled, mb)
        loss_sum += float(batch_loss.item()) * int(B)
        count_sum += int(B)

        pred_scaled_np = pred_scaled.detach().cpu().numpy()
        yb_scaled_np = yb_scaled.detach().cpu().numpy()
        mb_np = mb.detach().cpu().numpy()

        for b in range(B):
            global_i = int(test_idx[base + b])
            m = float(Y_means_all[global_i])
            s = float(Y_stds_all[global_i]) + 1e-8

            pred_real = pred_scaled_np[b] * s + m
            true_real = yb_scaled_np[b] * s + m
            mask_real = mb_np[b]

            # 无信息点（mask==0）置零：与训练/评估口径保持一致
            pred_real[mask_real < 0.5] = 0.0
            true_real[mask_real < 0.5] = 0.0

            nrmse, mae, mse, r2, grad_mse = _masked_metrics_per_sample(pred_real, true_real, mask_real)
            NRMSE_list.append(nrmse)
            MAE_list.append(mae)
            MSE_list.append(mse)
            R2_list.append(r2)
            GradMSE_list.append(grad_mse)

            if SAVE_PRED_NPY or SAVE_VALID_POINTS_CSV or SAVE_FULL_GRID_CSV:
                save_prediction_artifacts(
                    out_dir=SAVE_DIR,
                    sample_id=global_i,
                    pred_real=pred_real,
                    true_real=true_real,
                    mask=mask_real,
                    lin_valid=lin,
                    x_unique=x_unique,
                    y_unique=y_unique,
                    z_unique=z_unique,
                    save_npy=SAVE_PRED_NPY,
                    save_csv_valid_points=SAVE_VALID_POINTS_CSV,
                    save_csv_full_grid=SAVE_FULL_GRID_CSV,
                )

        base += B

if pred_batches > 0:
    avg_batch = pred_time_sum / pred_batches
    avg_sample = pred_time_sum / max(1, pred_samples)
    print("===== Inference timing (forward only) =====")
    print(f"Total forward time: {pred_time_sum:.6f} s")
    print(f"Avg per batch     : {avg_batch:.6f} s")
    print(f"Avg per sample    : {avg_sample:.6f} s")

if count_sum > 0:
    print("===== Test masked_loss (scaled space) =====")
    print(f"loss={loss_sum / count_sum:.6f}")

print("===== Per-sample metrics (mask==1) =====")
for i in range(len(test_idx)):
    print(
        f"sample={int(test_idx[i])} | "
        f"NRMSE={NRMSE_list[i]:.6f} | MAE={MAE_list[i]:.6f} | MSE={MSE_list[i]:.6f} | "
        f"R2={R2_list[i]:.6f} | GradMSE={GradMSE_list[i]:.6e}"
    )

NRMSE_arr = np.asarray(NRMSE_list, dtype=np.float64)
MAE_arr = np.asarray(MAE_list, dtype=np.float64)
MSE_arr = np.asarray(MSE_list, dtype=np.float64)
R2_arr = np.asarray(R2_list, dtype=np.float64)
GradMSE_arr = np.asarray(GradMSE_list, dtype=np.float64)

print("===== Summary on test set =====")
print(f"NRMSE  : mean={np.nanmean(NRMSE_arr):.6f}, std={np.nanstd(NRMSE_arr):.6f}")
print(f"MAE    : mean={np.nanmean(MAE_arr):.6f}, std={np.nanstd(MAE_arr):.6f}")
print(f"MSE    : mean={np.nanmean(MSE_arr):.6f}, std={np.nanstd(MSE_arr):.6f}")
print(f"R2     : mean={np.nanmean(R2_arr):.6f}, std={np.nanstd(R2_arr):.6f}")
print(f"GradMSE: mean={np.nanmean(GradMSE_arr):.6e}, std={np.nanstd(GradMSE_arr):.6e}")

# =============================================================
# 轻量化/效率对比：测试脚本可测指标输出（推理侧为主）
# - Params (M)
# - Checkpoint size (MB)
# - GFLOPs / forward (B=1, 仅估算 Conv/Linear)
# - Peak VRAM inference (GB)
# - Max batch size (fp32 / fp16) —— 推理 forward-only 近似上限
# - Inference latency (ms/sample) —— 已在上面输出 forward-only
# =============================================================

def _ckpt_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024.0 * 1024.0)
    except OSError:
        return float("nan")

def _params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def estimate_gflops_forward(model: nn.Module, bc_example: torch.Tensor) -> float:
    """
    通过一次前向的 hook 估算 GFLOPs（只统计 Conv3d / Linear；FLOPs≈2*MACs）。
    这是“模型结构复杂度”的近似指标：不同 GPU/库实现会导致实际时间不同，但 GFLOPs 可用于论文对比。
    """
    macs = 0

    def conv3d_hook(mod: nn.Conv3d, inp, out):
        nonlocal macs
        out_t = out
        B, Cout, Ox, Oy, Oz = out_t.shape
        Cin = mod.in_channels
        kx, ky, kz = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size,) * 3
        groups = mod.groups
        macs += int(B) * int(Cout) * int(Ox) * int(Oy) * int(Oz) * (int(Cin) // int(groups)) * int(kx) * int(ky) * int(kz)

    def linear_hook(mod: nn.Linear, inp, out):
        nonlocal macs
        out_t = out
        out_elems = out_t.numel()
        macs += int(out_elems) * int(mod.in_features)

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            hooks.append(m.register_forward_hook(conv3d_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(bc_example)
    for h in hooks:
        h.remove()
    if model_was_training:
        model.train()

    gflops = (2.0 * float(macs)) / 1e9
    return gflops

def probe_max_batch_inference(model: nn.Module, device: torch.device, fp16: bool, max_cap: int = 512) -> int:
    """
    仅 forward-only 的 batch 上限探测（用于论文“可跑的最大 batch”对比）。
    采用指数增长 + 二分搜索；每次尝试都会 synchronize，避免异步导致的误判。
    """
    if device.type != "cuda":
        return 0

    def try_bs(bs: int) -> bool:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            bc = torch.zeros((bs, 6), device=device, dtype=torch.float32)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=fp16, dtype=torch.float16):
                    _ = model(bc)
            torch.cuda.synchronize()
            return True
        except RuntimeError as e:
            msg = str(e).lower()
            if ("out of memory" in msg) or ("cuda" in msg and "memory" in msg):
                return False
            raise

    lo, hi = 1, 1
    while hi < max_cap and try_bs(hi):
        lo = hi
        hi *= 2
    hi = min(hi, max_cap)

    best = lo if try_bs(lo) else 0
    l, r = best, hi
    while l <= r:
        mid = (l + r) // 2
        if try_bs(mid):
            best = mid
            l = mid + 1
        else:
            r = mid - 1
    return best

def measure_peak_vram_inference(model: nn.Module, device: torch.device, fp16: bool, bs: int = 1) -> float:
    """返回 peak allocated GB（forward-only）。"""
    if device.type != "cuda":
        return float("nan")
    was_training = model.training
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    bc = torch.zeros((bs, 6), device=device, dtype=torch.float32)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=fp16, dtype=torch.float16):
            _ = model(bc)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    if was_training:
        model.train()
    return float(peak) / (1024.0 ** 3)

print("\n===== Lightweightness / Efficiency report (test-side) =====")
print(f"Checkpoint size (MB): {_ckpt_size_mb(CKPT_PATH):.3f}")

core_bench = model.module if isinstance(model, nn.DataParallel) else model
print(f"Params (M): {_params_m(core_bench):.3f}")

# GFLOPs / forward (B=1) —— bc_example 只需要 (1,6)
try:
    bc1 = torch.zeros((1, 6), device=device, dtype=torch.float32)
    gflops = estimate_gflops_forward(core_bench.to(device), bc1)
    print(f"GFLOPs / forward (B=1, est.): {gflops:.3f}")
except Exception as e:
    print(f"GFLOPs / forward (B=1, est.): NA ({e})")

# Peak VRAM inference / Max batch size
if device.type == "cuda":
    peak_fp32 = measure_peak_vram_inference(model, device=device, fp16=False, bs=1)
    peak_fp16 = measure_peak_vram_inference(model, device=device, fp16=True, bs=1)
    print(f"Peak VRAM inference (GB) fp32 (B=1): {peak_fp32:.3f}")
    print(f"Peak VRAM inference (GB) fp16 (B=1): {peak_fp16:.3f}")

    try:
        max_bs_fp32 = probe_max_batch_inference(model, device=device, fp16=False, max_cap=512)
        max_bs_fp16 = probe_max_batch_inference(model, device=device, fp16=True, max_cap=1024)
        print(f"Max batch size fp32 (forward-only, cap): {max_bs_fp32}")
        print(f"Max batch size fp16 (forward-only, cap): {max_bs_fp16}")
    except Exception as e:
        print(f"Max batch size probe: NA ({e})")

# CUDA_VISIBLE_DEVICES=2,4 python "test_CNN_Concat_sdf+bm_GN_RemoveStem.py"
