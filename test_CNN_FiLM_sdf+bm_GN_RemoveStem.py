#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================
# test_3dcnn_film_voxel7_batch.py
#
# ✅ 批量测试 FiLM voxel7 模型（对齐“多 seed + best/last ckpt”的训练改造）
#
# 你训练后会得到一批：
#   {CKPT_PREFIX}_seed{seed}_best.pth
#   {CKPT_PREFIX}_seed{seed}_last.pth
#
# 本脚本会：
#   - 自动 glob 扫描这些 pth
#   - 对每个 ckpt 在 test split 上做一次完整评估
#   - 输出一个汇总 CSV（每个 ckpt 一行）
#
# 环境变量（推荐）：
#   CKPT_PREFIX     : ckpt 文件前缀（默认 best_CNN_FiLM_sdf+bm_GN_RemoveStem）
#   CKPT_DIR        : ckpt 所在目录（默认 .）
#   CKPT_KIND       : best / last / both（默认 both）
#   OUTPUT_CSV      : 输出汇总 CSV 路径（默认 test_ckpt_summary.csv）
#   BATCH_SIZE      : 测试 batch（默认 16）
#
# 数据路径：
#   DATA_BC         : boundary_condition.csv（默认 data/boundary_condition.csv）
#   DATA_TEMP       : Temp_all.csv（默认 data/Temp_all.csv）
#   DATA_VOXEL      : cnn_input_channels_no_normals.csv（默认 data/cnn_input_channels_no_normals.csv）
#
# （可选）保存预测产物（默认关）：
#   SAVE_DIR, SAVE_PRED_NPY, SAVE_VALID_POINTS_CSV, SAVE_FULL_GRID_CSV
#
# 注意：
# - 每个 ckpt 自带 x_mean/x_scale 与 Y_means/Y_stds，本脚本按 ckpt 自己的统计去构造 test 的 scaled-space label
# - 如果你的 seeds 都用同一个数据集/同一个归一化口径，结果应一致；这里仍按 ckpt 分开计算以避免 silent mismatch
# =============================================================

import os
import re
import glob
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# --------------------- 设备 ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    torch.cuda.init()
    print(f"CUDA devices: {torch.cuda.device_count()} visible.")


# =============================================================
# ENV config
# =============================================================
CKPT_PREFIX = os.environ.get("CKPT_PREFIX", "best_CNN_FiLM_sdf+bm_GN_RemoveStem").strip()
CKPT_DIR = os.environ.get("CKPT_DIR", ".").strip()
CKPT_KIND = os.environ.get("CKPT_KIND", "both").strip().lower()  # best / last / both
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "test_ckpt_summary.csv").strip()

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))

DATA_BC = os.environ.get("DATA_BC", "data/boundary_condition.csv").strip()
DATA_TEMP = os.environ.get("DATA_TEMP", "data/Temp_all.csv").strip()
DATA_VOXEL = os.environ.get("DATA_VOXEL", "data/cnn_input_channels_no_normals.csv").strip()

# ---- optional artifacts ----
SAVE_DIR = os.environ.get("SAVE_DIR", "test_outputs_voxel7").strip()
SAVE_PRED_NPY = os.environ.get("SAVE_PRED_NPY", "0").strip() in ("1", "true", "yes")
SAVE_VALID_POINTS_CSV = os.environ.get("SAVE_VALID_POINTS_CSV", "0").strip() in ("1", "true", "yes")
SAVE_FULL_GRID_CSV = os.environ.get("SAVE_FULL_GRID_CSV", "0").strip() in ("1", "true", "yes")
CSV_FULLGRID_CHUNKSIZE = int(os.environ.get("CSV_FULLGRID_CHUNKSIZE", "1000000"))

# -------------------------------------------------------------
# 全局：体素输入 + mask（每个 ckpt 可能不同 voxel_cols / mask）
# -------------------------------------------------------------
GEOM_MASK = None   # (1,1,nx,ny,nz) torch
VOXEL_INPUT = None # (1,7,nx,ny,nz) torch


def make_gn(C: int, max_groups: int = 8) -> nn.GroupNorm:
    G = min(max_groups, C)
    while C % G != 0:
        G -= 1
    return nn.GroupNorm(G, C)


# =============================================================
# 模型定义（与你贴的测试脚本一致）
# =============================================================
class FiLMResidualBlock(nn.Module):
    """(Conv -> GN -> FiLM -> GELU) x2 -> Dropout3d + residual"""

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = make_gn(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = make_gn(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout3d(dropout_p)
        self.residual_proj = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = out * (1.0 + gamma) + beta
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = out * (1.0 + gamma) + beta
        out = self.act(out)

        out = self.drop(out)
        return out + residual


def build_model_from_ckpt(ckpt: dict) -> nn.Module:
    """从 ckpt meta 重建最新版 FiLM-A(voxel7) 网络结构。"""

    model_type = str(ckpt.get("model_type", ""))
    allowed_prefixes = {"FiLM_A_voxel7_no_stem_pooldown_fullres_light"}
    if not any(model_type.startswith(p) for p in allowed_prefixes):
        raise ValueError(
            f"ckpt.model_type={model_type!r} 与该测试脚本不匹配（期望 FiLM_A_voxel7*）。"
        )

    input_dim = int(ckpt["input_dim"])
    nx = int(ckpt["nx"])
    ny = int(ckpt["ny"])
    nz = int(ckpt["nz"])

    depth = int(ckpt.get("depth", 3))
    if depth != 3:
        raise ValueError(f"该测试脚本对齐的是 depth=3，但 ckpt.depth={depth}")

    base_ch = int(ckpt.get("base_ch", 24))
    dropout_p = float(ckpt.get("dropout_p", 0.1))

    film_hidden = int(ckpt.get("film_hidden", 96))
    film_scale = float(ckpt.get("film_scale", 1.0))
    film_mlp_dropout = float(ckpt.get("film_mlp_dropout", 0.0))

    class FiLMGen(nn.Module):
        """bc -> {gamma,beta}; 输出按 film_scale 缩放；MLP 内含 dropout 与训练端一致"""

        def __init__(self, input_dim: int, ch_list: list[int], hidden: int, scale: float, mlp_dropout: float):
            super().__init__()
            self.ch_list = ch_list
            self.scale = float(scale)
            out_dim = 2 * sum(ch_list)
            p = float(mlp_dropout)
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Dropout(p=p),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(p=p),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, bc: torch.Tensor):
            B = bc.size(0)
            v = self.net(bc)
            gammas, betas = [], []
            offset = 0
            s = self.scale
            for C in self.ch_list:
                g_raw = v[:, offset: offset + C]
                offset += C
                b_raw = v[:, offset: offset + C]
                offset += C
                gammas.append((s * g_raw).view(B, C, 1, 1, 1))
                betas.append((s * b_raw).view(B, C, 1, 1, 1))
            return gammas, betas

    class CNN3D_FiLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.nx, self.ny, self.nz = nx, ny, nz
            self.depth = 3
            self.base_ch = base_ch
            self.film_hidden = film_hidden
            self.film_scale = film_scale
            self.film_mlp_dropout = film_mlp_dropout

            assert VOXEL_INPUT is not None, "VOXEL_INPUT 未初始化"
            assert GEOM_MASK is not None, "GEOM_MASK 未初始化"
            self.register_buffer("voxel_input", VOXEL_INPUT)  # (1,7,nx,ny,nz)
            self.register_buffer("geom_mask", GEOM_MASK)      # (1,1,nx,ny,nz)

            c0 = max(8, base_ch // 2)

            self.enc0_full = FiLMResidualBlock(7, c0, dropout_p=dropout_p)  # /1
            self.pool0 = nn.MaxPool3d(kernel_size=2, stride=2)              # /2

            self.enc0 = FiLMResidualBlock(c0, base_ch, dropout_p=dropout_p)          # /2
            self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)                       # /4
            self.enc1 = FiLMResidualBlock(base_ch, base_ch * 2, dropout_p=dropout_p) # /4
            self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)                       # /8
            self.enc2 = FiLMResidualBlock(base_ch * 2, base_ch * 4, dropout_p=dropout_p)  # /8

            bottleneck_ch = base_ch * 4
            self.bottleneck = FiLMResidualBlock(bottleneck_ch, bottleneck_ch, dropout_p=dropout_p)

            self.up2_conv = FiLMResidualBlock(bottleneck_ch + base_ch * 2, base_ch * 2, dropout_p=dropout_p)  # /4
            self.up1_conv = FiLMResidualBlock(base_ch * 2 + base_ch, base_ch, dropout_p=dropout_p)            # /2

            self.out_proj = nn.Conv3d(base_ch, base_ch, kernel_size=1)

            self.up0_conv = FiLMResidualBlock(base_ch + c0, c0, dropout_p=dropout_p)  # /1
            self.final_conv = nn.Conv3d(c0, 1, kernel_size=1)

            ch_list = [
                c0,         # enc0_full
                base_ch,     # enc0
                base_ch * 2, # enc1
                base_ch * 4, # enc2
                base_ch * 4, # bottleneck
                base_ch * 2, # up2
                base_ch,     # up1
                c0,          # up0
            ]
            self.film = FiLMGen(
                input_dim=input_dim,
                ch_list=ch_list,
                hidden=film_hidden,
                scale=film_scale,
                mlp_dropout=film_mlp_dropout,
            )

        def forward(self, bc: torch.Tensor) -> torch.Tensor:
            B = bc.size(0)
            vox = self.voxel_input.expand(B, -1, -1, -1, -1)
            mask_ch = self.geom_mask.expand(B, -1, -1, -1, -1)

            gammas, betas = self.film(bc)
            gi = 0

            x_full_skip = self.enc0_full(vox, gammas[gi], betas[gi]); gi += 1
            x = self.pool0(x_full_skip)

            x0 = self.enc0(x, gammas[gi], betas[gi]); gi += 1
            x1 = self.pool1(x0)
            x1 = self.enc1(x1, gammas[gi], betas[gi]); gi += 1
            x2 = self.pool2(x1)
            x2 = self.enc2(x2, gammas[gi], betas[gi]); gi += 1

            xb = self.bottleneck(x2, gammas[gi], betas[gi]); gi += 1

            x_up2 = F.interpolate(xb, size=x1.shape[2:], mode="trilinear", align_corners=False)
            x_dec2 = self.up2_conv(torch.cat([x_up2, x1], dim=1), gammas[gi], betas[gi]); gi += 1

            x_up1 = F.interpolate(x_dec2, size=x0.shape[2:], mode="trilinear", align_corners=False)
            x_dec1 = self.up1_conv(torch.cat([x_up1, x0], dim=1), gammas[gi], betas[gi]); gi += 1

            x_dec = self.out_proj(x_dec1)

            x_up0 = F.interpolate(x_dec, size=x_full_skip.shape[2:], mode="trilinear", align_corners=False)
            x0_full = self.up0_conv(torch.cat([x_up0, x_full_skip], dim=1), gammas[gi], betas[gi]); gi += 1

            x_full = self.final_conv(x0_full)

            out = x_full.squeeze(1)
            out = out * mask_ch.squeeze(1)
            return out

    return CNN3D_FiLM()


# =============================================================
# Loss + metrics
# =============================================================
def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_elem = F.smooth_l1_loss(pred, target, reduction="none")
    masked = per_elem * mask
    return masked.sum() / (mask.sum() + 1e-8)


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


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# =============================================================
# Optional: saving artifacts (kept from your script, trimmed usage)
# =============================================================
def _lin_to_ijk(lin: np.ndarray, ny: int, nz: int):
    lin = lin.astype(np.int64)
    ix = lin // (ny * nz)
    rem = lin % (ny * nz)
    iy = rem // nz
    iz = rem % nz
    return ix, iy, iz


def _full_grid_xyz_columns(x_unique: np.ndarray, y_unique: np.ndarray, z_unique: np.ndarray):
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
            {"x": xs.astype(np.float32), "y": ys.astype(np.float32), "z": zs.astype(np.float32),
             "true": y.astype(np.float32), "pred": p.astype(np.float32), "err": e.astype(np.float32)}
        )
        df.to_csv(os.path.join(out_dir, f"sample_{sample_id:04d}_valid_points.csv"), index=False)

    if save_csv_full_grid:
        xs, ys, zs = _full_grid_xyz_columns(x_unique, y_unique, z_unique)

        pred_flat = pred_real.astype(np.float32, copy=False).ravel(order="C")
        true_flat = true_real.astype(np.float32, copy=False).ravel(order="C")
        mask_flat = mask.astype(np.float32, copy=False).ravel(order="C")
        err_flat = (pred_flat - true_flat).astype(np.float32, copy=False)

        df_full = pd.DataFrame(
            {"x": xs.astype(np.float32, copy=False), "y": ys.astype(np.float32, copy=False), "z": zs.astype(np.float32, copy=False),
             "Temp_pred": pred_flat, "Temp_true": true_flat, "Temp_err": err_flat, "mask": mask_flat}
        )
        out_path = os.path.join(out_dir, f"sample_{sample_id:04d}_fullgrid.csv")
        df_full.to_csv(out_path, index=False, chunksize=csv_fullgrid_chunksize)


# =============================================================
# Data pre-load once (voxel/temp/bc) — shared by all ckpts
# =============================================================
print(f"\nLoading voxel CSV: {DATA_VOXEL}")
df_vox = pd.read_csv(DATA_VOXEL)

required_cols = ["x", "y", "z", "C0", "C1", "C2", "C3", "C4", "C5", "sdf"]
missing = [c for c in required_cols if c not in df_vox.columns]
if missing:
    raise KeyError(f"{DATA_VOXEL} 缺少列: {missing}")

xv = df_vox["x"].to_numpy(dtype=np.float32)
yv = df_vox["y"].to_numpy(dtype=np.float32)
zv = df_vox["z"].to_numpy(dtype=np.float32)

x_unique = np.sort(np.unique(xv))
y_unique = np.sort(np.unique(yv))
z_unique = np.sort(np.unique(zv))
(nx, ny, nz) = (len(x_unique), len(y_unique), len(z_unique))
print(f"Voxel grid: nx={nx}, ny={ny}, nz={nz}")

x_index = {float(v): i for i, v in enumerate(x_unique)}
y_index = {float(v): i for i, v in enumerate(y_unique)}
z_index = {float(v): i for i, v in enumerate(z_unique)}

# Precompute lin_all mapping (Temp_all full-grid case uses this)
ix_all = np.array([x_index[float(v)] for v in xv], dtype=np.int64)
iy_all = np.array([y_index[float(v)] for v in yv], dtype=np.int64)
iz_all = np.array([z_index[float(v)] for v in zv], dtype=np.int64)
lin_all = ix_all * (ny * nz) + iy_all * nz + iz_all
total_points = int(df_vox.shape[0])

# Cache voxel grids by voxel_cols tuple (most seeds identical)
_voxel_cache: Dict[Tuple[str, ...], np.ndarray] = {}

def build_voxel_grid_by_cols(col_order: List[str]) -> np.ndarray:
    key = tuple(col_order)
    if key in _voxel_cache:
        return _voxel_cache[key]
    for c in col_order:
        if c not in df_vox.columns:
            raise KeyError(f"voxel_cols 包含不存在列: {c}")
    voxel_grid = np.zeros((len(col_order), nx, ny, nz), dtype=np.float32)
    cols_np = [df_vox[c].to_numpy(dtype=np.float32) for c in col_order]
    for i in range(df_vox.shape[0]):
        ix = ix_all[i]
        iy = iy_all[i]
        iz = iz_all[i]
        voxel_grid[:, ix, iy, iz] = np.array([c[i] for c in cols_np], dtype=np.float32)
    _voxel_cache[key] = voxel_grid
    return voxel_grid


print(f"\nLoading Temp CSV: {DATA_TEMP}")
T_np = pd.read_csv(DATA_TEMP).to_numpy(dtype=np.float32)

def _as_samples_first(a: np.ndarray, n_points: int) -> np.ndarray:
    if a.shape[0] == n_points:
        return a.T
    if a.shape[1] == n_points:
        return a
    raise ValueError(f"Temp_all.csv 维度 {a.shape} 与点数 {n_points} 不匹配。")

# We cannot know valid_points without ckpt (lin_valid). We'll keep T_np raw and interpret per-ckpt.

print(f"\nLoading BC CSV: {DATA_BC}")
df_bc = pd.read_csv(DATA_BC)
X_data = df_bc.iloc[:, :6].to_numpy(dtype=np.float32)
split_raw = df_bc.iloc[:, 6].to_numpy()

if split_raw.dtype.kind in "OUS":
    split = np.array([str(s).strip().lower() for s in split_raw])
    test_idx = np.where(split == "test")[0]
else:
    test_idx = np.where(split_raw == 2)[0]

print(f"Test samples: {len(test_idx)}")


# =============================================================
# Helpers: ckpt parsing, file discovery
# =============================================================
_seed_re = re.compile(r"_seed(\d+)_")
_kind_re = re.compile(r"_(best|last)\.pth$")

def parse_seed_and_kind(path: str) -> Tuple[Optional[int], Optional[str]]:
    seed = None
    kind = None
    m = _seed_re.search(os.path.basename(path))
    if m:
        seed = int(m.group(1))
    k = _kind_re.search(path)
    if k:
        kind = k.group(1)
    return seed, kind


def discover_ckpts() -> List[str]:
    pats = []
    if CKPT_KIND in ("both", "best"):
        pats.append(os.path.join(CKPT_DIR, f"{CKPT_PREFIX}_seed*_best.pth"))
    if CKPT_KIND in ("both", "last"):
        pats.append(os.path.join(CKPT_DIR, f"{CKPT_PREFIX}_seed*_last.pth"))
    paths = []
    for pat in pats:
        paths.extend(glob.glob(pat))
    paths = sorted(set(paths))
    return paths


def ckpt_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024.0 * 1024.0)
    except OSError:
        return float("nan")


def count_parameters(m: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


# =============================================================
# Core: evaluate one ckpt on test split
# =============================================================
@dataclass
class EvalResult:
    ckpt_path: str
    seed: Optional[int]
    ckpt_kind: Optional[str]

    # test metrics summary
    loss_scaled_mean: float
    nrmse_mean: float
    nrmse_std: float
    mae_mean: float
    mae_std: float
    mse_mean: float
    mse_std: float
    r2_mean: float
    r2_std: float
    gradmse_mean: float
    gradmse_std: float

    # timing
    forward_time_s: float
    avg_ms_per_sample: float

    # model/ckpt info
    ckpt_size_mb: float
    params_total: int
    params_trainable: int

    # meta from ckpt if present
    train_seed_in_ckpt: Optional[int]
    best_epoch: Optional[int]
    best_val: Optional[float]
    last_epoch: Optional[int]
    last_val: Optional[float]

    dropout_p: Optional[float]
    film_hidden: Optional[int]
    film_scale: Optional[float]
    film_mlp_dropout: Optional[float]
    lr: Optional[float]
    film_lr_mult: Optional[float]


def evaluate_ckpt(ckpt_path: str) -> EvalResult:
    global VOXEL_INPUT, GEOM_MASK

    print(f"\n=============================================================")
    print(f"Evaluating ckpt: {ckpt_path}")
    print("=============================================================")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # --- voxel cols & grids ---
    col_order = ckpt.get("voxel_cols", ["C0", "C1", "C2", "C3", "C4", "C5", "sdf"])
    voxel_grid = build_voxel_grid_by_cols(list(col_order))
    VOXEL_INPUT = torch.tensor(voxel_grid[None, ...], dtype=torch.float32, device=device)

    # --- mask / lin_valid ---
    if "geom_mask_np" in ckpt and ckpt["geom_mask_np"] is not None:
        geom_mask_np = np.asarray(ckpt["geom_mask_np"], dtype=np.float32)
        if geom_mask_np.shape != (nx, ny, nz):
            raise ValueError(f"ckpt.geom_mask_np shape={geom_mask_np.shape} != {(nx,ny,nz)}")
    else:
        if "C0" not in col_order:
            raise ValueError(f"voxel_cols 必须包含 C0 用于 mask，但现在是: {col_order}")
        geom_mask_np = (voxel_grid[list(col_order).index("C0")] > 0.5).astype(np.float32)

    GEOM_MASK = torch.tensor(geom_mask_np[None, None, ...], dtype=torch.float32, device=device)

    if "lin_valid" in ckpt and ckpt["lin_valid"] is not None:
        lin_valid = np.asarray(ckpt["lin_valid"], dtype=np.int64)
    else:
        lin_valid = np.where(geom_mask_np.reshape(-1) > 0.5)[0].astype(np.int64)

    valid_points = int(lin_valid.shape[0])

    # --- temp parsing for this ckpt's valid_points / total_points ---
    if (T_np.shape[0] == valid_points) or (T_np.shape[1] == valid_points):
        Y_valid = _as_samples_first(T_np, valid_points)  # (num_samples, valid_points)
        num_samples = int(Y_valid.shape[0])
        Y_grid_flat = np.zeros((num_samples, nx * ny * nz), dtype=np.float32)
        Y_grid_flat[:, lin_valid] = Y_valid
    elif (T_np.shape[0] == total_points) or (T_np.shape[1] == total_points):
        Y_all = _as_samples_first(T_np, total_points)  # (num_samples, total_points)
        num_samples = int(Y_all.shape[0])
        Y_grid_flat = np.zeros((num_samples, nx * ny * nz), dtype=np.float32)
        Y_grid_flat[:, lin_all] = Y_all
    else:
        raise ValueError(
            f"Temp_all.csv 维度 {T_np.shape} 既不匹配 valid_points={valid_points}，也不匹配 total_points={total_points}。"
        )

    # --- scaling params from ckpt ---
    x_mean = ckpt.get("x_mean", None)
    x_scale = ckpt.get("x_scale", None)
    if x_mean is None or x_scale is None:
        raise ValueError("ckpt 中未找到 x_mean/x_scale（无法对齐训练时 X 标准化）")
    x_mean = np.asarray(x_mean, dtype=np.float32)
    x_scale = np.asarray(x_scale, dtype=np.float32)

    Y_means_all = ckpt.get("Y_means", None)
    Y_stds_all = ckpt.get("Y_stds", None)
    if Y_means_all is None or Y_stds_all is None:
        raise ValueError("ckpt 中未找到 Y_means/Y_stds（无法对齐训练时 Y 样本级归一化）")
    Y_means_all = np.asarray(Y_means_all, dtype=np.float32)
    Y_stds_all = np.asarray(Y_stds_all, dtype=np.float32)

    # sanity: test_idx must exist in these arrays
    if int(np.max(test_idx)) >= int(Y_means_all.shape[0]):
        raise ValueError(f"test_idx max={int(np.max(test_idx))} >= ckpt.Y_means len={int(Y_means_all.shape[0])}")

    # --- build X_test scaled ---
    X_scaled = (X_data - x_mean) / (x_scale + 1e-12)
    x_test = X_scaled[test_idx].astype(np.float32)

    # --- build y_test_scaled for this ckpt ---
    mask_valid_full = np.broadcast_to(geom_mask_np[None, ...], (num_samples, nx, ny, nz)).astype(np.float32)
    mask_test = mask_valid_full[test_idx].astype(np.float32)

    y_test_scaled = np.zeros((len(test_idx), nx * ny * nz), dtype=np.float32)
    for j, i in enumerate(test_idx):
        m = float(Y_means_all[i])
        s = float(Y_stds_all[i]) + 1e-8
        y_valid_i = Y_grid_flat[i, lin_valid]
        y_test_scaled[j, lin_valid] = (y_valid_i - m) / s
    y_test_scaled = y_test_scaled.reshape((len(test_idx), nx, ny, nz))

    # --- build model & load weights ---
    model = build_model_from_ckpt(ckpt).to(device)

    state_dict = ckpt["state_dict"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    core = model.module if isinstance(model, nn.DataParallel) else model
    total_params, trainable_params = count_parameters(core)

    # --- dataloader ---
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

    # --- inference + metrics ---
    nrmse_list, mae_list, mse_list, r2_list, grad_list = [], [], [], [], []

    loss_sum = 0.0
    cnt_sum = 0

    pred_time_sum = 0.0
    pred_batches = 0
    pred_samples = 0

    if SAVE_PRED_NPY or SAVE_VALID_POINTS_CSV or SAVE_FULL_GRID_CSV:
        _ensure_dir(SAVE_DIR)

    with torch.no_grad():
        base = 0
        for xb, yb_scaled, mb in test_loader:
            B = int(xb.size(0))
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
            pred_samples += B

            batch_loss = masked_loss(pred_scaled, yb_scaled, mb)
            loss_sum += float(batch_loss.item()) * B
            cnt_sum += B

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

                pred_real[mask_real < 0.5] = 0.0
                true_real[mask_real < 0.5] = 0.0

                nrmse, mae, mse, r2, grad_mse = _masked_metrics_per_sample(pred_real, true_real, mask_real)
                nrmse_list.append(nrmse)
                mae_list.append(mae)
                mse_list.append(mse)
                r2_list.append(r2)
                grad_list.append(grad_mse)

                if SAVE_PRED_NPY or SAVE_VALID_POINTS_CSV or SAVE_FULL_GRID_CSV:
                    # 注意：sample_id 使用 global_i，便于和原数据索引一致
                    save_prediction_artifacts(
                        out_dir=SAVE_DIR,
                        sample_id=global_i,
                        pred_real=pred_real,
                        true_real=true_real,
                        mask=mask_real,
                        lin_valid=lin_valid,
                        x_unique=x_unique,
                        y_unique=y_unique,
                        z_unique=z_unique,
                        save_npy=SAVE_PRED_NPY,
                        save_csv_valid_points=SAVE_VALID_POINTS_CSV,
                        save_csv_full_grid=SAVE_FULL_GRID_CSV,
                        csv_fullgrid_chunksize=CSV_FULLGRID_CHUNKSIZE,
                    )

            base += B

    loss_scaled_mean = float(loss_sum / max(1, cnt_sum))

    nrmse_arr = np.asarray(nrmse_list, dtype=np.float64)
    mae_arr = np.asarray(mae_list, dtype=np.float64)
    mse_arr = np.asarray(mse_list, dtype=np.float64)
    r2_arr = np.asarray(r2_list, dtype=np.float64)
    grad_arr = np.asarray(grad_list, dtype=np.float64)

    forward_time_s = float(pred_time_sum)
    avg_ms_per_sample = float(1000.0 * pred_time_sum / max(1, pred_samples))

    seed_from_name, kind_from_name = parse_seed_and_kind(ckpt_path)

    # meta from ckpt if available
    train_seed_in_ckpt = ckpt.get("train_seed", None)
    best_epoch = ckpt.get("best_epoch", ckpt.get("best_epoch_ema", None))
    best_val = ckpt.get("best_val", ckpt.get("best_val_ema", None))
    last_epoch = ckpt.get("last_epoch", None)
    last_val = ckpt.get("last_val", None)

    # hyper/meta fields (for traceability)
    dropout_p = ckpt.get("dropout_p", None)
    film_hidden = ckpt.get("film_hidden", None)
    film_scale = ckpt.get("film_scale", None)
    film_mlp_dropout = ckpt.get("film_mlp_dropout", None)
    lr = ckpt.get("lr", None)
    film_lr_mult = ckpt.get("film_lr_mult", None)

    print(f"[DONE] loss_scaled_mean={loss_scaled_mean:.6f} | NRMSE mean={np.nanmean(nrmse_arr):.6f} | "
          f"MAE mean={np.nanmean(mae_arr):.6f} | R2 mean={np.nanmean(r2_arr):.6f} | "
          f"avg_ms/sample={avg_ms_per_sample:.3f}")

    return EvalResult(
        ckpt_path=ckpt_path,
        seed=seed_from_name,
        ckpt_kind=kind_from_name,

        loss_scaled_mean=loss_scaled_mean,

        nrmse_mean=float(np.nanmean(nrmse_arr)),
        nrmse_std=float(np.nanstd(nrmse_arr)),
        mae_mean=float(np.nanmean(mae_arr)),
        mae_std=float(np.nanstd(mae_arr)),
        mse_mean=float(np.nanmean(mse_arr)),
        mse_std=float(np.nanstd(mse_arr)),
        r2_mean=float(np.nanmean(r2_arr)),
        r2_std=float(np.nanstd(r2_arr)),
        gradmse_mean=float(np.nanmean(grad_arr)),
        gradmse_std=float(np.nanstd(grad_arr)),

        forward_time_s=forward_time_s,
        avg_ms_per_sample=avg_ms_per_sample,

        ckpt_size_mb=float(ckpt_size_mb(ckpt_path)),
        params_total=int(total_params),
        params_trainable=int(trainable_params),

        train_seed_in_ckpt=(int(train_seed_in_ckpt) if train_seed_in_ckpt is not None else None),
        best_epoch=(int(best_epoch) if best_epoch is not None else None),
        best_val=(float(best_val) if best_val is not None else None),
        last_epoch=(int(last_epoch) if last_epoch is not None else None),
        last_val=(float(last_val) if last_val is not None else None),

        dropout_p=(float(dropout_p) if dropout_p is not None else None),
        film_hidden=(int(film_hidden) if film_hidden is not None else None),
        film_scale=(float(film_scale) if film_scale is not None else None),
        film_mlp_dropout=(float(film_mlp_dropout) if film_mlp_dropout is not None else None),
        lr=(float(lr) if lr is not None else None),
        film_lr_mult=(float(film_lr_mult) if film_lr_mult is not None else None),
    )


# =============================================================
# Main: discover ckpts -> eval all -> save CSV
# =============================================================
ckpt_paths = discover_ckpts()
if not ckpt_paths:
    raise FileNotFoundError(
        f"No ckpt found. CKPT_DIR={CKPT_DIR}, CKPT_PREFIX={CKPT_PREFIX}, CKPT_KIND={CKPT_KIND}. "
        f"Example expected: {CKPT_PREFIX}_seed42_best.pth"
    )

print("\n===== Batch test config =====")
print(f"CKPT_DIR={CKPT_DIR}")
print(f"CKPT_PREFIX={CKPT_PREFIX}")
print(f"CKPT_KIND={CKPT_KIND}")
print(f"Found ckpts: {len(ckpt_paths)}")
print(f"OUTPUT_CSV={OUTPUT_CSV}")
print(f"BATCH_SIZE={BATCH_SIZE}")
print(f"DATA_BC={DATA_BC}")
print(f"DATA_TEMP={DATA_TEMP}")
print(f"DATA_VOXEL={DATA_VOXEL}")

rows = []
for p in ckpt_paths:
    try:
        r = evaluate_ckpt(p)
        rows.append(r.__dict__)
    except Exception as e:
        # 不中断全局：把失败原因记到 CSV
        print(f"[ERROR] ckpt={p} failed: {e}")
        seed, kind = parse_seed_and_kind(p)
        rows.append({
            "ckpt_path": p,
            "seed": seed,
            "ckpt_kind": kind,
            "error": str(e),
        })

df = pd.DataFrame(rows)

# Sort: best/last, then seed
if "ckpt_kind" in df.columns and "seed" in df.columns:
    kind_order = {"best": 0, "last": 1}
    df["_kind_order"] = df["ckpt_kind"].map(lambda x: kind_order.get(str(x), 99))
    df["_seed_order"] = df["seed"].fillna(10**9).astype(int)
    df = df.sort_values(["_kind_order", "_seed_order", "ckpt_path"]).drop(columns=["_kind_order", "_seed_order"])

os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"\nSaved batch test summary CSV: {OUTPUT_CSV}")
print(df)


# =============================================================
# 命令行用法示例
# =============================================================
# 1) 在当前目录测试所有 best+last：
#    CKPT_PREFIX="best_CNN_FiLM_sdf+bm_GN_RemoveStem" CKPT_KIND=both python test_3dcnn_film_voxel7_batch.py
#
# 2) 指定 ckpt 目录（比如 ./ckpt），只测 best：
#    CKPT_DIR="./ckpt" CKPT_PREFIX="best_CNN_FiLM_sdf+bm_GN_RemoveStem" CKPT_KIND=best OUTPUT_CSV="ckpt/test_best.csv" \
#    python test_3dcnn_film_voxel7_batch.py
#
# 3) 你想同时开保存 full-grid CSV（很大！）：
#    SAVE_FULL_GRID_CSV=1 SAVE_DIR="test_outputs" python test_3dcnn_film_voxel7_batch.py
#
# 4) 用不同 batch：
#    BATCH_SIZE=8 python test_3dcnn_film_voxel7_batch.py
