#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================
# test_CNN_boundary_channel_aligned.py
# 对齐 CNN_boundary_channel.py 的测试脚本
# -------------------------------------------------------------
# 对齐点：
# - 固定体素输入：使用训练 ckpt 里保存的 voxel_cols 顺序构建 VOXEL_INPUT
# - mask：优先使用 ckpt["geom_mask_np"]；否则从 df_vox["C0"] 构建
# - valid 点：优先使用 ckpt["lin_valid"]；否则从 mask 推导
# - X 标准化：使用 ckpt["x_mean"], ckpt["x_scale"]
# - split：按训练脚本逻辑从 boundary_condition.csv 第7列解析 train/val/test
# - Temp_all.csv：按训练脚本逻辑重建 full-grid Y_grid（raw 温度）
# - 模型结构：严格对齐训练脚本 CNN3D_NoFiLM(depth=2)：
#   * stem 输入通道=13 (=7 voxel + 6 bc)
#   * full-res 融合通道=base_ch+13
#   * stem_stride/base_ch/dropout_p 来自 ckpt
# - 输出：预测 pred_scaled/pred_real 与 true_real 的评估指标（仅在 mask==1 的点上）
# =============================================================

from __future__ import annotations

import os
import json
import math
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# --------------------- small utils ---------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """兼容 DataParallel 导致的 'module.' 前缀"""
    if not isinstance(state, dict) or len(state) == 0:
        return state
    keys = list(state.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state.items()}
    return state

def _extract_state_dict(ckpt_obj) -> Dict[str, torch.Tensor]:
    """兼容多种保存方式"""
    if isinstance(ckpt_obj, dict):
        for k in ("state_dict", "model_state", "model"):
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        keys = list(ckpt_obj.keys())
        if any((".weight" in k) or (".bias" in k) for k in keys):
            return ckpt_obj
        raise KeyError("ckpt 里找不到 state_dict/model_state/model，也不像纯 state_dict。")
    return ckpt_obj

def masked_nrmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0.5
    if valid.sum() <= 0:
        return float("nan")
    err = pred[valid] - target[valid]
    rmse = math.sqrt(float(np.mean(err**2)))
    t = target[valid]
    denom = float(t.max() - t.min())
    return rmse if denom <= 1e-12 else rmse / denom

def masked_r2(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0.5
    if valid.sum() <= 1:
        return float("nan")
    y = target[valid]
    yhat = pred[valid]
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot

def grad_mse_3d(pred3d: np.ndarray, true3d: np.ndarray, mask3d: np.ndarray) -> float:
    eps = 1e-12
    m = mask3d > 0.5

    def grad_axis(a: np.ndarray, axis: int) -> np.ndarray:
        g = np.zeros_like(a, dtype=np.float32)
        slc_center = [slice(None)] * 3
        slc_prev = [slice(None)] * 3
        slc_next = [slice(None)] * 3

        slc_center[axis] = slice(1, -1)
        slc_prev[axis] = slice(0, -2)
        slc_next[axis] = slice(2, None)
        g[tuple(slc_center)] = 0.5 * (a[tuple(slc_next)] - a[tuple(slc_prev)])

        slc0 = [slice(None)] * 3
        slc1 = [slice(None)] * 3
        slc0[axis] = 0
        slc1[axis] = 1
        g[tuple(slc0)] = a[tuple(slc1)] - a[tuple(slc0)]

        slc_last = [slice(None)] * 3
        slc_lastm1 = [slice(None)] * 3
        slc_last[axis] = -1
        slc_lastm1[axis] = -2
        g[tuple(slc_last)] = a[tuple(slc_last)] - a[tuple(slc_lastm1)]
        return g

    gp_x = grad_axis(pred3d, 0)
    gp_y = grad_axis(pred3d, 1)
    gp_z = grad_axis(pred3d, 2)

    gt_x = grad_axis(true3d, 0)
    gt_y = grad_axis(true3d, 1)
    gt_z = grad_axis(true3d, 2)

    diff2 = (gp_x - gt_x) ** 2 + (gp_y - gt_y) ** 2 + (gp_z - gt_z) ** 2
    if m.sum() <= 0:
        return float("nan")
    return float(diff2[m].mean() + eps)

def save_prediction_artifacts(
    out_dir: str,
    sample_id: int,
    pred_scaled3d: np.ndarray,
    pred_real3d: np.ndarray,
    true_real3d: np.ndarray,
    mask3d: np.ndarray,
    save_npy: bool = False,
):
    ensure_dir(out_dir)
    if save_npy:
        np.save(os.path.join(out_dir, f"sample_{sample_id:04d}_pred_scaled.npy"), pred_scaled3d.astype(np.float32))
        np.save(os.path.join(out_dir, f"sample_{sample_id:04d}_pred_real.npy"), pred_real3d.astype(np.float32))
        np.save(os.path.join(out_dir, f"sample_{sample_id:04d}_true_real.npy"), true_real3d.astype(np.float32))
        np.save(os.path.join(out_dir, f"sample_{sample_id:04d}_mask.npy"), mask3d.astype(np.float32))
        np.save(os.path.join(out_dir, f"sample_{sample_id:04d}_err.npy"), (pred_real3d - true_real3d).astype(np.float32))

# =============================================================
# 模型定义：严格对齐训练脚本 CNN3D_NoFiLM(depth=2)
# =============================================================

VOXEL_INPUT = None  # (1,7,nx,ny,nz)
GEOM_MASK = None    # (1,1,nx,ny,nz)

class ConvBlock(nn.Module):
    """与训练脚本一致：Conv3d → BN → GELU → Conv3d → BN → GELU → Dropout3d + residual"""
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

def build_model_from_ckpt(ckpt: Dict, device: torch.device, nx: int, ny: int, nz: int) -> nn.Module:
    base_ch = int(ckpt.get("base_ch", 24))
    dropout_p = float(ckpt.get("dropout_p", 0.1))
    stem_stride = int(ckpt.get("stem_stride", 2))

    class CNN3D_NoFiLM(nn.Module):
        def __init__(self):
            super().__init__()
            assert VOXEL_INPUT is not None, "VOXEL_INPUT 未初始化"
            assert GEOM_MASK is not None, "GEOM_MASK 未初始化"

            self.nx, self.ny, self.nz = nx, ny, nz
            self.base_ch = base_ch
            self.stem_stride = stem_stride

            self.register_buffer("voxel_input", VOXEL_INPUT)  # (1,7,nx,ny,nz)
            self.register_buffer("geom_mask", GEOM_MASK)      # (1,1,nx,ny,nz)

            # 训练脚本：in_ch_stem = 13 (7 voxel + 6 bc)
            in_ch_stem = 13
            self.stem = nn.Sequential(
                nn.Conv3d(in_ch_stem, base_ch, kernel_size=3, padding=1, stride=self.stem_stride),
                nn.BatchNorm3d(base_ch),
                nn.GELU(),
            )

            # Encoder (depth=2)
            self.enc0 = ConvBlock(base_ch, base_ch, dropout_p=dropout_p)
            self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
            self.enc1 = ConvBlock(base_ch, base_ch * 2, dropout_p=dropout_p)

            # Bottleneck
            bottleneck_ch = base_ch * 2
            self.bottleneck = ConvBlock(bottleneck_ch, bottleneck_ch, dropout_p=dropout_p)

            # Decoder (depth=2)：训练脚本用 interpolate + cat + ConvBlock
            self.up1_conv = ConvBlock(bottleneck_ch + base_ch, base_ch, dropout_p=dropout_p)
            self.out_proj = nn.Conv3d(base_ch, base_ch, kernel_size=1)

            # 全分辨率融合：decoder_feat + 7(voxel) + 6(bc) => base_ch + 13
            in_ch_full = base_ch + 13
            self.geom_block = ConvBlock(in_ch_full, base_ch, dropout_p=dropout_p)
            self.final_conv = nn.Conv3d(base_ch, 1, kernel_size=1)

        def forward(self, bc: torch.Tensor) -> torch.Tensor:
            """
            bc: (B,6) —— 标准化后的边界条件
            return: (B,nx,ny,nz) —— 预测的【scaled】温度场（仅inside区域有效）
            """
            B = bc.size(0)

            vox = self.voxel_input.expand(B, -1, -1, -1, -1)          # (B,7,nx,ny,nz)
            mask_ch = self.geom_mask.expand(B, -1, -1, -1, -1)        # (B,1,nx,ny,nz)

            bc_grid = bc.view(B, 6, 1, 1, 1).expand(B, 6, self.nx, self.ny, self.nz)
            x_in = torch.cat([vox, bc_grid], dim=1)  # (B,13,nx,ny,nz)

            x = self.stem(x_in)  # (B,base_ch,nx/2,ny/2,nz/2)

            x0 = self.enc0(x)
            x1 = self.pool1(x0)
            x1 = self.enc1(x1)

            xb = self.bottleneck(x1)

            x_up = F.interpolate(xb, size=x0.shape[2:], mode="trilinear", align_corners=False)
            x_cat = torch.cat([x_up, x0], dim=1)
            x_dec = self.up1_conv(x_cat)

            x_dec = self.out_proj(x_dec)  # (B,base_ch,nx/2,ny/2,nz/2)

            x_dec_full = F.interpolate(x_dec, size=(self.nx, self.ny, self.nz),
                                       mode="trilinear", align_corners=False)

            x_full = torch.cat([x_dec_full, vox, bc_grid], dim=1)  # (B,base_ch+13,nx,ny,nz)
            x_full = self.geom_block(x_full)
            x_full = self.final_conv(x_full)  # (B,1,nx,ny,nz)

            out = x_full.squeeze(1)
            out = out * mask_ch.squeeze(1)  # 输出门控：仅 inside 区域
            return out

    model = CNN3D_NoFiLM().to(device)

    state = _extract_state_dict(ckpt)
    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load_state] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing examples:", missing[:10])
    if unexpected:
        print("  unexpected examples:", unexpected[:10])
    model.eval()
    return model

# =============================================================
# main
# =============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    CKPT_PATH = "best_CNN_boundary_channel_aligned_bench.pth"   # 训练脚本保存的名字
    datapath_bc = "data/boundary_condition.csv"
    datapath_temp = "data/Temp_all.csv"
    datapath_voxel = "data/cnn_input_channels_no_normals.csv"

    SAVE_DIR = "test_outputs_aligned"
    SAVE_PRED_NPY = False
    BATCH_SIZE = 16

    ensure_dir(SAVE_DIR)

    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    print(
        f"[ckpt] model_type={ckpt.get('model_type', 'NA')}, stem_stride={ckpt.get('stem_stride','NA')}, "
        f"depth={ckpt.get('depth', 'NA')}, base_ch={ckpt.get('base_ch', 'NA')}, dropout_p={ckpt.get('dropout_p', 'NA')}"
    )

    # ------------------ 读取体素输入 & mask（对齐训练） ------------------
    global VOXEL_INPUT, GEOM_MASK

    df_vox = pd.read_csv(datapath_voxel)
    for c in ["x", "y", "z", "C0"]:
        if c not in df_vox.columns:
            raise KeyError(f"{datapath_voxel} 缺少列: {c}")

    xv = df_vox["x"].to_numpy(dtype=np.float32)
    yv = df_vox["y"].to_numpy(dtype=np.float32)
    zv = df_vox["z"].to_numpy(dtype=np.float32)

    x_unique = np.sort(np.unique(xv))
    y_unique = np.sort(np.unique(yv))
    z_unique = np.sort(np.unique(zv))
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    print(f"[voxel] grid: nx={nx}, ny={ny}, nz={nz}")

    if all(k in ckpt for k in ("nx", "ny", "nz")):
        ckpt_grid = (int(ckpt["nx"]), int(ckpt["ny"]), int(ckpt["nz"]))
        if (nx, ny, nz) != ckpt_grid:
            raise ValueError(f"当前 voxel 网格 {(nx,ny,nz)} 与 ckpt 网格 {ckpt_grid} 不一致。")

    x_index = {float(v): i for i, v in enumerate(x_unique)}
    y_index = {float(v): i for i, v in enumerate(y_unique)}
    z_index = {float(v): i for i, v in enumerate(z_unique)}

    # ✅ 对齐训练：用 ckpt["voxel_cols"]（而不是简单取 last7）
    voxel_cols = ckpt.get("voxel_cols", None)
    if voxel_cols is None:
        voxel_cols = list(df_vox.columns[-7:])
        print(f"[warn] ckpt 缺少 voxel_cols，fallback 到 df_vox 后7列: {voxel_cols}")
    else:
        voxel_cols = list(voxel_cols)
        print(f"[voxel] using ckpt voxel_cols: {voxel_cols}")

    C = len(voxel_cols)
    if C != 7:
        raise ValueError(f"当前实现按训练脚本假设 voxel 通道数=7，但 voxel_cols={C}。请检查训练脚本保存内容。")

    voxel_grid = np.zeros((C, nx, ny, nz), dtype=np.float32)
    cols_np = [df_vox[c].to_numpy(dtype=np.float32) for c in voxel_cols]

    for i in range(df_vox.shape[0]):
        ix = x_index[float(xv[i])]
        iy = y_index[float(yv[i])]
        iz = z_index[float(zv[i])]
        voxel_grid[:, ix, iy, iz] = np.array([c[i] for c in cols_np], dtype=np.float32)

    # ✅ mask：优先 ckpt["geom_mask_np"]，否则从 df_vox["C0"] 构建
    if "geom_mask_np" in ckpt:
        geom_mask_np = np.array(ckpt["geom_mask_np"], dtype=np.float32)
        if geom_mask_np.shape != (nx, ny, nz):
            raise ValueError(f"ckpt.geom_mask_np 形状 {geom_mask_np.shape} != {(nx,ny,nz)}")
        print("[mask] using ckpt geom_mask_np")
    else:
        geom_mask_np = np.zeros((nx, ny, nz), dtype=np.float32)
        c0_np = df_vox["C0"].to_numpy(dtype=np.float32)
        for i in range(df_vox.shape[0]):
            ix = x_index[float(xv[i])]
            iy = y_index[float(yv[i])]
            iz = z_index[float(zv[i])]
            geom_mask_np[ix, iy, iz] = 1.0 if c0_np[i] > 0.5 else 0.0
        print("[mask] built from df_vox[C0]")

    GEOM_MASK = torch.tensor(geom_mask_np[None, None, ...], dtype=torch.float32, device=device)
    VOXEL_INPUT = torch.tensor(voxel_grid[None, ...], dtype=torch.float32, device=device)
    print(f"[mask] C0(inside_mask) ratio: {geom_mask_np.mean() * 100:.3f}%")

    # ✅ valid 点：优先 ckpt["lin_valid"]
    if "lin_valid" in ckpt:
        lin_valid = np.array(ckpt["lin_valid"], dtype=np.int64)
        print("[mask] using ckpt lin_valid")
    else:
        lin_valid = np.where(geom_mask_np.reshape(-1) > 0.5)[0].astype(np.int64)
        print("[mask] lin_valid derived from mask")

    # ------------------ 构建模型（对齐训练） ------------------
    model = build_model_from_ckpt(ckpt, device=device, nx=nx, ny=ny, nz=nz)

    # ------------------ 读取边界条件 & split（对齐训练） ------------------
    df_bc = pd.read_csv(datapath_bc)
    if df_bc.shape[1] < 7:
        raise ValueError(f"{datapath_bc} 列数不足 7（至少需要前6列BC + 第7列split）。")

    X_data = df_bc.iloc[:, :6].to_numpy(dtype=np.float32)  # (N,6)
    split_raw = df_bc.iloc[:, 6].to_numpy()

    if split_raw.dtype.kind in "OUS":
        split = np.array([str(s).strip().lower() for s in split_raw])
        test_idx = np.where(split == "test")[0]
    else:
        test_idx = np.where(split_raw == 2)[0]

    print(f"[split] test size = {len(test_idx)}")

    # X 标准化：ckpt["x_mean"]/["x_scale"]
    if "x_mean" not in ckpt or "x_scale" not in ckpt:
        raise KeyError("ckpt 缺少 x_mean/x_scale（训练脚本保存的 X scaler）。")

    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt["x_mean"], dtype=np.float64)
    scaler.scale_ = np.array(ckpt["x_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]

    X_test = scaler.transform(X_data[test_idx]).astype(np.float32)

    # ------------------ 读取温度场 Temp_all.csv（对齐训练：full-grid） ------------------
    T_np = pd.read_csv(datapath_temp).to_numpy(dtype=np.float32)
    num_points = int(df_vox.shape[0])

    if T_np.shape[0] == num_points:
        Y_raw = T_np.T
    elif T_np.shape[1] == num_points:
        Y_raw = T_np
    else:
        raise ValueError(f"Temp_all.csv 维度 {T_np.shape} 与点数 num_points={num_points} 不匹配。")

    num_samples = int(Y_raw.shape[0])
    print(f"[temp] samples={num_samples}, points(full csv rows)={num_points}")

    # 把“点表”映射到 full-grid
    ix_all = np.array([x_index[float(v)] for v in xv], dtype=np.int64)
    iy_all = np.array([y_index[float(v)] for v in yv], dtype=np.int64)
    iz_all = np.array([z_index[float(v)] for v in zv], dtype=np.int64)
    lin_all = ix_all * (ny * nz) + iy_all * nz + iz_all

    Y_grid = np.zeros((num_samples, nx * ny * nz), dtype=np.float32)
    Y_grid[:, lin_all] = Y_raw
    Y_grid = Y_grid.reshape((num_samples, nx, ny, nz))  # raw 温度

    # ------------------ Y 反标准化参数（训练保存：样本级 mean/std） ------------------
    if "Y_means" not in ckpt or "Y_stds" not in ckpt:
        raise KeyError("ckpt 缺少 Y_means/Y_stds（训练脚本保存的样本级归一化参数）。")

    Y_means = np.array(ckpt["Y_means"], dtype=np.float32)
    Y_stds = np.array(ckpt["Y_stds"], dtype=np.float32)
    if Y_means.shape[0] != num_samples or Y_stds.shape[0] != num_samples:
        raise ValueError(f"ckpt.Y_means/Y_stds 长度与 num_samples 不一致：{Y_means.shape[0]} vs {num_samples}")

    # ------------------ DataLoader ------------------
    dl = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32)),
                    batch_size=BATCH_SIZE, shuffle=False)

    # ------------------ 推理与评估（real space） ------------------
    NRMSE_list: List[float] = []
    MAE_list: List[float] = []
    MSE_list: List[float] = []
    R2_list: List[float] = []
    GradMSE_list: List[float] = []

    # forward-only timing counters (scope: main)
    pred_time_sum = 0.0
    pred_batches = 0
    pred_samples = 0

    model.eval()
    with torch.no_grad():
        offset = 0
        for (xb,) in dl:
            B = xb.size(0)
            xb = xb.to(device)

            if device.type == "cuda":

                torch.cuda.synchronize()

            t0 = time.perf_counter()

            pred_scaled_t = model(xb)  # (B,nx,ny,nz)  —— scaled & masked

            if device.type == "cuda":

                torch.cuda.synchronize()

            t1 = time.perf_counter()

            pred_time_sum += (t1 - t0)

            pred_batches += 1

            pred_samples += int(B)
            pred_scaled = pred_scaled_t.detach().cpu().numpy().astype(np.float32)

            batch_ids = test_idx[offset: offset + B]

            for bi, sid in enumerate(batch_ids):
                sid = int(sid)

                true_real3d = Y_grid[sid].astype(np.float32)
                mask3d = geom_mask_np.astype(np.float32)

                mu = float(Y_means[sid])
                sd = max(float(Y_stds[sid]), 1e-8)

                # pred_real：只对 valid 点做反标准化；其他点保持 0
                pred_real3d = np.zeros((nx, ny, nz), dtype=np.float32)
                pred_real3d.reshape(-1)[lin_valid] = pred_scaled[bi].reshape(-1)[lin_valid] * sd + mu

                # true_real：保持 raw（来自 Temp_all.csv 的真实温度），mask 外点大概率是 0（或无意义）
                pred_valid = pred_real3d.reshape(-1)[lin_valid]
                true_valid = true_real3d.reshape(-1)[lin_valid]
                mask_valid = np.ones_like(true_valid, dtype=np.float32)

                nrmse = masked_nrmse(pred_valid, true_valid, mask_valid)
                mae = float(np.mean(np.abs(pred_valid - true_valid)))
                mse = float(np.mean((pred_valid - true_valid) ** 2))
                r2 = masked_r2(pred_valid, true_valid, mask_valid)
                gmse = grad_mse_3d(pred_real3d, true_real3d, mask3d)

                NRMSE_list.append(nrmse)
                MAE_list.append(mae)
                MSE_list.append(mse)
                R2_list.append(r2)
                GradMSE_list.append(gmse)

                print(f"sample={sid} | NRMSE={nrmse:.6f} | MAE={mae:.6f} | MSE={mse:.6f} | R2={r2:.6f} | GradMSE={gmse:.6e}")

                save_prediction_artifacts(
                    out_dir=SAVE_DIR,
                    sample_id=sid,
                    pred_scaled3d=pred_scaled[bi],
                    pred_real3d=pred_real3d,
                    true_real3d=true_real3d,
                    mask3d=mask3d,
                    save_npy=SAVE_PRED_NPY,
                )

            offset += B

    if pred_batches > 0:
        avg_batch_s = pred_time_sum / pred_batches
        avg_sample_s = pred_time_sum / max(1, pred_samples)
        print("===== Inference timing (forward only) =====")
        print(f"Total forward time: {pred_time_sum:.6f} s")
        print(f"Avg per batch     : {avg_batch_s:.6f} s")
        print(f"Avg per sample    : {avg_sample_s:.6f} s")

    def mean_std(x: List[float]) -> Tuple[float, float]:
        arr = np.array(x, dtype=np.float64)
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    nrmse_m, nrmse_s = mean_std(NRMSE_list)
    mae_m, mae_s = mean_std(MAE_list)
    mse_m, mse_s = mean_std(MSE_list)
    r2_m, r2_s = mean_std(R2_list)
    gm_m, gm_s = mean_std(GradMSE_list)

    print("===== Summary on test set (real space, mask==1) =====")
    print(f"NRMSE  : mean={nrmse_m:.6f}, std={nrmse_s:.6f}")
    print(f"MAE    : mean={mae_m:.6f}, std={mae_s:.6f}")
    print(f"MSE    : mean={mse_m:.6f}, std={mse_s:.6f}")
    print(f"R2     : mean={r2_m:.6f}, std={r2_s:.6f}")
    print(f"GradMSE: mean={gm_m:.6e}, std={gm_s:.6e}")

    summary = {
        "ckpt_path": CKPT_PATH,
        "model_type": ckpt.get("model_type", ""),
        "voxel_csv": datapath_voxel,
        "voxel_cols": voxel_cols,
        "bc_csv": datapath_bc,
        "temp_csv": datapath_temp,
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "test_idx": test_idx.tolist(),
        "metrics": {
            "NRMSE_mean": nrmse_m, "NRMSE_std": nrmse_s,
            "MAE_mean": mae_m, "MAE_std": mae_s,
            "MSE_mean": mse_m, "MSE_std": mse_s,
            "R2_mean": r2_m, "R2_std": r2_s,
            "GradMSE_mean": gm_m, "GradMSE_std": gm_s,
        },
    }
    with open(os.path.join(SAVE_DIR, "test_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    
    # =============================================================
    # 轻量化/效率对比：测试脚本可测指标输出（推理侧为主）
    # - Params (M)
    # - Checkpoint size (MB)
    # - GFLOPs / forward (B=1, 仅估算 Conv/Linear)
    # - Peak VRAM inference (GB)
    # - Max batch size (fp32 / fp16) —— 推理 forward-only 近似上限
    # 注：Peak VRAM training / Train step time 更适合放训练脚本；此处提供可选“一步反传”基准（默认关闭）。
    # =============================================================

    def _ckpt_size_mb(path: str) -> float:
        try:
            return os.path.getsize(path) / (1024.0 * 1024.0)
        except OSError:
            return float("nan")

    def _params_m(model: nn.Module) -> float:
        return sum(p.numel() for p in model.parameters()) / 1e6

    def estimate_gflops_forward(model: nn.Module, bc_example: torch.Tensor) -> float:
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

        was_training = model.training
        model.eval()
        with torch.no_grad():
            _ = model(bc_example)
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

        return (2.0 * float(macs)) / 1e9

    def probe_max_batch_inference(model: nn.Module, device: torch.device, fp16: bool, max_cap: int = 512) -> int:
        if device.type != "cuda":
            return 0

        def try_bs(bs: int) -> bool:
            try:
                torch.cuda.empty_cache()
                bc = torch.zeros((bs, 6), device=device, dtype=torch.float32)
                with torch.no_grad():
                    if fp16:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            _ = model(bc)
                    else:
                        _ = model(bc)
                return True
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
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
        if device.type != "cuda":
            return float("nan")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        bc = torch.zeros((bs, 6), device=device, dtype=torch.float32)
        with torch.no_grad():
            if fp16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = model(bc)
            else:
                _ = model(bc)
        peak = torch.cuda.max_memory_allocated()
        return float(peak) / (1024.0 ** 3)

    RUN_ONE_TRAIN_STEP_BENCH = False

    def one_train_step_bench(model: nn.Module, device: torch.device, fp16: bool, bs: int = 1) -> tuple[float, float]:
        if device.type != "cuda":
            return float("nan"), float("nan")
        model.train()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        bc = torch.zeros((bs, 6), device=device, dtype=torch.float32)
        with torch.no_grad():
            y_shape = model(bc).shape
        y = torch.zeros(y_shape, device=device, dtype=torch.float32)

        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        if fp16:
            scaler = torch.cuda.amp.GradScaler()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred = model(bc)
                loss = F.mse_loss(pred, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            pred = model(bc)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        peak_gb = float(torch.cuda.max_memory_allocated()) / (1024.0 ** 3)
        step_ms = (t1 - t0) * 1000.0
        model.eval()
        return peak_gb, step_ms

    print("\n===== Lightweightness / Efficiency report (test-side) =====")
    print(f"Checkpoint size (MB): {_ckpt_size_mb(CKPT_PATH):.3f}")
    print(f"Params (M): {_params_m(model):.3f}")

    try:
        bc1 = torch.zeros((1, 6), device=device, dtype=torch.float32)
        gflops = estimate_gflops_forward(model, bc1)
        print(f"GFLOPs / forward (B=1, est.): {gflops:.3f}")
    except Exception as e:
        print(f"GFLOPs / forward (B=1, est.): NA ({e})")

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

        if RUN_ONE_TRAIN_STEP_BENCH:
            p32, t32 = one_train_step_bench(model, device=device, fp16=False, bs=1)
            p16, t16 = one_train_step_bench(model, device=device, fp16=True, bs=1)
            print(f"Peak VRAM training (GB) fp32 (B=1, 1 step): {p32:.3f}")
            print(f"Peak VRAM training (GB) fp16 (B=1, 1 step): {p16:.3f}")
            print(f"Train step time (ms/iter) fp32 (B=1, 1 step): {t32:.2f}")
            print(f"Train step time (ms/iter) fp16 (B=1, 1 step): {t16:.2f}")
    else:
        print("Peak VRAM / Max batch size: NA (CUDA not available)")

    print("测试完成。")

if __name__ == "__main__":
    main()
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python "test_CNN_boundary_channel_aligned_bench.py"