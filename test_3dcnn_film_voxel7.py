# =============================================================
# test_3dcnn_film_voxel7.py
# 加载训练好的 FiLM-A(voxel7) 3D UNet 模型并在测试集上评估
#
# 与最新版训练脚本（CNN_FiLM_sdf&bm_7.py）对齐要点：
# - 模型：FiLM-UNet(depth=2)，固定 7 通道体素输入（来自 cnn_input_channels_no_normals.csv）
# - FiLM 注入：ResidualBlock 内采用 BN -> FiLM -> GELU（而不是 block 输出后再 apply_film）
# - Stem 也条件化：Conv -> BN -> FiLM -> GELU（让 bc 从第一层参与特征提取）
# - FiLM 超参：hidden = base_ch * film_mult；gamma/beta 采用 film_scale 缩放
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


# =============================================================
# 模型定义（与最新版训练脚本一致）
# =============================================================

class FiLMResidualBlock(nn.Module):
    """(Conv -> BN -> FiLM -> GELU) x2 -> Dropout3d + residual"""

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


def build_model_from_ckpt(ckpt: dict) -> nn.Module:
    """从 best_3dcnn_film_voxel7.pth 重建最新版 FiLM-A(voxel7) 网络结构。"""

    model_type = str(ckpt.get("model_type", ""))
    allowed_prefixes = {
        "FiLM_A_voxel7_stem_film_scaled",
        "FiLM_A_voxel7_stem_film",
        "FiLM_A_voxel7",
        "FiLM_A",
    }
    if not any(model_type.startswith(p) for p in allowed_prefixes):
        raise ValueError(
            f"ckpt.model_type={model_type!r} 与该测试脚本不匹配（期望 FiLM_A_voxel7*）。"
        )

    input_dim = int(ckpt["input_dim"])
    nx = int(ckpt["nx"])
    ny = int(ckpt["ny"])
    nz = int(ckpt["nz"])

    # 训练脚本已固定 depth=2，但这里仍允许从 ckpt 读取并强约束
    depth = int(ckpt.get("depth", 2))
    if depth != 2:
        raise ValueError(f"该测试脚本对齐的是 depth=2，但 ckpt.depth={depth}")

    base_ch = int(ckpt.get("base_ch", 24))
    dropout_p = float(ckpt.get("dropout_p", 0.1))
    stem_stride = int(ckpt.get("stem_stride", 2) or 2)

    # ===== FiLM 超参：优先使用 film_mult + film_scale（训练脚本保存的）=====
    film_mult = ckpt.get("film_mult", None)
    if film_mult is None:
        # 兼容旧 ckpt：若只有 film_hidden，则反推一个近似；否则给默认
        film_hidden = ckpt.get("film_hidden", None)
        if film_hidden is None or film_hidden is False:
            film_mult = 4
        else:
            film_mult = max(1, int(round(float(film_hidden) / float(base_ch))))
    film_mult = int(film_mult)
    film_hidden = int(base_ch * film_mult)

    film_scale = float(ckpt.get("film_scale", 1.0))

    class FiLMGen(nn.Module):
        """bc -> {gamma,beta}; 输出按 film_scale 缩放以稳定训练/推理"""

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
                g = s * g_raw
                b = s * b_raw
                gammas.append(g.view(B, C, 1, 1, 1))
                betas.append(b.view(B, C, 1, 1, 1))
            return gammas, betas

    class CNN3D_FiLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.nx, self.ny, self.nz = nx, ny, nz
            self.depth = 2
            self.base_ch = base_ch
            self.film_mult = film_mult
            self.film_hidden = film_hidden
            self.film_scale = film_scale
            self.stem_stride = stem_stride

            # 固定输入：7 通道体素输入 + mask
            assert VOXEL_INPUT is not None, "VOXEL_INPUT 未初始化：请先读取 cnn_input_channels_no_normals.csv"
            assert GEOM_MASK is not None, "GEOM_MASK 未初始化：请先从 C0 构造 mask"
            self.register_buffer("voxel_input", VOXEL_INPUT)  # (1,7,nx,ny,nz)
            self.register_buffer("geom_mask", GEOM_MASK)      # (1,1,nx,ny,nz)

            # Stem：Conv -> BN -> FiLM -> GELU
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

            # Decoder (depth=2 只有 up1)
            self.up1_conv = FiLMResidualBlock(bottleneck_ch + base_ch, base_ch, dropout_p=dropout_p)
            self.out_proj = nn.Conv3d(base_ch, base_ch, kernel_size=1)

            # 全分辨率融合：decoder_feat + 7 通道体素输入
            self.geom_block = FiLMResidualBlock(base_ch + 7, base_ch, dropout_p=dropout_p)
            self.final_conv = nn.Conv3d(base_ch, 1, kernel_size=1)

            # FiLM 注入点通道列表（严格对齐训练脚本的 forward 顺序）
            # 0) stem, 1) enc0, 2) enc1, 3) bottleneck, 4) up1, 5) geom_block
            ch_list = [base_ch, base_ch, base_ch * 2, base_ch * 2, base_ch, base_ch]
            self.film = FiLMGen(input_dim=input_dim, ch_list=ch_list, hidden=film_hidden, scale=film_scale)

        def forward(self, bc: torch.Tensor) -> torch.Tensor:
            B = bc.size(0)
            vox = self.voxel_input.expand(B, -1, -1, -1, -1)    # (B,7,nx,ny,nz)
            mask_ch = self.geom_mask.expand(B, -1, -1, -1, -1)  # (B,1,nx,ny,nz)

            gammas, betas = self.film(bc)
            gi = 0

            # Stem: Conv -> BN -> FiLM -> GELU
            x = self.stem_conv(vox)
            x = self.stem_bn(x)
            x = x * (1.0 + gammas[gi]) + betas[gi]
            x = self.stem_act(x)
            gi += 1

            # Encoder
            x0 = self.enc0(x, gammas[gi], betas[gi]); gi += 1
            x1 = self.pool1(x0)
            x1 = self.enc1(x1, gammas[gi], betas[gi]); gi += 1

            # Bottleneck
            xb = self.bottleneck(x1, gammas[gi], betas[gi]); gi += 1

            # Decoder up1
            x_up = F.interpolate(xb, size=x0.shape[2:], mode="trilinear", align_corners=False)
            x_cat = torch.cat([x_up, x0], dim=1)
            x_dec = self.up1_conv(x_cat, gammas[gi], betas[gi]); gi += 1
            x_dec = self.out_proj(x_dec)

            # Back to full resolution
            x_dec_full = F.interpolate(x_dec, size=(self.nx, self.ny, self.nz), mode="trilinear", align_corners=False)

            # Fuse with voxel input at full res
            vox_full = self.voxel_input.expand(B, -1, -1, -1, -1)
            x_full = torch.cat([x_dec_full, vox_full], dim=1)  # (B,base_ch+7,nx,ny,nz)
            x_full = self.geom_block(x_full, gammas[gi], betas[gi]); gi += 1
            x_full = self.final_conv(x_full)

            out = x_full.squeeze(1)
            out = out * mask_ch.squeeze(1)
            return out

    return CNN3D_FiLM()


# =============================================================
# Masked Loss（与训练脚本一致；用于在测试集上报告 scaled-space loss）
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
        masked = ((pred - target) ** 2) * mask
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
        grad_loss += (((pd - td) ** 2) * md).sum() / (md.sum() + 1e-8)
    grad_loss = grad_loss / 3.0

    return base_loss + grad_weight * grad_loss


# =============================================================
# 指标（仅在 mask==1 的真实点上计算）
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

    # Gradient MSE：只在相邻两点都为有效点的位置计算
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
# 参数统计
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
# 结果保存：预测场 + 有效点 CSV + ✅整场 CSV
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

CKPT_PATH = "best_3dcnn_film_voxel7.pth"
datapath_bc = "data/boundary_condition.csv"
datapath_temp = "data/Temp_all.csv"
datapath_voxel = "data/cnn_input_channels_no_normals.csv"

# ===================== 预测保存设置 =====================
SAVE_DIR = "test_outputs_voxel7"

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
    f"dropout_p={ckpt.get('dropout_p','NA')}, stem_stride={ckpt.get('stem_stride','NA')}"
)
print(
    f"film_mult={ckpt.get('film_mult','NA')}, film_scale={ckpt.get('film_scale','NA')}, "
    f"lr={ckpt.get('lr','NA')}, weight_decay={ckpt.get('weight_decay','NA')}, "
    f"loss_type={ckpt.get('loss_type','NA')}, grad_weight={ckpt.get('grad_weight','NA')}"
)

# -------------------------------------------------------------
# 读取 7 通道体素输入（与训练脚本一致）
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

# C0 作为 inside_mask
if "C0" not in col_order:
    raise ValueError(f"voxel_cols 必须包含 C0 用于 mask，但现在是: {col_order}")
geom_mask_np = (voxel_grid[col_order.index("C0")] > 0.5).astype(np.float32)
GEOM_MASK = torch.tensor(geom_mask_np[None, None, ...], dtype=torch.float32, device=device)
VOXEL_INPUT = torch.tensor(voxel_grid[None, ...], dtype=torch.float32, device=device)
print(f"C0(inside_mask) 占比: {geom_mask_np.mean() * 100:.3f}%")

lin_ckpt = ckpt.get("lin_valid", None)
if lin_ckpt is None:
    raise ValueError("ckpt 中未找到 lin_valid；请使用新版训练脚本保存的模型。")
lin = np.asarray(lin_ckpt, dtype=np.int64)

# -------------------------------------------------------------
# 读取温度：支持两种格式
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
# 边界条件与 split
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
# 恢复标准化参数（与训练脚本保存一致）
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

# 构建测试集 y_scaled：只在有效点上归一化
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
# 构建模型并加载权重
# -------------------------------------------------------------
model = build_model_from_ckpt(ckpt).to(device)

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 张 GPU 进行数据并行推理...")
    model = nn.DataParallel(model)

state_dict = ckpt["state_dict"]
# 兼容保存时带 module. 前缀的情况
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}

model_core = model.module if isinstance(model, nn.DataParallel) else model
missing, unexpected = model_core.load_state_dict(state_dict, strict=False)
if missing:
    print("[Warning] Missing keys:", missing)
if unexpected:
    print("[Warning] Unexpected keys:", unexpected)

model.eval()

# 参数统计
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
# DataLoader
# -------------------------------------------------------------
BATCH_SIZE = 32

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
# 推理 + 评估
# =============================================================
NRMSE_list, MAE_list, MSE_list, R2_list, GradMSE_list = [], [], [], [], []

loss_type_ckpt = ckpt.get("loss_type", "mse")
grad_weight_ckpt = float(ckpt.get("grad_weight", 0.1))

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

        batch_loss = masked_loss(pred_scaled, yb_scaled, mb, loss_type=loss_type_ckpt, grad_weight=grad_weight_ckpt)
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
    print(f"loss={loss_sum / count_sum:.6f} (loss_type={loss_type_ckpt}, grad_weight={grad_weight_ckpt})")

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

print("测试完成。")
# CUDA_VISIBLE_DEVICES=1,2,3,4,5 python "test_3dcnn_film_voxel7.py"
