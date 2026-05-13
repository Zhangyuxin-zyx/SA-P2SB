# from typing import Literal
from typing import Literal, Dict, Union 
from omegaconf import DictConfig

import torch
from einops import reduce
from torch import Tensor
from torch.nn.functional import l1_loss, mse_loss

from metrics.emd_assignment import emd_module as EMD

from .structural_loss import calc_selective_chamfer, calc_repulsion_loss, calc_laplacian_loss

def mean_squared_error(pred: Tensor, gt: Tensor) -> Tensor:
    loss = mse_loss(pred, gt, reduction="none")
    loss = reduce(loss, "b ... -> b", "mean")
    return loss


def mean_squared_error_sum(pred: Tensor, gt: Tensor) -> Tensor:
    loss = mse_loss(pred, gt, reduction="none")
    loss = reduce(loss, "b ... -> b", "sum")
    return loss


def l1(pred: Tensor, gt: Tensor) -> Tensor:
    loss = l1_loss(pred, gt, reduction="none")
    loss = reduce(loss, "b ... -> b", "mean")
    return loss


class EmdLoss:
    def __init__(self):
        self.emd = EMD.emdModule()

    def __call__(self, pred: Tensor, gt: Tensor) -> Tensor:
        if pred.shape[-1] != 3:
            pred = pred.transpose(1, 2)
        if gt.shape[-1] != 3:
            gt = gt.transpose(1, 2)

        distances, _ = self.emd(pred, gt, eps=0.005, iters=50)

        loss = torch.sqrt(distances)
        loss = reduce(loss, "b ... -> b", "mean")
        return loss

class CombinedLoss:
    def __init__(self, cfg: Union[Dict, DictConfig]):
        """
        初始化混合损失函数，从 Config 中读取权重和参数。
        实现热插拔：如果权重为 0，则不计算对应的 Loss。
        """
        # 兼容字典和 OmegaConf
        diff_cfg = cfg.diffusion if hasattr(cfg, "diffusion") else cfg.get("diffusion", {})
        
        # 读取权重 (默认 0.0 表示关闭)
        self.w_scd = diff_cfg.get("scd_weight", 0.0)
        self.w_rep = diff_cfg.get("repulsion_weight", 0.0)
        self.w_lap = diff_cfg.get("laplacian_weight", 0.0)

        # 读取参数
        self.scd_lambda = diff_cfg.get("scd_lambda", 0.98)
        self.rep_k = diff_cfg.get("repulsion_k", 4)
        self.rep_h = diff_cfg.get("repulsion_h", 0.03)

    def __call__(self, pred: Tensor, gt: Tensor) -> Tensor:
        # 1. 基础 MSE (始终计算，作为 Base Loss)
        total_loss = mean_squared_error(pred, gt) # Returns [B]

        # 2. SCD (Selective Chamfer Distance)
        # 只有在权重 > 0 时才计算，节省显存和计算量
        if self.w_scd > 0:
            scd_val = calc_selective_chamfer(pred, gt, lambda_ratio=self.scd_lambda) # Returns [B]
            total_loss = total_loss + (self.w_scd * scd_val)

        # 3. Repulsion Loss
        if self.w_rep > 0:
            rep_val = calc_repulsion_loss(pred, k=self.rep_k, h=self.rep_h) # Returns [B]
            total_loss = total_loss + (self.w_rep * rep_val)

        # 计算 Laplacian Loss
        if self.w_lap > 0:
            lap_val = calc_laplacian_loss(pred, k=10) 
            total_loss = total_loss + (self.w_lap * lap_val)

        return total_loss


def get_loss(cfg: Union[Dict, DictConfig]) -> callable:
    """
    根据配置返回对应的 Loss 函数。
    
    Args:
        cfg: 整个配置对象，包含 diffusion 部分。

    Returns:
        callable: Loss function instance.
    """
    # 获取 diffusion 配置块
    diff_cfg = cfg.diffusion if hasattr(cfg, "diffusion") else cfg.get("diffusion", {})
    
    loss_type = diff_cfg.get("loss_type", "mse")

    # 策略 1: 混合结构损失 (Hybrid Structural Loss)
    # 只要 loss_type 显式指定了混合损失，或者 scd_weight 大于 0，就使用 CombinedLoss
    if loss_type == "mse_scd_repulsion" or diff_cfg.get("scd_weight", 0) > 0:
        return CombinedLoss(cfg)

    # 策略 2: 传统/单一 Loss 
    if loss_type == "mse":
        return mean_squared_error
    if loss_type == "mse_sum":
        return mean_squared_error_sum
    if loss_type == "l1":
        return l1
    if loss_type == "emd":
        return EmdLoss()

    raise ValueError(f"Unknown loss type: {loss_type}")