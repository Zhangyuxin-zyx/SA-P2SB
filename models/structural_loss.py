import torch
import torch.nn as nn

try:
    from pytorch3d.ops import knn_points
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False

def _ensure_last_channel_is_3(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 3:
        return x
    elif x.shape[-2] == 3:
        return x.transpose(-1, -2).contiguous()
    else:
        raise ValueError(f"Unexpected shape: {x.shape}, expected [B, N, 3] or [B, 3, N]")

def calc_selective_chamfer(pred: torch.Tensor, gt: torch.Tensor, lambda_ratio: float = 0.98) -> torch.Tensor:
    """
    计算选择性倒角距离 (Selective Chamfer Distance)
    只计算最小的前 lambda_ratio 部分的距离，忽略离群点。
    """
    if not PYTORCH3D_AVAILABLE:
        raise ImportError("Please install pytorch3d for optimized SCD calculation.")

    # 1. 维度调整 -> [B, N, 3]
    pred = _ensure_last_channel_is_3(pred)
    gt = _ensure_last_channel_is_3(gt)

    B, N, _ = pred.shape
    _, M, _ = gt.shape

    # 2. 计算双向 KNN (K=1)
    # pred 到 gt 的最近距离
    x_nn = knn_points(pred, gt, K=1)
    dists_x2y = x_nn.dists.squeeze(-1)  # [B, N]

    # gt 到 pred 的最近距离
    y_nn = knn_points(gt, pred, K=1)
    dists_y2x = y_nn.dists.squeeze(-1)  # [B, M]

    # 3. 选择性过滤 (Selective)
    # 保留距离最小的 k 个点 (过滤掉最大的 1-lambda_ratio)
    k_x = max(1, int(N * lambda_ratio))
    k_y = max(1, int(M * lambda_ratio))

    # topk with largest=False 取最小值
    val_x, _ = torch.topk(dists_x2y, k=k_x, dim=1, largest=False)
    val_y, _ = torch.topk(dists_y2x, k=k_y, dim=1, largest=False)

    # 4. 求和并平均
    loss = val_x.mean(dim=1) + val_y.mean(dim=1)
    
    # 返回每个 batch 的 loss [B]，需要在外部取 mean
    return loss

def calc_laplacian_loss(pred: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    计算拉普拉斯平滑损失 (Laplacian Smoothness Loss)
    让每个点靠近其 K 个邻居的重心，从而消除高频抖动/浮噪。
    """
    if not PYTORCH3D_AVAILABLE:
        return torch.tensor(0.0).to(pred.device)

    # 1. 维度调整 [B, N, 3]
    pred = _ensure_last_channel_is_3(pred)
    
    # 2. 找 K 个邻居
    # pred: [B, N, 3]
    knn_res = knn_points(pred, pred, K=k+1) # K+1 因为包含自己
    idx = knn_res.idx[:, :, 1:] # [B, N, k] 去掉自己
    
    # 3. 获取邻居坐标
    B, N, _ = pred.shape
    # 展平 batch 索引以便 gather
    batch_idx = torch.arange(B, device=pred.device).view(B, 1, 1).repeat(1, N, k)
    
    # gather 邻居坐标: [B, N, k, 3]
    neighbors = pred[batch_idx, idx, :]
    
    # 4. 计算邻居重心 (Centroid)
    centroids = neighbors.mean(dim=2) # [B, N, 3]
    
    # 5. 计算点到重心的距离 (L2)
    loss = torch.norm(pred - centroids, dim=-1).mean()
    
    return loss

def calc_repulsion_loss(pred: torch.Tensor, k: int = 4, h: float = 0.03) -> torch.Tensor:
    """
    计算排斥损失 (Repulsion Loss)
    防止点云聚集/塌陷。
    """
    if not PYTORCH3D_AVAILABLE:
        raise ImportError("Please install pytorch3d for optimized Repulsion calculation.")

    # 1. 维度调整 -> [B, N, 3]
    pred = _ensure_last_channel_is_3(pred)

    # 2. 计算自身 KNN
    # K+1 因为包含自己
    knn_res = knn_points(pred, pred, K=k + 1)
    dists = knn_res.dists # [B, N, K+1]

    # 3. 去掉自己到自己的距离 (通常是第0个，距离为0)
    dists = dists[:, :, 1:] 
    
    # 4. 计算斥力
    # dists 是平方距离，先开根号得到欧氏距离，加上 epsilon 防止梯度爆炸
    dists = torch.sqrt(dists + 1e-8) 
    
    # 高斯核衰减：距离越近，斥力越大
    repulsion = -dists * torch.exp(-(dists**2) / (h**2))
    
    # 对所有点和邻居取平均
    return repulsion.mean(dim=[1, 2]) # 返回 [B]