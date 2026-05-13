import torch.nn as nn
import torch.nn.functional as F
import torch

def knn(x, k):
    # x: [B, C, N]
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    # x: [B, C, N]
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    # Feature: [x_j - x_i, x_i] (Relative coords + Global coords)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.final_norm = nn.InstanceNorm1d(out_channels)

    def forward(self, x):
        # x: [B, C, N]
        x_graph = get_graph_feature(x, k=self.k) 
        x = self.conv(x_graph)
        # x = x.max(dim=-1, keepdim=False)[0] # Max pooling over neighbors
        x = x.mean(dim=-1, keepdim=False) 
        x = self.final_norm(x)
        return x

class TimeGatedGeometryBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.GroupNorm(8, out_channels), 
            nn.SiLU() 
        )
        
        self.proj_back = nn.Conv1d(out_channels, 3, 1)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        nn.init.zeros_(self.proj_back.weight)
        nn.init.zeros_(self.proj_back.bias)

    def forward(self, x, noise_level):
        # x: [B, 3, N]
        # noise_level: [B]

        with torch.no_grad():
            idx = knn(x, self.k)
        x_graph = get_graph_feature(x, k=self.k, idx=idx) 
        
        feat = self.conv(x_graph)
        feat = feat.max(dim=-1)[0] # [B, out, N]

        t_in = noise_level.view(-1, 1).float()

        gate = (1.0 - t_in) * self.time_mlp(t_in) # [B, 1]
        gate = gate.view(-1, 1, 1) 
        
        feat_residual = self.proj_back(feat) # [B, 3, N]
        
        return feat_residual * gate