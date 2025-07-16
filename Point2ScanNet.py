import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PointPillarsEncoder(nn.Module):
    def __init__(self, num_features=3, voxel_size=(0.1, 0.1, 0.2), 
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1)):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(num_features, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.backbone = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, 2),
            ResidualBlock(128, 256),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, points):
        # 简化版体素化（实际应用中需使用高效实现）
        B, N, C = points.shape
        
        # 坐标归一化到体素网格
        pc_range = torch.tensor(self.point_cloud_range, device=points.device)
        voxel_size = torch.tensor(self.voxel_size, device=points.device)
        grid_size = ((pc_range[3:6] - pc_range[0:3]) / voxel_size).long()
        
        # 将点云坐标转换为体素索引
        voxels = torch.zeros(B, grid_size[2], grid_size[1], grid_size[0], C, device=points.device)
        
        # 简化处理：只保留每个体素中的第一个点
        indices = ((points - pc_range[0:3]) / voxel_size).long()
        indices = torch.clamp(indices, min=0, max=grid_size.unsqueeze(0).unsqueeze(0) - 1)
        
        for b in range(B):
            for i in range(N):
                x, y, z = indices[b, i, 0], indices[b, i, 1], indices[b, i, 2]
                voxels[b, z, y, x] = points[b, i]
        
        # 转换为伪图像格式
        voxels = voxels.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, Z, Y, X)
        voxels = voxels.view(B, C, -1)  # (B, C, Z*Y*X)
        
        # 特征提取
        voxel_features = self.feature_extractor(voxels)  # (B, 64, Z*Y*X)
        
        # 重塑为2D伪图像
        pseudo_image = voxel_features.view(B, 64, grid_size[2], grid_size[1]*grid_size[0])
        
        # 骨干网络提取特征
        features = self.backbone(pseudo_image)
        return features


class DynamicAttentionPooling(nn.Module):
    def __init__(self, feature_channels=256):
        super().__init__()
        self.attention_generator = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_channels)
        )
    
    def forward(self, features, num_points=360):
        B, C, H, W = features.shape
        
        # 生成角度位置编码
        angles = torch.linspace(0, 2*torch.pi, num_points, device=features.device)
        pos_encoding = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)  # (M, 2)
        
        # 生成注意力权重
        attention_weights = self.attention_generator(pos_encoding)  # (M, C)
        
        # 全局特征池化
        global_features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)  # (B, C)
        
        # 注意力加权
        direction_features = torch.matmul(attention_weights, global_features.unsqueeze(-1)).squeeze(-1)  # (B, M, C)
        
        return direction_features.permute(0, 2, 1)  # (B, C, M)


class CoordinateRegressionHead(nn.Module):
    def __init__(self, feature_channels=256):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Conv1d(feature_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)  # 直接输出x,y,z坐标
        )
    
    def forward(self, x):
        coordinates = self.regressor(x).permute(0, 2, 1)  # (B, M, 3)
        return coordinates


class Point2Scan3DNet(nn.Module):
    def __init__(self, num_features=3, max_points=1024):
        super().__init__()
        self.encoder = PointPillarsEncoder(num_features)
        self.pooling = DynamicAttentionPooling()
        self.regressor = CoordinateRegressionHead()
        self.max_points = max_points
    
    def forward(self, points, num_scan_points=360):
        num_scan_points = min(num_scan_points, self.max_points)
        features = self.encoder(points)
        direction_features = self.pooling(features, num_scan_points)
        coordinates = self.regressor(direction_features)
        return coordinates