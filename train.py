import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

from AutoPoint2ScanNet import AutoPoint2ScanNet


# 自定义数据集类
class PointCloud2LaserScanDataset(Dataset):
    def __init__(self, data_dir, max_points=1024, augment=True):
        self.data_dir = data_dir
        self.pointcloud_files = sorted([f for f in os.listdir(data_dir) if f.startswith('pointcloud_')])
        self.laserscan_files = sorted([f for f in os.listdir(data_dir) if f.startswith('laserscan_')])
        self.max_points = max_points
        self.augment = augment
        
        # 确保点云和LaserScan文件匹配
        assert len(self.pointcloud_files) == len(self.laserscan_files), "点云和LaserScan文件数量不匹配"
    
    def __len__(self):
        return len(self.pointcloud_files)
    
    def __getitem__(self, idx):
        # 加载点云
        pc_file = os.path.join(self.data_dir, self.pointcloud_files[idx])
        pointcloud = np.loadtxt(pc_file, dtype=np.float32)
        
        # 加载LaserScan
        ls_file = os.path.join(self.data_dir, self.laserscan_files[idx])
        laserscan = np.loadtxt(ls_file, dtype=np.float32)
        
        # 数据增强
        if self.augment:
            # 随机旋转（绕Z轴）
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            
            pointcloud = np.dot(pointcloud, R)
            laserscan = np.dot(laserscan, R)
            
            # 随机添加噪声
            pointcloud += np.random.normal(0, 0.01, size=pointcloud.shape).astype(np.float32)
        
        # 随机采样点云（如果点数超过最大值）
        if len(pointcloud) > self.max_points:
            indices = np.random.choice(len(pointcloud), self.max_points, replace=False)
            pointcloud = pointcloud[indices]
        
        # 计算真实LaserScan点数
        num_points = len(laserscan)
        
        return {
            'pointcloud': pointcloud,
            'laserscan': laserscan,
            'num_points': num_points
        }


# 自定义损失函数
class PointCloud2LaserScanLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predicted_coords, predicted_points, target_coords, target_points):
        batch_size = predicted_coords.shape[0]
        total_loss = 0
        
        # 点数预测损失（归一化到[0,1]范围）
        points_loss = self.mse_loss(
            predicted_points.float() / predicted_coords.shape[1], 
            target_points.float() / predicted_coords.shape[1]
        )
        
        # 坐标损失（对每个样本分别计算）
        coord_loss = 0
        for b in range(batch_size):
            # 如果预测点数和目标点数不同，需要对齐
            pred_points = predicted_coords[b, :predicted_points[b]]
            target = target_coords[b, :target_points[b]]
            
            # 使用最近邻匹配计算损失
            pred_np = pred_points.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            
            # 构建KD树用于最近邻搜索
            tree = KDTree(target_np)
            distances, indices = tree.query(pred_np, k=1)
            
            # 计算匹配点的MSE损失
            matched_target = torch.tensor(target_np[indices[:, 0]], device=pred_points.device)
            coord_loss += self.mse_loss(pred_points, matched_target)
        
        coord_loss /= batch_size
        
        # 总损失 = 坐标损失 + 点数预测损失
        total_loss = coord_loss + 0.1 * points_loss
        
        return total_loss, coord_loss, points_loss


# 评估指标计算
def calculate_metrics(pred_coords, target_coords, pred_points, target_points):
    batch_size = pred_coords.shape[0]
    all_distances = []
    
    for b in range(batch_size):
        pred = pred_coords[b, :pred_points[b]].detach().cpu().numpy()
        target = target_coords[b, :target_points[b]].detach().cpu().numpy()
        
        # 计算每个预测点到最近目标点的距离
        tree = KDTree(target)
        distances, _ = tree.query(pred, k=1)
        all_distances.extend(distances.flatten())
    
    # 计算平均误差和最大误差
    mean_error = np.mean(all_distances)
    max_error = np.max(all_distances)
    rmse = np.sqrt(np.mean(np.square(all_distances)))
    
    return {
        'mean_error': mean_error,
        'max_error': max_error,
        'rmse': rmse
    }


# 训练函数
def train(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 创建保存模型的目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据集
    train_dataset = PointCloud2LaserScanDataset(
        data_dir=args.train_data_dir,
        max_points=args.max_points,
        augment=True
    )
    
    val_dataset = PointCloud2LaserScanDataset(
        data_dir=args.val_data_dir,
        max_points=args.max_points,
        augment=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 初始化模型
    model = AutoPoint2ScanNet(
        num_features=3,
        max_points=args.max_points
    ).to(args.device)
    
    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 定义学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 定义损失函数
    criterion = PointCloud2LaserScanLoss()
    
    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_coord_loss = 0.0
        train_points_loss = 0.0
        
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for batch in train_progress:
            # 准备数据
            pointclouds = batch['pointcloud'].to(args.device)
            laserscans = batch['laserscan'].to(args.device)
            num_points = batch['num_points'].to(args.device)
            
            # 前向传播
            optimizer.zero_grad()
            predicted_coords, predicted_points = model(pointclouds)
            
            # 计算损失
            loss, coord_loss, points_loss = criterion(
                predicted_coords, predicted_points,
                laserscans, num_points
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录损失
            train_loss += loss.item()
            train_coord_loss += coord_loss.item()
            train_points_loss += points_loss.item()
            
            # 更新进度条
            train_progress.set_postfix({
                'loss': loss.item(),
                'coord_loss': coord_loss.item(),
                'points_loss': points_loss.item()
            })
        
        # 计算平均损失
        train_loss /= len(train_loader)
        train_coord_loss /= len(train_loader)
        train_points_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_metrics = {'mean_error': 0, 'max_error': 0, 'rmse': 0}
        
        val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
        with torch.no_grad():
            for batch in val_progress:
                # 准备数据
                pointclouds = batch['pointcloud'].to(args.device)
                laserscans = batch['laserscan'].to(args.device)
                num_points = batch['num_points'].to(args.device)
                
                # 前向传播
                predicted_coords, predicted_points = model(pointclouds)
                
                # 计算损失
                loss, _, _ = criterion(
                    predicted_coords, predicted_points,
                    laserscans, num_points
                )
                
                # 计算评估指标
                metrics = calculate_metrics(
                    predicted_coords, laserscans,
                    predicted_points, num_points
                )
                
                # 记录损失和指标
                val_loss += loss.item()
                val_metrics['mean_error'] += metrics['mean_error']
                val_metrics['max_error'] += metrics['max_error']
                val_metrics['rmse'] += metrics['rmse']
                
                # 更新进度条
                val_progress.set_postfix({
                    'loss': loss.item(),
                    'mean_error': metrics['mean_error'],
                    'rmse': metrics['rmse']
                })
        
        # 计算平均损失和指标
        val_loss /= len(val_loader)
        val_metrics['mean_error'] /= len(val_loader)
        val_metrics['max_error'] /= len(val_loader)
        val_metrics['rmse'] /= len(val_loader)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'Train Loss: {train_loss:.6f} (Coord: {train_coord_loss:.6f}, Points: {train_points_loss:.6f})')
        print(f'Val Loss: {val_loss:.6f}, Mean Error: {val_metrics["mean_error"]:.6f}, RMSE: {val_metrics["rmse"]:.6f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f'Best model saved at epoch {epoch+1} with val loss: {best_val_loss:.6f}')
        
        # 保存当前模型
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'current_model.pth'))
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(os.path.join(args.save_dir, 'loss_curve.png'))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AutoPoint2ScanNet 训练脚本')
    parser.add_argument('--train_data_dir', type=str, default='', required=True, help='训练数据集目录')
    parser.add_argument('--val_data_dir', type=str, default='', required=True, help='验证数据集目录')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda', help='使用的设备 (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--max_points', type=int, default=1024, help='最大点云点数')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    train(args)