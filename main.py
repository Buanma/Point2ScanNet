import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

# 加载点云数据
def load_point_cloud(file_path):
    """从文本文件加载点云数据"""
    points = np.loadtxt(file_path, dtype=np.float32)
    return points


# 保存LaserScan结果（xyz格式）
def save_laserscan_xyz(scan_points, file_path):
    """将LaserScan的三维坐标点保存到文本文件"""
    np.savetxt(file_path, scan_points, fmt='%.6f')


# 主函数
def main(args):
    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = Point2Scan3DNet(num_features=3, max_points=args.max_points)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 加载点云数据
    print(f"加载点云数据: {args.input_file}")
    points = load_point_cloud(args.input_file)
    
    # 转换为张量并添加批次维度
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N, 3)
    
    # 运行模型推理
    print(f"运行模型推理 (输出点数: {args.num_points})")
    with torch.no_grad():
        scan_result = model(points_tensor, num_scan_points=args.num_points)
    
    # 转换为numpy数组并保存结果
    scan_result_np = scan_result.cpu().numpy()[0]  # (M, 3)
    save_laserscan_xyz(scan_result_np, args.output_file)
    
    print(f"LaserScan结果已保存到: {args.output_file}")
    print(f"输出格式: {args.num_points} 个点 (每行包含 x y z 坐标)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='点云到三维LaserScan转换工具')
    parser.add_argument('--input_file', type=str, required=True, help='输入点云文件路径 (txt格式)')
    parser.add_argument('--output_file', type=str, required=True, help='输出LaserScan文件路径 (txt格式)')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--num_points', type=int, default=360, help='输出LaserScan的点数')
    parser.add_argument('--max_points', type=int, default=1024, help='模型支持的最大点数')
    parser.add_argument('--device', type=str, default='auto', help='使用的设备 (auto/cpu/gpu)')
    
    args = parser.parse_args()
    main(args)