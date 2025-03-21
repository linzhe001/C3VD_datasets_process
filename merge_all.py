import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
import argparse
import re

def load_poses(pose_path):
    """从 pose.txt 加载所有位姿（直接加载 T_cam_to_world），并自动修正格式。"""
    print(f"正在加载位姿文件：{pose_path}")
    poses = []
    with open(pose_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = list(map(float, line.strip().split(',')))
            if len(data) != 16:
                raise ValueError(f"位姿行数据长度错误，期望16个数字，实际为{len(data)}，内容：{line}")
            T = np.array(data).reshape(4, 4)
            if not np.allclose(T[3, :], [0, 0, 0, 1]):
                if np.allclose(T[:3, 3], np.zeros(3)) and not np.allclose(T[3, :3], np.zeros(3)):
                    T_fixed = np.eye(4)
                    T_fixed[:3, :3] = T[:3, :3]
                    T_fixed[:3, 3] = T[3, :3]
                    T = T_fixed
                else:
                    raise ValueError(f"无效的变换矩阵，行：{line}")
            poses.append(T)
    print(f"共加载了 {len(poses)} 个位姿")
    return poses

def merge_point_clouds(input_dir, output_path, voxel_size=0.05, pose_path=None):
    """
    将目录中所有点云文件合并为一个点云
    
    参数:
        input_dir: 包含点云文件的目录
        output_path: 合并后点云的保存路径
        voxel_size: 下采样的体素大小
        pose_path: 可选的位姿文件路径，如果提供则应用位姿变换
    """
    # 支持的点云文件扩展名
    supported_extensions = ['.pcd', '.ply', '.xyz', '.pts']
    
    # 查找目录中的所有点云文件
    pcd_files = []
    for file in sorted(os.listdir(input_dir)):
        if any(file.lower().endswith(ext) for ext in supported_extensions):
            pcd_path = os.path.join(input_dir, file)
            pcd_files.append(pcd_path)
    
    if not pcd_files:
        raise ValueError(f"在目录 {input_dir} 中未找到任何支持的点云文件")
    
    print(f"找到 {len(pcd_files)} 个点云文件")
    
    # 如果提供了位姿文件，则加载位姿
    poses = None
    if pose_path and os.path.exists(pose_path):
        poses = load_poses(pose_path)
    
    # 合并所有点云
    combined_pcd = o3d.geometry.PointCloud()
    batch_count = 0
    
    for pcd_path in tqdm(pcd_files, desc="合并点云"):
        # 读取点云
        try:
            pcd = o3d.io.read_point_cloud(pcd_path)
            if not pcd.has_points():
                print(f"警告: 文件 {pcd_path} 无法加载有效点云，跳过")
                continue
            
            # 如果提供了位姿，尝试应用位姿变换
            if poses is not None:
                # 从文件名中提取帧索引
                filename = os.path.basename(pcd_path)
                match = re.search(r'(\d+)', filename)
                if match:
                    frame_idx = int(match.group(1))
                    if frame_idx < len(poses):
                        pcd.transform(poses[frame_idx])
                    else:
                        print(f"警告: 帧索引 {frame_idx} 超出位姿数组范围，不应用变换")
            
            # 对单帧点云进行下采样以减少数据量
            pcd_ds = pcd.voxel_down_sample(voxel_size)
            combined_pcd += pcd_ds
            
            # 每添加10个点云后进行一次体素下采样控制点数
            batch_count += 1
            if batch_count % 10 == 0:
                combined_pcd = combined_pcd.voxel_down_sample(voxel_size)
                print(f"已处理 {batch_count} 个点云，当前点数: {len(combined_pcd.points)}")
        except Exception as e:
            print(f"处理文件 {pcd_path} 时出错: {e}")
    
    # 最终下采样合并后的点云
    print(f"执行最终下采样 (当前点数: {len(combined_pcd.points)})...")
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size)
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存合并后的点云
    print(f"保存合并点云到: {output_path}")
    o3d.io.write_point_cloud(output_path, combined_pcd)
    print(f"点云合并完成，共有 {len(combined_pcd.points)} 个点")
    
    return combined_pcd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将目录中所有点云文件合并为一个点云")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="包含点云文件的输入目录")
    parser.add_argument("--output_path", type=str, required=True,
                        help="合并后点云的保存路径")
    parser.add_argument("--voxel_size", type=float, default=0.05,
                        help="下采样的体素大小")
    parser.add_argument("--pose_path", type=str, default=None,
                        help="可选的位姿文件路径，如果提供则应用位姿变换")
    
    args = parser.parse_args()
    
    merge_point_clouds(args.input_dir, args.output_path, args.voxel_size, args.pose_path)
