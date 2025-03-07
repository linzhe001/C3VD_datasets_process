"""
全向相机深度图转点云工具

此脚本实现了将全向相机深度图转换为3D点云的功能。
基于Scaramuzza等人的全向相机模型。

使用说明:
python depth_to_pointcloud.py --depth_dir /path/to/depth/images --output_dir /path/to/output --config /path/to/config.ini

参数:
  - depth_dir: 深度图文件夹路径
  - output_dir: 输出点云文件夹路径
  - config: 配置文件路径，包含相机内参
  - pose_file: (可选) 包含相机位姿的文件
"""

import os
import argparse
import numpy as np
import cv2
import glob
import configparser
from tqdm import tqdm
from pathlib import Path
import open3d as o3d

class OmnidirectionalCamera:
    """全向相机模型类"""
    
    def __init__(self, width, height, cx, cy, a0, a2, a3, a4, c, d, e):
        """
        初始化全向相机模型
        
        参数:
            width, height: 图像尺寸
            cx, cy: 图像中心点
            a0, a2, a3, a4: 多项式系数
            c, d, e: 拉伸矩阵参数
        """
        self.width = width
        self.height = height
        self.center = np.array([cx, cy])
        self.poly_coeff = np.array([a0, a2, a3, a4])
        
        # 创建拉伸矩阵
        self.stretch_mat = np.array([[c, d], [e, 1.0]])
        
        # 计算行列式
        det = c - d*e
        
        # 检查矩阵是否奇异，如果接近奇异则进行调整
        if abs(det) < 1e-10:
            print("警告：拉伸矩阵接近奇异，使用默认值")
            # 使用轻微修改的单位矩阵
            self.stretch_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
            self.stretch_mat_inv = np.array([[1.0, 0.0], [0.0, 1.0]])
        else:
            # 计算逆矩阵
            self.stretch_mat_inv = np.linalg.inv(self.stretch_mat)
    
    def pixel_to_ray(self, px):
        """
        将像素坐标转换为射线方向
        
        参数:
            px: 像素坐标 [x, y]
            
        返回:
            ray: 归一化的射线方向 [x, y, z]
        """
        # 转换到屏幕空间
        uvp = px - self.center
        
        # 使用拉伸矩阵校正畸变
        uvpp = self.stretch_mat_inv @ uvp
        
        # 计算径向距离
        rho = np.sqrt(uvpp[0]**2 + uvpp[1]**2)
        
        # 使用多项式模型计算z值
        z = self.poly_coeff[0] + \
            self.poly_coeff[1] * rho**2 + \
            self.poly_coeff[2] * rho**3 + \
            self.poly_coeff[3] * rho**4
        
        # 组合并归一化射线方向
        ray = np.array([uvpp[0], uvpp[1], z])
        ray = ray / np.linalg.norm(ray)
        
        return ray

def read_config(config_path):
    """读取配置文件"""
    # 检查文件扩展名
    with open(config_path, 'r') as f:
        values = f.readline().strip().split()
        
    if len(values) < 11:
        raise ValueError(f"参数文件格式错误，期望至少11个参数，实际获得{len(values)}个")
        
    width = int(values[0])      # Width
    height = int(values[1])     # height
    cx = float(values[2])       # cx
    cy = float(values[3])       # cy
    a0 = float(values[4])       # a0
    a2 = float(values[5])       # a2
    a3 = float(values[6])       # a3
    a4 = float(values[7])       # a4
    e = float(values[8])        # e
    c = float(values[9])        # f 对应代码中的c参数
    d = float(values[10])       # g 对应代码中的d参数
    
    print(f"从文本文件加载相机参数: {config_path}")
    return OmnidirectionalCamera(width, height, cx, cy, a0, a2, a3, a4, c, d, e)

def read_pose_file(pose_path, frame_idx=0):
    """
    读取相机位姿文件
    
    参数:
        pose_path: 位姿文件路径
        frame_idx: 要读取的帧索引
        
    返回:
        4x4变换矩阵
    """
    if not os.path.exists(pose_path):
        # 如果没有位姿文件，返回单位矩阵
        return np.eye(4)
    
    poses = []
    with open(pose_path, 'r') as f:
        for line in f:
            values = [float(v) for v in line.strip().split(',')]
            if len(values) == 16:
                pose = np.array(values).reshape(4, 4)
                poses.append(pose)
    
    if frame_idx < len(poses):
        return poses[frame_idx]
    else:
        print(f"警告: 位姿文件中没有索引为 {frame_idx} 的帧，使用单位矩阵")
        return np.eye(4)

def depth_to_pointcloud(depth_image, camera, camera_pose=None, max_depth=100.0, min_depth=0.1):
    """
    将深度图转换为点云
    
    参数:
        depth_image: 16位深度图像
        camera: 全向相机模型
        camera_pose: 相机位姿矩阵 (4x4)
        max_depth: 最大深度值
        min_depth: 最小有效深度值
        
    返回:
        点云 (N x 3)
    """
    if camera_pose is None:
        camera_pose = np.eye(4)
    
    # 归一化深度图 (16位 -> 实际深度值)
    normalized_depth = depth_image.astype(float) / 65535.0 * max_depth
    
    # 创建像素网格
    y_coords, x_coords = np.mgrid[0:camera.height, 0:camera.width]
    pixels = np.stack([x_coords.flatten(), y_coords.flatten()], axis=-1)
    
    # 获取有效深度像素 - 添加最小深度过滤
    valid_mask = (normalized_depth.flatten() > min_depth) & (normalized_depth.flatten() < max_depth)
    valid_pixels = pixels[valid_mask]
    valid_depths = normalized_depth.flatten()[valid_mask]
    
    # 初始化点云数组
    points = []
    
    # 每次处理一批点以提高效率
    batch_size = 10000
    for i in range(0, len(valid_pixels), batch_size):
        batch_pixels = valid_pixels[i:i+batch_size]
        batch_depths = valid_depths[i:i+batch_size]
        
        batch_points = []
        for px, depth in zip(batch_pixels, batch_depths):
            # 获取射线方向 (相机坐标系)
            ray_dir_local = camera.pixel_to_ray(px)
            
            # 转换射线到世界坐标系
            ray_dir_world = (camera_pose[:3, :3] @ ray_dir_local).flatten()
            
            # 获取相机位置
            camera_position = camera_pose[:3, 3]
            
            # 计算3D点
            point = camera_position + depth * ray_dir_world
            
            # 检查点是否在合理范围内
            if np.all(np.abs(point) < 1000):  # 过滤异常值
                batch_points.append(point)
        
        points.extend(batch_points)
    
    # 转换为数组
    points_array = np.array(points)
    
    # 如果点云为空，返回空数组
    if len(points_array) == 0:
        return np.zeros((0, 3))
    
    # 过滤离群点 - 使用统计过滤
    mean = np.mean(points_array, axis=0)
    std = np.std(points_array, axis=0)
    mask = np.all(np.abs(points_array - mean) < 3 * std, axis=1)
    
    return points_array[mask]

def save_pointcloud(points, output_path, voxel_size=0.01, max_points=None):
    """
    保存点云到文件
    
    参数:
        points: 点云数组 (N x 3)
        output_path: 输出文件路径
        voxel_size: 体素下采样大小
        max_points: 最大点数
    """
    if len(points) == 0:
        print(f"警告: 点云为空，未保存 {output_path}")
        return
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 估计法向量
    pcd.estimate_normals()
    
    # 体素下采样
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    
    # 限制点数
    if max_points is not None and len(pcd.points) > max_points:
        pcd = pcd.random_down_sample(max_points / len(pcd.points))
    
    # 保存点云
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云已保存到 {output_path}，共 {len(pcd.points)} 个点")

def process_directory(args):
    """处理整个目录的深度图，包括子目录中的不同场景"""
    # 读取相机内参
    camera = read_config(args.config)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 遍历所有子目录
    total_files_processed = 0
    for root, dirs, files in os.walk(args.depth_dir):
        # 对每个子目录（场景）处理深度图
        depth_files = sorted([f for f in files if f.endswith('_depth.tiff') or f.endswith('.tiff')])
        
        if not depth_files:
            continue
            
        # 检查当前场景的位姿文件 - 查找 pose.txt
        scene_pose_file = os.path.join(root, 'pose.txt')
            
        if not os.path.exists(scene_pose_file) and args.pose_file:
            # 如果场景没有位姿文件，使用传入的默认位姿文件
            scene_pose_file = args.pose_file
        elif not os.path.exists(scene_pose_file):
            # 如果没有找到位姿文件，使用None
            scene_pose_file = None
            print(f"警告: 场景 {root} 没有找到位姿文件 (pose.txt)，将使用单位矩阵")
        else:
            print(f"使用位姿文件: {scene_pose_file}")
        
        print(f"处理场景: {root}")
        print(f"找到 {len(depth_files)} 个深度图文件")
        
        # 处理每个深度图
        for depth_file in tqdm(depth_files, desc=f"处理场景 {os.path.basename(root)} 中的深度图"):
            # 获取文件名
            file_name = os.path.basename(depth_file)
            file_base = os.path.splitext(file_name)[0]
            
            # 提取帧索引 - 尝试多种格式
            if '_depth' in file_base:
                frame_idx_str = file_base.split('_')[0]
            else:
                frame_idx_str = file_base
                
            try:
                frame_idx = int(frame_idx_str)
            except ValueError:
                # 如果无法提取帧索引，使用文件顺序作为索引
                frame_idx = total_files_processed
                
            # 读取深度图
            depth_img_path = os.path.join(root, depth_file)
            depth_img = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
            
            # 读取相机位姿 (如果有的话)
            camera_pose = None
            if scene_pose_file and os.path.exists(scene_pose_file):
                camera_pose = read_pose_file(scene_pose_file, frame_idx)
            
            # 转换为点云
            points = depth_to_pointcloud(depth_img, camera, camera_pose)
            
            # 保持原始文件夹结构
            rel_path = os.path.relpath(root, args.depth_dir)
            scene_output_dir = os.path.join(args.output_dir, rel_path)
            os.makedirs(scene_output_dir, exist_ok=True)
            
            # 保存点云
            output_path = os.path.join(scene_output_dir, f"{file_base}.ply")
            save_pointcloud(points, output_path)
            
            total_files_processed += 1
    
    print(f"共处理 {total_files_processed} 个深度图文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全向相机深度图转点云工具")
    parser.add_argument("--depth_dir", required=True, help="深度图文件夹路径")
    parser.add_argument("--output_dir", required=True, help="输出点云文件夹路径")
    parser.add_argument("--config", required=True, help="包含相机内参的配置文件路径")
    parser.add_argument("--pose_file", help="包含相机位姿的文件路径 (可选)")
    
    args = parser.parse_args()
    process_directory(args)

