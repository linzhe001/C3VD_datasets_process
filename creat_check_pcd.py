import cv2
import numpy as np
import open3d as o3d
import os
import argparse
import gc
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def depth_image_to_point_cloud(depth_image_path, fx, fy, cx, cy, k1, k2, k3, k4, scale_factor=0.001, voxel_size=None, batch_size=10, max_points=30000):
    # 读取深度图
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
    
    # 打印深度图信息
    print(f"深度图形状: {depth_image.shape}, 数据类型: {depth_image.dtype}")
    print(f"深度值范围: {np.min(depth_image)} - {np.max(depth_image)}")
    
    # 获取深度图的尺寸
    height, width = depth_image.shape
    
    # 设置相机内参矩阵
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    # 设置畸变系数 - 针对鱼眼相机模型
    D = np.array([k1, k2, k3, k4])
    
    try:
        # 使用鱼眼相机模型校正畸变
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (width, height), cv2.CV_32FC1
        )
        undistorted_depth = cv2.remap(depth_image, map1, map2, cv2.INTER_LINEAR)
        # 释放原始深度图内存
        del depth_image, map1, map2
    except Exception as e:
        print(f"畸变校正失败，使用原始深度图: {e}")
        undistorted_depth = depth_image
        del depth_image
    
    # 创建有效深度的掩码
    valid_mask = undistorted_depth > 0
    valid_depth_count = np.sum(valid_mask)
    print(f"有效深度像素数: {valid_depth_count}")
    
    if valid_depth_count == 0:
        print("警告：未找到有效深度值")
        return np.array([[0, 0, 0]])
    
    # 向量化计算点云（替代嵌套循环）
    # 创建网格坐标
    v, u = np.indices((height, width))
    
    # 仅选择有效深度的像素
    z = undistorted_depth[valid_mask] * scale_factor
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    
    # 计算3D坐标
    x = (u_valid - cx) / fx * z
    y = (v_valid - cy) / fy * z
    
    # 合并坐标创建点云
    point_cloud = np.column_stack((x, y, z))
    
    # 释放不再需要的内存
    del undistorted_depth, valid_mask, v, u, z, u_valid, v_valid, x, y
    
    # 如果需要下采样
    if voxel_size is not None and voxel_size > 0:
        # 使用Open3D的内置体素下采样方法（比手动实现更快）
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        # 转回NumPy数组
        point_cloud = np.asarray(pcd.points)
        
        # 清理内存
        del pcd
    
    print(f"生成点云包含 {len(point_cloud)} 个点")
    return point_cloud

def compute_range(point_cloud):
    """
    计算点云的 bounding box 对角线长度。
    如果返回值异常，可能说明单位或内参出现问题。
    """
    if len(point_cloud) == 0:
        return 0.0
    min_xyz = np.min(point_cloud, axis=0)
    max_xyz = np.max(point_cloud, axis=0)
    return np.linalg.norm(max_xyz - min_xyz)

def process_single_file(file_path, output_folder, root_folder, camera_params, voxel_size, max_points):
    try:
        print(file_path)
        
        fx, fy, cx, cy, k1, k2, k3, k4 = camera_params
        
        # 生成点云数据
        point_cloud = depth_image_to_point_cloud(
            file_path, fx, fy, cx, cy, k1, k2, k3, k4, 
            scale_factor=0.001, voxel_size=voxel_size, max_points=max_points
        )

        # 计算点云范围，仅供调试参考
        pc_range = compute_range(point_cloud)
        print(f"点云范围: {pc_range:.2f}")
        
        # 检查点云是否为空
        if len(point_cloud) <= 1:
            print(f"警告: {os.path.basename(file_path)} 生成的点云为空或仅包含默认点，跳过处理")
            return False

        # 创建 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        
        # 根据原始文件夹结构，在输出文件夹中创建相应子文件夹
        root_dir = os.path.dirname(file_path)
        rel_path = os.path.relpath(root_dir, root_folder)
        target_folder = os.path.join(output_folder, rel_path)
        os.makedirs(target_folder, exist_ok=True)

        # 生成并保存点云文件（PLY 格式）
        filename = os.path.basename(file_path)
        pc_file_name = filename[:-5] + '_pcd.ply'  # 保留原文件名，并添加后缀
        pc_file_path = os.path.join(target_folder, pc_file_name)
        o3d.io.write_point_cloud(pc_file_path, pcd)
        
        return True
    except Exception as e:
        print(f"处理文件 {os.path.basename(file_path)} 时发生错误: {e}")
        return False
    finally:
        # 每处理完一个文件后清理内存
        gc.collect()

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='从深度图生成点云文件')
    parser.add_argument('--input', type=str, required=True, help='输入深度图文件的根目录路径')
    parser.add_argument('--output', type=str, required=True, help='输出点云文件的根目录路径')
    parser.add_argument('--voxel_size', type=float, default=0.005, help='点云降采样的体素大小')
    parser.add_argument('--max_points', type=int, default=30000, help='点云的最大点数')
    parser.add_argument('--threads', type=int, default=4, help='并行处理的线程数')
    args = parser.parse_args()
    
    # 相机内参和畸变系数（根据实际数据设置）
    fx = 767.3861511125845
    fy = 767.5058656118406
    cx = 679.054265997005
    cy = 543.646891684636
    k1 = -0.18867185058223412
    k2 = -0.003927337093919806
    k3 = 0.030524814153620117
    k4 = -0.012756926010904904
    camera_params = (fx, fy, cx, cy, k1, k2, k3, k4)
    
    # 使用命令行参数指定的路径
    root_folder = args.input
    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)
    
    # 设置点云降采样的体素大小和最大点数限制
    voxel_size = args.voxel_size
    max_points = args.max_points
    
    # 收集所有需要处理的深度图文件
    depth_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('_depth.tiff'):
                depth_files.append(os.path.join(root, file))
    
    total_files = len(depth_files)
    print(f"找到 {total_files} 个深度图文件需要处理")
    
    # 创建部分函数，固定部分参数
    process_file_fn = partial(
        process_single_file, 
        output_folder=output_folder, 
        root_folder=root_folder,
        camera_params=camera_params,
        voxel_size=voxel_size,
        max_points=max_points
    )
    
    # 使用多线程并行处理文件
    successful = 0
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        results = list(executor.map(process_file_fn, depth_files))
        successful = sum(1 for r in results if r)
    
    print(f"处理完成: {successful}/{total_files} 个文件成功处理")

if __name__ == '__main__':
    main()
