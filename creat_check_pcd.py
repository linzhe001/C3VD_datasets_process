import cv2
import numpy as np
import open3d as o3d
import os

def depth_image_to_point_cloud(depth_image_path, fx, fy, cx, cy, k1, k2, k3, k4, scale_factor=0.001):
    # 读取深度图
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)

    # 获取深度图的尺寸
    height, width = depth_image.shape

    # 初始化点云列表
    point_cloud = []

    for v in range(height):
        for u in range(width):
            depth_value = depth_image[v, u]
            if depth_value > 0:  # 只处理有效的深度值
                # 考虑镜头畸变校正像素坐标
                # 注意：如果深度图已与去畸变图像对齐，可以跳过此步骤
                undistorted_u, undistorted_v = correct_distortion(
                    u, v, fx, fy, cx, cy, k1, k2, k3, k4
                )

                # 将深度值从毫米转换为米（如果需要）
                Z = depth_value * scale_factor  # 深度值
                
                # 计算归一化坐标
                x_norm = (undistorted_u - cx) / fx
                y_norm = (undistorted_v - cy) / fy
                
                # 反投影到相机坐标系
                X = Z * x_norm
                Y = Z * y_norm
                
                # 将相机坐标系下的坐标添加到点云列表
                point_cloud.append([X, Y, Z])

    return np.array(point_cloud)

def correct_distortion(u, v, fx, fy, cx, cy, k1, k2, k3, k4, p1=0, p2=0):
    # 计算归一化坐标
    x_raw = (u - cx) / fx
    y_raw = (v - cy) / fy

    # 计算径向畸变因子
    r2 = x_raw ** 2 + y_raw ** 2
    r4 = r2 ** 2
    r6 = r4 * r2
    r8 = r6 * r2
    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8

    # 应用完整的畸变校正公式（包括切向畸变）
    x_corrected = x_raw * radial + 2*p1*x_raw*y_raw + p2*(r2 + 2*x_raw**2)
    y_corrected = y_raw * radial + p1*(r2 + 2*y_raw**2) + 2*p2*x_raw*y_raw

    # 转回像素坐标
    undistorted_u = x_corrected * fx + cx
    undistorted_v = y_corrected * fy + cy

    return undistorted_u, undistorted_v

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

# 相机内参和畸变系数（根据实际数据设置）
fx = 767.3861511125845
fy = 767.5058656118406
cx = 679.054265997005
cy = 543.646891684636
k1 = -0.18867185058223412
k2 = -0.003927337093919806
k3 = 0.030524814153620117
k4 = -0.012756926010904904

# 根文件夹路径（请替换为实际路径）
root_folder = "/home/linzhe_linux/C3VD_datasets/C3VD"

# 最终保存 ply 文件的根文件夹路径
output_folder = "/home/linzhe_linux/C3VD_datasets/C3VD_ply"
os.makedirs(output_folder, exist_ok=True)

# 设置点云降采样的体素大小（可根据实际情况调整）
VOXEL_SIZE = 0.005

# 初始化文件计数器和总文件数
file_count = 0
total_files = 0
for root, dirs, files in os.walk(root_folder):
    total_files += len([f for f in files if f.endswith('_depth.tiff')])

# 遍历根文件夹及其子文件夹下的所有深度图文件
for root, dirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith('_depth.tiff'):
            depth_image_path = os.path.join(root, file)
            print(depth_image_path)

            # 1) 生成点云数据
            point_cloud = depth_image_to_point_cloud(
                depth_image_path, fx, fy, cx, cy, k1, k2, k3, k4, scale_factor=0.001
            )

            # 2) 计算点云范围，仅供调试参考
            pc_range = compute_range(point_cloud)
            print(f"点云范围: {pc_range:.2f}")

            # 3) 创建 Open3D 点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)

            # 4) 对点云进行降采样
            pcd_down = pcd.voxel_down_sample(VOXEL_SIZE)
            
            # 5) 检查点云数量是否超过3万点限制
            MAX_POINTS = 30000
            if len(pcd_down.points) > MAX_POINTS:
                # 如果下采样后仍超过3万点，进行随机采样
                print(f"体素下采样后点数({len(pcd_down.points)})仍超过限制，进行随机采样...")
                sampling_ratio = MAX_POINTS / len(pcd_down.points)
                pcd_down = pcd_down.random_down_sample(sampling_ratio)
                print(f"随机采样后点数: {len(pcd_down.points)}")

            # 6) 根据原始文件夹结构，在输出文件夹中创建相应子文件夹
            rel_path = os.path.relpath(root, root_folder)
            target_folder = os.path.join(output_folder, rel_path)
            os.makedirs(target_folder, exist_ok=True)

            # 生成并保存点云文件（PLY 格式）
            pc_file_name = file[:-5] + '_pcd.ply'  # 保留原文件名，并添加后缀
            pc_file_path = os.path.join(target_folder, pc_file_name)
            o3d.io.write_point_cloud(pc_file_path, pcd_down)

            file_count += 1
            print(f'已处理 {file_count}/{total_files} 个文件: {file}')
