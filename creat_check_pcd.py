import cv2
import numpy as np
import open3d as o3d
import os
import argparse
import gc

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
        del depth_image
    except Exception as e:
        print(f"畸变校正失败，使用原始深度图: {e}")
        undistorted_depth = depth_image
        del depth_image
    
    # 检查是否需要进行体素下采样
    if voxel_size is not None and voxel_size > 0 and batch_size > 0:
        valid_depth_count = 0
        total_points_processed = 0
        
        # 临时点云和结果点云
        temp_points = []  # 存储临时点云，达到batch_size后会进行下采样
        final_points = []  # 存储所有下采样后的点云
        
        # 处理所有像素点
        for v in range(height):
            for u in range(width):
                depth_value = undistorted_depth[v, u]
                if depth_value > 0:  # 只处理有效深度值
                    valid_depth_count += 1
                    Z = depth_value * scale_factor
                    X = (u - cx) / fx * Z
                    Y = (v - cy) / fy * Z
                    temp_points.append([X, Y, Z])
                    
                    # 每收集 batch_size 个点就进行一次下采样
                    if len(temp_points) >= batch_size:
                        total_points_processed += len(temp_points)
                        
                        # 将临时点云添加到最终点云
                        final_points.extend(temp_points)
                        
                        # 固定体素大小，不进行动态调整
                        current_voxel_size = voxel_size
                        
                        # 对整个点云进行体素下采样
                        if len(final_points) > 0:
                            final_np = np.array(final_points)
                            voxel_dict = {}
                            
                            for point in final_np:
                                # 计算体素索引（始终使用固定的 voxel_size）
                                voxel_x = int(np.floor(point[0] / current_voxel_size))
                                voxel_y = int(np.floor(point[1] / current_voxel_size))
                                voxel_z = int(np.floor(point[2] / current_voxel_size))
                                voxel_idx = (voxel_x, voxel_y, voxel_z)
                                
                                if voxel_idx in voxel_dict:
                                    voxel_dict[voxel_idx][0] += point[0]
                                    voxel_dict[voxel_idx][1] += point[1]
                                    voxel_dict[voxel_idx][2] += point[2]
                                    voxel_dict[voxel_idx][3] += 1
                                else:
                                    voxel_dict[voxel_idx] = [point[0], point[1], point[2], 1]
                            
                            # 重置最终点云并添加下采样结果
                            final_points = []
                            for voxel, (x_sum, y_sum, z_sum, count) in voxel_dict.items():
                                final_points.append([x_sum/count, y_sum/count, z_sum/count])
                        
                        # 清空临时点云列表
                        temp_points = []
                        
                        if total_points_processed % 100000 == 0:
                            print(f"已处理 {total_points_processed} 个点，当前点云大小: {len(final_points)}")
        
        # 处理剩余的点
        if len(temp_points) > 0:
            # 将临时点云转换为NumPy数组
            temp_np = np.array(temp_points)
            
            # 对临时点云进行体素下采样
            voxel_dict = {}
            for point in temp_np:
                # 计算体素索引
                voxel_x = int(np.floor(point[0] / voxel_size))
                voxel_y = int(np.floor(point[1] / voxel_size))
                voxel_z = int(np.floor(point[2] / voxel_size))
                voxel_idx = (voxel_x, voxel_y, voxel_z)
                
                # 更新体素字典
                if voxel_idx in voxel_dict:
                    voxel_dict[voxel_idx][0] += point[0]
                    voxel_dict[voxel_idx][1] += point[1]
                    voxel_dict[voxel_idx][2] += point[2]
                    voxel_dict[voxel_idx][3] += 1
                else:
                    voxel_dict[voxel_idx] = [point[0], point[1], point[2], 1]
            
            # 将体素下采样结果添加到最终点云
            for voxel, (x_sum, y_sum, z_sum, count) in voxel_dict.items():
                final_points.append([x_sum/count, y_sum/count, z_sum/count])
        
        print(f"总有效深度像素数: {valid_depth_count}")
        print(f"最终点云包含 {len(final_points)} 个点")
        point_cloud = np.array(final_points) if final_points else np.array([[0, 0, 0]])
    else:
        # 原始逻辑：不进行实时下采样，但我们仍然去掉sample_step，处理所有点
        point_cloud = []
        valid_depth_count = 0
        
        for v in range(height):
            for u in range(width):
                depth_value = undistorted_depth[v, u]
                if depth_value > 0:  # 只过滤深度为0的点
                    valid_depth_count += 1
                    Z = depth_value * scale_factor
                    X = (u - cx) / fx * Z
                    Y = (v - cy) / fy * Z
                    point_cloud.append([X, Y, Z])
        
        print(f"有效深度像素数: {valid_depth_count}")
        
        # 将点云列表转换为NumPy数组
        if len(point_cloud) > 0:
            point_cloud = np.array(point_cloud)
        else:
            print("警告：未能生成有效点云，返回默认点")
            point_cloud = np.array([[0, 0, 0]])
    
    # 释放校正后的深度图内存
    del undistorted_depth
    
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

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='从深度图生成点云文件')
    parser.add_argument('--input', type=str, required=True, help='输入深度图文件的根目录路径')
    parser.add_argument('--output', type=str, required=True, help='输出点云文件的根目录路径')
    parser.add_argument('--voxel_size', type=float, default=0.005, help='点云降采样的体素大小')
    parser.add_argument('--max_points', type=int, default=30000, help='点云的最大点数')
    args = parser.parse_args()
    
    # 相机内参和畸变系数（根据实际数据设置）
    fx = 767.3861511125845
    fy = 767.5058656118406
    cx = 679.054265997005
    cy = 543.646891684636
    k1 = -0.18867185058223412
    k2 = -0.003927337093919806
    k3 = 0.030524814153620117  # 在标准模型中这可能是p1
    k4 = -0.012756926010904904 # 在标准模型中这可能是p2
    
    # 使用命令行参数指定的路径
    root_folder = args.input
    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)
    
    # 设置点云降采样的体素大小（可通过命令行参数调整）
    VOXEL_SIZE = args.voxel_size
    
    # 设置最大点数限制（可通过命令行参数调整）
    MAX_POINTS = args.max_points
    
    # 初始化文件计数器和总文件数
    file_count = 0
    total_files = 0
    for root, dirs, files in os.walk(root_folder):
        total_files += len([f for f in files if f.endswith('_depth.tiff')])
    
    # 遍历根文件夹及其子文件夹下的所有深度图文件
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('_depth.tiff'):
                try:
                    depth_image_path = os.path.join(root, file)
                    print(depth_image_path)
    
                    # 1) 生成点云数据，每batch_size个点进行一次下采样
                    point_cloud = depth_image_to_point_cloud(
                        depth_image_path, fx, fy, cx, cy, k1, k2, k3, k4, 
                        scale_factor=0.001, voxel_size=VOXEL_SIZE, batch_size=1000, max_points=MAX_POINTS
                    )
    
                    # 2) 计算点云范围，仅供调试参考
                    pc_range = compute_range(point_cloud)
                    print(f"点云范围: {pc_range:.2f}")
                    
                    # 检查点云是否为空
                    if len(point_cloud) <= 1:
                        print(f"警告: {file} 生成的点云为空或仅包含默认点，跳过处理")
                        continue
    
                    # 3) 创建 Open3D 点云对象
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
                    # 6) 根据原始文件夹结构，在输出文件夹中创建相应子文件夹
                    rel_path = os.path.relpath(root, root_folder)
                    target_folder = os.path.join(output_folder, rel_path)
                    os.makedirs(target_folder, exist_ok=True)
    
                    # 生成并保存点云文件（PLY 格式）
                    pc_file_name = file[:-5] + '_pcd.ply'  # 保留原文件名，并添加后缀
                    pc_file_path = os.path.join(target_folder, pc_file_name)
                    o3d.io.write_point_cloud(pc_file_path, pcd)
    
                    file_count += 1
                    print(f'已处理 {file_count}/{total_files} 个文件: {file}')
                    
                    # 每处理完一个文件后主动清理内存
                    gc.collect()  # 触发垃圾回收
                    
                except Exception as e:
                    print(f"处理文件 {file} 时发生错误: {e}")
                    # 错误恢复后也清理内存
                    gc.collect()
                    continue

if __name__ == '__main__':
    main()
