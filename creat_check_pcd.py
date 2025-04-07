import cv2
import numpy as np
import open3d as o3d
import os
import argparse
import gc
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import copy

def depth_image_to_point_cloud(depth_image_path, fx, fy, cx, cy, k1, k2, k3, k4, voxel_size=None, batch_size=10, max_points=30000):
    # 读取深度图
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
    
    # 将深度图转换为浮点型并缩放
    depth_image = (depth_image.astype(np.float32)/(2**16-1))*100
    
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
    z = undistorted_depth[valid_mask]
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

def load_poses(pose_path):
    """从 pose.txt 加载所有位姿（按列主序存储的4×4矩阵）。
    
    每行包含 16 个数字（逗号分隔），但是按列主序排列，
    即前4个数字是第一列，接下来4个数字是第二列，依此类推。
    """
    print(f"正在加载：{pose_path}")
    poses = []
    with open(pose_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = list(map(float, line.strip().split(',')))
            if len(data) != 16:
                raise ValueError(f"位姿行数据长度错误，期望16个数字，实际为{len(data)}，内容：{line}")
            
            # 按列主序重构4×4矩阵
            T = np.zeros((4, 4))
            for i in range(4):  # 列
                for j in range(4):  # 行
                    T[j, i] = data[i * 4 + j]
            
            # 验证矩阵格式
            if not np.allclose(T[3, :], [0, 0, 0, 1]):
                raise ValueError(f"无效的转换矩阵，最后一行应为[0,0,0,1]：\n{T}")
            
            poses.append(T)
            
    print(f"成功加载了 {len(poses)} 个位姿矩阵")
    return poses

def extract_rotation_translation_from_pose(pose):
    """从位姿矩阵中提取旋转矩阵和平移向量"""
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return rotation, translation

def apply_scale_to_point_cloud(pcd, scale_factor):
    """对点云应用缩放变换，以坐标原点为中心"""
    pcd_scaled = copy.deepcopy(pcd)
    
    # 使用坐标原点(0,0,0)为中心进行缩放
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale_factor
    pcd_scaled.transform(scale_matrix)
    
    return pcd_scaled

def get_rotation_matrix(rotation):
    """根据3×3旋转矩阵创建4×4变换矩阵（只包含旋转）"""
    rot_matrix = np.eye(4)
    rot_matrix[:3, :3] = rotation
    return rot_matrix

def get_translation_matrix(translation):
    """根据平移向量创建4×4变换矩阵（只包含平移）"""
    trans_matrix = np.eye(4)
    trans_matrix[:3, 3] = translation
    return trans_matrix

def extract_scene_name(file_path, root_folder):
    """从文件路径中提取场景名称"""
    # 获取相对路径
    rel_path = os.path.relpath(file_path, root_folder)
    # 第一级目录通常是场景名
    parts = rel_path.split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return None

def find_pose_file(pose_root, scene_name):
    """查找场景对应的pose文件"""
    pose_path = os.path.join(pose_root, scene_name, "pose.txt")
    if os.path.exists(pose_path):
        return pose_path
    return None

def process_single_file(file_path, output_folder, root_folder, camera_params, voxel_size, max_points, pose_matrix=None, apply_transform=False):
    try:
        print(file_path)
        
        fx, fy, cx, cy, k1, k2, k3, k4 = camera_params
        
        # 生成点云数据
        point_cloud = depth_image_to_point_cloud(
            file_path, fx, fy, cx, cy, k1, k2, k3, k4, 
            voxel_size=voxel_size, max_points=max_points
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
        
        # 如果需要应用变换
        if apply_transform and pose_matrix is not None:
            # 记录原始点云中心
            original_center = pcd.get_center()
            print(f"原始点云中心: {original_center}")
            
            # 从位姿矩阵中提取旋转和平移部分
            rotation, translation = extract_rotation_translation_from_pose(pose_matrix)
            
            # 打印当前使用的位姿信息
            print(f"使用位姿信息:")
            print(f"旋转矩阵:\n{rotation}")
            print(f"平移向量: {translation}")
            
            # 步骤1：先应用旋转（以坐标原点为中心）
            rotation_matrix = get_rotation_matrix(rotation)
            rotated_pcd = copy.deepcopy(pcd)
            rotated_pcd.transform(rotation_matrix)
            rotated_center = rotated_pcd.get_center()
            print(f"旋转后的点云中心: {rotated_center}")
            
            # 步骤3：最后应用平移变换
            translation_matrix = get_translation_matrix(translation)
            transformed_pcd = copy.deepcopy(rotated_pcd)
            transformed_pcd.transform(translation_matrix)
            final_center = transformed_pcd.get_center()
            print(f"最终变换后点云中心: {final_center}")
            
            # 使用变换后的点云
            pcd = transformed_pcd
        
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
    parser.add_argument('--pose_path', type=str, help='位姿文件的路径（单个文件）')
    parser.add_argument('--pose_root', type=str, help='位姿文件的根目录（自动查找各场景pose文件）')
    parser.add_argument('--apply_transform', action='store_true', help='是否应用旋转和平移变换')
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
    
    # 处理文件
    successful = 0
    
    # 如果需要应用变换
    if args.apply_transform:
        # 创建场景->位姿数据的映射
        scene_poses = {}
        
        if args.pose_path and os.path.exists(args.pose_path):
            # 如果只提供了单个位姿文件，则所有场景共用此位姿
            poses = load_poses(args.pose_path)
            print(f"从单个位姿文件加载了 {len(poses)} 个位姿矩阵")
            # 对所有场景使用相同的位姿
            for file_path in depth_files:
                scene_name = extract_scene_name(file_path, root_folder)
                if scene_name and scene_name not in scene_poses:
                    scene_poses[scene_name] = poses
        
        elif args.pose_root and os.path.exists(args.pose_root):
            # 如果提供了位姿根目录，为每个场景查找对应的位姿文件
            print(f"尝试从位姿根目录 {args.pose_root} 加载位姿文件")
            # 收集所有场景名
            scene_names = set()
            for file_path in depth_files:
                scene_name = extract_scene_name(file_path, root_folder)
                if scene_name:
                    scene_names.add(scene_name)
            
            # 为每个场景加载位姿数据
            for scene_name in scene_names:
                pose_file = find_pose_file(args.pose_root, scene_name)
                if pose_file:
                    print(f"为场景 {scene_name} 加载位姿文件: {pose_file}")
                    scene_poses[scene_name] = load_poses(pose_file)
                else:
                    print(f"警告: 未找到场景 {scene_name} 的位姿文件")
        
        else:
            print("错误: 未指定有效的位姿文件或位姿根目录")
            return
        
        # 处理每个文件
        for file_path in depth_files:
            try:
                # 提取场景名和帧索引
                scene_name = extract_scene_name(file_path, root_folder)
                frame_idx = int(os.path.basename(file_path).split('_')[0])
                
                # 如果找到对应场景的位姿数据
                if scene_name in scene_poses:
                    poses = scene_poses[scene_name]
                    
                    # 确保帧索引在范围内
                    if frame_idx < len(poses):
                        print(f"处理文件: {file_path} (场景: {scene_name}, 帧: {frame_idx})")
                        result = process_single_file(
                            file_path, 
                            output_folder, 
                            root_folder,
                            camera_params,
                            voxel_size,
                            max_points,
                            pose_matrix=poses[frame_idx],
                            apply_transform=True
                        )
                        if result:
                            successful += 1
                    else:
                        print(f"警告：帧索引 {frame_idx} 超出场景 {scene_name} 位姿数据范围({len(poses)})，跳过变换")
                        result = process_single_file(
                            file_path, 
                            output_folder, 
                            root_folder,
                            camera_params,
                            voxel_size,
                            max_points
                        )
                        if result:
                            successful += 1
                else:
                    print(f"警告：未找到场景 {scene_name} 的位姿数据，跳过变换")
                    result = process_single_file(
                        file_path, 
                        output_folder, 
                        root_folder,
                        camera_params,
                        voxel_size,
                        max_points
                    )
                    if result:
                        successful += 1
            except Exception as e:
                print(f"处理文件 {os.path.basename(file_path)} 时发生错误: {e}")
                continue
    else:
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
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            results = list(executor.map(process_file_fn, depth_files))
            successful = sum(1 for r in results if r)
    
    print(f"处理完成: {successful}/{total_files} 个文件成功处理")

if __name__ == '__main__':
    main()
