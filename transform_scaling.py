import os
import numpy as np
import open3d as o3d
import argparse
from glob import glob
import copy
import re
from tqdm import tqdm

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

def compute_scale_factor(source_pcd, target_pcd):
    """计算两个点云之间的缩放因子，基于它们的边界盒大小"""
    source_bbox = source_pcd.get_axis_aligned_bounding_box()
    target_bbox = target_pcd.get_axis_aligned_bounding_box()
    
    source_extent = source_bbox.get_extent()
    target_extent = target_bbox.get_extent()
    
    # 计算每个维度的比例
    scale_x = target_extent[0] / source_extent[0] if source_extent[0] != 0 else 1.0
    scale_y = target_extent[1] / source_extent[1] if source_extent[1] != 0 else 1.0
    scale_z = target_extent[2] / source_extent[2] if source_extent[2] != 0 else 1.0
    
    # 使用平均缩放因子
    scale_factor = (scale_x + scale_y + scale_z) / 3.0
    
    print(f"源点云尺寸: {source_extent}")
    print(f"目标点云尺寸: {target_extent}")
    print(f"计算得到的缩放因子: {scale_factor:.4f} (x={scale_x:.4f}, y={scale_y:.4f}, z={scale_z:.4f})")
    
    return scale_factor

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

def extract_frame_index(filename, pattern_type):
    """从不同模式的文件名中提取帧索引
    
    参数:
        filename: 文件名
        pattern_type: 'source' 或 'target'，表示文件名模式类型
    
    返回:
        提取的帧索引
    """
    try:
        if pattern_type == 'source':
            # 源点云: "0000_depth_pcd.ply"
            return int(os.path.basename(filename).split('_')[0])
        elif pattern_type == 'target':
            # 目标点云: "frame_0000_visible.ply"
            match = re.search(r'frame_(\d+)_visible', os.path.basename(filename))
            if match:
                return int(match.group(1))
        else:
            raise ValueError(f"未知的模式类型: {pattern_type}")
    except:
        return None

def process_point_clouds(source_root, target_root, pose_root, output_root, scene_name):
    """处理指定场景的所有点云文件：先旋转，再缩放，最后平移（均以坐标原点为中心）"""
    # 构建场景相关目录
    source_scene_dir = os.path.join(source_root, scene_name)
    target_scene_dir = os.path.join(target_root, scene_name)
    pose_path = os.path.join(pose_root, scene_name, "pose.txt")
    output_scene_dir = os.path.join(output_root, scene_name)
    os.makedirs(output_scene_dir, exist_ok=True)
    
    # 检查文件夹和文件是否存在
    if not os.path.exists(source_scene_dir):
        print(f"错误：源场景文件夹不存在 - {source_scene_dir}")
        return
    if not os.path.exists(target_scene_dir):
        print(f"错误：目标场景文件夹不存在 - {target_scene_dir}")
        return
    if not os.path.exists(pose_path):
        print(f"错误：位姿文件不存在 - {pose_path}")
        return
    
    # 加载位姿数据
    poses = load_poses(pose_path)
    
    # 获取所有点云文件
    source_pcd_files = sorted(glob(os.path.join(source_scene_dir, "*_depth_pcd.ply")))
    target_pcd_files = sorted(glob(os.path.join(target_scene_dir, "frame_*_visible.ply")))
    
    print(f"找到 {len(source_pcd_files)} 个源点云文件")
    print(f"找到 {len(target_pcd_files)} 个目标点云文件")
    
    # 创建目标点云字典，便于根据帧索引查找对应目标点云
    target_pcd_dict = {}
    for target_file in target_pcd_files:
        idx = extract_frame_index(target_file, 'target')
        if idx is not None:
            target_pcd_dict[idx] = target_file
    
    center_differences = []
    
    for source_file in tqdm(source_pcd_files, desc=f"处理场景 {scene_name} 的点云"):
        # 提取源文件帧索引
        source_idx = extract_frame_index(source_file, 'source')
        if source_idx is None:
            print(f"警告：无法从文件名 {os.path.basename(source_file)} 解析出索引")
            continue
        
        # 查找对应的目标点云文件
        if source_idx not in target_pcd_dict:
            print(f"警告：未找到索引为 {source_idx} 的目标点云文件，跳过")
            continue
        
        target_file = target_pcd_dict[source_idx]
        
        # 检查位姿数据是否足够（应和点云序号匹配）
        if source_idx >= len(poses):
            print(f"警告：索引 {source_idx} 超出位姿数据范围({len(poses)})，跳过")
            continue
        
        # 加载点云
        source_pcd = o3d.io.read_point_cloud(source_file)
        target_pcd = o3d.io.read_point_cloud(target_file)
        if not source_pcd.has_points() or not target_pcd.has_points():
            print(f"警告：点云 {source_idx} 无点或为空，跳过")
            continue
        
        # 记录原始源点云中心
        original_source_center = source_pcd.get_center()
        
        # 从位姿矩阵中提取旋转和平移部分
        T_pose = poses[source_idx]
        rotation, translation = extract_rotation_translation_from_pose(T_pose)
        
        # 打印当前使用的位姿信息（用于调试）
        print(f"\n点云 {source_idx} 的位姿信息:")
        print(f"旋转矩阵:\n{rotation}")
        print(f"平移向量: {translation}")
        
        # 步骤1：先应用旋转（以坐标原点为中心）
        rotation_matrix = get_rotation_matrix(rotation)
        rotated_source = copy.deepcopy(source_pcd)
        rotated_source.transform(rotation_matrix)
        rotated_center = rotated_source.get_center()
        
        # 步骤2：计算旋转后的点云与目标点云之间的缩放因子
        scale_factor = compute_scale_factor(rotated_source, target_pcd)
        
        # 步骤3：应用缩放变换（以坐标原点为中心）
        scaled_source = apply_scale_to_point_cloud(rotated_source, scale_factor)
        scaled_center = scaled_source.get_center()
        
        # 步骤4：最后应用平移变换
        translation_matrix = get_translation_matrix(translation)
        transformed_source = copy.deepcopy(scaled_source)
        transformed_source.transform(translation_matrix)
        
        # 计算调整后点云与目标点云中心的差异
        adjusted_center = transformed_source.get_center()
        target_center = target_pcd.get_center()
        center_diff = adjusted_center - target_center
        center_diff_norm = np.linalg.norm(center_diff)
        center_differences.append(center_diff_norm)
        
        print(f"\n点云 {source_idx} 处理结果:")
        print(f"  源点云文件: {os.path.basename(source_file)}")
        print(f"  目标点云文件: {os.path.basename(target_file)}")
        print(f"  原始源点云中心: {original_source_center}")
        print(f"  旋转后的点云中心: {rotated_center}")
        print(f"  旋转+缩放后的点云中心: {scaled_center}")
        print(f"  最终变换后点云中心: {adjusted_center}")
        print(f"  目标点云中心: {target_center}")
        print(f"  中心差异: {center_diff}, 距离: {center_diff_norm:.4f}")
        
        # 只保存最终调整后的点云（经过旋转、缩放和平移）
        output_file = os.path.join(output_scene_dir, f"{source_idx:04d}_adjusted.ply")
        o3d.io.write_point_cloud(output_file, transformed_source)
    
    # 统计所有点云的中心差异
    if center_differences:
        avg_diff = np.mean(center_differences)
        max_diff = np.max(center_differences)
        min_diff = np.min(center_differences)
        print(f"\n场景 {scene_name} 中心差异统计:")
        print(f"  平均差异: {avg_diff:.4f}")
        print(f"  最大差异: {max_diff:.4f}")
        print(f"  最小差异: {min_diff:.4f}")

def main():
    parser = argparse.ArgumentParser(description='调整源点云数据集（先旋转，再缩放，最后平移，均以坐标原点为中心）以匹配目标点云数据集')
    parser.add_argument('--source_root', type=str, required=True, help='源点云数据集根目录')
    parser.add_argument('--target_root', type=str, required=True, help='目标点云数据集根目录')
    parser.add_argument('--pose_root', type=str, required=True, help='位姿文件所在根目录')
    parser.add_argument('--output_root', type=str, required=True, help='调整后点云的输出根目录')
    parser.add_argument('--scene', type=str, help='要处理的单个场景名称')
    parser.add_argument('--scenes', nargs='+', help='要处理的多个场景名称列表')
    
    args = parser.parse_args()
    
    if args.scene:
        process_point_clouds(args.source_root, args.target_root, args.pose_root, args.output_root, args.scene)
    elif args.scenes:
        for scene in args.scenes:
            print(f"\n开始处理场景: {scene}")
            process_point_clouds(args.source_root, args.target_root, args.pose_root, args.output_root, scene)
    else:
        source_scenes = set(os.listdir(args.source_root))
        target_scenes = set(os.listdir(args.target_root))
        pose_scenes = set(os.listdir(args.pose_root))
        common_scenes = sorted(list(source_scenes.intersection(target_scenes).intersection(pose_scenes)))
        print(f"在三个数据集中找到 {len(common_scenes)} 个共有场景: {common_scenes}")
        for scene in common_scenes:
            print(f"\n开始处理场景: {scene}")
            process_point_clouds(args.source_root, args.target_root, args.pose_root, args.output_root, scene)

if __name__ == "__main__":
    main()