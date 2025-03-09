import os
import numpy as np
import open3d as o3d
import argparse
from glob import glob
import copy
import re
from tqdm import tqdm

def compute_scale_factor(source_pcd, target_pcd):
    """计算两个点云之间的缩放因子，基于它们的边界框大小"""
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

def apply_scale_to_point_cloud(pcd, scale_factor, center=None):
    """对点云应用缩放变换，以指定中心为基准"""
    if center is None:
        center = pcd.get_center()
    
    # 创建缩放矩阵
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale_factor
    
    # 创建以中心为基准的变换
    center_to_origin = np.eye(4)
    center_to_origin[:3, 3] = -center
    
    origin_to_center = np.eye(4)
    origin_to_center[:3, 3] = center
    
    # 组合变换：先移到原点，然后缩放，再移回原位
    transform = np.matmul(origin_to_center, np.matmul(scale_matrix, center_to_origin))
    
    # 应用变换
    pcd_scaled = copy.deepcopy(pcd)
    pcd_scaled.transform(transform)
    
    return pcd_scaled

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

def process_point_clouds(source_root, target_root, output_root, scene_name):
    """处理指定场景的所有点云文件，只应用缩放变换"""
    
    # 构建场景文件夹路径
    source_scene_dir = os.path.join(source_root, scene_name)
    target_scene_dir = os.path.join(target_root, scene_name)
    
    # 创建输出目录
    output_scene_dir = os.path.join(output_root, scene_name)
    os.makedirs(output_scene_dir, exist_ok=True)
    
    # 检查文件夹是否存在
    if not os.path.exists(source_scene_dir):
        print(f"错误：源场景文件夹不存在 - {source_scene_dir}")
        return
    
    if not os.path.exists(target_scene_dir):
        print(f"错误：目标场景文件夹不存在 - {target_scene_dir}")
        return
    
    # 获取所有点云文件
    source_pcd_files = sorted(glob(os.path.join(source_scene_dir, "*_depth_pcd.ply")))
    target_pcd_files = sorted(glob(os.path.join(target_scene_dir, "frame_*_visible.ply")))
    
    print(f"找到 {len(source_pcd_files)} 个源点云文件")
    print(f"找到 {len(target_pcd_files)} 个目标点云文件")
    
    # 创建目标点云索引字典，便于查找
    target_pcd_dict = {}
    for target_file in target_pcd_files:
        idx = extract_frame_index(target_file, 'target')
        if idx is not None:
            target_pcd_dict[idx] = target_file
    
    # 记录中心点差异
    center_differences = []
    
    # 处理每个源点云文件
    for source_file in tqdm(source_pcd_files, desc=f"处理场景 {scene_name} 的点云"):
        # 提取源文件索引
        source_idx = extract_frame_index(source_file, 'source')
        if source_idx is None:
            print(f"警告：无法从文件名 {os.path.basename(source_file)} 解析出索引")
            continue
        
        # 查找对应的目标点云文件
        if source_idx not in target_pcd_dict:
            print(f"警告：未找到索引为 {source_idx} 的目标点云文件，跳过")
            continue
        
        target_file = target_pcd_dict[source_idx]
        
        # 加载点云
        source_pcd = o3d.io.read_point_cloud(source_file)
        target_pcd = o3d.io.read_point_cloud(target_file)
        
        if not source_pcd.has_points() or not target_pcd.has_points():
            print(f"警告：点云 {source_idx} 无点或为空，跳过")
            continue
        
        # 记录原始源点云中心
        original_source_center = source_pcd.get_center()
        
        # 只进行缩放处理，不进行旋转和平移
        scale_factor = compute_scale_factor(source_pcd, target_pcd)
        scaled_source = apply_scale_to_point_cloud(source_pcd, scale_factor)
        
        # 计算调整后的点云与目标点云中心的差异
        adjusted_center = scaled_source.get_center()
        target_center = target_pcd.get_center()
        center_diff = adjusted_center - target_center
        center_diff_norm = np.linalg.norm(center_diff)
        center_differences.append(center_diff_norm)
        
        print(f"\n点云 {source_idx} 处理结果:")
        print(f"  源点云文件: {os.path.basename(source_file)}")
        print(f"  目标点云文件: {os.path.basename(target_file)}")
        print(f"  原始源点云中心: {original_source_center}")
        print(f"  缩放后点云中心: {adjusted_center}")
        print(f"  目标点云中心: {target_center}")
        print(f"  中心差异: {center_diff}, 距离: {center_diff_norm:.4f}")
        
        # 保存调整后的点云
        output_file = os.path.join(output_scene_dir, f"{source_idx:04d}_scaled_only.ply")
        o3d.io.write_point_cloud(output_file, scaled_source)
    
    # 统计中心差异
    if center_differences:
        avg_diff = np.mean(center_differences)
        max_diff = np.max(center_differences)
        min_diff = np.min(center_differences)
        print(f"\n场景 {scene_name} 中心差异统计:")
        print(f"  平均差异: {avg_diff:.4f}")
        print(f"  最大差异: {max_diff:.4f}")
        print(f"  最小差异: {min_diff:.4f}")

def main():
    parser = argparse.ArgumentParser(description='对源点云数据集只应用缩放变换，不进行旋转和平移')
    parser.add_argument('--source_root', type=str, required=True, help='源点云数据集根目录')
    parser.add_argument('--target_root', type=str, required=True, help='目标点云数据集根目录')
    parser.add_argument('--output_root', type=str, required=True, help='调整后点云的输出根目录')
    parser.add_argument('--scene', type=str, help='要处理的单个场景名称')
    parser.add_argument('--scenes', nargs='+', help='要处理的多个场景名称列表')
    
    args = parser.parse_args()
    
    if args.scene:
        # 处理单个场景
        process_point_clouds(args.source_root, args.target_root, args.output_root, args.scene)
    elif args.scenes:
        # 处理多个场景
        for scene in args.scenes:
            print(f"\n开始处理场景: {scene}")
            process_point_clouds(args.source_root, args.target_root, args.output_root, scene)
    else:
        # 自动检测所有共有的场景
        source_scenes = set(os.listdir(args.source_root))
        target_scenes = set(os.listdir(args.target_root))
        common_scenes = sorted(list(source_scenes.intersection(target_scenes)))
        
        print(f"在源点云和目标点云数据集中找到 {len(common_scenes)} 个共有场景: {common_scenes}")
        
        for scene in common_scenes:
            print(f"\n开始处理场景: {scene}")
            process_point_clouds(args.source_root, args.target_root, args.output_root, scene)

if __name__ == "__main__":
    main()
