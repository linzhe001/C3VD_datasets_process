import os
import json
import numpy as np
import open3d as o3d
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import copy  # 新增导入

# ================== 工具函数 ==================
def load_poses(pose_path):
    """从 pose.txt 加载所有位姿（直接加载 T_cam_to_world），并自动修正格式。
    
    如果载入的矩阵最后一行不为 [0, 0, 0, 1]，
    但前三行的第4列全为0，而第四行前三个元素非零，则把平移向量由第四行转换到第4列。
    """
    print(f"正在加载：{pose_path}")  # 输出当前加载的文件路径
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
                    raise ValueError(f"Invalid transformation matrix in line: {line}")
            poses.append(T)
    return poses

def compute_relative_pose(pose_i, pose_j):
    """计算相对位姿: T_ij = T_j @ inv(T_i) (适用于 T_cam_to_world)"""
    return pose_j @ np.linalg.inv(pose_i)

def multi_scale_icp(src_pcd, tgt_pcd, 
                    voxel_size=[0.1, 0.05, 0.025], 
                    max_iter=[100, 50, 30], 
                    init_trans=np.eye(4)):
    """多尺度ICP细化，对齐 src_pcd 到 tgt_pcd，返回最终转换矩阵。"""
    current_trans = init_trans
    for scale in range(len(voxel_size)):
        src_down = src_pcd.voxel_down_sample(voxel_size[scale])
        tgt_down = tgt_pcd.voxel_down_sample(voxel_size[scale])
        result_icp = o3d.pipelines.registration.registration_icp(
            src_down, tgt_down,
            max_correspondence_distance=voxel_size[scale] * 2,
            init=current_trans,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter[scale])
        )
        current_trans = result_icp.transformation
    return current_trans

# ================== 全局 ICP 配准与点云融合 ==================
def align_scene_with_icp(pose_root, ply_root, scene_name,
                         individual_voxel=0.05, final_voxel=0.05,
                         icp_parallel=True):
    """
    针对单个子场景进行全局 ICP 配准（已优化以处理不完整的点云数据集）
    """
    # 扫描可用的点云文件
    pcd_dir = os.path.join(ply_root, scene_name)
    available_indices, file_paths = scan_available_pointclouds(pcd_dir)
    
    if not available_indices:
        print(f"错误: 场景 {scene_name} 没有找到任何有效的点云文件")
        return
    
    print(f"场景 {scene_name} 找到 {len(available_indices)} 个有效点云文件")
    
    # 只加载需要的位姿数据
    pose_path = os.path.join(pose_root, scene_name, "pose.txt")
    poses = load_selective_poses(pose_path, available_indices)
    
    # 加载点云数据
    all_pcds = {}
    for idx in available_indices:
        pcd = o3d.io.read_point_cloud(file_paths[idx])
        if not pcd.has_points():
            print(f"警告: 点云 {idx} 中无点，跳过")
            continue
        all_pcds[idx] = pcd
    
    # 找出存在连续索引的帧对，用于计算 ICP
    continuous_pairs = []
    for i in range(len(available_indices) - 1):
        idx1 = available_indices[i]
        idx2 = available_indices[i + 1]
        
        # 只处理连续或接近的帧索引
        if idx2 - idx1 <= 3 and idx1 in all_pcds and idx2 in all_pcds:
            continuous_pairs.append((idx1, idx2))
    
    # 计算相对变换
    relative_transforms = {}
    
    def icp_task(i, j):
        src_pcd = all_pcds[i]
        tgt_pcd = all_pcds[j]
        init_trans = compute_relative_pose(poses[i], poses[j])
        final_trans = multi_scale_icp(src_pcd, tgt_pcd, init_trans=init_trans)
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            src_pcd, tgt_pcd, 0.1, final_trans
        )
        return i, j, final_trans, info
    
    if icp_parallel and continuous_pairs:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(icp_task, i, j): (i, j) for i, j in continuous_pairs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="ICP 配准"):
                i, j, final_trans, info = future.result()
                relative_transforms[(i, j)] = (final_trans, info)
    else:
        for i, j in tqdm(continuous_pairs, desc="ICP 配准"):
            _, _, final_trans, info = icp_task(i, j)
            relative_transforms[(i, j)] = (final_trans, info)
    
    # 构建 PoseGraph
    pose_graph = o3d.pipelines.registration.PoseGraph()
    
    # 添加所有节点
    first_idx = available_indices[0]
    abs_poses = {first_idx: np.eye(4)}
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
    
    # 添加所有边
    for pair, (trans, info) in relative_transforms.items():
        i, j = pair
        
        # 添加节点（如果不存在）
        for idx in [i, j]:
            if idx != first_idx and idx not in abs_poses:
                # 使用相对于第一帧的累计变换
                path_to_idx = find_path(first_idx, idx, relative_transforms)
                if path_to_idx:
                    abs_pose = calculate_absolute_pose(path_to_idx, relative_transforms, first_idx)
                    abs_poses[idx] = abs_pose
                    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(abs_pose))
        
        # 添加边
        if i in abs_poses and j in abs_poses:
            i_idx = available_indices.index(i)
            j_idx = available_indices.index(j)
            edge = o3d.pipelines.registration.PoseGraphEdge(i_idx, j_idx, trans, info, uncertain=False)
            pose_graph.edges.append(edge)
    
    # 全局优化 PoseGraph
    if pose_graph.edges:
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=0.1,
            edge_prune_threshold=0.25,
            reference_node=0
        )
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option
        )
    
    # 根据优化后的位姿融合点云
    combined_pcd = o3d.geometry.PointCloud()
    for i, idx in enumerate(available_indices):
        if idx in all_pcds:
            pcd = all_pcds[idx]
            if i < len(pose_graph.nodes):
                pcd.transform(pose_graph.nodes[i].pose)
            else:
                print(f"警告: 索引 {i} 超出优化节点范围，使用原始位姿")
                pcd.transform(poses[idx])
            pcd_ds = pcd.voxel_down_sample(individual_voxel)
            combined_pcd += pcd_ds
    
    combined_pcd = combined_pcd.voxel_down_sample(final_voxel)
    output_ply = os.path.join(ply_root, scene_name, f"{scene_name}_combined_downsampled_{final_voxel:.3f}.ply")
    o3d.io.write_point_cloud(output_ply, combined_pcd)
    print(f"保存配准结果到文件：{output_ply}")

def find_path(start, end, relative_transforms):
    """查找从起始帧到目标帧的路径"""
    # 简化版的广度优先搜索
    visited = set()
    queue = [(start, [start])]
    
    while queue:
        node, path = queue.pop(0)
        if node == end:
            return path
            
        if node in visited:
            continue
            
        visited.add(node)
        
        # 寻找所有相邻节点
        for pair in relative_transforms:
            i, j = pair
            if i == node and j not in visited:
                queue.append((j, path + [j]))
            elif j == node and i not in visited:
                queue.append((i, path + [i]))
    
    return None  # 没找到路径

def calculate_absolute_pose(path, relative_transforms, first_idx):
    """根据路径计算绝对位姿"""
    abs_pose = np.eye(4)
    
    for i in range(len(path) - 1):
        curr = path[i]
        next_node = path[i + 1]
        
        if (curr, next_node) in relative_transforms:
            trans, _ = relative_transforms[(curr, next_node)]
            abs_pose = abs_pose @ trans
        elif (next_node, curr) in relative_transforms:
            trans, _ = relative_transforms[(next_node, curr)]
            abs_pose = abs_pose @ np.linalg.inv(trans)
    
    return abs_pose

# ================== 子场景帧组融合（含ICP refine） ==================
def fuse_scene_groups(pose_root, ply_root, scene_name, fuse_frame_count=5, 
                      individual_voxel=0.05, final_voxel=0.05, new_dataset_root="new_dataset",
                      max_points=30000):  # 添加最大点数参数
    """
    针对单个子场景进行帧组融合 - 根据fuse_frame_count进行分组，并限制点云大小
    """
    # 扫描可用的点云文件
    pcd_dir = os.path.join(ply_root, scene_name)
    available_indices, file_paths = scan_available_pointclouds(pcd_dir)
    
    if not available_indices:
        print(f"错误: 场景 {scene_name} 没有找到任何有效的点云文件")
        return []
    
    print(f"场景 {scene_name} 找到 {len(available_indices)} 个有效点云文件")
    
    # 只加载需要的位姿数据
    pose_path = os.path.join(pose_root, scene_name, "pose.txt")
    poses = load_selective_poses(pose_path, available_indices)
    
    # 每个子场景单独处理
    fused_scene_dir = os.path.join(new_dataset_root, scene_name)
    os.makedirs(fused_scene_dir, exist_ok=True)
    
    fusion_info = []
    
    # 按照fuse_frame_count进行分组，不再使用固定的最大组大小
    available_groups = []
    i = 0
    while i < len(available_indices):
        # 检查是否有足够的帧来形成完整的组
        if i + fuse_frame_count <= len(available_indices):
            # 有足够的帧，取完整组
            group_indices = available_indices[i:i+fuse_frame_count]
        else:
            # 没有足够的帧，取剩余所有帧作为最后一组
            group_indices = available_indices[i:]
            # 如果最后一组少于2帧，跳过处理
            if len(group_indices) < 2:
                break
                
        available_groups.append(group_indices)
        i += fuse_frame_count
    
    print(f"将 {len(available_indices)} 个点云按照每组 {fuse_frame_count} 帧分成 {len(available_groups)} 组处理")
    
    # 处理每组点云
    for group_idx, group_indices in enumerate(available_groups):
        print(f"处理组 {group_idx+1}/{len(available_groups)}: 帧索引 {group_indices}")
            
        start_idx = group_indices[0]
        end_idx = group_indices[-1]
        
        # 使用空点云作为容器
        combined_pcd = o3d.geometry.PointCloud()
        
        # 处理每一帧
        for idx in group_indices:
            # 加载并处理点云
            pcd = o3d.io.read_point_cloud(file_paths[idx])
            if not pcd.has_points():
                print(f"警告: 点云 {idx} 中无点，跳过")
                continue
                
            # 转换到全局坐标系
            pcd.transform(poses[idx])
            
            # 下采样
            pcd_ds = pcd.voxel_down_sample(individual_voxel)
            
            # 如果是第一帧以外的帧，用ICP细化
            if idx != start_idx and combined_pcd.has_points():
                icp_result = o3d.pipelines.registration.registration_icp(
                    pcd_ds, combined_pcd,
                    max_correspondence_distance=0.1,
                    init=np.eye(4),
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
                )
                pcd_ds.transform(icp_result.transformation)
            
            # 直接添加到结果点云
            combined_pcd += pcd_ds
            
            # 释放内存
            del pcd, pcd_ds
        
        # 最终下采样并控制点数
        if combined_pcd.has_points():
            # 初始下采样
            combined_pcd = combined_pcd.voxel_down_sample(final_voxel)
            
            # 检查点数并进一步下采样直到低于最大点数
            current_voxel_size = final_voxel
            points_array = np.asarray(combined_pcd.points)
            num_points = len(points_array)
            
            while num_points > max_points:
                # 增加体素大小以减少点数
                current_voxel_size *= 1.2
                print(f"点云过大 ({num_points} > {max_points})，增加体素大小至 {current_voxel_size:.4f}")
                combined_pcd = combined_pcd.voxel_down_sample(current_voxel_size)
                
                points_array = np.asarray(combined_pcd.points)
                num_points = len(points_array)
            
            print(f"最终点云点数: {num_points}，使用体素大小: {current_voxel_size:.4f}")
            
            # 保存结果
            frame_count = len(group_indices)
            fused_filename = f"merged_{start_idx:04d}_{end_idx:04d}_{frame_count}frames.ply"
            output_path = os.path.join(fused_scene_dir, fused_filename)
            
            try:
                o3d.io.write_point_cloud(output_path, combined_pcd)
                print(f"子场景 {scene_name}: 保存融合点云至 {output_path}")
                
                fusion_info.append({
                    "group_name": fused_filename,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "frame_count": frame_count,
                    "frame_indices": group_indices,
                    "first_pose": poses[start_idx].tolist(),
                    "points_count": num_points,
                    "voxel_size": current_voxel_size
                })
            except Exception as e:
                print(f"保存点云时出错: {e}")
        
        # 清理内存
        del combined_pcd
        import gc
        gc.collect()
    
    return fusion_info

def process_all_scenes(pose_root, ply_root, scene_list, fuse_frame_count=5, 
                       individual_voxel=0.05, final_voxel=0.05, new_dataset_root="new_dataset", 
                       json_out="fused_info.json", max_points=30000):
    """
    对多个子场景进行点云融合处理，并将各子场景的融合信息写入同一 JSON 文件。
    输出 JSON 文件结构示例如下：
    
    {
        "cecum_t1_a": {
             "5": [{融合组信息1}, {融合组信息2}, ...],
             "10": [{...}],
             "15": [{...}],
             "20": [{...}]
        },
        "cecum_t1_b": {
            ...
        },
        ...
    }
    """
    # 如果 JSON 文件已存在，则加载现有数据，否则初始化为空字典
    if os.path.exists(json_out):
        with open(json_out, "r", encoding="utf-8") as f:
            all_fusion_info = json.load(f)
    else:
        all_fusion_info = {}

    for scene_name in scene_list:
        print(f"处理子场景：{scene_name}")
        fusion_info = fuse_scene_groups(pose_root, ply_root, scene_name, fuse_frame_count, 
                                        individual_voxel, final_voxel, new_dataset_root,
                                        max_points=max_points)
        # 针对每个场景，先按场景进行区分
        if scene_name not in all_fusion_info:
            all_fusion_info[scene_name] = {}
        # 再根据 fuse_frame_count（转换为字符串）进行区分保存
        all_fusion_info[scene_name][str(fuse_frame_count)] = fusion_info

    # 写入 JSON 文件
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_fusion_info, f, indent=4, ensure_ascii=False)
    print(f"所有融合信息已写入 JSON 文件: {json_out}")

def scan_available_pointclouds(pcd_dir, filename_pattern="{i:04d}_depth_pcd.ply"):
    """
    扫描目录，找出所有实际存在的点云文件，返回帧索引列表
    参数:
      pcd_dir: 点云文件目录
      filename_pattern: 文件名格式模板
    返回:
      存在的点云文件索引列表和对应的文件路径字典
    """
    available_indices = []
    file_paths = {}
    
    # 检查目录是否存在
    if not os.path.exists(pcd_dir):
        print(f"警告: 目录 {pcd_dir} 不存在")
        return available_indices, file_paths
    
    # 扫描目录中的所有文件
    for filename in os.listdir(pcd_dir):
        if filename.endswith("_depth_pcd.ply"):
            try:
                # 从文件名提取索引，例如从 "0004_depth_pcd.ply" 提取 4
                index = int(filename.split("_")[0])
                path = os.path.join(pcd_dir, filename)
                available_indices.append(index)
                file_paths[index] = path
            except ValueError:
                print(f"警告: 无法从文件名 {filename} 提取索引")
    
    # 排序索引，确保按顺序处理
    available_indices.sort()
    return available_indices, file_paths

def load_selective_poses(pose_path, indices):
    """
    只加载指定索引的位姿数据
    参数:
      pose_path: 位姿文件路径
      indices: 需要加载的帧索引列表
    返回:
      包含指定索引位姿的列表
    """
    print(f"选择性加载位姿: {pose_path}")
    poses = {}  # 使用字典存储，键为索引，值为位姿矩阵
    
    with open(pose_path, 'r') as f:
        lines = f.readlines()
        
        # 确保有足够的行
        max_index = max(indices) if indices else 0
        if max_index >= len(lines):
            print(f"警告: 请求的最大索引 {max_index} 超出位姿文件行数 {len(lines)}")
        
        # 只加载需要的索引的位姿
        for idx in indices:
            if idx < len(lines):
                line = lines[idx].strip()
                if not line:
                    print(f"警告: 位姿文件中第 {idx} 行为空")
                    poses[idx] = np.eye(4)  # 使用单位矩阵作为默认值
                    continue
                
                data = list(map(float, line.split(',')))
                if len(data) != 16:
                    print(f"警告: 位姿行 {idx} 数据长度错误，使用单位矩阵")
                    poses[idx] = np.eye(4)
                    continue
                
                T = np.array(data).reshape(4, 4)
                if not np.allclose(T[3, :], [0, 0, 0, 1]):
                    if np.allclose(T[:3, 3], np.zeros(3)) and not np.allclose(T[3, :3], np.zeros(3)):
                        T_fixed = np.eye(4)
                        T_fixed[:3, :3] = T[:3, :3]
                        T_fixed[:3, 3] = T[3, :3]
                        T = T_fixed
                    else:
                        print(f"警告: 第 {idx} 行的转换矩阵无效，使用单位矩阵")
                        T = np.eye(4)
                        
                poses[idx] = T
            else:
                print(f"警告: 索引 {idx} 超出位姿文件范围，使用单位矩阵")
                poses[idx] = np.eye(4)
                
    return poses

# ================== 主流程入口 ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点云配准和融合处理")
    parser.add_argument("--mode", type=str, choices=["total", "partial"], default="total",
                        help="选择处理模式：'total' 进行全局ICP配准与融合，'partial' 进行子场景帧组融合（带ICP refine）")
    # 分别指定 pose 与 ply 数据集的根目录
    parser.add_argument("--pose_root", type=str, default="/path/to/pose_dataset",
                        help="pose 数据集根目录，每个场景文件夹下包含 pose.txt")
    parser.add_argument("--ply_root", type=str, default="/path/to/ply_dataset",
                        help="ply 数据集根目录，每个场景文件夹下包含 fragments_ply")
    parser.add_argument("--scene", type=str, default="cecum_t1_a",
                        help="单个子场景名称（用于 total 模式）")
    parser.add_argument("--scene_list", nargs="+", default=["scene1", "scene2", "scene3"],
                        help="多个子场景名称列表（用于 partial 模式）")
    parser.add_argument("--fuse_frame_count", type=int, default=5,
                        help="每次融合的帧数")
    parser.add_argument("--individual_voxel", type=float, default=0.05,
                        help="各个点云预下采样体素尺寸")
    parser.add_argument("--final_voxel", type=float, default=0.05,
                        help="融合后整体点云下采样体素尺寸")
    parser.add_argument("--new_dataset_root", type=str, default="/path/to/your/new_dataset",
                        help="新数据集保存目录（用于 partial 模式）")
    parser.add_argument("--json_out", type=str, default="fused_info.json",
                        help="保存融合信息的 JSON 文件名称")
    parser.add_argument("--max_points", type=int, default=30000,
                       help="融合点云的最大点数")
    args = parser.parse_args()

    if args.mode == "total":
        align_scene_with_icp(args.pose_root, args.ply_root, args.scene,
                             individual_voxel=args.individual_voxel,
                             final_voxel=args.final_voxel,
                             icp_parallel=True)
    elif args.mode == "partial":
        process_all_scenes(args.pose_root, args.ply_root, args.scene_list, args.fuse_frame_count,
                           individual_voxel=args.individual_voxel,
                           final_voxel=args.final_voxel,
                           new_dataset_root=args.new_dataset_root,
                           json_out=args.json_out,
                           max_points=args.max_points)