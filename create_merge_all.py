import os
import json
import numpy as np
import open3d as o3d
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import copy  # 新增导入
import re

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

def invert_rigid_transform(T):
    """
    计算 4x4 刚体变换矩阵的逆
    :param T: numpy 数组 (4,4) 刚体变换矩阵
    :return: numpy 数组 (4,4) 逆变换矩阵
    """
    R = T[:3, :3]  # 提取旋转部分 (3x3)
    t = T[:3, 3]   # 提取平移向量 (3x1)
    
    # 计算逆变换
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T  # 旋转矩阵转置
    T_inv[:3, 3] = -R.T @ t  # 平移部分取负并乘以 R^T
    
    return T_inv

def compute_relative_pose(pose_i, pose_j):
    """计算相对位姿: T_ij = T_j @ inv(T_i) (适用于 T_cam_to_world)"""
    return pose_j @ invert_rigid_transform(pose_i)

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
    针对单个子场景进行全局 ICP 配准：
      1. 从 pose_root 加载绝对位姿，从 ply_root 加载点云（fragments_ply）。
      2. 计算连续帧 ICP 相对变换，构造 PoseGraph 进行全局优化。
      3. 根据优化后的位姿融合所有点云，并保存结果。
    """
    # 加载 pose.txt 中所有绝对位姿
    pose_path = os.path.join(pose_root, scene_name, "pose.txt")
    all_poses = load_poses(pose_path)
    num_frames = len(all_poses)

    # 从 ply_root 加载所有点云数据
    pcd_dir = os.path.join(ply_root, scene_name)
    all_pcds = []
    
    # 寻找所有含有 "_depth_pcd.ply" 的点云文件
    pcd_files = []
    for file in sorted(os.listdir(pcd_dir)):
        if "_depth_pcd.ply" in file:
            pcd_path = os.path.join(pcd_dir, file)
            pcd_files.append(pcd_path)
    
    if not pcd_files:
        raise ValueError(f"在目录 {pcd_dir} 中未找到任何包含 '_depth_pcd.ply' 的点云文件")
    
    print(f"找到 {len(pcd_files)} 个单帧点云文件")
    
    for pcd_path in tqdm(pcd_files, desc="加载点云"):
        if not os.path.exists(pcd_path):
            print(f"警告: 文件 {pcd_path} 不存在，跳过此帧")
            all_pcds.append(o3d.geometry.PointCloud())
            continue
        pcd = o3d.io.read_point_cloud(pcd_path)
        if not pcd.has_points():
            print(f"警告: 文件 {pcd_path} 无法加载有效点云")
        all_pcds.append(pcd)

    # 计算连续帧之间的 ICP 配准
    relative_results = [None] * (len(all_pcds) - 1)
    def icp_task(i):
        src_pcd = all_pcds[i]
        tgt_pcd = all_pcds[i + 1]
        # 从文件名中提取帧索引
        src_filename = os.path.basename(pcd_files[i])
        tgt_filename = os.path.basename(pcd_files[i + 1])
        src_idx = int(src_filename.split('_')[1])
        tgt_idx = int(tgt_filename.split('_')[1])
        
        init_trans = compute_relative_pose(all_poses[src_idx], all_poses[tgt_idx])
        final_trans = multi_scale_icp(src_pcd, tgt_pcd, init_trans=init_trans)
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            src_pcd, tgt_pcd, 0.1, final_trans
        )
        return i, final_trans, info

    if icp_parallel:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(icp_task, i): i for i in range(len(all_pcds) - 1)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="ICP 配准"):
                i, final_trans, info = future.result()
                relative_results[i] = (final_trans, info)
    else:
        for i in tqdm(range(len(all_pcds) - 1), desc="ICP 配准"):
            _, final_trans, info = icp_task(i)
            relative_results[i] = (final_trans, info)

    # 构造 PoseGraph，累计计算各帧绝对位姿
    pose_graph = o3d.pipelines.registration.PoseGraph()
    abs_poses = [np.eye(4)]
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
    for i in range(len(all_pcds) - 1):
        trans, info = relative_results[i]
        abs_pose = abs_poses[i] @ trans
        abs_poses.append(abs_pose)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(abs_pose))
        edge = o3d.pipelines.registration.PoseGraphEdge(i, i + 1, trans, info, uncertain=False)
        pose_graph.edges.append(edge)

    # 全局优化 PoseGraph
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
    for i in tqdm(range(len(all_pcds)), desc="合并点云"):
        pcd = all_pcds[i]
        pcd.transform(pose_graph.nodes[i].pose)
        pcd_ds = pcd.voxel_down_sample(individual_voxel)
        combined_pcd += pcd_ds
    combined_pcd = combined_pcd.voxel_down_sample(final_voxel)
    output_ply = os.path.join(ply_root, scene_name, f"{scene_name}_combined_all_1frames.ply")
    o3d.io.write_point_cloud(output_ply, combined_pcd)
    print(f"保存配准结果到文件：{output_ply}")

# ================== 子场景帧组融合（含ICP refine） ==================
def fuse_scene_groups(pose_root, ply_root, scene_name, fuse_frame_count=5, 
                      individual_voxel=0.05, final_voxel=0.05, new_dataset_root="new_dataset"):
    """
    针对单个子场景进行帧组融合：
      1. 从 pose_root 加载该场景的 pose.txt，从 ply_root 加载点云（fragments_ply）。
      2. 将每帧点云根据对应的绝对位姿转换到全局坐标系。
      3. 按 fuse_frame_count 分组，每组以第一帧为基准，并对后续帧使用较少迭代次数的 ICP refine 对齐后融合。
      4. 融合结果保存到 new_dataset_root 下对应场景目录中，同时记录每组第一帧的 pose 以写入 JSON 文件。
    """
    # 加载 pose.txt
    pose_path = os.path.join(pose_root, scene_name, "pose.txt")
    poses = load_poses(pose_path)
    
    # 加载含有 "_depth_pcd.ply" 的点云文件
    pcd_dir = os.path.join(ply_root, scene_name)
    pcd_files = []
    for file in sorted(os.listdir(pcd_dir)):
        if "_depth_pcd.ply" in file:
            pcd_path = os.path.join(pcd_dir, file)
            pcd_files.append(pcd_path)
    
    if not pcd_files:
        raise ValueError(f"在目录 {pcd_dir} 中未找到任何包含 '_depth_pcd.ply' 的点云文件")
    
    # 每个子场景单独处理，不混淆
    fused_scene_dir = os.path.join(new_dataset_root, scene_name)
    os.makedirs(fused_scene_dir, exist_ok=True)
    
    fusion_info = []
    
    # 按非重叠方式分组
    group_start_indices = list(range(0, len(pcd_files), fuse_frame_count))
    # 处理最后一组（如果剩余不足fuse_frame_count帧）
    if len(pcd_files) % fuse_frame_count != 0 and len(pcd_files) > fuse_frame_count:
        last_start = max(0, len(pcd_files) - fuse_frame_count)
        if last_start not in group_start_indices:
            group_start_indices.append(last_start)
    group_start_indices = sorted(list(set(group_start_indices)))
    
    # 对每组进行融合，并对后续帧进行 ICP refine
    for group_idx, start_idx in enumerate(group_start_indices):
        end_idx = min(start_idx + fuse_frame_count - 1, len(pcd_files) - 1)
        
        # 以第一帧作为基准
        base_pcd_path = pcd_files[start_idx]
        base_pcd = o3d.io.read_point_cloud(base_pcd_path)
        
        # 从文件名中提取帧索引
        base_filename = os.path.basename(base_pcd_path)
        base_frame_idx = int(base_filename.split('_')[1])
        
        base_pcd.transform(poses[base_frame_idx])
        base_pcd_ds = base_pcd.voxel_down_sample(individual_voxel)
        combined_pcd = base_pcd_ds
        valid_group = True
        
        # 后续帧: 先根据绝对pose变换，再ICP refine对齐
        for j in range(start_idx + 1, end_idx + 1):
            curr_pcd_path = pcd_files[j]
            curr_pcd = o3d.io.read_point_cloud(curr_pcd_path)
            
            if not curr_pcd.has_points():
                print(f"警告: 点云 {curr_pcd_path} 中点云为空，跳过当前融合组")
                valid_group = False
                break
                
            # 从文件名中提取帧索引
            curr_filename = os.path.basename(curr_pcd_path)
            curr_frame_idx = int(curr_filename.split('_')[1])
            
            curr_pcd.transform(poses[curr_frame_idx])
            curr_pcd_ds = curr_pcd.voxel_down_sample(individual_voxel)
            
            icp_result = o3d.pipelines.registration.registration_icp(
                curr_pcd_ds, base_pcd_ds,
                max_correspondence_distance=0.1,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            curr_pcd_ds.transform(icp_result.transformation)
            combined_pcd += curr_pcd_ds
            
        if not valid_group:
            print(f"警告: 子场景 {scene_name} 中融合组 {group_idx} 被跳过")
            continue
            
        combined_pcd = combined_pcd.voxel_down_sample(final_voxel)
        
        # 文件命名：显示起始帧索引，终止帧索引和融合帧数
        start_frame_idx = int(os.path.basename(pcd_files[start_idx]).split('_')[1])
        end_frame_idx = int(os.path.basename(pcd_files[end_idx]).split('_')[1])
        actual_frame_count = end_idx - start_idx + 1
        
        fused_filename = f"merged_{start_frame_idx:04d}_{end_frame_idx:04d}_{actual_frame_count}frames.ply"
        output_path = os.path.join(fused_scene_dir, fused_filename)
        o3d.io.write_point_cloud(output_path, combined_pcd)
        print(f"子场景 {scene_name}: 保存融合点云至 {output_path}")
        
        fusion_info.append({
            "group_name": fused_filename,
            "start_idx": start_frame_idx,
            "end_idx": end_frame_idx,
            "fuse_frame_count": actual_frame_count,
            "first_pose": poses[base_frame_idx].tolist()
        })
        
    return fusion_info

def process_all_scenes(pose_root, ply_root, scene_list, fuse_frame_count=5, 
                       individual_voxel=0.05, final_voxel=0.05, new_dataset_root="new_dataset", 
                       json_out="fused_info.json"):
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
        fusion_info = fuse_scene_groups(pose_root, ply_root, scene_name, fuse_frame_count, individual_voxel, final_voxel, new_dataset_root)
        # 针对每个场景，先按场景进行区分
        if scene_name not in all_fusion_info:
            all_fusion_info[scene_name] = {}
        # 再根据 fuse_frame_count（转换为字符串）进行区分保存
        all_fusion_info[scene_name][str(fuse_frame_count)] = fusion_info

    # 写入 JSON 文件
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_fusion_info, f, indent=4, ensure_ascii=False)
    print(f"所有融合信息已写入 JSON 文件: {json_out}")

def merge_all_point_clouds(pose_path, pcd_dir, output_path, voxel_size=0.05):
    """
    将场景中所有单帧点云合成为一个完整点云
    
    参数:
        pose_path: pose.txt 文件路径
        pcd_dir: 包含所有单帧点云的目录
        output_path: 输出的合成点云文件路径
        voxel_size: 下采样的体素大小
        max_points: 最终点云的最大点数(默认30000)
    """
    # 加载所有位姿
    poses = load_poses(pose_path)
    num_frames = len(poses)
    print(f"共加载了 {num_frames} 个位姿")
    
    # 寻找所有含有 "_depth_pcd.ply" 的点云文件
    pcd_files = []
    for file in sorted(os.listdir(pcd_dir)):
        if "_depth_pcd.ply" in file:
            pcd_path = os.path.join(pcd_dir, file)
            pcd_files.append(pcd_path)
    
    if not pcd_files:
        raise ValueError(f"在目录 {pcd_dir} 中未找到任何包含 '_depth_pcd.ply' 的点云文件")
    
    print(f"找到 {len(pcd_files)} 个单帧点云文件")
    
    # 合并所有点云
    combined_pcd = o3d.geometry.PointCloud()
    batch_count = 0

    for pcd_path in tqdm(pcd_files, desc="合并点云"):
        # 读取点云
        pcd = o3d.io.read_point_cloud(pcd_path)
        if not pcd.has_points():
            print(f"警告: 文件 {pcd_path} 无法加载有效点云，跳过")
            continue
        
        # 从文件名中提取帧索引，使用正则表达式以提高鲁棒性
        filename = os.path.basename(pcd_path)
        match = re.search(r'(\d+)', filename)
        if match:
            frame_idx = int(match.group(1))
        else:
            print(f"警告：无法从文件名 {filename} 中提取帧索引，跳过此文件")
            continue
        
        # 确保帧索引在位姿数组范围内
        if frame_idx < len(poses):
            pcd.transform(poses[frame_idx])
            # 对单帧点云进行下采样以减少数据量
            pcd_ds = pcd.voxel_down_sample(voxel_size)
            combined_pcd += pcd_ds
            
            # 每添加10个点云后进行一次体素下采样控制点数
            batch_count += 1
            if batch_count % 10 == 0:
                # 进行体素下采样以减少大量重复点
                combined_pcd = combined_pcd.voxel_down_sample(voxel_size)
        else:
            print(f"警告: 帧索引 {frame_idx} 超出位姿数组范围，跳过")
    
    # 最终下采样合并后的点云
    print(f"执行最终下采样 (当前点数: {len(combined_pcd.points)})...")
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size)
    
    # 保存合并后的点云
    print(f"保存合成点云到: {output_path}")
    o3d.io.write_point_cloud(output_path, combined_pcd)
    print(f"点云合成完成，共有 {len(combined_pcd.points)} 个点")
    
    return combined_pcd

def load_point_clouds(pcd_dir, num_frames, scene_name):
    """
    辅助函数：加载指定目录下所有含有 "_depth_pcd.ply" 的点云文件
    参数：
      pcd_dir: 点云数据文件所在目录
      num_frames: 应该加载的帧数（实际加载的可能不同）
      scene_name: 当前处理的场景名，用于日志输出
    """
    all_pcds = []
    pcd_files = []

    for file in sorted(os.listdir(pcd_dir)):
        if "_depth_pcd.ply" in file:
            pcd_path = os.path.join(pcd_dir, file)
            pcd_files.append(pcd_path)

    # 限制加载帧的数量，仅保留前 num_frames 个文件
    if num_frames is not None:
        pcd_files = pcd_files[:num_frames]

    if not pcd_files:
        raise ValueError(f"在目录 {pcd_dir} 中未找到任何包含 '_depth_pcd.ply' 的点云文件")

    for pcd_path in tqdm(pcd_files, desc="加载点云"):
        if not os.path.exists(pcd_path):
            print(f"警告: 场景 {scene_name} 中文件 {pcd_path} 不存在，添加空点云占位")
            all_pcds.append(o3d.geometry.PointCloud())
            continue
        pcd = o3d.io.read_point_cloud(pcd_path)
        if not pcd.has_points():
            print(f"警告: 文件 {pcd_path} 无法加载有效点云")
        all_pcds.append(pcd)

    return all_pcds

# ================== 主流程入口 ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点云配准和融合处理")
    parser.add_argument("--mode", type=str, choices=["align", "fuse", "merge"], default="merge",
                        help="选择处理模式：'align' 进行全局ICP配准与融合，'fuse' 进行子场景帧组融合（带ICP refine），'merge' 直接合并所有点云")
    # 分别指定 pose 与 ply 数据集的根目录
    parser.add_argument("--pose_root", type=str, default="/SAN/medic/MRpcr/C3VD",
                        help="pose 数据集根目录，每个场景文件夹下包含 pose.txt")
    parser.add_argument("--ply_root", type=str, default="/SAN/medic/MRpcr/fused_C3VD",
                        help="ply 数据集根目录，每个场景文件夹包含点云文件")
    parser.add_argument("--scene", type=str, default="cecum_t1_a",
                        help="单个子场景名称")
    parser.add_argument("--scene_list", nargs="+", default=["cecum_t1_a"],
                        help="多个子场景名称列表（用于 fuse 模式）")
    parser.add_argument("--fuse_frame_count", type=int, default=5,
                        help="每次融合的帧数")
    parser.add_argument("--individual_voxel", type=float, default=0.05,
                        help="各个点云预下采样体素尺寸")
    parser.add_argument("--final_voxel", type=float, default=0.05,
                        help="融合后整体点云下采样体素尺寸")
    parser.add_argument("--new_dataset_root", type=str, default="/SAN/medic/MRpcr/fused_all_C3VD",
                        help="新数据集保存目录（用于 fuse 模式）")
    parser.add_argument("--json_out", type=str, default="fused_info.json",
                        help="保存融合信息的 JSON 文件名称")
    parser.add_argument("--pose_path", type=str, default="/SAN/medic/MRpcr/C3VD/cecum_t1_a/pose.txt",
                        help="pose.txt 文件路径（用于 merge 模式）")
    parser.add_argument("--pcd_dir", type=str, default="/SAN/medic/MRpcr/fused_C3VD/cecum_t1_a",
                        help="包含所有单帧点云的目录（用于 merge 模式）")
    parser.add_argument("--output_path", type=str, default="/SAN/medic/MRpcr/fused_C3VD/cecum_t1_a/cecum_t1_a_merged_all.ply",
                        help="输出的合成点云文件路径（用于 merge 模式）")
    parser.add_argument("--voxel_size", type=float, default=0.05,
                        help="下采样的体素大小（用于 merge 模式）")
    args = parser.parse_args()

    if args.mode == "align":
        align_scene_with_icp(args.pose_root, args.ply_root, args.scene,
                             individual_voxel=args.individual_voxel,
                             final_voxel=args.final_voxel,
                             icp_parallel=True)
    elif args.mode == "fuse":
        process_all_scenes(args.pose_root, args.ply_root, args.scene_list, args.fuse_frame_count,
                           individual_voxel=args.individual_voxel,
                           final_voxel=args.final_voxel,
                           new_dataset_root=args.new_dataset_root,
                           json_out=args.json_out)
    elif args.mode == "merge":
        merge_all_point_clouds(args.pose_path, args.pcd_dir, args.output_path, args.voxel_size) 