#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本脚本遍历数据集下的所有场景文件夹。每个场景文件夹应具有如下结构：
       
  （C3VD结构方案）：
    C3VD/
       ├── C3VD_obj/            # 参考模型(.obj)
       └── C3VD_ply/            # 包含各个场景子目录
          └── [场景名称]/        # 场景目录(如 cecum_t1_a)，包含多个.ply文件

脚本执行以下步骤：
  1. 对场景文件夹中的参考 3D 模型采样生成点云，并计算包围盒信息（仅计算一次，保存到 JSON 文件中）；
  2. 对每个 ply 文件进行点云加载、去噪与包围盒计算，
  3. 根据参考点云（旧结构中为 .obj 文件或新结构中 ref 文件）的包围盒与源点云的包围盒，计算缩放系数（支持两种模式：'power'——仅允许10的次方倍；'relative'——直接采用平均比例）；
  4. 对每个原始点云进行缩放，并保存缩放后的文件（文件名前添加 "scaled_" 前缀），同时记录每个文件对应的缩放系数；
  5. 将每个场景的缩放系数和参考模型包围盒信息保存到该场景文件夹下的 JSON 文件中。
"""

import os
import argparse
import json
import math
import numpy as np
import open3d as o3d

def sample_obj_point_cloud(obj_file, num_points=10000):
    """
    从 obj 文件加载网格，并均匀采样生成点云。
    """
    mesh = o3d.io.read_triangle_mesh(obj_file)
    if mesh.is_empty():
        print(f"加载网格失败：{obj_file}")
        return None
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd

def compute_aabb_dimensions(pcd):
    """
    计算点云的轴对齐包围盒（AABB），返回 (dims, min_bound, max_bound)
    dims 为 (长, 宽, 高) 的差值。
    """
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = np.array(bbox.get_min_bound())
    max_bound = np.array(bbox.get_max_bound())
    dims = max_bound - min_bound
    return dims, min_bound, max_bound

def remove_outliers_iqr(pcd):
    """
    使用 IQR 算法对点云数据的每个坐标进行过滤，
    过滤条件：某一维度上小于 Q1 - 1.5 * IQR 或大于 Q3 + 1.5 * IQR 的点视为异常点。
    """
    pts = np.asarray(pcd.points)
    mask = np.ones(pts.shape[0], dtype=bool)
    for i in range(3):
        coord = pts[:, i]
        Q1 = np.percentile(coord, 25)
        Q3 = np.percentile(coord, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = mask & ((coord >= lower_bound) & (coord <= upper_bound))
    filtered_pts = pts[mask]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_pts)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)[mask]
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors)
    return filtered_pcd

def adjust_point_cloud_scaling(pcd, scaling_factor):
    """
    对点云进行均匀缩放
    """
    pcd_scaled = o3d.geometry.PointCloud()
    pts = np.asarray(pcd.points)
    pts_scaled = pts * scaling_factor
    pcd_scaled.points = o3d.utility.Vector3dVector(pts_scaled)
    if pcd.has_colors():
        pcd_scaled.colors = pcd.colors
    return pcd_scaled

def process_scene_folder(scene_folder, mode, output_scaling):
    """
    处理单个场景文件夹（旧结构处理逻辑）
    """
    # 查找参考模型 .obj 文件
    files_in_scene = os.listdir(scene_folder)
    obj_files = [f for f in files_in_scene if f.lower().endswith('.obj')]
    if not obj_files:
        print(f"场景文件夹 {scene_folder} 中没有找到 .obj 文件，跳过")
        return
    ref_obj_path = os.path.join(scene_folder, obj_files[0])
    
    # 定位 fragments_ply 文件夹
    fragments_dir = os.path.join(scene_folder, "fragments_ply")
    if not os.path.exists(fragments_dir) or not os.path.isdir(fragments_dir):
        print(f"场景文件夹 {scene_folder} 中没有找到 fragments_ply 文件夹，跳过")
        return

    ply_files = [f for f in os.listdir(fragments_dir) if f.lower().endswith('.ply')]
    if not ply_files:
        print(f"fragments_ply 文件夹 {fragments_dir} 中没有找到 .ply 文件，跳过")
        return
    
    # 计算参考模型的包围盒（如果尚未计算）
    global ref_bbox_dict
    if scene_folder in ref_bbox_dict:
        ref_dims = np.array(ref_bbox_dict[scene_folder]['dims'])
    else:
        ref_pcd = sample_obj_point_cloud(ref_obj_path, num_points=10000)
        if ref_pcd is None:
            print(f"参考 3D 模型 {ref_obj_path} 加载失败")
            return
        ref_dims, ref_min, ref_max = compute_aabb_dimensions(ref_pcd)
        ref_bbox_dict[scene_folder] = {
            'dims': ref_dims.tolist(),
            'min_bound': ref_min.tolist(),
            'max_bound': ref_max.tolist()
        }
    
    # 在输出字典中为该场景创建条目
    if scene_folder not in output_scaling:
        output_scaling[scene_folder] = {}
    
    # 遍历 fragments_ply 中的每个 ply 文件
    for ply_file in ply_files:
        ply_path = os.path.join(fragments_dir, ply_file)
        
        # 加载 ply 点云
        depth_pcd = o3d.io.read_point_cloud(ply_path)
        if depth_pcd.is_empty():
            print(f"点云 {ply_path} 为空，跳过")
            continue
        
        # IQR 去噪
        depth_pcd_filtered = remove_outliers_iqr(depth_pcd)
        depth_dims, depth_min, depth_max = compute_aabb_dimensions(depth_pcd_filtered)
        
        # 若任一维度为 0，则无法计算缩放
        if np.any(depth_dims == 0):
            print(f"点云 {ply_path} 在某个维度上尺寸为 0，无法计算缩放，跳过")
            continue
        
        # 计算各维度比例，取平均值作为缩放系数
        ratios = ref_dims / depth_dims
        raw_scale_factor = np.mean(ratios)
    
        # 若缩放系数接近 1，则不进行缩放
        if abs(raw_scale_factor - 1.0) < 0.1:
            scaled_factor = 1.0
        else:
            if mode == 'power':
                log_scale = math.log10(raw_scale_factor)
                rounded_log = round(log_scale)
                scaled_factor = 10 ** rounded_log
            else:
                scaled_factor = raw_scale_factor
        
        # 记录该 ply 文件的缩放系数
        output_scaling[scene_folder][ply_file] = scaled_factor
        
        # 对原始 ply 点云缩放
        depth_pcd_scaled = adjust_point_cloud_scaling(depth_pcd, scaled_factor)
        
        # 保存缩放后的 ply 文件，文件名前添加 "scaled_"
        scaled_ply_filename = "scaled_" + ply_file
        scaled_ply_path = os.path.join(fragments_dir, scaled_ply_filename)
        o3d.io.write_point_cloud(scaled_ply_path, depth_pcd_scaled)
        print(f"场景 [{scene_folder}] 中 ply 文件 [{ply_file}]：缩放系数 {scaled_factor:.4f}，已保存至 {scaled_ply_path}")
        
        #（调试输出）保存包围盒信息到文本文件
        bb_info_filename = f"bounding_box_info_{os.path.splitext(ply_file)[0]}.txt"
        bb_info_path = os.path.join(fragments_dir, bb_info_filename)
        with open(bb_info_path, "w") as f:
            f.write("参考 3D 模型包围盒 (长, 宽, 高): {}\n".format(ref_dims.tolist()))
            f.write("深度图生成点云包围盒 (长, 宽, 高): {}\n".format(depth_dims.tolist()))
    
    # --- 清理 fragments_ply 文件夹，只保留缩放后的 ply 文件 ---
    for filename in os.listdir(fragments_dir):
        # 如果文件名不以 "scaled_" 开头或者扩展名不为 .ply，则删除该文件
        if not (filename.startswith("scaled_") and filename.lower().endswith(".ply")):
            file_path = os.path.join(fragments_dir, filename)
            os.remove(file_path)
            print(f"已删除非缩放文件: {file_path}")

def process_colon_scene_folder(scene_folder, mode, output_scaling):
    """
    针对新结构的 colon 文件夹进行处理（不改变 power 模式）。
    参考点云来自 scene_folder/ref，下属文件为 .obj 或 .ply；
    源点云来自 scene_folder/test 和 scene_folder/train，
    通过比对文件名前面的部分（公共前缀）进行配对，并计算缩放系数。
    """
    print(f"处理 colon 文件夹结构: {scene_folder}")
    ref_folder = os.path.join(scene_folder, "ref")
    source_folders = []
    test_folder = os.path.join(scene_folder, "test")
    train_folder = os.path.join(scene_folder, "train")
    if os.path.exists(test_folder) and os.path.isdir(test_folder):
        source_folders.append(test_folder)
    if os.path.exists(train_folder) and os.path.isdir(train_folder):
        source_folders.append(train_folder)
    if not os.path.exists(ref_folder):
        print(f"场景文件夹 {scene_folder} 中未找到 ref 文件夹，跳过")
        return

    # 获取 ref 文件夹下所有可作为参考的文件（.obj 或 .ply）
    ref_files = [f for f in os.listdir(ref_folder) if f.lower().endswith(('.obj', '.ply'))]
    if not ref_files:
        print(f"ref 文件夹 {ref_folder} 中没有找到有效的参考文件，跳过")
        return

    # 初始化该场景的缩放系数记录
    if scene_folder not in output_scaling:
        output_scaling[scene_folder] = {}

    # 使用全局字典记录该场景中所有参考文件的包围盒信息
    global ref_bbox_dict
    if scene_folder not in ref_bbox_dict:
        ref_bbox_dict[scene_folder] = {}  # 结构： { ref_file: {dims, min_bound, max_bound} }

    # 定义计算两个字符串公共前缀长度的辅助函数
    def common_prefix_length(a, b):
        count = 0
        for ca, cb in zip(a, b):
            if ca == cb:
                count += 1
            else:
                break
        return count

    # 遍历各源文件夹（test 和 train）
    for src_folder in source_folders:
        src_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.ply', '.obj'))]
        for src_file in src_files:
            src_path = os.path.join(src_folder, src_file)
            # 在 ref 文件中查找与当前源文件前面若干字符匹配最长的文件
            best_match = None
            best_len = 0
            src_basename = os.path.splitext(src_file)[0]
            for ref_file in ref_files:
                ref_basename = os.path.splitext(ref_file)[0]
                cp_len = common_prefix_length(src_basename, ref_basename)
                if cp_len > best_len:
                    best_len = cp_len
                    best_match = ref_file
            # 设置一个基本的阈值，确保匹配合理
            if best_match is None or best_len < 3:
                print(f"未找到与 {src_file} 匹配的参考文件，跳过")
                continue

            ref_file_path = os.path.join(ref_folder, best_match)
            # 加载参考点云：若为 obj 文件，则采样；否则直接加载
            ext = os.path.splitext(best_match)[1].lower()
            if ext == ".obj":
                ref_pcd = sample_obj_point_cloud(ref_file_path, num_points=10000)
            else:
                ref_pcd = o3d.io.read_point_cloud(ref_file_path)
            if ref_pcd is None or ref_pcd.is_empty():
                print(f"参考文件 {ref_file_path} 加载失败，跳过")
                continue
            ref_dims, ref_min, ref_max = compute_aabb_dimensions(ref_pcd)
            # 记录参考文件的包围盒信息（若未记录过）
            if best_match not in ref_bbox_dict[scene_folder]:
                ref_bbox_dict[scene_folder][best_match] = {
                    'dims': ref_dims.tolist(),
                    'min_bound': ref_min.tolist(),
                    'max_bound': ref_max.tolist()
                }
            
            # 加载源点云
            src_ext = os.path.splitext(src_file)[1].lower()
            if src_ext == ".obj":
                src_pcd = sample_obj_point_cloud(src_path, num_points=10000)
            else:
                src_pcd = o3d.io.read_point_cloud(src_path)
            if src_pcd is None or src_pcd.is_empty():
                print(f"源文件 {src_path} 加载失败或为空，跳过")
                continue
            # IQR 去噪
            src_pcd_filtered = remove_outliers_iqr(src_pcd)
            src_dims, src_min, src_max = compute_aabb_dimensions(src_pcd_filtered)
            if np.any(src_dims == 0):
                print(f"源文件 {src_path} 在某个维度上尺寸为 0，跳过")
                continue

            # 计算各维度比例，取平均值作为缩放系数
            ratios = ref_dims / src_dims
            raw_scale_factor = np.mean(ratios)

            if abs(raw_scale_factor - 1.0) < 0.1:
                scaled_factor = 1.0
            else:
                if mode == 'power':
                    log_scale = math.log10(raw_scale_factor)
                    rounded_log = round(log_scale)
                    scaled_factor = 10 ** rounded_log
                else:  # relative 模式
                    scaled_factor = raw_scale_factor

            # 记录该源文件对应的缩放系数（以相对路径记录，便于区分 test 与 train 文件）
            relative_src_file = os.path.join(os.path.basename(src_folder), src_file)
            output_scaling[scene_folder][relative_src_file] = scaled_factor

            # 对源点云进行缩放
            src_pcd_scaled = adjust_point_cloud_scaling(src_pcd, scaled_factor)
            # 保存缩放后的点云，文件名前添加 "scaled_"
            scaled_filename = src_file + "_scaled"
            scaled_path = os.path.join(src_folder, scaled_filename)
            o3d.io.write_point_cloud(scaled_path, src_pcd_scaled)
            print(f"文件 [{src_file}] 使用参考文件 [{best_match}] 缩放系数 {scaled_factor:.4f}，已保存至 {scaled_path}")

        # 清理 src_folder 中非 "scaled_" 开头的点云文件
        for filename in os.listdir(src_folder):
            if not (filename.startswith("scaled_") and filename.lower().endswith(('.ply', '.obj'))):
                file_path = os.path.join(src_folder, filename)
                try:
                    os.remove(file_path)
                    print(f"已删除非缩放文件: {file_path}")
                except Exception as e:
                    print(f"无法删除 {file_path}: {e}")

def process_c3vd_folder(base_folder, mode, output_scaling):
    """
    处理C3VD数据集的特殊结构。
    C3VD数据集结构：
    - C3VD_obj/: 包含参考模型(.obj)
    - C3VD_ply/: 包含场景子目录，每个子目录包含多个.ply文件
    """
    print(f"处理C3VD文件夹结构: {base_folder}")
    
    # 检查C3VD_obj和C3VD_ply文件夹是否存在
    obj_folder = os.path.join(base_folder, "C3VD_obj")
    ply_base_folder = os.path.join(base_folder, "C3VD_ply")
    
    if not os.path.exists(obj_folder) or not os.path.isdir(obj_folder):
        print(f"C3VD_obj文件夹不存在于 {base_folder}，跳过")
        return
    
    if not os.path.exists(ply_base_folder) or not os.path.isdir(ply_base_folder):
        print(f"C3VD_ply文件夹不存在于 {base_folder}，跳过")
        return
    
    # 获取参考模型文件
    obj_files = [f for f in os.listdir(obj_folder) if f.lower().endswith('.obj')]
    if not obj_files:
        print(f"C3VD_obj文件夹中没有找到.obj文件，跳过")
        return
    
    # 获取场景子目录
    scene_folders = [d for d in os.listdir(ply_base_folder) 
                     if os.path.isdir(os.path.join(ply_base_folder, d))]
    if not scene_folders:
        print(f"C3VD_ply文件夹中没有找到任何子目录，跳过")
        return
    
    # 对每个场景子目录执行处理
    for scene_name in scene_folders:
        scene_folder = os.path.join(ply_base_folder, scene_name)
        print(f"处理场景: {scene_name}")
        
        # 找到与场景名称最匹配的参考模型
        best_match = None
        best_score = 0
        for obj_file in obj_files:
            obj_basename = os.path.splitext(obj_file)[0]
            # 简单比较场景名和obj文件名的相似度
            common_chars = sum(1 for a, b in zip(scene_name, obj_basename) if a == b)
            if common_chars > best_score:
                best_score = common_chars
                best_match = obj_file
        
        if best_match is None:
            print(f"未找到与场景 {scene_name} 匹配的参考模型，使用第一个.obj文件")
            best_match = obj_files[0]
        
        ref_obj_path = os.path.join(obj_folder, best_match)
        print(f"为场景 {scene_name} 选择参考模型: {best_match}")
        
        # 初始化该场景的缩放系数记录
        if scene_folder not in output_scaling:
            output_scaling[scene_folder] = {}
        
        # 计算参考模型的包围盒
        global ref_bbox_dict
        scene_key = f"C3VD_{scene_name}"  # 使用唯一标识符作为字典键
        if scene_key in ref_bbox_dict:
            ref_dims = np.array(ref_bbox_dict[scene_key]['dims'])
        else:
            ref_pcd = sample_obj_point_cloud(ref_obj_path, num_points=10000)
            if ref_pcd is None:
                print(f"参考模型 {ref_obj_path} 加载失败，跳过该场景")
                continue
            ref_dims, ref_min, ref_max = compute_aabb_dimensions(ref_pcd)
            ref_bbox_dict[scene_key] = {
                'obj_file': best_match,
                'dims': ref_dims.tolist(),
                'min_bound': ref_min.tolist(),
                'max_bound': ref_max.tolist()
            }
        
        # 处理场景中的每个.ply文件
        ply_files = [f for f in os.listdir(scene_folder) if f.lower().endswith('.ply')]
        if not ply_files:
            print(f"场景 {scene_name} 中没有找到.ply文件，跳过")
            continue
        
        for ply_file in ply_files:
            # 跳过已处理的文件
            if ply_file.startswith("scaled_"):
                continue
                
            ply_path = os.path.join(scene_folder, ply_file)
            
            # 加载点云
            src_pcd = o3d.io.read_point_cloud(ply_path)
            if src_pcd is None or src_pcd.is_empty():
                print(f"点云 {ply_path} 加载失败或为空，跳过")
                continue
            
            # IQR去噪
            src_pcd_filtered = remove_outliers_iqr(src_pcd)
            src_dims, src_min, src_max = compute_aabb_dimensions(src_pcd_filtered)
            
            # 检查维度
            if np.any(src_dims == 0):
                print(f"点云 {ply_path} 在某个维度上尺寸为0，跳过")
                continue
            
            # 计算缩放系数
            ratios = ref_dims / src_dims
            raw_scale_factor = np.mean(ratios)
            
            if abs(raw_scale_factor - 1.0) < 0.1:
                scaled_factor = 1.0
            else:
                if mode == 'power':
                    log_scale = math.log10(raw_scale_factor)
                    rounded_log = round(log_scale)
                    scaled_factor = 10 ** rounded_log
                else:  # relative模式
                    scaled_factor = raw_scale_factor
            
            # 记录缩放系数
            output_scaling[scene_folder][ply_file] = scaled_factor
            
            # 缩放点云
            src_pcd_scaled = adjust_point_cloud_scaling(src_pcd, scaled_factor)
            
            # 保存缩放后的点云
            scaled_ply_filename = "scaled_" + ply_file
            scaled_ply_path = os.path.join(scene_folder, scaled_ply_filename)
            o3d.io.write_point_cloud(scaled_ply_path, src_pcd_scaled)
            print(f"场景 [{scene_name}] 中点云 [{ply_file}]: 缩放系数 {scaled_factor:.4f}，已保存至 {scaled_ply_path}")
            
            # 保存包围盒信息到文本文件（调试用）
            bb_info_filename = f"bounding_box_info_{os.path.splitext(ply_file)[0]}.txt"
            bb_info_path = os.path.join(scene_folder, bb_info_filename)
            with open(bb_info_path, "w") as f:
                f.write(f"参考模型: {best_match}\n")
                f.write("参考模型包围盒 (长, 宽, 高): {}\n".format(ref_dims.tolist()))
                f.write("点云包围盒 (长, 宽, 高): {}\n".format(src_dims.tolist()))
                f.write(f"缩放系数: {scaled_factor:.4f}\n")
        
        # 保存场景的参考模型包围盒信息
        scene_bbox_path = os.path.join(scene_folder, f"ref_bounding_boxes_{scene_name}.json")
        with open(scene_bbox_path, "w") as f:
            json.dump({best_match: ref_bbox_dict[scene_key]}, f, indent=4)
        print(f"场景 [{scene_name}] 参考模型包围盒信息已保存到 {scene_bbox_path}")
        
        # 保存场景的缩放系数
        scene_scaling_path = os.path.join(scene_folder, f"scaling_factors_{scene_name}.json")
        with open(scene_scaling_path, "w") as f:
            json.dump(output_scaling[scene_folder], f, indent=4)
        print(f"场景 [{scene_name}] 缩放系数已保存到 {scene_scaling_path}")
        
        # 清理非缩放文件（可选，取消注释以启用）
        # for filename in os.listdir(scene_folder):
        #     if filename.endswith('.ply') and not filename.startswith("scaled_"):
        #         file_path = os.path.join(scene_folder, filename)
        #         os.remove(file_path)
        #         print(f"已删除非缩放文件: {file_path}")

def process_folder_dual_mode(folder_path, ref_model_path, output_scaling):
    """
    使用双模式处理点云文件夹：先使用POWER模式，然后对结果应用Relative模式。
    直接覆盖原始文件。
    """
    print(f"处理文件夹: {folder_path}")
    print(f"参考模型: {ref_model_path}")
    
    # 确保文件夹存在
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"文件夹 {folder_path} 不存在或不是目录，跳过")
        return
    
    # 加载参考模型
    ref_pcd = None
    ref_extension = os.path.splitext(ref_model_path)[1].lower()
    if ref_extension == '.obj':
        ref_pcd = sample_obj_point_cloud(ref_model_path, num_points=10000)
    else:  # 假设是点云文件
        ref_pcd = o3d.io.read_point_cloud(ref_model_path)
    
    if ref_pcd is None or ref_pcd.is_empty():
        print(f"参考模型 {ref_model_path} 加载失败，跳过")
        return
    
    # 计算参考模型的包围盒
    ref_dims, ref_min, ref_max = compute_aabb_dimensions(ref_pcd)
    ref_info = {
        'model': os.path.basename(ref_model_path),
        'dims': ref_dims.tolist(),
        'min_bound': ref_min.tolist(),
        'max_bound': ref_max.tolist()
    }
    
    # 在输出字典中为该文件夹创建条目
    folder_key = os.path.abspath(folder_path)
    if folder_key not in output_scaling:
        output_scaling[folder_key] = {
            'reference': ref_info,
            'files': {}
        }
    
    # 查找所有点云文件
    ply_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.ply')]
    if not ply_files:
        print(f"文件夹 {folder_path} 中没有找到 .ply 文件，跳过")
        return
    
    # 第一阶段：使用POWER模式处理
    print("===== 第一阶段：POWER模式 =====")
    power_mode_results = {}
    
    for ply_file in ply_files:
        ply_path = os.path.join(folder_path, ply_file)
        
        # 加载点云
        pcd = o3d.io.read_point_cloud(ply_path)
        if pcd.is_empty():
            print(f"点云 {ply_path} 为空，跳过")
            continue
        
        # IQR去噪
        pcd_filtered = remove_outliers_iqr(pcd)
        pcd_dims, pcd_min, pcd_max = compute_aabb_dimensions(pcd_filtered)
        
        # 若任一维度为0，则无法计算缩放
        if np.any(pcd_dims == 0):
            print(f"点云 {ply_path} 在某个维度上尺寸为0，无法计算缩放，跳过")
            continue
        
        # 计算各维度比例，取平均值作为缩放系数
        ratios = ref_dims / pcd_dims
        raw_scale_factor = np.mean(ratios)
        
        # 应用POWER模式
        if abs(raw_scale_factor - 1.0) < 0.1:
            power_scale_factor = 1.0
        else:
            log_scale = math.log10(raw_scale_factor)
            rounded_log = round(log_scale)
            power_scale_factor = 10 ** rounded_log
        
        # 记录POWER模式的缩放系数
        power_mode_results[ply_file] = {
            'raw_factor': raw_scale_factor,
            'power_factor': power_scale_factor,
            'dims': pcd_dims.tolist()
        }
        
        # 对点云应用POWER模式缩放
        pcd_power_scaled = adjust_point_cloud_scaling(pcd, power_scale_factor)
        
        # 创建临时文件
        temp_path = os.path.join(folder_path, f"temp_{ply_file}")
        o3d.io.write_point_cloud(temp_path, pcd_power_scaled)
        print(f"POWER模式：文件 [{ply_file}] 缩放系数 {power_scale_factor:.4f}，已保存临时文件")
    
    # 第二阶段：对POWER模式结果使用Relative模式
    print("===== 第二阶段：Relative模式 =====")
    
    for ply_file in ply_files:
        if ply_file not in power_mode_results:
            continue
        
        # 加载POWER模式处理后的临时文件
        temp_path = os.path.join(folder_path, f"temp_{ply_file}")
        pcd = o3d.io.read_point_cloud(temp_path)
        
        # IQR去噪
        pcd_filtered = remove_outliers_iqr(pcd)
        pcd_dims, pcd_min, pcd_max = compute_aabb_dimensions(pcd_filtered)
        
        # 计算Relative模式的缩放系数
        ratios = ref_dims / pcd_dims
        relative_scale_factor = np.mean(ratios)
        
        # 对点云应用Relative模式缩放
        pcd_final = adjust_point_cloud_scaling(pcd, relative_scale_factor)
        
        # 保存到原始文件路径，覆盖原始文件
        original_path = os.path.join(folder_path, ply_file)
        o3d.io.write_point_cloud(original_path, pcd_final)
        
        # 删除临时文件
        os.remove(temp_path)
        
        # 计算总缩放系数（POWER * Relative）
        power_factor = power_mode_results[ply_file]['power_factor']
        total_scale_factor = power_factor * relative_scale_factor
        
        # 记录缩放信息
        output_scaling[folder_key]['files'][ply_file] = {
            'power_mode': power_mode_results[ply_file],
            'relative_factor': relative_scale_factor,
            'total_factor': total_scale_factor
        }
        
        print(f"Relative模式：文件 [{ply_file}] 相对缩放系数 {relative_scale_factor:.4f}，总缩放系数 {total_scale_factor:.4f}，已覆盖原始文件")
    
    # 保存缩放系数信息到JSON文件
    info_filename = "scaling_factors_info.json"
    info_path = os.path.join(folder_path, info_filename)
    with open(info_path, "w") as f:
        json.dump(output_scaling[folder_key], f, indent=4)
    print(f"文件夹 [{folder_path}] 的缩放信息已保存到 {info_path}")

def main():
    parser = argparse.ArgumentParser(description="对点云文件进行缩放处理，先用POWER模式，再用Relative模式")
    parser.add_argument("--ref_model", type=str, required=True, help="参考3D模型路径(.obj或.ply)")
    parser.add_argument("--folder", type=str, required=True, help="要处理的点云文件夹路径")
    args = parser.parse_args()
    
    # 检查参考模型是否存在
    if not os.path.exists(args.ref_model):
        print(f"参考模型 {args.ref_model} 不存在")
        return
    
    # 存储缩放系数信息
    scaling_info = {}
    
    # 处理文件夹
    process_folder_dual_mode(args.folder, args.ref_model, scaling_info)
    
    print("处理完成！")

if __name__ == "__main__":
    main()