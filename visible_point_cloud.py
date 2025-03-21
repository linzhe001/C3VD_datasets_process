#!/usr/bin/env python
import os
import numpy as np
import open3d as o3d
import cv2
import argparse
from glob import glob

# project_points_to_image_all projects 3D world points onto the 2D camera image plane.
# It returns the pixel coordinates, a visibility flag (1 if visible), and depth values.
def project_points_to_image_all(world_points, camera_pose, intrinsic, width, height):
    """
    投影点云到图像平面，针对不同的坐标约定做了调整
    """
    # 将3D点转换为齐次坐标
    homogeneous_points = np.hstack((world_points, np.ones((world_points.shape[0], 1))))
    
    # 将世界坐标系转到相机坐标系
    camera_points = (camera_pose @ homogeneous_points.T).T
    
    # 修改：根据Z值范围判断相机看向的方向，如果所有Z值都为正，则判断Z>0的点在前方
    z_min = camera_points[:, 2].min()
    z_max = camera_points[:, 2].max()
    
    # 由于相机坐标系下Z值都为正，我们应该使用Z>0作为前方判断条件
    in_front = camera_points[:, 2] > 0  # 修改条件
    
    # Initialize arrays for projected pixel coordinates, visibility flags, and depth values.
    pixel_coords = np.zeros((len(world_points), 2))
    visibility = np.zeros(len(world_points))
    depths = camera_points[:, 2]
 
    # Only points in front of the camera have a positive z-coordinate.
    if np.any(in_front):
        # Apply perspective projection using the intrinsic matrix.
        pixel_x = (intrinsic[0, 0] * camera_points[in_front, 0] / camera_points[in_front, 2]) + intrinsic[0, 2]
        pixel_y = (intrinsic[1, 1] * camera_points[in_front, 1] / camera_points[in_front, 2]) + intrinsic[1, 2]
        # Get indices of the points that are in front.
        front_indices = np.where(in_front)[0]
        pixel_coords[front_indices, 0] = pixel_x
        pixel_coords[front_indices, 1] = pixel_y
 
        # Check if the projected pixel positions are within the image frame's boundaries.
        in_frame = ((pixel_x >= 0) & (pixel_x < width) & (pixel_y >= 0) & (pixel_y < height))
        frame_indices = front_indices[in_frame]
        visibility[frame_indices] = 1
 
    return pixel_coords, visibility, depths
 
# handle_occlusion_raycasting_all uses Open3D's raycasting capabilities to update the visibility
# of points by checking if they are occluded (i.e., not directly visible from the camera).
def handle_occlusion_raycasting_all(world_coords, camera_pos, mesh, visibility, ray_step=0.2):
    # Create a raycasting scene and add the mesh converted to Open3D's tensor format.
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)
 
    # Process only points initially marked as visible.
    visible_indices = np.where(visibility == 1)[0]
    if len(visible_indices) == 0:
        return visibility
 
    # Prepare an array of rays. Each ray consists of its origin (camera position) and a normalized direction.
    rays = np.zeros((len(visible_indices), 6))
    rays[:, 0:3] = camera_pos
    ray_dirs = world_coords[visible_indices] - camera_pos
    ray_lens = np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    ray_dirs = ray_dirs / ray_lens
    rays[:, 3:6] = ray_dirs
 
    # Convert the rays array to a tensor required by the Open3D raycasting API.
    rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays_t)
    t_hit = ans['t_hit'].numpy()
 
    # Update the visibility flag for points that are occluded.
    updated_visibility = visibility.copy()
    for i, (idx, t, ray_len) in enumerate(zip(visible_indices, t_hit, ray_lens.flatten())):
        if abs(t - ray_len) > 0.5:  # 改为更大的阈值
            updated_visibility[idx] = 0
 
    return updated_visibility

def visualize_cameras_and_model(mesh, poses, output_path):
    # 创建可视化几何体
    vis_geometries = [mesh]
    
    # 为每个相机位姿创建一个小的坐标系可视化
    for i, pose in enumerate(poses[:10]):  # 仅可视化前10个相机
        # 创建相机坐标系
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        camera_frame.transform(pose)
        vis_geometries.append(camera_frame)
        
        # 创建指示相机位置的球体
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(pose[:3, 3])
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
        vis_geometries.append(sphere)
    
    # 保存为可视化文件
    o3d.io.write_triangle_mesh(output_path, mesh)  # 保存模型
    
    return vis_geometries

def process_scene(scene_path, output_root, intrinsic_params, distortion_params, depth_scale=0.001, depth_threshold=0.1):
    """
    处理单个场景文件夹：加载 mesh、pose.txt 及所有深度图，并生成每帧的可见点云
    """
    scene_name = os.path.basename(os.path.normpath(scene_path))
    print(f"正在处理场景: {scene_name}")
    
    # 查找 mesh 文件（假设后缀为 .obj）
    mesh_files = [f for f in os.listdir(scene_path) if f.endswith(".obj")]
    if not mesh_files:
        print(f"场景 {scene_name} 未找到 mesh 文件，跳过。")
        return
    mesh_file_path = os.path.join(scene_path, mesh_files[0])
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    mesh.compute_vertex_normals()
    
    # 从 mesh 上均匀采样候选点（方法 B）
    pcd_mesh = mesh.sample_points_uniformly(number_of_points=100000)
    world_points = np.asarray(pcd_mesh.points)
    
    # 读取 pose.txt 文件
    pose_file = os.path.join(scene_path, "pose.txt")
    if not os.path.exists(pose_file):
        print(f"场景 {scene_name} 未找到 pose.txt，跳过。")
        return
        
    # 正确读取逗号分隔的位姿数据
    poses_raw = np.loadtxt(pose_file, delimiter=",")
    if poses_raw.ndim == 1 and poses_raw.size == 16:
        pose_raw_matrix = poses_raw.reshape((4,4))
        # 重新格式化为标准的变换矩阵
        standard_pose = np.eye(4)
        standard_pose[:3, :3] = pose_raw_matrix[:3, :3]  # 复制旋转部分
        standard_pose[:3, 3] = pose_raw_matrix[3, :3]    # 平移向量从最后一行移到最后一列
        poses = [standard_pose]
    elif poses_raw.ndim == 2 and poses_raw.shape[1] == 16:
        poses = []
        for i in range(poses_raw.shape[0]):
            pose_raw_matrix = poses_raw[i].reshape((4,4))
            # 重新格式化为标准的变换矩阵
            standard_pose = np.eye(4)
            standard_pose[:3, :3] = pose_raw_matrix[:3, :3]  # 复制旋转部分
            standard_pose[:3, 3] = pose_raw_matrix[3, :3]    # 平移向量从最后一行移到最后一列
            poses.append(standard_pose)
    else:
        print(f"场景 {scene_name} pose.txt 格式不符合预期，跳过。")
        return
    
    # 查找所有的深度图文件（假设命名模式 *_depth.tiff）
    depth_files = sorted(glob(os.path.join(scene_path, "*_depth.tiff")))
    if len(depth_files) == 0:
        print(f"场景 {scene_name} 未找到深度图文件，跳过。")
        return
    
    num_frames = min(len(depth_files), len(poses))
    print(f"场景 {scene_name} 共找到 {num_frames} 帧深度图与对应 pose。")
    
    # 创建输出目录：output_root/scene_name/
    scene_output_dir = os.path.join(output_root, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)
    
    # 构造内参矩阵和畸变系数数组
    fx, fy, cx, cy = intrinsic_params
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
    k1, k2, k3, k4 = distortion_params
    D = np.array([k1, k2, k3, k4])
    
    # 固定深度比例因子为0.001（毫米转米）
    depth_scale = 0.001
    print(f"使用固定深度比例: {depth_scale}（毫米转米）")
    
    for i in range(num_frames):  # 处理所有帧
        depth_path = depth_files[i]
        print(f"  处理帧 {i}: {os.path.basename(depth_path)}")
        # 加载深度图
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            print(f"    无法读取深度图 {depth_path}，跳过此帧。")
            continue
        
        height, width = depth_img.shape
        
        # 执行去畸变处理
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (width, height), cv2.CV_32FC1
        )
        undistorted_depth = cv2.remap(depth_img, map1, map2, cv2.INTER_LINEAR)
        
        # 读取第 i 帧对应的 pose
        frame_pose = poses[i]
        
        # 正确获取相机位置 - 位姿矩阵的最后一列前三行是平移向量
        camera_position = frame_pose[:3, 3]  # 从最后一列读取平移向量
        
        # 修改：不翻转Z轴，因为点Z值已经为正，这表明相机坐标系中Z轴指向相机前方
        camera_pose = np.linalg.inv(frame_pose)  # 取逆获得从世界到相机的变换
        
        # 检查是否有NaN值
        if np.isnan(camera_pose).any() or np.isnan(frame_pose).any():
            print("警告: 位姿矩阵中包含NaN值!")
            continue
        
        # 检查矩阵值范围
        if np.max(np.abs(frame_pose)) > 1000 or np.max(np.abs(camera_pose)) > 1000:
            print("警告: 位姿矩阵中包含异常大的值!")
        
        # 投影模型采样点到图像平面
        pixel_coords, visibility, depths = project_points_to_image_all(
            world_points, camera_pose, K, width, height
        )
        
        # 直接保存所有投影到图像范围内的点，不做任何额外过滤
        initial_visible_points = world_points[visibility == 1]
        if len(initial_visible_points) > 0:
            pcd_visible = o3d.geometry.PointCloud()
            pcd_visible.points = o3d.utility.Vector3dVector(initial_visible_points)
            output_filename = os.path.join(scene_output_dir, f"frame_{i:04d}_projection_only.ply")
            o3d.io.write_point_cloud(output_filename, pcd_visible)
            print(f"    保存投影点云 ({len(initial_visible_points)}点): {output_filename}")
        else:
            print("    警告: 投影后无可见点!")
        
        # 如果经过 project_points_to_image_all 函数后，可见点数量已经为0，则直接跳过后续处理
        visible_count = np.sum(visibility)
        if visible_count == 0:
            print("    警告: 投影后无可见点，跳过射线检测")
            # 创建一个空点云并保存
            pcd_visible = o3d.geometry.PointCloud()
            output_filename = os.path.join(scene_output_dir, f"frame_{i:04d}_visible.ply")
            o3d.io.write_point_cloud(output_filename, pcd_visible)
            print(f"    保存可见点云: {output_filename}")
            continue
        
        # 更新射线检测的相机位置参数
        camera_pos = frame_pose[:3, 3]  # 从变换矩阵正确位置获取相机位置
        visibility = handle_occlusion_raycasting_all(world_points, camera_pos, mesh, visibility, ray_step=0.5)
        
        # 筛选可见点
        visible_points = world_points[visibility == 1]
        
        # 构造点云并保存为 PLY 文件
        pcd_visible = o3d.geometry.PointCloud()
        pcd_visible.points = o3d.utility.Vector3dVector(visible_points)
        output_filename = os.path.join(scene_output_dir, f"frame_{i:04d}_visible.ply")
        o3d.io.write_point_cloud(output_filename, pcd_visible)
        print(f"    保存可见点云: {output_filename}")

    # 在读取完mesh和poses后调用
    visualization_path = os.path.join(scene_output_dir, "cameras_and_model.ply")
    vis_geometries = visualize_cameras_and_model(mesh, poses[:10], visualization_path)
    print(f"相机和模型可视化已保存到: {visualization_path}")
    
    # 添加: 删除中间文件，只保留最终可见点云
    print("删除中间文件，只保留最终可见点云...")
    # 删除projection_only文件
    projection_files = glob(os.path.join(scene_output_dir, "*_projection_only.ply"))
    for file in projection_files:
        try:
            os.remove(file)
            print(f"  已删除: {os.path.basename(file)}")
        except Exception as e:
            print(f"  无法删除 {file}: {e}")
    
    # 删除cameras_and_model文件
    cameras_model_file = os.path.join(scene_output_dir, "cameras_and_model.ply")
    if os.path.exists(cameras_model_file):
        try:
            os.remove(cameras_model_file)
            print(f"  已删除: cameras_and_model.ply")
        except Exception as e:
            print(f"  无法删除 cameras_and_model.ply: {e}")

def main():
    parser = argparse.ArgumentParser(description="生成每一帧深度图对应的可见点云，处理所有场景")
    parser.add_argument("--input", type=str, required=True, help="C3VD 文件夹路径，每个子文件夹表示一个场景")
    parser.add_argument("--output", type=str, required=True, help="可见点云输出根目录")
    args = parser.parse_args()
    
    input_root = args.input
    output_root = args.output
    
    # 定义相机内参和畸变系数（参见 creat_check_pcd.py）
    intrinsic_params = (767.3861511125845, 767.5058656118406, 679.054265997005, 543.646891684636)
    distortion_params = (-0.18867185058223412, -0.003927337093919806, 0.030524814153620117, -0.012756926010904904)
    
    # 获取输入目录下的所有场景文件夹
    scene_dirs = [os.path.join(input_root, d) for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    if len(scene_dirs) == 0:
        print("未在输入路径下找到任何场景文件夹。")
        return
    
    for scene_path in scene_dirs:
        process_scene(scene_path, output_root, intrinsic_params, distortion_params)
    
    print("所有场景的可见点云生成任务已完成！")

if __name__ == "__main__":
    main()