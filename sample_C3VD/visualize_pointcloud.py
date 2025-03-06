import open3d as o3d
import numpy as np
import colorsys
import os

def apply_transform(pcd, transform_matrix):
    """
    对点云应用4x4变换矩阵
    参数:
        pcd: Open3D点云对象
        transform_matrix: 4x4变换矩阵，包含旋转和平移信息
    返回:
        变换后的点云对象
    """
    return pcd.transform(transform_matrix)

def visualize_manual_pointcloud(file_path, manual_color=[1, 0, 0], transform_matrix=None):
    """
    显示单个点云，并为所有点统一设置一个颜色
    参数:
        file_path: 点云文件路径（例如 .pcd 或 .ply 文件）
        manual_color: 指定颜色 [R, G, B]（取值范围为 0-1），例如 [1, 0, 0] 表示红色
        transform_matrix: 可选的4x4变换矩阵，用于在显示前变换点云
    """
    pcd = o3d.io.read_point_cloud(file_path)
    
    # 应用变换矩阵（如果提供）
    if transform_matrix is not None:
        pcd = apply_transform(pcd, transform_matrix)
        
    points = np.asarray(pcd.points)
    colors = np.tile(manual_color, (len(points), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, coordinate_frame])

def visualize_multiple_pointclouds(file_paths, transform_matrices=None):
    """
    加载多个点云，并为每个点云分配固定顺序的颜色
    参数：
        file_paths: 点云文件路径的列表（例如一组 .pcd 或 .ply 文件路径）
        transform_matrices: 可选的变换矩阵列表，与file_paths一一对应，用于在显示前变换每个点云
    """
    if len(file_paths) == 0:
        print("未找到任何点云文件")
        return

    # 预定义固定顺序的颜色列表 - RGB值 (取值范围为 0-1)
    FIXED_COLORS = [
        [1.0, 0.0, 0.0],     # 红色
        [0.0, 1.0, 0.0],     # 绿色
        [0.0, 0.0, 1.0],     # 蓝色
        [1.0, 1.0, 0.0],     # 黄色
        [1.0, 0.0, 1.0],     # 洋红色
        [0.0, 1.0, 1.0],     # 青色
        [1.0, 0.5, 0.0],     # 橙色
        [0.5, 0.0, 1.0],     # 紫色
        [0.0, 0.5, 0.0],     # 深绿色
        [0.5, 0.5, 0.5],     # 灰色
        [1.0, 0.75, 0.8],    # 粉色
        [0.0, 0.0, 0.5],     # 深蓝色
    ]

    pcd_list = []
    for idx, file_path in enumerate(file_paths):
        # 使用固定颜色列表，循环使用
        color_idx = idx % len(FIXED_COLORS)
        color = FIXED_COLORS[color_idx]

        # 读取点云数据
        pcd = o3d.io.read_point_cloud(file_path)
        # 应用变换矩阵（如果提
        if transform_matrices is not None and idx < len(transform_matrices) and transform_matrices[idx] is not None:
            transform_matrices_i = np.linalg.inv(transform_matrices[idx])
            pcd = apply_transform(pcd, transform_matrices_i)
            print(f"已应用变换矩阵到 {os.path.basename(file_path)}")
            
        # 赋予颜色
        points = np.asarray(pcd.points)
        colors = np.tile(color, (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_list.append(pcd)
        
        print(f"已加载: {os.path.basename(file_path)}，分配的固定颜色: {color}")
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(pcd_list + [coordinate_frame])

if __name__ == "__main__":
    # 示例1: 单个点云文件
    # 此文件将被统一着色为红色 [1, 0, 0]
    single_file = r"path/to/your/single_point_cloud.ply"  # 请修改为实际路径
    multiple_files = [
         r"C:\Users\asus\Downloads\fused_C3VD\C3VD\C3VD_ply\cecum_t1_a\merged_0020_0024_5frames.ply",   
         r"C:\Users\asus\Downloads\fused_C3VD\C3VD\C3VD_ply\cecum_t1_a\merged_0110_0114_5frames.ply",  
         r"C:\Users\asus\Downloads\fused_C3VD\C3VD\C3VD_ply\cecum_t1_a\merged_0265_0269_5frames.ply",  
    ]
    
    # 示例变换矩阵 - 形如用户提供的JSON文件中的矩阵
    transform_matrix1 = np.array([
                    [
                        0.948122,
                        0.310597,
                        -0.0677803,
                        55.0535
                    ],
                    [
                        -0.314651,
                        0.947271,
                        -0.0606028,
                        39.4183
                    ],
                    [
                        0.0453833,
                        0.078786,
                        0.995858,
                        -101.707
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0
                    ]

    ])

    transform_matrix2 = np.array([
                    [
                        0.948115,
                        0.310742,
                        -0.0672112,
                        49.7588
                    ],
                    [
                        -0.314766,
                        0.94721,
                        -0.0609528,
                        44.7614
                    ],
                    [
                        0.0447226,
                        0.0789461,
                        0.995875,
                        -86.5937
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0
                    ]

    ])

    transform_matrix3 = np.array([
                    [
                        0.951373,
                        0.307609,
                        -0.0163203,
                        50.2705
                    ],
                    [
                        -0.306228,
                        0.938702,
                        -0.158311,
                        53.7402
                    ],
                    [
                        -0.0333781,
                        0.155611,
                        0.987255,
                        -68.5532
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0
                    ]

    ])
    
    # 可以为每个点云指定不同的变换矩阵
    transform_matrices = [transform_matrix1, transform_matrix2, transform_matrix3]  # 第一个点云使用变换，第二个不变换
    
    visualize_multiple_pointclouds(multiple_files)
    
    # 使用变换矩阵的例子:
    visualize_multiple_pointclouds(multiple_files, transform_matrices=transform_matrices)