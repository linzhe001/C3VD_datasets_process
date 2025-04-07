import os
import numpy as np
import open3d as o3d
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class PointCloudAnomalyDetector:
    def __init__(self, main_folder, ref_folder_name, src_folder_name):
        """
        初始化点云异常检测器
        
        参数:
            main_folder: 主文件夹路径
            ref_folder_name: 参考点云文件夹名
            src_folder_name: 源点云文件夹名
        """
        self.main_folder = Path(main_folder)
        self.ref_folder = self.main_folder / ref_folder_name
        self.src_folder = self.main_folder / src_folder_name
        self.results = {
            'ref': {},
            'src': {}
        }
        self.pair_results = []
        self.anomalies = []
        
    def extract_id(self, filename):
        """从文件名中提取场景ID和文件ID"""
        import re
        basename = os.path.basename(filename)
        
        # 提取场景名
        scene_name = os.path.basename(os.path.dirname(filename))
        
        # 处理不同类型的文件
        if "coverage_mesh.ply" in basename:
            # 对于参考点云，使用特殊标识
            return f"{scene_name}_coverage"
        else:
            # 对于源点云，提取数字ID
            match = re.search(r'(\d+)', basename)
            file_id = match.group(1) if match else "unknown"
            return f"{scene_name}_{file_id}"
        
    def load_point_cloud(self, file_path):
        """加载PLY点云文件"""
        try:
            pcd = o3d.io.read_point_cloud(str(file_path))
            points = np.asarray(pcd.points)
            return points, True
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
            self.anomalies.append(f"文件 {file_path} 无法加载: {e}")
            return None, False
            
    def analyze_point_cloud(self, points, file_path):
        """分析单个点云的异常情况"""
        if points is None or len(points) == 0:
            self.anomalies.append(f"文件 {file_path} 异常: 点云为空或加载失败")
            return {
                'file': str(file_path),
                'status': '错误',
                'points_count': 0,
                'zero_points': 0,
                'outliers': 0,
                'error_msg': '点云为空或加载失败',
                'is_anomalous': True
            }
            
        # 基本统计信息
        point_count = len(points)
        
        # 检查零点或接近零点
        zero_mask = np.all(np.abs(points) < 1e-6, axis=1)
        zero_points = np.sum(zero_mask)
        
        # 使用IQR(四分位距)方法检测异常值
        q1 = np.percentile(points, 25, axis=0)
        q3 = np.percentile(points, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 检查各个维度上是否有超出边界的点
        outlier_mask = np.zeros(point_count, dtype=bool)
        for dim in range(3):  # x, y, z维度
            dim_outliers = (points[:, dim] < lower_bound[dim]) | (points[:, dim] > upper_bound[dim])
            outlier_mask = outlier_mask | dim_outliers
            
        outlier_count = np.sum(outlier_mask)
        
        # 点云整体是否异常(如果异常点占比超过20%或有大量零点)
        is_anomalous = (outlier_count / point_count > 0.2) or (zero_points > point_count * 0.1)
        
        if is_anomalous:
            reason = []
            if outlier_count / point_count > 0.2:
                reason.append(f"异常点比例过高: {outlier_count/point_count*100:.2f}%")
            if zero_points > point_count * 0.1:
                reason.append(f"零点/接近零点比例过高: {zero_points/point_count*100:.2f}%")
            
            self.anomalies.append(f"文件 {file_path} 异常: {', '.join(reason)}")
        
        return {
            'file': str(file_path),
            'status': '异常' if is_anomalous else '正常',
            'points_count': point_count,
            'zero_points': int(zero_points),
            'zero_percent': float(zero_points / point_count * 100),
            'outliers': int(outlier_count),
            'outlier_percent': float(outlier_count / point_count * 100),
            'is_anomalous': is_anomalous
        }
    
    def process_files(self, folder, folder_type):
        """处理指定文件夹中的所有PLY文件"""
        all_scenes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        
        for scene in all_scenes:
            scene_path = os.path.join(folder, scene)
            for file in tqdm(os.listdir(scene_path), desc=f"处理 {folder_type} - {scene}"):
                if file.endswith('.ply'):
                    file_path = os.path.join(scene_path, file)
                    points, success = self.load_point_cloud(file_path)
                    if success:
                        result = self.analyze_point_cloud(points, file_path)
                        file_id = self.extract_id(file)
                        if file_id:
                            self.results[folder_type][f"{scene}_{file_id}"] = result
    
    def downsample_point_cloud(self, points, target_count):
        """将点云降采样到指定点数"""
        if points is None or len(points) <= target_count:
            return points
            
        # 随机采样到目标点数
        indices = np.random.choice(len(points), target_count, replace=False)
        return points[indices]
        
    def compare_pairs(self):
        """比较参考点云和源点云对，每个场景中所有源点云对应一个参考点云"""
        # 按场景分组整理数据
        scenes = {}
        
        # 提取场景名称
        for key in self.results['ref'].keys():
            scene_name = key.split('_')[0]  # 假设格式为"场景名_ID"
            if scene_name not in scenes:
                scenes[scene_name] = {'ref': None, 'src': []}
        
        for key in self.results['src'].keys():
            scene_name = key.split('_')[0]
            if scene_name not in scenes:
                scenes[scene_name] = {'ref': None, 'src': []}
        
        # 分配参考点云和源点云
        for scene_name in scenes:
            # 查找该场景的参考点云
            ref_key = None
            for key in self.results['ref'].keys():
                if key.startswith(f"{scene_name}_"):
                    if "coverage" in self.results['ref'][key]['file']:  # 假设参考点云包含"coverage"关键字
                        ref_key = key
                        break
            
            if ref_key:
                scenes[scene_name]['ref'] = self.results['ref'][ref_key]
            
            # 查找该场景的所有源点云
            for key in self.results['src'].keys():
                if key.startswith(f"{scene_name}_"):
                    scenes[scene_name]['src'].append(self.results['src'][key])
        
        # 加载原始点云数据缓存
        original_point_clouds = {}
        
        # 检查每个场景
        for scene_name, data in scenes.items():
            # 检查是否有参考点云
            if data['ref'] is None:
                self.anomalies.append(f"场景 {scene_name} 缺少参考点云")
                continue
            
            # 检查是否有源点云
            if not data['src']:
                self.anomalies.append(f"场景 {scene_name} 没有源点云")
                continue
            
            # 加载参考点云（如果尚未加载）
            ref_file = data['ref']['file']
            if ref_file not in original_point_clouds:
                ref_points, success = self.load_point_cloud(ref_file)
                if not success:
                    continue
                original_point_clouds[ref_file] = ref_points
            
            # 对比参考点云和源点云
            ref_data = data['ref']
            for src_data in data['src']:
                src_file = os.path.basename(src_data['file'])
                
                # 加载源点云（如果尚未加载）
                if src_data['file'] not in original_point_clouds:
                    src_points, success = self.load_point_cloud(src_data['file'])
                    if not success:
                        continue
                    original_point_clouds[src_data['file']] = src_points
                
                # 将参考点云降采样到与源点云相同的点数
                src_points = original_point_clouds[src_data['file']]
                ref_points = original_point_clouds[ref_file]
                downsampled_ref_points = self.downsample_point_cloud(ref_points, len(src_points))
                
                # 这里可以添加更多基于几何特性的比较，而不是仅仅比较点数
                # 例如，可以比较点云的分布、密度、主方向等
                
                # 仅在有其他异常情况时添加到异常列表
                # 由于我们已经降采样匹配了点数，不再使用点数差异判断异常
    
    def generate_simple_report(self, output_file):
        """生成简单的异常报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("点云数据集异常检测报告\n")
            f.write("=" * 50 + "\n\n")
            
            if not self.anomalies:
                f.write("未发现异常点云。\n")
            else:
                f.write(f"发现 {len(self.anomalies)} 个异常:\n\n")
                for i, anomaly in enumerate(self.anomalies, 1):
                    f.write(f"{i}. {anomaly}\n")
            
        print(f"简单报告已保存到 {output_file}")
            
    def run(self, output_file='./点云异常报告.txt'):
        """运行完整的分析流程"""
        print("开始分析参考点云...")
        self.process_files(self.ref_folder, 'ref')
        
        print("开始分析源点云...")
        self.process_files(self.src_folder, 'src')
        
        print("比较点云对...")
        self.compare_pairs()
        
        print("生成报告...")
        self.generate_simple_report(output_file)

if __name__ == "__main__":
    # 使用示例
    import argparse
    
    parser = argparse.ArgumentParser(description='点云数据集异常检测工具')
    parser.add_argument('--main_folder', type=str, required=True, help='主文件夹路径')
    parser.add_argument('--ref_folder', type=str, default='reference', help='参考点云文件夹名')
    parser.add_argument('--src_folder', type=str, default='source', help='源点云文件夹名')
    parser.add_argument('--output', type=str, default='./点云异常报告.txt', help='输出报告路径')
    
    args = parser.parse_args()
    
    detector = PointCloudAnomalyDetector(
        args.main_folder,
        args.ref_folder,
        args.src_folder
    )
    detector.run(args.output)