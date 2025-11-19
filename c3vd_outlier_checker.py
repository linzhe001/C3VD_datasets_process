import os
import numpy as np
import open3d as o3d
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class PointCloudAnomalyDetector:
    def __init__(self, main_folder, ref_folder_name, src_folder_name):
        """
        Initialize point cloud anomaly detector

        Parameters:
            main_folder: Main folder path
            ref_folder_name: Reference point cloud folder name
            src_folder_name: Source point cloud folder name
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
        """Extract scene ID and file ID from filename"""
        import re
        basename = os.path.basename(filename)

        # Extract scene name
        scene_name = os.path.basename(os.path.dirname(filename))

        # Handle different types of files
        if "coverage_mesh.ply" in basename:
            # For reference point cloud, use special identifier
            return f"{scene_name}_coverage"
        else:
            # For source point cloud, extract numeric ID
            match = re.search(r'(\d+)', basename)
            file_id = match.group(1) if match else "unknown"
            return f"{scene_name}_{file_id}"

    def load_point_cloud(self, file_path):
        """Load PLY point cloud file"""
        try:
            pcd = o3d.io.read_point_cloud(str(file_path))
            points = np.asarray(pcd.points)
            return points, True
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            self.anomalies.append(f"File {file_path} cannot be loaded: {e}")
            return None, False

    def analyze_point_cloud(self, points, file_path):
        """Analyze anomalies in a single point cloud"""
        if points is None or len(points) == 0:
            self.anomalies.append(f"File {file_path} anomaly: Point cloud is empty or failed to load")
            return {
                'file': str(file_path),
                'status': 'Error',
                'points_count': 0,
                'zero_points': 0,
                'outliers': 0,
                'error_msg': 'Point cloud is empty or failed to load',
                'is_anomalous': True
            }


        # Basic statistics
        point_count = len(points)

        # Check zero points or near-zero points
        zero_mask = np.all(np.abs(points) < 1e-6, axis=1)
        zero_points = np.sum(zero_mask)

        # Use IQR (Interquartile Range) method to detect outliers
        q1 = np.percentile(points, 25, axis=0)
        q3 = np.percentile(points, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Check if there are points exceeding the bounds in each dimension
        outlier_mask = np.zeros(point_count, dtype=bool)
        for dim in range(3):  # x, y, z dimensions
            dim_outliers = (points[:, dim] < lower_bound[dim]) | (points[:, dim] > upper_bound[dim])
            outlier_mask = outlier_mask | dim_outliers

        outlier_count = np.sum(outlier_mask)

        # Is the overall point cloud anomalous (if outlier ratio exceeds 20% or has many zero points)
        is_anomalous = (outlier_count / point_count > 0.2) or (zero_points > point_count * 0.1)

        if is_anomalous:
            reason = []
            if outlier_count / point_count > 0.2:
                reason.append(f"Outlier ratio too high: {outlier_count/point_count*100:.2f}%")
            if zero_points > point_count * 0.1:
                reason.append(f"Zero/near-zero point ratio too high: {zero_points/point_count*100:.2f}%")

            self.anomalies.append(f"File {file_path} anomaly: {', '.join(reason)}")

        return {
            'file': str(file_path),
            'status': 'Anomalous' if is_anomalous else 'Normal',
            'points_count': point_count,
            'zero_points': int(zero_points),
            'zero_percent': float(zero_points / point_count * 100),
            'outliers': int(outlier_count),
            'outlier_percent': float(outlier_count / point_count * 100),
            'is_anomalous': is_anomalous
        }

    def process_files(self, folder, folder_type):
        """Process all PLY files in the specified folder"""
        all_scenes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

        for scene in all_scenes:
            scene_path = os.path.join(folder, scene)
            for file in tqdm(os.listdir(scene_path), desc=f"Processing {folder_type} - {scene}"):
                if file.endswith('.ply'):
                    file_path = os.path.join(scene_path, file)
                    points, success = self.load_point_cloud(file_path)
                    if success:
                        result = self.analyze_point_cloud(points, file_path)
                        file_id = self.extract_id(file)
                        if file_id:
                            self.results[folder_type][f"{scene}_{file_id}"] = result

    def downsample_point_cloud(self, points, target_count):
        """Downsample point cloud to specified point count"""
        if points is None or len(points) <= target_count:
            return points

        # Random sampling to target point count
        indices = np.random.choice(len(points), target_count, replace=False)
        return points[indices]

    def compare_pairs(self):
        """Compare reference point cloud and source point cloud pairs, all source point clouds in each scene correspond to one reference point cloud"""
        # Organize data by scene grouping
        scenes = {}

        # Extract scene names
        for key in self.results['ref'].keys():
            scene_name = key.split('_')[0]  # Assume format is "scene_name_ID"
            if scene_name not in scenes:
                scenes[scene_name] = {'ref': None, 'src': []}
        
        for key in self.results['src'].keys():
            scene_name = key.split('_')[0]
            if scene_name not in scenes:
                scenes[scene_name] = {'ref': None, 'src': []}

        # Assign reference point clouds and source point clouds
        for scene_name in scenes:
            # Find reference point cloud for this scene
            ref_key = None
            for key in self.results['ref'].keys():
                if key.startswith(f"{scene_name}_"):
                    if "coverage" in self.results['ref'][key]['file']:  # Assume reference point cloud contains "coverage" keyword
                        ref_key = key
                        break

            if ref_key:
                scenes[scene_name]['ref'] = self.results['ref'][ref_key]

            # Find all source point clouds for this scene
            for key in self.results['src'].keys():
                if key.startswith(f"{scene_name}_"):
                    scenes[scene_name]['src'].append(self.results['src'][key])

        # Load original point cloud data cache
        original_point_clouds = {}

        # Check each scene
        for scene_name, data in scenes.items():
            # Check if reference point cloud exists
            if data['ref'] is None:
                self.anomalies.append(f"Scene {scene_name} is missing reference point cloud")
                continue

            # Check if source point clouds exist
            if not data['src']:
                self.anomalies.append(f"Scene {scene_name} has no source point clouds")
                continue

            # Load reference point cloud (if not yet loaded)
            ref_file = data['ref']['file']
            if ref_file not in original_point_clouds:
                ref_points, success = self.load_point_cloud(ref_file)
                if not success:
                    continue
                original_point_clouds[ref_file] = ref_points

            # Compare reference point cloud and source point clouds
            ref_data = data['ref']
            for src_data in data['src']:
                src_file = os.path.basename(src_data['file'])

                # Load source point cloud (if not yet loaded)
                if src_data['file'] not in original_point_clouds:
                    src_points, success = self.load_point_cloud(src_data['file'])
                    if not success:
                        continue
                    original_point_clouds[src_data['file']] = src_points

                # Downsample reference point cloud to same point count as source point cloud
                src_points = original_point_clouds[src_data['file']]
                ref_points = original_point_clouds[ref_file]
                downsampled_ref_points = self.downsample_point_cloud(ref_points, len(src_points))

                # More comparisons based on geometric features can be added here, instead of just comparing point counts
                # For example, compare distribution, density, principal directions, etc.

                # Only add to anomaly list when there are other anomalous conditions
                # Since we've already downsampled to match point counts, no longer use point count difference to determine anomalies

    def generate_simple_report(self, output_file):
        """Generate simple anomaly report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Point Cloud Dataset Anomaly Detection Report\n")
            f.write("=" * 50 + "\n\n")

            if not self.anomalies:
                f.write("No anomalous point clouds found.\n")
            else:
                f.write(f"Found {len(self.anomalies)} anomalies:\n\n")
                for i, anomaly in enumerate(self.anomalies, 1):
                    f.write(f"{i}. {anomaly}\n")

        print(f"Simple report has been saved to {output_file}")

    def run(self, output_file='./point_cloud_anomaly_report.txt'):
        """Run complete analysis workflow"""
        print("Starting analysis of reference point clouds...")
        self.process_files(self.ref_folder, 'ref')

        print("Starting analysis of source point clouds...")
        self.process_files(self.src_folder, 'src')

        print("Comparing point cloud pairs...")
        self.compare_pairs()

        print("Generating report...")
        self.generate_simple_report(output_file)

if __name__ == "__main__":
    # Usage example
    import argparse

    parser = argparse.ArgumentParser(description='Point cloud dataset anomaly detection tool')
    parser.add_argument('--main_folder', type=str, required=True, help='Main folder path')
    parser.add_argument('--ref_folder', type=str, default='reference', help='Reference point cloud folder name')
    parser.add_argument('--src_folder', type=str, default='source', help='Source point cloud folder name')
    parser.add_argument('--output', type=str, default='./point_cloud_anomaly_report.txt', help='Output report path')
    
    args = parser.parse_args()
    
    detector = PointCloudAnomalyDetector(
        args.main_folder,
        args.ref_folder,
        args.src_folder
    )
    detector.run(args.output)