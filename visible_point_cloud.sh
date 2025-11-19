#!/usr/bin/bash
#! /usr/bin/bash

#$ -l tmem=28G
#$ -l h_vmem=28G            # Increase memory limit
#$ -l h_rt=7200              # Runtime limit (in seconds or hours:minutes:seconds format)
#$ -l gpu=true
#$ -pe gpu 1
#$ -N visible_pcd
#$ -o /SAN/medic/MRpcr/logs/visible_pcd_output.log
#$ -e /SAN/medic/MRpcr/logs/visible_pcd_error.log
#$ -wd /SAN/medic/MRpcr

# Activate Conda environment
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pcd

# Absolute path to data folder (users can modify as needed)
DATA_ROOT="/SAN/medic/MRpcr/C3VD_datasets"


# Input folder: path containing all scene folders (including .obj and pose.txt)
INPUT_PATH="$DATA_ROOT/Original_C3VD"

# Point cloud source folder: contains point cloud files for all scenes
POINT_CLOUD_SOURCE="$DATA_ROOT/C3VD_Raycasting10K_source"

# Output directory: root directory to save visible point clouds for all scenes
OUTPUT_DIR="$DATA_ROOT/C3VD_Raycasting10K_target"

# Create output directory (if it doesn't exist)
mkdir -p "$OUTPUT_DIR"

# Run visible point cloud generation script, passing input, point cloud source and output directories as arguments
python3 /SAN/medic/MRpcr/C3VD_datasets_process/visible_point_cloud.py --input "$INPUT_PATH" --point_cloud_source "$POINT_CLOUD_SOURCE" --output "$OUTPUT_DIR"

echo "Visible point cloud generation task completed for all scenes!"