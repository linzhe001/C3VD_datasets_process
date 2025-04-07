#!/usr/bin/bash
#! /usr/bin/bash

#$ -l tmem=28G
#$ -l h_vmem=28G            # 增加内存限制
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

# 数据文件夹的绝对路径（用户可以根据需要修改）
DATA_ROOT="/SAN/medic/MRpcr/C3VD_datasets"


# 输入文件夹：包含所有场景文件夹的路径（包含.obj和pose.txt）
INPUT_PATH="$DATA_ROOT/C3VD"

# 点云源文件夹：包含所有场景的点云文件
POINT_CLOUD_SOURCE="$DATA_ROOT/C3VD_ply_source"

# 输出目录：保存所有场景的可见点云的根目录
OUTPUT_DIR="$DATA_ROOT/visible_point_cloud_ply_depth"

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 运行可见点云生成脚本，将输入、点云源和输出目录作为参数传入
python3 /SAN/medic/MRpcr/C3VD_datasets_process/visible_point_cloud.py --input "$INPUT_PATH" --point_cloud_source "$POINT_CLOUD_SOURCE" --output "$OUTPUT_DIR"

echo "所有场景的可见点云生成任务已完成！"