#!/usr/bin/bash

# 数据文件夹的绝对路径（用户可以根据需要修改）
DATA_ROOT="/mnt/c/Users/asus/Downloads/C3VD_datasets"

# 输入文件夹：包含所有场景文件夹的路径（例如 DATA_ROOT/C3VD 下每个子文件夹为一个场景）
INPUT_PATH="$DATA_ROOT/C3VD"

# 输出目录：保存所有场景的可见点云的根目录
OUTPUT_DIR="$DATA_ROOT/visible_point_cloud_ply"

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 运行可见点云生成脚本，将输入和输出目录作为参数传入
python3 visible_point_cloud.py --input "$INPUT_PATH" --output "$OUTPUT_DIR"

echo "所有场景的可见点云生成任务已完成！"