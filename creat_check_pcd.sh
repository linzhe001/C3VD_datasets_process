#! /usr/bin/bash

# 数据文件夹的绝对路径（用户可以根据需要修改）
DATA_ROOT="/mnt/c/Users/asus/Downloads/C3VD_datasets"

# 设置输入和输出的相对路径
INPUT_PATH="$DATA_ROOT/C3VD"          # 深度图文件的根目录
OUTPUT_PATH="$DATA_ROOT/C3VD_ply"     # 点云文件的输出目录

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_PATH"

# 运行点云生成脚本
python3 creat_check_pcd.py --input "$INPUT_PATH" \
                           --output "$OUTPUT_PATH" \
                           --voxel_size 0.005 \
                           --max_points 30000

echo "深度图到点云转换任务已完成！"
#EOF
