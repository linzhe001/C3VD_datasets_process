#!/bin/bash

# 数据文件夹的绝对路径（用户可以根据需要修改）
DATA_ROOT="/mnt/c/Users/asus/Downloads/C3VD_datasets"

# 直接在脚本中设置参数
DEPTH_DIR="$DATA_ROOT/C3VD"  # 深度图目录
OUTPUT_DIR="$DATA_ROOT/C3VD_ply"  # 输出目录
CONFIG_FILE="cfhq190l_omnidirectional_params/cfhq190l_omnidirectional_values.txt"  # 相机内参配置文件 - 确保此文件存在

# 运行Python脚本
python Omni_pcd.py \
    --depth_dir "$DEPTH_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG_FILE"

echo "点云生成完成，输出结果保存在 $OUTPUT_DIR 文件夹中"