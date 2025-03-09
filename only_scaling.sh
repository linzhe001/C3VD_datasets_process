#! /usr/bin/bash

# 数据文件夹的绝对路径（用户可以根据需要修改）
DATA_ROOT="/mnt/c/Users/asus/Downloads/C3VD_datasets"

# 直接在脚本中设置参数
SOURCE_ROOT="$DATA_ROOT/C3VD_ply"            # 源点云数据集根目录（需要调整的点云）
TARGET_ROOT="$DATA_ROOT/visible_point_cloud_ply"  # 目标点云数据集根目录（参考点云）
OUTPUT_ROOT="$DATA_ROOT/C3VD_ply_scaled_only"    # 仅缩放后点云的输出目录
SCENE_LIST="cecum_t1_a"       # 要处理的子场景列表，空格分隔多个场景

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_ROOT"

# 限制Python内存使用
export PYTHONMEM=1024M

# 运行Python脚本
python3 only_scaling.py \
  --source_root "$SOURCE_ROOT" \
  --target_root "$TARGET_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --scenes $SCENE_LIST

echo "点云缩放处理完成!"

#EOF
