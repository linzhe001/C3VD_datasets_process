#!/bin/bash

# 数据文件夹的绝对路径（用户可以根据需要修改）
DATA_ROOT="/mnt/c/Users/asus/Downloads/C3VD_datasets"

# 直接在脚本中设置参数
SOURCE_ROOT="$DATA_ROOT/C3VD_ply"                      # 源点云数据集根目录（需要调整的点云）
TARGET_ROOT="$DATA_ROOT/visible_point_cloud_ply"       # 目标点云数据集根目录（参考点云）
POSE_ROOT="$DATA_ROOT/C3VD"                            # 位姿文件所在目录
OUTPUT_ROOT="$DATA_ROOT/C3VD_ply_rot_scale_trans"      # 新名称，反映变换顺序：先旋转，再缩放，最后平移
SCENE_LIST="cecum_t1_a"                                # 要处理的子场景列表，空格分隔多个场景

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_ROOT"

# 运行Python脚本（使用先旋转，再缩放，最后平移的变换顺序）
python3 transform_scaling.py \
  --source_root "$SOURCE_ROOT" \
  --target_root "$TARGET_ROOT" \
  --pose_root "$POSE_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --scenes $SCENE_LIST

echo "点云调整处理完成! (使用优化的变换顺序：先旋转，再缩放，最后平移)"
#EOF
