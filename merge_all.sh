#! /usr/bin/bash

# 数据文件夹的绝对路径（用户可以根据需要修改）
DATA_ROOT="/mnt/c/Users/asus/Downloads/C3VD_datasets"

# 设置当前工作目录的相对路径
# 请根据您的实际文件结构调整这些路径
PCD_DIR="$DATA_ROOT/C3VD_ply_rot_scale_trans/cecum_t1_a"  # 相对于当前目录的点云文件夹
OUTPUT_PATH="$DATA_ROOT/fused_all_C3VD/cecum_t1_a"        # 相对于当前目录的输出文件夹
OUTPUT_FILE="cecum_t1_a_merged_all.ply"                   # 输出文件名

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_PATH"

# 运行修改后的点云合并脚本
python merge_all.py \
  --input_dir "$PCD_DIR" \
  --output_path "$OUTPUT_PATH/$OUTPUT_FILE" \
  --voxel_size 0.5  # 增加体素大小以减少内存需求

echo "点云合并任务已完成！"
#EOF
