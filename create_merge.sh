#! /usr/bin/bash

# 数据文件夹的绝对路径（用户可以根据需要修改）
DATA_ROOT="/home/linzhe_linux/C3VD_datasets_process"

# 直接在脚本中设置参数
POSE_ROOT="$DATA_ROOT/C3VD"               # pose数据集根目录：每个场景文件夹下包含 pose.txt
PLY_ROOT="$DATA_ROOT/C3VD_ply"            # ply数据集根目录：每个场景文件夹下包含 fragments_ply
NEW_DATASET_ROOT="$DATA_ROOT/fused_C3VD_new"  # 新数据集保存目录，用于 partial 模式输出
SCENE_LIST="cecum_t1_a"          # 要处理的子场景列表
INDIVIDUAL_VOXEL="0.1"           # 增大下采样体素尺寸 (从0.05增大到0.1)
FINAL_VOXEL="0.1"                # 增大最终体素尺寸
FUSE_COUNT="5"                   # 每组的帧数

# 限制Python内存使用
export PYTHONMEM=1024M

# 运行Python脚本
python3 create_merge.py \
  --mode partial \
  --pose_root "$POSE_ROOT" \
  --ply_root "$PLY_ROOT" \
  --scene_list "$SCENE_LIST" \
  --fuse_frame_count "$FUSE_COUNT" \
  --individual_voxel "$INDIVIDUAL_VOXEL" \
  --final_voxel "$FINAL_VOXEL" \
  --new_dataset_root "$NEW_DATASET_ROOT" \
  --json_out "fused_info_${FUSE_COUNT}.json"

echo "处理完成!"

#EOF 