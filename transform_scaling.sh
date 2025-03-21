#! /usr/bin/bash

#$ -l tmem=28G
#$ -l h_vmem=28G            # 增加内存限制
#$ -l h_rt=7200              # Runtime limit (in seconds or hours:minutes:seconds format)
#$ -l gpu=true
#$ -pe gpu 2
#$ -N transform_scaling
#$ -o /SAN/medic/MRpcr/logs/transform_scaling_output.log
#$ -e /SAN/medic/MRpcr/logs/transform_scaling_error.log
#$ -wd /SAN/medic/MRpcr

# Activate Conda environment
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pcd
# 数据文件夹的绝对路径（用户可以根据需要修改）
DATA_ROOT="/SAN/medic/MRpcr/C3VD_datasets"

# 直接在脚本中设置参数
SOURCE_ROOT="$DATA_ROOT/C3VD_ply"                      # 源点云数据集根目录（需要调整的点云）
TARGET_ROOT="$DATA_ROOT/visible_point_cloud_ply"       # 目标点云数据集根目录（参考点云）
POSE_ROOT="$DATA_ROOT/C3VD"                            # 位姿文件所在目录
OUTPUT_ROOT="$DATA_ROOT/C3VD_ply_rot_scale_trans"      # 新名称，反映变换顺序：先旋转，再缩放，最后平移
SCENE_LIST="cecum_t1_a cecum_t1_b cecum_t2_a cecum_t2_b cecum_t2_c cecum_t3_a cecum_t4_a cecum_t4_b desc_t4_a sigmoid_t1_a sigmoid_t2_a sigmoid_t3_a sigmoid_t3_b trans_t1_a trans_t1_b trans_t2_a trans_t2_b trans_t2_c trans_t3_a trans_t3_b trans_t4_a trans_t4_b"                                # 要处理的子场景列表，空格分隔多个场景

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_ROOT"

# 运行Python脚本（使用先旋转，再缩放，最后平移的变换顺序）
python3 /SAN/medic/MRpcr/C3VD_datasets_process/transform_scaling.py\
  --source_root "$SOURCE_ROOT" \
  --target_root "$TARGET_ROOT" \
  --pose_root "$POSE_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --scenes $SCENE_LIST

echo "点云调整处理完成! (使用优化的变换顺序：先旋转，再缩放，最后平移)"
#EOF
