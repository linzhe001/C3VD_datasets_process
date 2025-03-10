#! /usr/bin/bash
#$ -l tmem=128G
#$ -l h_vmem=128G            
#$ -l h_rt=72000   
#$ -l gpu=true
#$ -pe gpu 1
#$ -N creat_pcd
#$ -o /SAN/medic/MRpcr/logs/creat_pcd_output.log
#$ -e /SAN/medic/MRpcr/logs/creat_pcd_error.log
#$ -wd /SAN/medic/MRpcr

# Activate Conda environment
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pcd
# 数据文件夹的绝对路径（用户可以根据需要修改）
DATA_ROOT="/SAN/medic/MRpcr/C3VD_datasets"

# 设置输入和输出的相对路径
INPUT_PATH="$DATA_ROOT/C3VD"          # 深度图文件的根目录
OUTPUT_PATH="$DATA_ROOT/C3VD_ply"     # 点云文件的输出目录

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_PATH"

# 运行点云生成脚本
python3 ./C3VD_datasets_process/creat_check_pcd.py --input "$INPUT_PATH" \
                           --output "$OUTPUT_PATH" \
                           --voxel_size 0.5 \
                           --max_points 50000

echo "深度图到点云转换任务已完成！"
#EOF
