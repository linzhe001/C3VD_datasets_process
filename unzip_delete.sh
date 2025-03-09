#! /usr/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G              # 硬内存限制
#$ -l h_rt=7200                # 运行时间限制（秒）
#$ -N unzip_delete_job
#$ -o /SAN/medic/MRpcr/logs/unzip_delete_output.log
#$ -e /SAN/medic/MRpcr/logs/unzip_delete_error.log
#$ -wd /SAN/medic/MRpcr

# 激活 Conda 环境
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pcd

# 设置处理参数
SOURCE_DIR="/SAN/medic/MRpcr/C3VD/C3VD"        # 源压缩包所在目录
TARGET_DIR="/SAN/medic/MRpcr/C3VD"         # 解压后文件的目标目录

# 执行Python脚本
echo "开始运行解压和清理脚本..."
python3 /SAN/medic/MRpcr/C3VD_datasets_process/unzip_delete.py \
  --source_dir ${SOURCE_DIR} \
  --target_dir ${TARGET_DIR}

echo "处理完成！"