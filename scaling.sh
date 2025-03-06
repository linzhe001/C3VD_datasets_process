#!/bin/bash

# 数据文件夹的绝对路径（用户可以根据需要修改）
DATA_ROOT="/home/linzhe_linux/C3VD_datasets_process"

# 直接在脚本中设置参数
REF_MODEL_PATH="$DATA_ROOT/C3VD/cecum_t1_a/coverage_mesh.obj"  # 参考模型路径
FOLDER_PATH="$DATA_ROOT/fused_C3VD_new"  # 要处理的点云文件夹路径

# 运行Python脚本
python3 scaling.py \
    --ref_model "$REF_MODEL_PATH" \
    --folder "$FOLDER_PATH"

echo "处理完成!" 