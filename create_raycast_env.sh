#! /usr/bin/bash
#$ -l tmem=128G
#$ -l h_vmem=128G            
#$ -l h_rt=72000   
#$ -l gpu=true
#$ -pe gpu 1
#$ -N creat_raycast
#$ -o /SAN/medic/MRpcr/logs/creat_raycast_output.log
#$ -e /SAN/medic/MRpcr/logs/creat_raycast_error.log
#$ -wd /SAN/medic/MRpcr


# 检查网络连接
ping -c 1 baidu.com
if [ $? -ne 0 ]; then
  echo "网络连接有问题，请检查网络设置"
  exit 1
fi

# 创建新环境，指定 Python 3.9
conda create -n Raycast python=3.9 -y

# 激活环境
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate Raycast

# 配置 pip 源
mkdir -p ~/.pip
echo "[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn" > ~/.pip/pip.conf

# 安装 Open3D 及依赖
pip install open3d numpy opencv-python

echo "环境 'Raycast' 创建完成，已安装 Open3D 和其他依赖。" 