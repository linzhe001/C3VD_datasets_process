#! /usr/bin/bash
#$ -l tmem=28G
#$ -l h_vmem=28G            
#$ -l h_rt=7200              
#$ -l gpu=true
#$ -pe gpu 1
#$ -N outlier_checker
#$ -o /SAN/medic/MRpcr/logs/outlier_checker_output.log
#$ -e /SAN/medic/MRpcr/logs/outlier_checker_error.log
#$ -wd /SAN/medic/MRpcr


# 点云数据集异常检测工具 - Bash脚本
# 此脚本用于执行c3vd_outlier_checker.py

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 显示使用说明
function show_usage {
    echo -e "${BLUE}点云数据集异常检测工具${NC}"
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --main-folder PATH   主文件夹路径 (默认: /SAN/medic/MRpcr/C3VD_datasets)"
    echo "  -r, --ref-folder PATH    参考点云文件夹路径 (默认: /SAN/medic/MRpcr/C3VD_datasets/C3VD_ply_rot_scale_trans)"
    echo "  -s, --src-folder PATH    源点云文件夹路径 (默认: /SAN/medic/MRpcr/C3VD_datasets/visible_point_cloud_ply)"
    echo "  -o, --output PATH        输出文件路径 (默认: /SAN/medic/MRpcr/result/c3vd_outlier_checker/点云异常报告.txt)"
    echo "  -h, --help               显示此帮助信息"
    echo ""
}

# 检查文件夹路径
check_folders() {
    if [ ! -d "$MAIN_FOLDER" ]; then
        echo -e "${YELLOW}警告: 主文件夹不存在: $MAIN_FOLDER${NC}"
    fi
    
    if [ ! -d "$REF_FOLDER" ]; then
        echo -e "${YELLOW}警告: 参考点云文件夹不存在: $REF_FOLDER${NC}"
    fi
    
    if [ ! -d "$SRC_FOLDER" ]; then
        echo -e "${YELLOW}警告: 源点云文件夹不存在: $SRC_FOLDER${NC}"
    fi
}

# 预设的数据路径
MAIN_FOLDER="/SAN/medic/MRpcr/C3VD_datasets"
REF_FOLDER="/SAN/medic/MRpcr/C3VD_datasets/C3VD_ref"
SRC_FOLDER="/SAN/medic/MRpcr/C3VD_datasets/C3VD_ply_rot_scale_trans"
OUTPUT_FILE="/SAN/medic/MRpcr/result/c3vd_outlier_checker/outlier_checker_report.txt"

# 确保输出目录存在
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    
    case $key in
        -m|--main-folder)
            MAIN_FOLDER="$2"
            shift 2
            ;;
        -r|--ref-folder)
            REF_FOLDER="$2"
            shift 2
            ;;
        -s|--src-folder)
            SRC_FOLDER="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# 主程序
echo -e "${BLUE}===== 点云数据集异常检测工具 =====${NC}"

# 激活Conda环境
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pcd

# 检查文件夹
check_folders

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 显示将要处理的内容
echo -e "${BLUE}开始处理:${NC}"
echo -e "  主文件夹: ${GREEN}$MAIN_FOLDER${NC}"
echo -e "  参考点云文件夹: ${GREEN}$REF_FOLDER${NC}"
echo -e "  源点云文件夹: ${GREEN}$SRC_FOLDER${NC}"
echo -e "  输出文件: ${GREEN}$OUTPUT_FILE${NC}"
echo ""

# 运行Python脚本
echo -e "${BLUE}执行点云异常检测...${NC}"
python3 /SAN/medic/MRpcr/C3VD_datasets_process/c3vd_outlier_checker.py \
    --main_folder "$MAIN_FOLDER" \
    --ref_folder "$REF_FOLDER" \
    --src_folder "$SRC_FOLDER" \
    --output "$OUTPUT_FILE"

# 检查Python脚本执行结果
if [ $? -eq 0 ]; then
    echo -e "${GREEN}点云异常检测已完成!${NC}"
    echo -e "结果保存在: ${GREEN}$OUTPUT_FILE${NC}"
    
    # 如果存在异常报告，显示其内容
    if [ -f "$OUTPUT_FILE" ]; then
        echo -e "${BLUE}异常报告内容:${NC}"
        echo "----------------------------------------"
        cat "$OUTPUT_FILE"
        echo "----------------------------------------"
    fi
    
    echo -e "${GREEN}检测完成。${NC}"
else
    echo -e "${RED}执行过程中发生错误!${NC}"
    exit 1
fi

exit 0