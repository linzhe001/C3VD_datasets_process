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


# Point Cloud Dataset Anomaly Detection Tool - Bash Script
# This script is used to execute c3vd_outlier_checker.py

# Set color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No color

# Display usage instructions
function show_usage {
    echo -e "${BLUE}Point Cloud Dataset Anomaly Detection Tool${NC}"
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --main-folder PATH   Main folder path (default: /SAN/medic/MRpcr/C3VD_datasets)"
    echo "  -r, --ref-folder PATH    Reference point cloud folder path (default: /SAN/medic/MRpcr/C3VD_datasets/C3VD_ply_rot_scale_trans)"
    echo "  -s, --src-folder PATH    Source point cloud folder path (default: /SAN/medic/MRpcr/C3VD_datasets/visible_point_cloud_ply)"
    echo "  -o, --output PATH        Output file path (default: /SAN/medic/MRpcr/result/c3vd_outlier_checker/outlier_report.txt)"
    echo "  -h, --help               Display this help information"
    echo ""
}

# Check folder paths
check_folders() {
    if [ ! -d "$MAIN_FOLDER" ]; then
        echo -e "${YELLOW}Warning: Main folder does not exist: $MAIN_FOLDER${NC}"
    fi

    if [ ! -d "$REF_FOLDER" ]; then
        echo -e "${YELLOW}Warning: Reference point cloud folder does not exist: $REF_FOLDER${NC}"
    fi

    if [ ! -d "$SRC_FOLDER" ]; then
        echo -e "${YELLOW}Warning: Source point cloud folder does not exist: $SRC_FOLDER${NC}"
    fi
}

# Preset data paths
MAIN_FOLDER="/SAN/medic/MRpcr/C3VD_datasets"
REF_FOLDER="/SAN/medic/MRpcr/C3VD_datasets/C3VD_Raycasting10K_target"
SRC_FOLDER="/SAN/medic/MRpcr/C3VD_datasets/C3VD_Raycasting10K_source"
OUTPUT_FILE="/SAN/medic/MRpcr/result/c3vd_outlier_checker/outlier_checker_report.txt"

# Ensure output directory exists
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")

# Parse command line arguments
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
            echo -e "${RED}Error: Unknown option $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Main program
echo -e "${BLUE}===== Point Cloud Dataset Anomaly Detection Tool =====${NC}"

# Activate Conda environment
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pcd

# Check folders
check_folders

# Create output directory (if it doesn't exist)
mkdir -p "$OUTPUT_DIR"

# Display what will be processed
echo -e "${BLUE}Starting processing:${NC}"
echo -e "  Main folder: ${GREEN}$MAIN_FOLDER${NC}"
echo -e "  Reference point cloud folder: ${GREEN}$REF_FOLDER${NC}"
echo -e "  Source point cloud folder: ${GREEN}$SRC_FOLDER${NC}"
echo -e "  Output file: ${GREEN}$OUTPUT_FILE${NC}"
echo ""

# Run Python script
echo -e "${BLUE}Executing point cloud anomaly detection...${NC}"
python3 /SAN/medic/MRpcr/C3VD_datasets_process/c3vd_outlier_checker.py \
    --main_folder "$MAIN_FOLDER" \
    --ref_folder "$REF_FOLDER" \
    --src_folder "$SRC_FOLDER" \
    --output "$OUTPUT_FILE"

# Check Python script execution result
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Point cloud anomaly detection completed!${NC}"
    echo -e "Results saved to: ${GREEN}$OUTPUT_FILE${NC}"

    # If anomaly report exists, display its contents
    if [ -f "$OUTPUT_FILE" ]; then
        echo -e "${BLUE}Anomaly report contents:${NC}"
        echo "----------------------------------------"
        cat "$OUTPUT_FILE"
        echo "----------------------------------------"
    fi

    echo -e "${GREEN}Detection completed.${NC}"
else
    echo -e "${RED}Error occurred during execution!${NC}"
    exit 1
fi

exit 0