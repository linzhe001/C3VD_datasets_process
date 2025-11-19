#! /usr/bin/bash

#$ -l tmem=28G
#$ -l h_vmem=28G            # Increase memory limit
#$ -l h_rt=7200              # Runtime limit (in seconds or hours:minutes:seconds format)
#$ -l gpu=true
#$ -pe gpu 2
#$ -N creat_pcd
#$ -o /SAN/medic/MRpcr/logs/creat_pcd_output.log
#$ -e /SAN/medic/MRpcr/logs/creat_pcd_error.log
#$ -wd /SAN/medic/MRpcr

# Activate Conda environment
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pcd

# Absolute path to data folder (users can modify as needed)
DATA_ROOT="/SAN/medic/MRpcr/C3VD_datasets"

# Set input and output relative paths
INPUT_PATH="$DATA_ROOT/Original_C3VD"          # Root directory of depth map files
OUTPUT_PATH="$DATA_ROOT/C3VD_Raycasting10K_source"     # Output directory for point cloud files
POSE_ROOT="$DATA_ROOT/Original_C3VD"          # Root directory of pose files

# Create output directory (if it doesn't exist)
mkdir -p "$OUTPUT_PATH"

# Set number of threads for parallel processing (adjust based on CPU cores)
THREADS=2

# Run point cloud generation script
python3 /SAN/medic/MRpcr/C3VD_datasets_process/creat_check_pcd.py --input "$INPUT_PATH" \
                          --output "$OUTPUT_PATH" \
                          --pose_root "$POSE_ROOT" \
                          --apply_transform \
                          --voxel_size 0.5 \
                          --max_points 50000 \
                          --threads $THREADS

echo "Depth map to point cloud conversion task completed!"
#EOF
