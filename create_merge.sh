#! /usr/bin/bash

# Absolute path to data folder (users can modify as needed)
DATA_ROOT="/home/linzhe_linux/C3VD_datasets"

# Set parameters directly in the script
POSE_ROOT="$DATA_ROOT/Original_C3VD"               # Pose dataset root directory: each scene folder contains pose.txt
PLY_ROOT="$DATA_ROOT/C3VD_Raycasting10K_source"            # PLY dataset root directory: each scene folder contains fragments_ply
NEW_DATASET_ROOT="$DATA_ROOT/fused_C3VD_new"  # New dataset save directory, for partial mode output
SCENE_LIST="cecum_t1_a"          # List of sub-scenes to process
INDIVIDUAL_VOXEL="0.1"           # Increase downsampling voxel size (from 0.05 to 0.1)
FINAL_VOXEL="0.1"                # Increase final voxel size
FUSE_COUNT="5"                   # Number of frames per group

# Limit Python memory usage
export PYTHONMEM=1024M

# Run Python script
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

echo "Processing completed!"

#EOF 