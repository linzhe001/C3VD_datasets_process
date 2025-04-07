# C3VD_datasets_process
## Introduction to Different Folders
Please build the datasets folder, and change ``DATA_ROOT`` to your folder path in each bash file. The sturcture of datasets folder is as follow:
```
├── C3VD_datasets
│   ├── C3VD
│   ├── C3VD_ply
│   └── fused_all_C3VD
```
### C3VD (only folder need prepared before)
Used to store the original dataset: depth maps, 3D models, pose files
```
├── C3VD
│   └── cecum_t1_a
│       ├── 0000_depth.tiff
│       ├── coverage_mesh.obj
│       └── pose.txt
```  
### C3VD_ply_source
This folder is generated by ``creat_check_pcd.py``, containing ply files synthesized from single-frame depth maps. It already includes point clouds of the first ten frames for subsequent testing.
```
├── C3VD_ply
│   └── cecum_t1_a
│       └── 0000_depth.ply
```
### fused_C3VD_new (not used)
This file is generated by ``create_merge.py``, containing multi-frame synthesized point clouds and json files with point cloud information
```
├── fused_all_C3VD
│   └── cecum_t1_a
│       └── cecum_t1_a_merged_all.ply
```

## Workflow

1. First run ``creat_check_pcd.sh`` to generate point clouds using depth maps

2. run ``visible_point_cloud.sh`` to generate visbible PCL by sampling the mesh for each frame.


The above is my complete process for generating synthesized point clouds. From step 2 onwards, I have not tested on my local computer, only used on the cluster.

## Description of Files Not Mentioned Above

### ``create_merge.py``
This python file used to merge different numbers of frames instead of all frames.

