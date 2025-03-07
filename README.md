# C3VD_datasets_process
## 不同的文件夹介绍

### C3VD
用来储存原数据集的：深度图，3D模型，pose文件
```
├── C3VD
│   └── cecum_t1_a
│       ├── 0000_depth.tiff
│       ├── coverage_mesh.obj
│       └── pose.txt
```
### C3VD_ply
该文件夹是由 ``creat_check_pcd.py`` 生成，里面包含着单帧深度图合成的ply文件。里面以及含有前十帧的单帧点云，用来后续的test。
```
├── C3VD_ply
│   └── cecum_t1_a
│       └── 0000_depth.ply
```
### fused_C3VD_new
该文件是由 ``create_merge.py``生成，里面包含着多帧合成的点云和含有点云信息的json文件
```
├── fused_all_C3VD
│   └── cecum_t1_a
│       └── cecum_t1_a_merged_all.ply
```

## 流程

1. 先用运行``creat_check_pcd.sh``使用深度图生成点云

2. 运行``create_merge_all.sh``生成多帧合成的点云储存在``fused_all_C3VD``文件夹中

3. 运行``scaling.sh``来对``fused_all_C3VD``按照3D模型的大小进行缩放。

以上就是我生成合成点云的全部过程，从第二步始我就没有在本地的电脑测试过，只在集群上使用过。

## 上面没有提到文件的介绍

### ``Omni_pcd.py`` 
该python文件，使用全向相机的方式使用深度图生成点云，但是给出的``camera_intrinsics``只是个sample，是不正确的。
