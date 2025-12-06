#!/bin/bash

# 测试点云变换函数的脚本

cd "$(dirname "$0")/.."

python test_pointcloud_transform.py \
    --data_path ../ArtImage-High-level/ArtImage \
    --batch_size 8 \
    --pose_mode rot_matrix \
    --acfm_rotation_type axis_angle \
    --device cuda \
    --num_points 2048 \
    --num_workers 4 \
    --output_dir ./output/pointcloud_test

