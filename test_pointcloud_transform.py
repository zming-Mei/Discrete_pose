"""
测试点云变换函数的可视化脚本
用于验证 _transform_pointcloud_by_pose 函数是否正确工作
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pytorch3d.transforms as pytorch3d_transforms

sys.path.append(os.path.dirname(__file__))
from configs.config import get_config
from datasets.dataloader import get_data_loaders_from_cfg, process_batch


def transform_pointcloud_by_pose(pts, pose, rotation_type='euler'):
    """
    Transform point cloud by inverse of pose.
    
    Args:
        pts: [bs, num_points, 3] point cloud in camera frame
        pose: [bs, 6] pose (rotation + translation)
        rotation_type: 'euler' or 'axis_angle'
        
    Returns:
        pts_transformed: [bs, num_points, 3] transformed point cloud
    """
    bs = pts.shape[0]
    
    if rotation_type == 'axis_angle':
        axis_angle = pose[:, :3]
        trans = pose[:, 3:]
        rot_matrix = pytorch3d_transforms.so3_exp_map(axis_angle)
    elif rotation_type == 'euler':
        angles = pose[:, :3]
        trans = pose[:, 3:]
        rot_matrix = pytorch3d_transforms.euler_angles_to_matrix(
            angles, convention='ZYX'
        )
    else:
        rotation_6d = pose[:, :6] 
        trans = pose[:, 6:]
        rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(rotation_6d)
    
    
    # Inverse transform: P' = R^T @ (P - t)
    pts_transformed = torch.matmul((pts - trans.unsqueeze(1)), rot_matrix.transpose(-2, -1))
    
    return pts_transformed


def visualize_pointclouds(pts_original, pts_transformed, pose_info, save_path=None):
    """
    可视化变换前后的点云
    
    Args:
        pts_original: [num_points, 3] 原始点云
        pts_transformed: [num_points, 3] 变换后点云
        pose_info: dict 位姿信息
        save_path: str 保存路径
    """
    fig = plt.figure(figsize=(16, 7))
    
    # 转换为numpy
    pts_orig = pts_original.cpu().numpy()
    pts_trans = pts_transformed.cpu().numpy()
    
    # 子图1: 原始点云
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pts_orig[:, 0], pts_orig[:, 1], pts_orig[:, 2], 
                c=pts_orig[:, 2], cmap='viridis', s=1, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Original Point Cloud (Camera Frame)', fontsize=14)
    
    # 设置相同的坐标范围
    all_pts = np.concatenate([pts_orig, pts_trans], axis=0)
    x_range = [all_pts[:, 0].min(), all_pts[:, 0].max()]
    y_range = [all_pts[:, 1].min(), all_pts[:, 1].max()]
    z_range = [all_pts[:, 2].min(), all_pts[:, 2].max()]
    
    max_range = max(x_range[1] - x_range[0], 
                   y_range[1] - y_range[0], 
                   z_range[1] - z_range[0]) / 2
    mid_x = (x_range[1] + x_range[0]) / 2
    mid_y = (y_range[1] + y_range[0]) / 2
    mid_z = (z_range[1] + z_range[0]) / 2
    
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 子图2: 变换后点云
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(pts_trans[:, 0], pts_trans[:, 1], pts_trans[:, 2], 
                c=pts_trans[:, 2], cmap='plasma', s=1, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Transformed Point Cloud (Object Frame)', fontsize=14)
    
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 添加位姿信息文本
    info_text = f"Pose Information:\n"
    info_text += f"Rotation (Euler ZYX): [{pose_info['rotation'][0]:.3f}, {pose_info['rotation'][1]:.3f}, {pose_info['rotation'][2]:.3f}] rad\n"
    info_text += f"Rotation (degrees): [{np.degrees(pose_info['rotation'][0]):.2f}, {np.degrees(pose_info['rotation'][1]):.2f}, {np.degrees(pose_info['rotation'][2]):.2f}]\n"
    info_text += f"Translation: [{pose_info['translation'][0]:.3f}, {pose_info['translation'][1]:.3f}, {pose_info['translation'][2]:.3f}] m"
    
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.close()  # 关闭图形，避免阻塞


def test_transform(cfg, data_loader, num_samples=3, rotation_type='euler'):
    """
    测试点云变换函数
    
    Args:
        cfg: 配置对象
        data_loader: 数据加载器
        num_samples: 测试样本数量
        rotation_type: 旋转表示类型
    """
    print("=" * 80)
    print("Point Cloud Transformation Test")
    print("=" * 80)
    print(f"Rotation type: {rotation_type}")
    print(f"Number of samples: {num_samples}")
    print("=" * 80)
    
    # 创建保存目录
    save_dir = os.path.join(cfg.output_dir, 'pointcloud_transform_test')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}\n")
    
    device = cfg.device
    sample_count = 0
    
    for batch_idx, batch in enumerate(data_loader):
        if sample_count >= num_samples:
            break
        
        # 处理batch
        batch = process_batch(batch, device, cfg.pose_mode, 
                            mini_batch_size=96, PTS_AUG_PARAMS=None)
        
        # 获取点云和位姿
        pts = batch['zero_mean_pts'].to(device)  # [bs, num_points, 3]
        pose_6d = batch['zero_mean_gt_pose'].to(device)  # [bs, 9] (6D rotation + 3D translation)
        
        bs = pts.shape[0]
        
        for i in range(min(bs, num_samples - sample_count)):
            sample_count += 1
            
            # 提取单个样本
            pts_single = pts[i:i+1]  # [1, num_points, 3]
            
            # 转换位姿格式
            rot_6d = pose_6d[i, :6]  # [6]
            trans = pose_6d[i, 6:] # [3]
            
            # 6D rotation -> rotation matrix -> euler angles
            rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(rot_6d.unsqueeze(0))
            euler_angles = pytorch3d_transforms.matrix_to_euler_angles(rot_matrix, convention='ZYX')
            
            # 构造位姿向量
            if rotation_type == 'axis_angle':
                axis_angle = pytorch3d_transforms.so3_log_map(rot_matrix)
                pose = torch.cat([axis_angle, trans.unsqueeze(0)], dim=1)  # [1, 6]
            elif rotation_type == 'euler':
                pose = torch.cat([euler_angles, trans.unsqueeze(0)], dim=1)  # [1, 6]
            else:
                pose = torch.cat([rot_6d, trans.unsqueeze(0)], dim=1)  # [1, 9]
            
            # 应用变换
            pts_transformed = transform_pointcloud_by_pose(pts_single, pose, rotation_type)
            
            # 统计信息
            print(f"\nSample {sample_count}:")
            print(f"  Original point cloud range:")
            print(f"    X: [{pts_single[0, :, 0].min():.3f}, {pts_single[0, :, 0].max():.3f}]")
            print(f"    Y: [{pts_single[0, :, 1].min():.3f}, {pts_single[0, :, 1].max():.3f}]")
            print(f"    Z: [{pts_single[0, :, 2].min():.3f}, {pts_single[0, :, 2].max():.3f}]")
            print(f"  Transformed point cloud range:")
            print(f"    X: [{pts_transformed[0, :, 0].min():.3f}, {pts_transformed[0, :, 0].max():.3f}]")
            print(f"    Y: [{pts_transformed[0, :, 1].min():.3f}, {pts_transformed[0, :, 1].max():.3f}]")
            print(f"    Z: [{pts_transformed[0, :, 2].min():.3f}, {pts_transformed[0, :, 2].max():.3f}]")
            
            # 位姿信息
            pose_info = {
                'rotation': euler_angles[0].cpu().numpy(),
                'translation': trans.cpu().numpy()
            }
            
            print(f"  Pose information:")
            print(f"    Rotation (Euler): {pose_info['rotation']}")
            print(f"    Rotation (degrees): {np.degrees(pose_info['rotation'])}")
            print(f"    Translation: {pose_info['translation']}")
            
            # 可视化
            save_path = os.path.join(save_dir, f'sample_{sample_count}.png')
            visualize_pointclouds(
                pts_single[0], 
                pts_transformed[0], 
                pose_info, 
                save_path
            )
        
        if sample_count >= num_samples:
            break
    
    print("\n" + "=" * 80)
    print(f"Test completed! Tested {sample_count} samples")
    print(f"Visualizations saved in: {save_dir}")
    print("=" * 80)


def main():
    # 获取配置
    cfg = get_config()
    
    # 设置输出目录
    if not hasattr(cfg, 'output_dir') or cfg.output_dir is None:
        cfg.output_dir = '/home/zming/diffpose/6D/code/DICArt/output'
    
    # 加载数据
    print("Loading data...")
    data_loaders = get_data_loaders_from_cfg(cfg, ['train'])
    train_loader = data_loaders['train_loader']
    print(f"Training set size: {len(train_loader)}")
    
    # 测试变换
    rotation_type = cfg.acfm_rotation_type if hasattr(cfg, 'acfm_rotation_type') else 'euler'
    test_transform(cfg, train_loader, num_samples=10, rotation_type=rotation_type)


if __name__ == "__main__":
    main()

