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

# Add pytorch3d to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pytorch3d'))
import pytorch3d.transforms as pytorch3d_transforms

sys.path.append(os.path.dirname(__file__))
from configs.config import get_config
from datasets.dataloader import get_data_loaders_from_cfg, process_batch


def get_rot_matrix_and_trans(pose, rotation_type='euler'):
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
    return rot_matrix, trans

def transform_pointcloud_by_pose(pts, pose, rotation_type='euler', mode='inverse'):
    """
    Transform point cloud by pose.
    
    Args:
        pts: [bs, num_points, 3] point cloud
        pose: [bs, 6] pose
        rotation_type: 'euler', 'axis_angle', '6d'
        mode: 'inverse' (P' = R^T @ (P - t)) or 'forward' (P' = R @ P + t)
    """
    rot_matrix, trans = get_rot_matrix_and_trans(pose, rotation_type)
    
    if mode == 'inverse':
        # Inverse transform: P' = R^T @ (P - t)
        # Note: For row vectors, this is (P - t) @ R
        pts_transformed = torch.matmul((pts - trans.unsqueeze(1)), rot_matrix.transpose(-2, -1))
    else:
        # Forward transform: P' = R @ P + t
        # Note: For row vectors, this is P @ R^T + t
        pts_transformed = torch.matmul(pts, rot_matrix.transpose(-2, -1)) + trans.unsqueeze(1)
    
    return pts_transformed


def visualize_comparison(pts_orig, pts_fwd, pts_inv, pose_info, save_path=None):
    """
    可视化对比正变换和逆变换
    """
    fig = plt.figure(figsize=(20, 7))
    
    pts_orig = pts_orig.cpu().numpy()
    pts_fwd = pts_fwd.cpu().numpy()
    pts_inv = pts_inv.cpu().numpy()
    
    # 计算差异
    diff = np.mean(np.linalg.norm(pts_fwd - pts_inv, axis=1))
    
    # 统一坐标范围
    all_pts = np.concatenate([pts_orig, pts_fwd, pts_inv], axis=0)
    x_range = [all_pts[:, 0].min(), all_pts[:, 0].max()]
    y_range = [all_pts[:, 1].min(), all_pts[:, 1].max()]
    z_range = [all_pts[:, 2].min(), all_pts[:, 2].max()]
    
    max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]) / 2
    mid_x = (x_range[1] + x_range[0]) / 2
    mid_y = (y_range[1] + y_range[0]) / 2
    mid_z = (z_range[1] + z_range[0]) / 2
    
    def set_axes(ax, title):
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, fontsize=12)

    # Subplot 1: Original
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(pts_orig[:, 0], pts_orig[:, 1], pts_orig[:, 2], c='k', s=1, alpha=0.3)
    set_axes(ax1, 'Original Point Cloud')
    
    # Subplot 2: Forward vs Inverse
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(pts_fwd[:, 0], pts_fwd[:, 1], pts_fwd[:, 2], c='r', s=1, alpha=0.5, label='Forward')
    ax2.scatter(pts_inv[:, 0], pts_inv[:, 1], pts_inv[:, 2], c='b', s=1, alpha=0.5, label='Inverse')
    set_axes(ax2, f'Forward (Red) vs Inverse (Blue)\nMean Diff: {diff:.4f}')
    ax2.legend()
    
    # Subplot 3: Forward only (for clarity)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(pts_fwd[:, 0], pts_fwd[:, 1], pts_fwd[:, 2], c='r', s=1, alpha=0.5, label='Forward')
    set_axes(ax3, 'Forward Transform')

    # Info text
    info_text = f"Rotation (deg): {np.degrees(pose_info['rotation'])}\nTranslation: {pose_info['translation']}"
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_pointclouds(pts_original, pts_transformed, pose_info, save_path=None):
    # 保留此函数以兼容旧代码（如果还需要），但我们主要用 visualize_comparison
    # ... (省略，或者我们可以直接重写调用逻辑)
    pass 



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
            pts_fwd = transform_pointcloud_by_pose(pts_single, pose, rotation_type, mode='forward')
            pts_inv = transform_pointcloud_by_pose(pts_single, pose, rotation_type, mode='inverse')
            
            # 统计信息
            print(f"\nSample {sample_count}:")
            diff_mean = torch.mean(torch.norm(pts_fwd - pts_inv, dim=2)).item()
            print(f"  Mean diff between Forward and Inverse: {diff_mean:.6f}")
            
            # 位姿信息
            pose_info = {
                'rotation': euler_angles[0].cpu().numpy(),
                'translation': trans.cpu().numpy()
            }
            
            # 可视化
            save_path = os.path.join(save_dir, f'sample_{sample_count}_comparison.png')
            visualize_comparison(
                pts_single[0], 
                pts_fwd[0], 
                pts_inv[0],
                pose_info, 
                save_path
            )
            print(f"  Saved comparison to {save_path}")
        
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

