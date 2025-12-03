"""
Utility functions for handling angle representations and residuals.
"""
import torch
import torch.nn.functional as F
import pytorch3d.transforms as pytorch3d_transforms


def normalize_angles(angles):
    """
    Normalize angles to [-pi, pi] range.
    
    Args:
        angles: [..., 3] Euler angles in radians
    
    Returns:
        normalized angles in [-pi, pi]
    """
    return torch.atan2(torch.sin(angles), torch.cos(angles))


def compute_angle_residual(target_angles, source_angles):
    """
    Compute angle residual considering periodicity.
    
    Args:
        target_angles: [..., 3] target Euler angles in radians
        source_angles: [..., 3] source Euler angles in radians
    
    Returns:
        residual: [..., 3] angle residual in [-pi, pi]
    """
    # Normalize both to [-pi, pi]
    target_norm = normalize_angles(target_angles)
    source_norm = normalize_angles(source_angles)
    
    # Compute difference
    diff = target_norm - source_norm
    
    # Normalize the difference to [-pi, pi]
    diff_norm = normalize_angles(diff)
    
    return diff_norm


def add_angle_residual(base_angles, residual):
    """
    Add residual to base angles with proper wrapping.
    
    Args:
        base_angles: [..., 3] base Euler angles
        residual: [..., 3] residual to add
    
    Returns:
        result angles in [-pi, pi]
    """
    result = base_angles + residual
    return normalize_angles(result)


def euler_to_6d(euler_angles, convention='XYZ'):
    """
    Convert Euler angles to 6D rotation representation.
    
    Args:
        euler_angles: [..., 3] Euler angles in radians
        convention: rotation convention
    
    Returns:
        rot_6d: [..., 6] 6D rotation representation
    """
    rot_mat = pytorch3d_transforms.euler_angles_to_matrix(euler_angles, convention)
    # Take first two columns of rotation matrix
    rot_6d = torch.cat([rot_mat[..., :, 0], rot_mat[..., :, 1]], dim=-1)
    return rot_6d


def rotation_6d_to_euler(rot_6d, convention='XYZ'):
    """
    Convert 6D rotation representation to Euler angles.
    
    Args:
        rot_6d: [..., 6] 6D rotation representation
        convention: rotation convention
    
    Returns:
        euler_angles: [..., 3] Euler angles in radians
    """
    rot_mat = pytorch3d_transforms.rotation_6d_to_matrix(rot_6d)
    euler_angles = pytorch3d_transforms.matrix_to_euler_angles(rot_mat, convention)
    return normalize_angles(euler_angles)


def compute_geodesic_rotation_loss(pred_rot, gt_rot):
    """
    Compute geodesic distance between two rotation matrices.
    More stable than Euler angle differences.
    
    Args:
        pred_rot: [..., 3, 3] predicted rotation matrices
        gt_rot: [..., 3, 3] ground truth rotation matrices
    
    Returns:
        geodesic distance in radians
    """
    # R_diff = R_pred^T @ R_gt
    R_diff = torch.matmul(pred_rot.transpose(-2, -1), gt_rot)
    
    # Geodesic distance: arccos((trace(R_diff) - 1) / 2)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    
    # Clamp for numerical stability
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    
    geodesic_dist = torch.acos(cos_angle)
    
    return geodesic_dist

