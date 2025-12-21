import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.transforms as pytorch3d_transforms
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from networks.pts_encoder.pointnets import PointNetfeat
from networks.pts_encoder.pointnet2 import Pointnet2ClsMSG
from networks.gf_algorithms.model_modules import *
from DICArt.networks.ACFM_condition import *


class DirectPoseMLP(nn.Module):
    """
    Direct Pose Prediction MLP network (Stage 2).

    Given the coarse pose from DFM and Top-K conditions, this network directly
    predicts the continuous GT pose using an MLP (no flow matching).
    """
    
    def __init__(self, cfg, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.eps = 1e-8
    
        # Rotation representation configuration
        self.rotation_type = cfg.acfm_rotation_type if hasattr(cfg, 'acfm_rotation_type') else 'euler'
        assert self.rotation_type in ['euler', 'axis_angle', '6d'], f"Unsupported rotation_type: {self.rotation_type}"
        if self.rotation_type == 'axis_angle':
            self.rotation_dim = 3  # 3D axis-angle (so(3) Lie algebra)
            print("DirectPoseMLP: Using 3D axis-angle (so(3)) representation")
        elif self.rotation_type == '6d':
            self.rotation_dim = 6  # 6D rotation representation
            print("DirectPoseMLP: Using 6D rotation representation")
        else:
            self.rotation_dim = 3  # Euler angles
            print("DirectPoseMLP: Using Euler angle representation")
        
        # Normalization configuration
        self.use_normalization = cfg.use_normalization if hasattr(cfg, 'use_normalization') else True
        print(f"DirectPoseMLP: Normalization {'enabled' if self.use_normalization else 'disabled'}")
        
        # DFM always outputs 3D Euler angles for Top-K conditions
        self.cond_rot_dim = 3
        self.translation_dim = 3
        
        # Prediction dimensions (what this network outputs)
        self.num_dimensions = self.rotation_dim + self.translation_dim
        # Condition dimensions (what DFM provides: coarse pose)
        self.cond_num_dimensions = self.cond_rot_dim + self.translation_dim
        
        # Loss weights
        self.rotation_weight = cfg.rotation_weight if hasattr(cfg, 'rotation_weight') else 1.0
        self.translation_weight = cfg.translation_weight if hasattr(cfg, 'translation_weight') else 1.0
        
        # Point cloud feature extractor
        if self.cfg.pts_encoder == 'pointnet':
            self.pts_encoder = PointNetfeat(num_points=self.cfg.num_points, out_dim=1024)
            self.pts_feat_dim = 1024
        elif self.cfg.pts_encoder == 'pointnet2':
            self.pts_encoder = Pointnet2ClsMSG(0)
            self.pts_feat_dim = 1024
        elif self.cfg.pts_encoder == 'pointnet_and_pointnet2':
            self.pts_pointnet = PointNetfeat(num_points=self.cfg.num_points, out_dim=1024)
            self.pts_pointnet2 = Pointnet2ClsMSG(0)
            self.fusion = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU()
            )
            self.pts_feat_dim = 1024
        else:
            raise NotImplementedError
        
        # === Coarse pose embedding ===
        # Embed the coarse pose from DFM (6D: 3 Euler + 3 Trans)
        self.coarse_pose_embedder = nn.Sequential(
            nn.Linear(self.cond_num_dimensions, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        ).to(device)
        
        # === Main network ===
        # Input dim: pts_feat (1024D) + coarse_pose_emb (256D)
        input_dim = self.pts_feat_dim + 256
        
        self.mlp_shared = SharedMLP(
            input_dim=input_dim,
            hidden_dim=1024,
            output_dim=512,
            dropout=0.1
        ).to(device)
        
        # Rotation and translation branches
        self.rotation_branch = PoseBranch(
            input_dim=512,
            hidden_dim=384,
            output_dim=256,
            dropout=0.1
        ).to(device)
        
        self.translation_branch = PoseBranch(
            input_dim=512,
            hidden_dim=384,
            output_dim=256,
            dropout=0.1
        ).to(device)
        
        # Self-attention
        self.self_attention_rotation = SelfAttentionBlock(
            embed_dim=256,
            num_heads=8,
            dropout=0.1
        ).to(device)
        
        self.self_attention_translation = SelfAttentionBlock(
            embed_dim=256,
            num_heads=8,
            dropout=0.1
        ).to(device)
        
        # Conditional attention (cross-branch interaction)
        self.conditional_attention_rotation = ConditionalAttention(
            embed_dim=256,
            condition_dim=256,
            num_heads=4,
            dropout=0.1
        ).to(device)
        
        self.conditional_attention_translation = ConditionalAttention(
            embed_dim=256,
            condition_dim=256,
            num_heads=4,
            dropout=0.1
        ).to(device)
        
        # Output heads (Direct pose prediction)
        self.rotation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.rotation_dim)
        ).to(device)
        
        self.translation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.translation_dim)
        ).to(device)
    
    def extract_pts_feature(self, pts):
        """Extract point cloud features."""
        if self.cfg.pts_encoder == 'pointnet':
            return self.pts_encoder(pts.permute(0, 2, 1))
        elif self.cfg.pts_encoder == 'pointnet2':
            return self.pts_encoder(pts)
        elif self.cfg.pts_encoder == 'pointnet_and_pointnet2':
            feat1 = self.pts_pointnet(pts.permute(0, 2, 1))
            feat2 = self.pts_pointnet2(pts)
            return self.fusion(torch.cat([feat1, feat2], dim=1))
        else:
            raise NotImplementedError
    
    def normalize_pose(self, pose, translation_status):
        """
        Normalize pose to [-1, 1] range.
        
        Args:
            pose: [bs, num_dimensions] - rotation + translation
            translation_status: [x_min, x_max, y_min, y_max, z_min, z_max]
        
        Returns:
            normalized_pose: [bs, num_dimensions] - normalized to [-1, 1]
        """
        if not self.use_normalization:
            return pose
        
        rotation = pose[:, :self.rotation_dim]
        translation = pose[:, self.rotation_dim:]
        
        # Normalize rotation based on type
        if self.rotation_type == 'euler':
            pi = np.pi
            rot_min = torch.tensor([-pi, -pi/2, -pi], device=self.device)
            rot_max = torch.tensor([pi, pi/2, pi], device=self.device)
        elif self.rotation_type == 'axis_angle':
            pi = np.pi
            rot_min = torch.tensor([-pi, -pi, -pi], device=self.device)
            rot_max = torch.tensor([pi, pi, pi], device=self.device)
        elif self.rotation_type == '6d':
            rot_min = torch.tensor([-1.0] * 6, device=self.device)
            rot_max = torch.tensor([1.0] * 6, device=self.device)
        else:
            raise ValueError(f"Unsupported rotation_type: {self.rotation_type}")
        
        rot_normalized = 2.0 * (rotation - rot_min) / (rot_max - rot_min + 1e-8) - 1.0
        
        trans_min = torch.tensor([
            translation_status[0],
            translation_status[2],
            translation_status[4]
        ], device=self.device)
        trans_max = torch.tensor([
            translation_status[1],
            translation_status[3],
            translation_status[5]
        ], device=self.device)
        
        trans_normalized = 2.0 * (translation - trans_min) / (trans_max - trans_min + 1e-8) - 1.0
        
        return torch.cat([rot_normalized, trans_normalized], dim=1)
    
    def denormalize_pose(self, normalized_pose, translation_status):
        """
        Denormalize pose from [-1, 1] to original range.
        
        Args:
            normalized_pose: [bs, num_dimensions] - normalized pose in [-1, 1]
            translation_status: [x_min, x_max, y_min, y_max, z_min, z_max]
        
        Returns:
            pose: [bs, num_dimensions] - denormalized pose
        """
        if not self.use_normalization:
            return normalized_pose
        
        rot_normalized = normalized_pose[:, :self.rotation_dim]
        trans_normalized = normalized_pose[:, self.rotation_dim:]
        
        if self.rotation_type == 'euler':
            pi = np.pi
            rot_min = torch.tensor([-pi, -pi/2, -pi], device=self.device)
            rot_max = torch.tensor([pi, pi/2, pi], device=self.device)
        elif self.rotation_type == 'axis_angle':
            pi = np.pi
            rot_min = torch.tensor([-pi, -pi, -pi], device=self.device)
            rot_max = torch.tensor([pi, pi, pi], device=self.device)
        elif self.rotation_type == '6d':
            rot_min = torch.tensor([-1.0] * 6, device=self.device)
            rot_max = torch.tensor([1.0] * 6, device=self.device)
        else:
            raise ValueError(f"Unsupported rotation_type: {self.rotation_type}")
        
        rotation = (rot_normalized + 1.0) / 2.0 * (rot_max - rot_min) + rot_min
        
        trans_min = torch.tensor([
            translation_status[0],
            translation_status[2],
            translation_status[4]
        ], device=self.device)
        trans_max = torch.tensor([
            translation_status[1],
            translation_status[3],
            translation_status[5]
        ], device=self.device)
        
        translation = (trans_normalized + 1.0) / 2.0 * (trans_max - trans_min) + trans_min
        
        return torch.cat([rotation, translation], dim=1)
    
    def compute_metric_loss(self, pred_pose, gt_pose):
        """
        Compute metric-based loss (rotation angle error + translation error).
        
        Args:
            pred_pose: [bs, num_dimensions] - predicted pose in original space
            gt_pose: [bs, num_dimensions] - ground truth pose in original space
            
        Returns:
            angle_loss: rotation angle loss (in radians, differentiable)
            trans_loss: translation loss (in meters)
            angle_error_deg: rotation angle error (in degrees, for logging)
            trans_error_m: translation error (in meters, for logging)
        """
        pred_rot = pred_pose[:, :self.rotation_dim]
        pred_trans = pred_pose[:, self.rotation_dim:]
        gt_rot = gt_pose[:, :self.rotation_dim]
        gt_trans = gt_pose[:, self.rotation_dim:]
        
        # Convert to rotation matrices
        if self.rotation_type == 'euler':
            pred_rot_matrix = pytorch3d_transforms.euler_angles_to_matrix(pred_rot, convention='ZYX')
            gt_rot_matrix = pytorch3d_transforms.euler_angles_to_matrix(gt_rot, convention='ZYX')
        elif self.rotation_type == 'axis_angle':
            pred_rot_matrix = pytorch3d_transforms.so3_exp_map(pred_rot)
            gt_rot_matrix = pytorch3d_transforms.so3_exp_map(gt_rot)
        elif self.rotation_type == '6d':
            pred_rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(pred_rot)
            gt_rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(gt_rot)
        else:
            raise ValueError(f"Unsupported rotation_type: {self.rotation_type}")
        
        # Compute rotation angle error (differentiable)
        rot_diff_matrix = torch.bmm(pred_rot_matrix, gt_rot_matrix.transpose(-2, -1))
        trace = rot_diff_matrix[:, 0, 0] + rot_diff_matrix[:, 1, 1] + rot_diff_matrix[:, 2, 2]
        cos_angle = (trace - 1.0) / 2.0
        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
        angle_error_rad = torch.acos(cos_angle)
        angle_error_deg = angle_error_rad * 180.0 / np.pi
        
        # Compute translation error (in meters)
        trans_error_m = torch.norm(pred_trans - gt_trans, dim=1)
        
        angle_loss = angle_error_rad.mean()
        trans_loss = trans_error_m.mean()
        
        return angle_loss, trans_loss, angle_error_deg.mean(), trans_error_m.mean()
    
    def _get_bin_stats(self, translation_status):
        """Calculate bin widths and min values for mapping bins to continuous values."""
        pi = np.pi
        rot_ranges = torch.tensor([2*pi, pi, 2*pi], device=self.device)
        rot_mins = torch.tensor([-pi, -pi/2, -pi], device=self.device)
        
        rot_bin_widths = rot_ranges / self.cfg.num_bins
        
        t_status = translation_status
        trans_ranges = torch.tensor([
            t_status[1] - t_status[0],
            t_status[3] - t_status[2],
            t_status[5] - t_status[4]
        ], device=self.device)
        trans_mins = torch.tensor([
            t_status[0],
            t_status[2],
            t_status[4]
        ], device=self.device)
        
        trans_bin_widths = trans_ranges / self.cfg.num_bins
        
        bin_widths = torch.cat([rot_bin_widths, trans_bin_widths])
        mins = torch.cat([rot_mins, trans_mins])
        
        return bin_widths, mins

    def get_coarse_pose_from_topk(self, topk_info, translation_status):
        """
        Get coarse pose from Top-K bins (use the top-1 bin center).
        
        Args:
            topk_info: dict containing:
                - topk_bins: [bs, 6, k] (DFM output: 3 rot + 3 trans)
                - topk_probs: [bs, 6, k]
                - coarse_pose: [bs, 6] (optional, if already computed)
            translation_status: list [min, max, min, max, min, max]
        
        Returns:
            coarse_pose: [bs, 6] Coarse pose (Euler + translation)
        """
        # If coarse_pose is already provided, use it
        if 'coarse_pose' in topk_info:
            coarse_bins = topk_info['coarse_pose']  # [bs, 6] bin indices
            bin_widths, mins = self._get_bin_stats(translation_status)
            bin_widths_batch = bin_widths.unsqueeze(0)
            mins_batch = mins.unsqueeze(0)
            # Convert bin indices to bin centers
            coarse_pose = mins_batch + (coarse_bins.float() + 0.5) * bin_widths_batch
            return coarse_pose
        
        # Otherwise, use top-1 from topk_bins
        topk_bins = topk_info['topk_bins']   # [bs, 6, k]
        top1_bins = topk_bins[:, :, 0]  # [bs, 6] - take top-1
        
        bin_widths, mins = self._get_bin_stats(translation_status)
        bin_widths_batch = bin_widths.unsqueeze(0)
        mins_batch = mins.unsqueeze(0)
        
        # Convert bin indices to bin centers
        coarse_pose = mins_batch + (top1_bins.float() + 0.5) * bin_widths_batch
        return coarse_pose

    def forward(self, pts_feat, coarse_pose_emb):
        """
        Forward pass: directly predict pose.

        Args:
            pts_feat: [bs, pts_feat_dim] point cloud features
            coarse_pose_emb: [bs, 256] embedded coarse pose
            
        Returns:
            pred_pose: [bs, num_dimensions] predicted pose (normalized if enabled)
        """
        bs = pts_feat.shape[0]
        
        # Concatenate features
        input_feat = torch.cat([pts_feat, coarse_pose_emb], dim=1)
        
        # Shared MLP
        shared_feat = self.mlp_shared(input_feat)
        
        # Rotation and translation branches
        rotation_feat = self.rotation_branch(shared_feat)
        translation_feat = self.translation_branch(shared_feat)
        
        # Self-attention
        rotation_feat = self.self_attention_rotation(rotation_feat)
        translation_feat = self.self_attention_translation(translation_feat)
        
        # Conditional attention (cross-branch)
        rotation_feat_enhanced = self.conditional_attention_rotation(
            rotation_feat, translation_feat
        )
        translation_feat_enhanced = self.conditional_attention_translation(
            translation_feat, rotation_feat
        )
        
        # Predict pose
        pred_rotation = self.rotation_head(rotation_feat_enhanced)
        pred_translation = self.translation_head(translation_feat_enhanced)
        
        pred_pose = torch.cat([pred_rotation, pred_translation], dim=1)
        
        return pred_pose
    
    def sample(self, pts_feat, topk_info, translation_status, step_size=None, method=None):
        """
        Sample (predict) pose directly from point cloud features and coarse pose.

        Args:
            pts_feat: [bs, pts_feat_dim]
            topk_info: dict with topk_bins, topk_probs, coarse_pose
            translation_status: list [min, max, min, max, min, max]
            step_size: (unused, for API compatibility)
            method: (unused, for API compatibility)
            
        Returns:
            pose: [bs, num_dimensions] predicted pose
        """
        # Get coarse pose from DFM
        coarse_pose = self.get_coarse_pose_from_topk(topk_info, translation_status)
        
        # Embed coarse pose
        coarse_pose_emb = self.coarse_pose_embedder(coarse_pose)
        
        # Forward pass
        pred_pose_normalized = self.forward(pts_feat, coarse_pose_emb)
        
        # Denormalize if needed
        if self.use_normalization:
            pred_pose = self.denormalize_pose(pred_pose_normalized, translation_status)
        else:
            pred_pose = pred_pose_normalized
        
        return pred_pose

    def loss(self, pose_gt, pts_feat, topk_info, translation_status):
        """
        Compute direct pose prediction loss.
        
        Args:
            pose_gt: [bs, num_dimensions] ground truth pose
            pts_feat: [bs, pts_feat_dim] point cloud features
            topk_info: dict with topk_bins, topk_probs, coarse_pose
            translation_status: list for normalization
            
        Returns:
            total_loss: scalar loss
            loss_description: string description
            loss_dict: dict of individual losses
        """
        bs = pose_gt.shape[0]
        
        # Get coarse pose from DFM
        coarse_pose = self.get_coarse_pose_from_topk(topk_info, translation_status)
        
        # Embed coarse pose
        coarse_pose_emb = self.coarse_pose_embedder(coarse_pose)
        
        # Forward pass to get predicted pose (normalized)
        pred_pose_normalized = self.forward(pts_feat, coarse_pose_emb)
        
        # Normalize GT if needed
        if self.use_normalization:
            gt_normalized = self.normalize_pose(pose_gt, translation_status)
        else:
            gt_normalized = pose_gt
        
        # L1 loss in normalized space
        pred_rot = pred_pose_normalized[:, :self.rotation_dim]
        pred_trans = pred_pose_normalized[:, self.rotation_dim:]
        gt_rot = gt_normalized[:, :self.rotation_dim]
        gt_trans = gt_normalized[:, self.rotation_dim:]
        
        rot_l1_loss = F.l1_loss(pred_rot, gt_rot)
        trans_l1_loss = F.l1_loss(pred_trans, gt_trans)
        
        # L2 loss in normalized space
        rot_l2_loss = F.mse_loss(pred_rot, gt_rot)
        trans_l2_loss = F.mse_loss(pred_trans, gt_trans)
        
        # Metric-based loss (in original space)
        if self.use_normalization:
            pred_pose_real = self.denormalize_pose(pred_pose_normalized, translation_status)
        else:
            pred_pose_real = pred_pose_normalized
        
        angle_loss, trans_metric_loss, angle_error_deg, trans_error_m = self.compute_metric_loss(
            pred_pose_real, pose_gt
        )
        
        # Combine losses
        l1_weight = 1.0
        l2_weight = 0.5
        metric_weight = self.cfg.metric_loss_weight if hasattr(self.cfg, 'metric_loss_weight') else 0.1
        
        total_loss = (
            l1_weight * (self.rotation_weight * rot_l1_loss + self.translation_weight * trans_l1_loss) +
            l2_weight * (self.rotation_weight * rot_l2_loss + self.translation_weight * trans_l2_loss) +
            metric_weight * (angle_loss + trans_metric_loss)
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'rot_l1_loss': rot_l1_loss.item(),
            'trans_l1_loss': trans_l1_loss.item(),
            'rot_l2_loss': rot_l2_loss.item(),
            'trans_l2_loss': trans_l2_loss.item(),
            'l1_loss': (rot_l1_loss + trans_l1_loss).item(),
            'l2_loss': (rot_l2_loss + trans_l2_loss).item(),
            'angle_error_deg': angle_error_deg.item(),
            'trans_error_m': trans_error_m.item(),
            'angle_loss': angle_loss.item(),
            'trans_metric_loss': trans_metric_loss.item()
        }
        
        loss_description = (
            f"Total: {total_loss:.4f}, "
            f"L1: {rot_l1_loss + trans_l1_loss:.4f}, "
            f"L2: {rot_l2_loss + trans_l2_loss:.4f}, "
            f"Angle: {angle_error_deg:.2f}°, "
            f"Trans: {trans_error_m*100:.2f}cm"
        )
        
        return total_loss, loss_description, loss_dict


# Alias for backward compatibility
AdaptiveContinuousFlowMatching = DirectPoseMLP
