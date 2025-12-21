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
from flow_matching.path import CondOTProbPath, AffineProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from DICArt.utils.angle_utils import compute_angle_residual, add_angle_residual, normalize_angles


class AdaptiveContinuousFlowMatching(nn.Module):
    """
    Adaptive Continuous Flow Matching network (Stage 2).

    Given the coarse pose from DFM and Top-K conditions, this network predicts
    the continuous GT pose directly (no longer residual).
    x0 is initialized based on DFM's Top-K bins.
    """
    
    def __init__(self, cfg, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.eps = 1e-8
        self.time_epsilon = 1e-3
        # If True: predict delta w.r.t. sampled coarse_pose_sample (from Top-K bins),
        # and compose it back to absolute pose during sampling / metric evaluation.
        self.predict_delta = cfg.acfm_predict_delta if hasattr(cfg, 'acfm_predict_delta') else False
    
        # Rotation representation configuration
        self.rotation_type = cfg.acfm_rotation_type if hasattr(cfg, 'acfm_rotation_type') else 'euler'
        assert self.rotation_type in ['euler', 'axis_angle', '6d'], f"Unsupported rotation_type: {self.rotation_type}"
        if self.rotation_type == 'axis_angle':
            self.rotation_dim = 3  # 3D axis-angle (so(3) Lie algebra)
            print("AdaptiveContinuousFlowMatching: Using 3D axis-angle (so(3)) representation")
        elif self.rotation_type == '6d':
            self.rotation_dim = 6  # 6D rotation representation
            print("AdaptiveContinuousFlowMatching: Using 6D rotation representation")
        else:
            self.rotation_dim = 3  # Euler angles
            print("AdaptiveContinuousFlowMatching: Using Euler angle representation")
        
        # Normalization configuration
        self.use_normalization = cfg.use_normalization if hasattr(cfg, 'use_normalization') else True
        print(f"AdaptiveContinuousFlowMatching: Normalization {'enabled' if self.use_normalization else 'disabled'}")
        if self.predict_delta:
            print("AdaptiveContinuousFlowMatching: Predicting DELTA (residual) w.r.t. coarse_pose_sample")
        
        # DFM always outputs 3D Euler angles for Top-K conditions
        self.cond_rot_dim = 3
        self.translation_dim = 3
        
        # Prediction dimensions (what ACFM outputs)
        self.num_dimensions = self.rotation_dim + self.translation_dim
        # Condition dimensions (what DFM provides)
        self.cond_num_dimensions = self.cond_rot_dim + self.translation_dim
        
        # Loss weights
        self.velocity_weight = cfg.velocity_weight if hasattr(cfg, 'velocity_weight') else 1.0
        self.rotation_weight = cfg.rotation_weight if hasattr(cfg, 'rotation_weight') else 1.0
        self.translation_weight = cfg.translation_weight if hasattr(cfg, 'translation_weight') else 1.0
        self.pose_prediction_weight = cfg.pose_prediction_weight if hasattr(cfg, 'pose_prediction_weight') else 0.1
        
        # Initialize Flow Matching path
        scheduler = PolynomialConvexScheduler(n=1)
        self.path = AffineProbPath(scheduler)
        
        # Point cloud feature extractor (shared with DFM)
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
        
        # === Time embedding ===
        self.time_embedder = TimestepEmbedder(
            hidden_size=256,
            frequency_embedding_size=128
        ).to(device)
        
        # === Feature fusion ===
        self.cond_time_proj = nn.Linear(self.pts_feat_dim, 256).to(device)
        self.cross_attention_fusion = CrossAttentionFusion(
            embed_dim=256,
            num_heads=8,
            dropout=0.1
        ).to(device)
        
        # === Main network ===
        # Input dim:
        # - absolute mode: x_t (num_dimensions) + time_emb (256D) + pts_feat (1024D)
        # - delta mode:     x_t (delta) + time_emb + pts_feat + coarse_pose_cond (num_dimensions)
        self.cond_pose_dim = self.num_dimensions if self.predict_delta else 0
        input_dim = self.num_dimensions + 256 + self.pts_feat_dim + self.cond_pose_dim
        
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
        
        # Output heads (Velocity only)
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

    def _rotation_to_matrix(self, rot):
        """Convert rotation (in current rotation_type) to rotation matrix."""
        if self.rotation_type == 'euler':
            return pytorch3d_transforms.euler_angles_to_matrix(rot, convention='ZYX')
        if self.rotation_type == 'axis_angle':
            return pytorch3d_transforms.so3_exp_map(rot)
        if self.rotation_type == '6d':
            return pytorch3d_transforms.rotation_6d_to_matrix(rot)
        raise ValueError(f"Unsupported rotation_type: {self.rotation_type}")

    def _matrix_to_rotation(self, rot_matrix):
        """Convert rotation matrix to rotation (in current rotation_type)."""
        if self.rotation_type == 'euler':
            angles = pytorch3d_transforms.matrix_to_euler_angles(rot_matrix, convention='ZYX')
            return normalize_angles(angles)
        if self.rotation_type == 'axis_angle':
            return pytorch3d_transforms.so3_log_map(rot_matrix)
        if self.rotation_type == '6d':
            return pytorch3d_transforms.matrix_to_rotation_6d(rot_matrix)
        raise ValueError(f"Unsupported rotation_type: {self.rotation_type}")

    def compute_delta_pose(self, pose_target, pose_base):
        """
        Compute delta such that apply_delta_pose(pose_base, delta) == pose_target (approximately).

        - Rotation: delta is computed on SO(3) (or wrapped Euler residual for 'euler').
        - Translation: simple additive residual in xyz.
        """
        base_rot = pose_base[:, :self.rotation_dim]
        base_trans = pose_base[:, self.rotation_dim:]
        target_rot = pose_target[:, :self.rotation_dim]
        target_trans = pose_target[:, self.rotation_dim:]

        # Rotation delta
        if self.rotation_type == 'euler':
            rot_delta = compute_angle_residual(target_rot, base_rot)
        else:
            R_base = self._rotation_to_matrix(base_rot)
            R_target = self._rotation_to_matrix(target_rot)
            R_delta = torch.bmm(R_target, R_base.transpose(-2, -1))  # R_delta @ R_base = R_target
            rot_delta = self._matrix_to_rotation(R_delta)

        # Translation delta (additive)
        trans_delta = target_trans - base_trans
        return torch.cat([rot_delta, trans_delta], dim=1)

    def apply_delta_pose(self, pose_base, delta_pose):
        """
        Compose delta onto base pose to get absolute pose.

        This is the inverse operation of compute_delta_pose().
        """
        base_rot = pose_base[:, :self.rotation_dim]
        base_trans = pose_base[:, self.rotation_dim:]
        delta_rot = delta_pose[:, :self.rotation_dim]
        delta_trans = delta_pose[:, self.rotation_dim:]

        # Rotation composition: R = R_delta @ R_base
        if self.rotation_type == 'euler':
            rot = add_angle_residual(base_rot, delta_rot)
        else:
            R_base = self._rotation_to_matrix(base_rot)
            R_delta = self._rotation_to_matrix(delta_rot)
            R = torch.bmm(R_delta, R_base)
            rot = self._matrix_to_rotation(R)

        # Translation composition: t = t_base + delta_t
        trans = base_trans + delta_trans
        return torch.cat([rot, trans], dim=1)
    
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
        # Allow config to disable normalization entirely
        if not self.use_normalization:
            return pose
        
        rotation = pose[:, :self.rotation_dim]
        translation = pose[:, self.rotation_dim:]
        
        # Normalize rotation based on type
        if self.rotation_type == 'euler':
            # Euler angles: [-pi, pi], [-pi/2, pi/2], [-pi, pi]
            pi = np.pi
            rot_min = torch.tensor([-pi, -pi/2, -pi], device=self.device)
            rot_max = torch.tensor([pi, pi/2, pi], device=self.device)
        elif self.rotation_type == 'axis_angle':
            # Axis-angle: each component in [-pi, pi]
            pi = np.pi
            rot_min = torch.tensor([-pi, -pi, -pi], device=self.device)
            rot_max = torch.tensor([pi, pi, pi], device=self.device)
        elif self.rotation_type == '6d':
            # 6D rotation: each component typically in [-1, 1] (unit vectors)
            rot_min = torch.tensor([-1.0] * 6, device=self.device)
            rot_max = torch.tensor([1.0] * 6, device=self.device)
        else:
            raise ValueError(f"Unsupported rotation_type: {self.rotation_type}")
        
        # Normalize rotation: [min, max] -> [-1, 1]
        rot_normalized = 2.0 * (rotation - rot_min) / (rot_max - rot_min + 1e-8) - 1.0
        
        # Normalize translation
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
        
        # Normalize translation: [min, max] -> [-1, 1]
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
        # Keep behavior consistent with normalization flag
        if not self.use_normalization:
            return normalized_pose
        
        rot_normalized = normalized_pose[:, :self.rotation_dim]
        trans_normalized = normalized_pose[:, self.rotation_dim:]
        
        # Denormalize rotation
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
        
        # Denormalize rotation: [-1, 1] -> [min, max]
        rotation = (rot_normalized + 1.0) / 2.0 * (rot_max - rot_min) + rot_min
        
        # Denormalize translation
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
        
        # Denormalize translation: [-1, 1] -> [min, max]
        translation = (trans_normalized + 1.0) / 2.0 * (trans_max - trans_min) + trans_min
        
        return torch.cat([rotation, translation], dim=1)

    def normalize_delta_pose(self, delta_pose, translation_status):
        """
        Normalize delta pose to [-1, 1] range.

        Rotation delta:
          - euler/axis_angle: each component in [-pi, pi]
          - 6d: in [-1, 1]
        Translation delta:
          - each axis in [-range, range], where range = (max-min) from translation_status
        """
        if not self.use_normalization:
            return delta_pose

        rot = delta_pose[:, :self.rotation_dim]
        trans = delta_pose[:, self.rotation_dim:]

        if self.rotation_type in ['euler', 'axis_angle']:
            pi = np.pi
            rot_min = torch.tensor([-pi, -pi, -pi], device=self.device)
            rot_max = torch.tensor([pi, pi, pi], device=self.device)
        elif self.rotation_type == '6d':
            rot_min = torch.tensor([-1.0] * 6, device=self.device)
            rot_max = torch.tensor([1.0] * 6, device=self.device)
        else:
            raise ValueError(f"Unsupported rotation_type: {self.rotation_type}")

        rot_norm = 2.0 * (rot - rot_min) / (rot_max - rot_min + 1e-8) - 1.0

        # symmetric translation range from dataset bounds
        t_status = translation_status
        trans_ranges = torch.tensor([
            t_status[1] - t_status[0],
            t_status[3] - t_status[2],
            t_status[5] - t_status[4]
        ], device=self.device)
        trans_min = -trans_ranges
        trans_max = trans_ranges
        trans_norm = 2.0 * (trans - trans_min) / (trans_max - trans_min + 1e-8) - 1.0

        return torch.cat([rot_norm, trans_norm], dim=1)

    def denormalize_delta_pose(self, delta_pose_normalized, translation_status):
        """Inverse of normalize_delta_pose()."""
        if not self.use_normalization:
            return delta_pose_normalized

        rot_norm = delta_pose_normalized[:, :self.rotation_dim]
        trans_norm = delta_pose_normalized[:, self.rotation_dim:]

        if self.rotation_type in ['euler', 'axis_angle']:
            pi = np.pi
            rot_min = torch.tensor([-pi, -pi, -pi], device=self.device)
            rot_max = torch.tensor([pi, pi, pi], device=self.device)
        elif self.rotation_type == '6d':
            rot_min = torch.tensor([-1.0] * 6, device=self.device)
            rot_max = torch.tensor([1.0] * 6, device=self.device)
        else:
            raise ValueError(f"Unsupported rotation_type: {self.rotation_type}")

        rot = (rot_norm + 1.0) / 2.0 * (rot_max - rot_min) + rot_min

        t_status = translation_status
        trans_ranges = torch.tensor([
            t_status[1] - t_status[0],
            t_status[3] - t_status[2],
            t_status[5] - t_status[4]
        ], device=self.device)
        trans_min = -trans_ranges
        trans_max = trans_ranges
        trans = (trans_norm + 1.0) / 2.0 * (trans_max - trans_min) + trans_min

        return torch.cat([rot, trans], dim=1)
    
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
        # R_diff = R_pred @ R_gt^T, angle = arccos((trace(R_diff) - 1) / 2)
        rot_diff_matrix = torch.bmm(pred_rot_matrix, gt_rot_matrix.transpose(-2, -1))
        trace = rot_diff_matrix[:, 0, 0] + rot_diff_matrix[:, 1, 1] + rot_diff_matrix[:, 2, 2]
        cos_angle = (trace - 1.0) / 2.0
        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
        angle_error_rad = torch.acos(cos_angle)  # [bs]
        angle_error_deg = angle_error_rad * 180.0 / np.pi
        
        # Compute translation error (in meters)
        trans_error_m = torch.norm(pred_trans - gt_trans, dim=1)  # [bs]
        
        # Mean losses
        angle_loss = angle_error_rad.mean()
        trans_loss = trans_error_m.mean()
        
        return angle_loss, trans_loss, angle_error_deg.mean(), trans_error_m.mean()
    
    def _get_bin_stats(self, translation_status):
        """Calculate bin widths and min values for mapping bins to continuous values."""
        # Rotation Ranges (Euler): [-pi, pi], [-pi/2, pi/2], [-pi, pi]
        pi = np.pi
        rot_ranges = torch.tensor([2*pi, pi, 2*pi], device=self.device)
        rot_mins = torch.tensor([-pi, -pi/2, -pi], device=self.device)
        
        rot_bin_widths = rot_ranges / self.cfg.num_bins
        
        # Translation Ranges
        t_status = translation_status # [min, max, min, max, min, max]
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

    def sample_noise(self, topk_info, translation_status, base_pose=None):
        """
        Sample x0 based on Stage 1 top-k bins.
        
        Note: DFM always outputs 6D bins (3 Euler angles + 3 translation),
        but ACFM may use different rotation representations (euler/axis_angle/6d).
        
        Args:
            topk_info: dict containing:
                - topk_bins: [bs, 6, k] (DFM output: 3 rot + 3 trans)
                - topk_probs: [bs, 6, k]
            translation_status: list [min, max, min, max, min, max]
            base_pose: [bs, num_dimensions] Optional base pose to compute delta
        
        Returns:
            x0: [bs, num_dimensions] Initial state sampled from bins
                If base_pose is provided, returns (delta_pose, x0_abs)
                where delta_pose = compute_delta_pose(x0_abs, base_pose)
        """
        topk_bins = topk_info['topk_bins']   # [bs, 6, k] from DFM
        topk_probs = topk_info['topk_probs'] # [bs, 6, k]
        
        bs, D, k = topk_bins.shape
        assert D == 6, f"Expected topk_bins to have 6 dimensions from DFM, got {D}"
        
        # 1. Select a bin for each dimension based on probabilities
        probs_sum = topk_probs.sum(dim=-1, keepdim=True)
        norm_probs = topk_probs / (probs_sum + 1e-10)
        
        # Flatten to sample
        flat_probs = norm_probs.view(-1, k) # [bs*6, k]
        selected_indices = torch.multinomial(flat_probs, 1).view(bs, D) # [bs, 6]
        
        # Gather the selected bin indices
        selected_bin_indices = torch.gather(topk_bins, 2, selected_indices.unsqueeze(-1)).squeeze(-1) # [bs, 6]
        
        # 2. Convert bin indices to continuous ranges (Euler angles + translation)
        bin_widths, mins = self._get_bin_stats(translation_status)
        
        # Expand stats for batch
        bin_widths_batch = bin_widths.unsqueeze(0) # [1, 6]
        mins_batch = mins.unsqueeze(0) # [1, 6]
        
        # Calculate range for selected bins
        # lower = min + bin_idx * width
        lower_bound = mins_batch + selected_bin_indices.float() * bin_widths_batch
        
        # 3. Sample uniformly within the bin
        noise = torch.rand(bs, D, device=self.device)
        x0_euler_trans = lower_bound + noise * bin_widths_batch  # [bs, 6]
        
        # 4. Convert to target rotation representation if needed
        if self.rotation_type == '6d':
            # Convert Euler angles to 6D rotation
            euler_angles = x0_euler_trans[:, :3]  # [bs, 3]
            translation = x0_euler_trans[:, 3:]   # [bs, 3]
            
            # Euler -> Rotation Matrix -> 6D
            rot_matrix = pytorch3d_transforms.euler_angles_to_matrix(euler_angles, convention='ZYX')
            rot_6d = pytorch3d_transforms.matrix_to_rotation_6d(rot_matrix)
            
            x0 = torch.cat([rot_6d, translation], dim=1)  # [bs, 9]
        elif self.rotation_type == 'axis_angle':
            # Convert Euler angles to axis-angle
            euler_angles = x0_euler_trans[:, :3]  # [bs, 3]
            translation = x0_euler_trans[:, 3:]   # [bs, 3]
            
            # Euler -> Rotation Matrix -> Axis-Angle
            rot_matrix = pytorch3d_transforms.euler_angles_to_matrix(euler_angles, convention='ZYX')
            axis_angle = pytorch3d_transforms.so3_log_map(rot_matrix)
            
            x0 = torch.cat([axis_angle, translation], dim=1)  # [bs, 6]
        else:  # euler
            x0 = x0_euler_trans  # [bs, 6]
        
        if base_pose is not None:
            # Return delta w.r.t. base_pose (sampled - base_pose)
            return self.compute_delta_pose(x0, base_pose), x0
        
        return x0

    def _predict_features(self, input_feat):
        """
        Extract features from the shared MLP and process through rotation/translation branches.
        
        Args:
            input_feat: [bs, input_dim] Concatenated input features
            
        Returns:
            rotation_feat_enhanced: [bs, 256] Enhanced rotation features
            translation_feat_enhanced: [bs, 256] Enhanced translation features
        """
        # 1. Shared MLP
        shared_feat = self.mlp_shared(input_feat)  # [bs, 512]
        
        # 2. Rotation and translation branches
        rotation_feat = self.rotation_branch(shared_feat)
        translation_feat = self.translation_branch(shared_feat)
        
        # 3. Self-attention blocks
        rotation_feat = self.self_attention_rotation(rotation_feat)
        translation_feat = self.self_attention_translation(translation_feat)
        
        # 4. Conditional attention (cross-branch)
        rotation_feat_enhanced = self.conditional_attention_rotation(
            rotation_feat, translation_feat
        )
        translation_feat_enhanced = self.conditional_attention_translation(
            translation_feat, rotation_feat
        )
        
        return rotation_feat_enhanced, translation_feat_enhanced

    def model_predict(self, x_t, t, pts_feat, coarse_pose_cond=None):
        """
        Predict the velocity field.

        Args:
            x_t: [bs, num_dimensions] current state
            t: [bs] time steps
            pts_feat: [bs, pts_feat_dim] point cloud features
            
        Returns:
            velocity: [bs, num_dimensions] predicted velocity field
        """
        bs = x_t.shape[0]
        
        # Make sure t has the correct shape
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(bs)
        elif t.shape[0] != bs:
            t = t.expand(bs)
        
        # 1. Time embedding
        t_emb = self.time_embedder(t)  # [bs, 256]
        
        # 2. Concatenate all conditioning features
        if self.predict_delta:
            if coarse_pose_cond is None:
                raise ValueError("predict_delta=True requires coarse_pose_cond")
            input_feat = torch.cat([x_t, t_emb, pts_feat, coarse_pose_cond], dim=1)
        else:
            input_feat = torch.cat([x_t, t_emb, pts_feat], dim=1)
        
        # 3. Get enhanced features
        rotation_feat_enhanced, translation_feat_enhanced = self._predict_features(input_feat)
        
        # 4. Predict velocity
        rotation_velocity = self.rotation_head(rotation_feat_enhanced)
        translation_velocity = self.translation_head(translation_feat_enhanced)
        
        velocity = torch.cat([
            rotation_velocity, translation_velocity
        ], dim=1)
        
        return velocity
    
    def sample(self, pts_feat, topk_info, translation_status, step_size=0.01, method='euler'):
        """
        Sample pose from noise (initialized from Top-K bins).

        Args:
            pts_feat: [bs, pts_feat_dim]
            topk_info: dict with topk_bins, topk_probs
            translation_status: list [min, max, min, max, min, max]
            step_size: ODE step size
            method: ODE solver method
            
        Returns:
            pose: [bs, num_dimensions] predicted pose
        """
        class VelocityModelWrapper(ModelWrapper):
            def __init__(self, model, pts_feat, coarse_pose_cond=None):
                super().__init__(model)
                self.pts_feat = pts_feat
                self.coarse_pose_cond = coarse_pose_cond
            
            def forward(self, x, t, **extras):
                vel = self.model.model_predict(x, t, self.pts_feat, coarse_pose_cond=self.coarse_pose_cond)
                return vel
        
        bs = pts_feat.shape[0]
        
        # 1. Initialize sampling state
        if self.predict_delta:
            # Pick a base pose for conditioning
            coarse_pose_sample = self.sample_noise(topk_info, translation_status)
            coarse_pose_cond = self.normalize_pose(coarse_pose_sample, translation_status) if self.use_normalization else coarse_pose_sample
            
            # Initial state x0 is another sample from Top-K bins relative to coarse_pose_sample
            x_init_real, _ = self.sample_noise(topk_info, translation_status, base_pose=coarse_pose_sample)
            x_init = self.normalize_delta_pose(x_init_real, translation_status) if self.use_normalization else x_init_real
        else:
            # Absolute pose mode: initialize x0 from Top-K bins
            x_init = self.sample_noise(topk_info, translation_status)
            coarse_pose_sample = None
            coarse_pose_cond = None

            # Apply normalization if enabled
            if self.use_normalization:
                x_init = self.normalize_pose(x_init, translation_status)
        
        # 2. Initialize ODE solver
        model_wrapper = VelocityModelWrapper(self, pts_feat, coarse_pose_cond=coarse_pose_cond)
        solver = ODESolver(velocity_model=model_wrapper)
        
        # 3. ODE sampling
        result = solver.sample(
            x_init=x_init,
            step_size=step_size,
            method=method,
            time_grid=torch.tensor([0.0, 1.0], device=self.device),
            return_intermediates=False
        )
        
        # 4. Decode to absolute pose
        if self.predict_delta:
            # result is delta (possibly normalized)
            delta = self.denormalize_delta_pose(result, translation_status) if self.use_normalization else result
            pose = self.apply_delta_pose(coarse_pose_sample, delta)
            return pose
        else:
            # result is absolute pose (possibly normalized)
            if self.use_normalization:
                result = self.denormalize_pose(result, translation_status)
            return result

    def loss(self, pose_gt, pts_feat, topk_info, translation_status):
        """Compute continuous flow matching loss with optional normalization."""
        bs = pose_gt.shape[0]
        
        # Sample time uniformly
        t = torch.rand(bs, device=self.device) * (1.0 - self.time_epsilon) + self.time_epsilon
        
        if self.predict_delta:
            # Coarse pose sample is used as conditioning
            coarse_pose_sample = self.sample_noise(topk_info, translation_status)
            # Target is true delta (GT - coarse_pose_sample)
            x_1_real = self.compute_delta_pose(pose_gt, coarse_pose_sample)
            
            # Initial noisy state x0 is another sample from Top-K bins relative to coarse_pose_sample (sampled - coarse)
            x_0_real, _ = self.sample_noise(topk_info, translation_status, base_pose=coarse_pose_sample)

            coarse_pose_cond = self.normalize_pose(coarse_pose_sample, translation_status) if self.use_normalization else coarse_pose_sample
            x_1 = self.normalize_delta_pose(x_1_real, translation_status) if self.use_normalization else x_1_real
            x_0 = self.normalize_delta_pose(x_0_real, translation_status) if self.use_normalization else x_0_real
        else:
            # Sample coarse pose from Top-K bins
            coarse_pose_sample = self.sample_noise(topk_info, translation_status)
            # Target is absolute GT pose, state starts from coarse_pose_sample
            x_0 = coarse_pose_sample
            coarse_pose_cond = None
            if self.use_normalization:
                x_0 = self.normalize_pose(x_0, translation_status)
                x_1 = self.normalize_pose(pose_gt, translation_status)
            else:
                x_1 = pose_gt
        
        # Flow matching in (normalized) space
        path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        x_t = path_sample.x_t
        dx_t = path_sample.dx_t
        
        predicted_velocity = self.model_predict(x_t, t, pts_feat, coarse_pose_cond=coarse_pose_cond)
        
        # Velocity loss (NO time weighting for standard Flow Matching)
        pred_rot_vel = predicted_velocity[:, :self.rotation_dim]
        pred_trans_vel = predicted_velocity[:, self.rotation_dim:]
        true_rot_vel = dx_t[:, :self.rotation_dim]
        true_trans_vel = dx_t[:, self.rotation_dim:]
        
        rot_loss = F.mse_loss(pred_rot_vel, true_rot_vel)
        trans_loss = F.mse_loss(pred_trans_vel, true_trans_vel)
        
        # Pose prediction loss (direct supervision in normalized space)
        pred_pose_normalized = self.path.velocity_to_target_broadcast(
            velocity=predicted_velocity, x_t=x_t, t=t
        )
        pred_rot = pred_pose_normalized[:, :self.rotation_dim]
        pred_trans = pred_pose_normalized[:, self.rotation_dim:]
        true_rot = x_1[:, :self.rotation_dim]
        true_trans = x_1[:, self.rotation_dim:]
        
        # Use L1 loss in normalized space
        rot_pred_loss = F.l1_loss(pred_rot, true_rot)
        trans_pred_loss = F.l1_loss(pred_trans, true_trans)
        
        # === Metric-based loss (angle in degrees, translation in meters) ===
        # Denormalize/compose to get real absolute pose values
        if self.predict_delta:
            pred_delta_real = self.denormalize_delta_pose(pred_pose_normalized, translation_status) if self.use_normalization else pred_pose_normalized
            pred_pose_real = self.apply_delta_pose(coarse_pose_sample, pred_delta_real)
        else:
            pred_pose_real = self.denormalize_pose(pred_pose_normalized, translation_status) if self.use_normalization else pred_pose_normalized
        
        # Compute metric loss using the helper function
        angle_loss, trans_metric_loss, angle_error_deg, trans_error_m = self.compute_metric_loss(
            pred_pose_real, pose_gt
        )
        
        # Loss weights
        pose_pred_weight = 0.5
        metric_weight = self.cfg.metric_loss_weight if hasattr(self.cfg, 'metric_loss_weight') else 0.1
        
        total_loss = (
            self.rotation_weight * rot_loss +
            self.translation_weight * trans_loss +
            pose_pred_weight * (self.rotation_weight * rot_pred_loss + 
                            self.translation_weight * trans_pred_loss) +
            metric_weight * (angle_loss + trans_metric_loss)
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'rot_velocity_loss': rot_loss.item(),
            'trans_velocity_loss': trans_loss.item(),
            'rot_pred_loss': rot_pred_loss.item(),
            'trans_pred_loss': trans_pred_loss.item(),
            'velocity_loss': (rot_loss + trans_loss).item(),
            'pose_pred_loss': (rot_pred_loss + trans_pred_loss).item(),
            'angle_error_deg': angle_error_deg.item(),
            'trans_error_m': trans_error_m.item(),
            'angle_loss': angle_loss.item(),
            'trans_metric_loss': trans_metric_loss.item()
        }
        
        loss_description = (
            f"Total: {total_loss:.4f}, "
            f"Vel: {rot_loss + trans_loss:.4f}, "
            f"Pose: {rot_pred_loss + trans_pred_loss:.4f}, "
            f"Angle: {angle_error_deg:.2f}°, "
            f"Trans: {trans_error_m*100:.2f}cm"
        )
        
        return total_loss, loss_description, loss_dict