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
        scheduler = PolynomialConvexScheduler(n=2)
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
        # Input dim: x_t (num_dimensions) + time_emb (256D) + pts_feat (1024D)
        input_dim = self.num_dimensions + 256 + self.pts_feat_dim
        
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

    def sample_noise(self, topk_info, translation_status):
        """
        Sample x0 based on Stage 1 top-k bins.
        
        Note: DFM always outputs 6D bins (3 Euler angles + 3 translation),
        but ACFM may use different rotation representations (euler/axis_angle/6d).
        
        Args:
            topk_info: dict containing:
                - topk_bins: [bs, 6, k] (DFM output: 3 rot + 3 trans)
                - topk_probs: [bs, 6, k]
            translation_status: list [min, max, min, max, min, max]
        
        Returns:
            x0: [bs, num_dimensions] Initial state sampled from bins
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

    def model_predict(self, x_t, t, pts_feat):
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
        input_feat = torch.cat([
            x_t,
            t_emb,
            pts_feat
        ], dim=1)
        
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
            def __init__(self, model, pts_feat):
                super().__init__(model)
                self.pts_feat = pts_feat
            
            def forward(self, x, t, **extras):
                vel = self.model.model_predict(x, t, self.pts_feat)
                return vel
        
        bs = pts_feat.shape[0]
        
        # 1. Initialize x0 from Top-K bins
        x_init = self.sample_noise(topk_info, translation_status)
        
        # 2. Initialize ODE solver
        model_wrapper = VelocityModelWrapper(self, pts_feat)
        solver = ODESolver(velocity_model=model_wrapper)
        
        # 3. ODE sampling
        result = solver.sample(
            x_init=x_init,
            step_size=step_size,
            method=method,
            time_grid=torch.tensor([0.0, 1.0], device=self.device),
            return_intermediates=False
        )
        
        return result

    def loss(self, pose_gt, pts_feat, topk_info, translation_status):
        """Compute continuous flow matching loss WITHOUT time weighting."""
        bs = pose_gt.shape[0]
        
        # Sample time uniformly
        t = torch.rand(bs, device=self.device) * (1.0 - self.time_epsilon) + self.time_epsilon
        
        # Sample path
        x_0 = self.sample_noise(topk_info, translation_status)
        x_1 = pose_gt
        path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        x_t = path_sample.x_t
        dx_t = path_sample.dx_t
        
        predicted_velocity = self.model_predict(x_t, t, pts_feat)
        
        # Velocity loss (NO time weighting for standard Flow Matching)
        pred_rot_vel = predicted_velocity[:, :self.rotation_dim]
        pred_trans_vel = predicted_velocity[:, self.rotation_dim:]
        true_rot_vel = dx_t[:, :self.rotation_dim]
        true_trans_vel = dx_t[:, self.rotation_dim:]
        
        rot_loss = F.mse_loss(pred_rot_vel, true_rot_vel)
        trans_loss = F.mse_loss(pred_trans_vel, true_trans_vel)
        
        # Pose prediction loss (direct supervision)
        pred_pose = self.path.velocity_to_target_broadcast(
            velocity=predicted_velocity, x_t=x_t, t=t
        )
        pred_rot = pred_pose[:, :self.rotation_dim]
        pred_trans = pred_pose[:, self.rotation_dim:]
        true_rot = x_1[:, :self.rotation_dim]
        true_trans = x_1[:, self.rotation_dim:]
        
        # Use L2 loss for better gradient stability
        rot_pred_loss = F.mse_loss(pred_rot, true_rot)
        trans_pred_loss = F.mse_loss(pred_trans, true_trans)
        
        # Increased pose prediction weight
        pose_pred_weight = 0.5  # Increased from 0.1
        
        total_loss = (
            self.rotation_weight * rot_loss +
            self.translation_weight * trans_loss +
            pose_pred_weight * (self.rotation_weight * rot_pred_loss + 
                            self.translation_weight * trans_pred_loss)
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'rot_velocity_loss': rot_loss.item(),
            'trans_velocity_loss': trans_loss.item(),
            'rot_pred_loss': rot_pred_loss.item(),
            'trans_pred_loss': trans_pred_loss.item(),
            'velocity_loss': (rot_loss + trans_loss).item(),
            'pose_pred_loss': (rot_pred_loss + trans_pred_loss).item()
        }
        
        loss_description = (
            f"Total: {total_loss:.4f}, "
            f"Vel: {rot_loss + trans_loss:.4f}, "
            f"Pose: {rot_pred_loss + trans_pred_loss:.4f}"
        )
        
        return total_loss, loss_description, loss_dict