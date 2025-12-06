import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.transforms as pytorch3d_transforms
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
    the continuous residual delta.
    """
    
    def __init__(self, cfg, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.k = cfg.topk_k
        self.eps = 1e-8
        self.time_epsilon = 1e-3
        
        # Rotation representation configuration
        self.rotation_type = cfg.acfm_rotation_type if hasattr(cfg, 'acfm_rotation_type') else 'euler'
        assert self.rotation_type in ['euler', 'axis_angle'], f"Unsupported rotation_type: {self.rotation_type}"
        if self.rotation_type == 'axis_angle':
            self.rotation_dim = 3  # 3D axis-angle (so(3) Lie algebra)
            print("AdaptiveContinuousFlowMatching: Using 3D axis-angle (so(3)) representation")
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
        scheduler = PolynomialConvexScheduler(n=1.5)
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
        
        # === Conditional encoders ===
        # 1. Coarse pose embedding (uses prediction dimensions)
        self.coarse_pose_embedder = CoarsePoseEmbedder(
            num_dimensions=self.num_dimensions,
            embed_dim=256,
            num_frequencies=64
        ).to(device)
        
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
        # Input dim: x_t (num_dimensions) + time_emb (256D) + pts_feat (1024D) + coarse_pose_emb (256D)
        input_dim = self.num_dimensions + 256 + self.pts_feat_dim + 256
        
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
        
        # Output heads
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
        """
        Extract point cloud features.
        
        Args:
            pts: [bs, num_points, 3] point cloud tensor
            
        Returns:
            features: [bs, feat_dim] extracted features
        """
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
    
    def model_predict(self, x_t, t, pts_feat, coarse_pose):
        """
        Predict the velocity field (residual).

        Args:
            x_t: [bs, num_dimensions] current state in residual space
            t: [bs] time steps
            pts_feat: [bs, pts_feat_dim] point cloud features
            coarse_pose: [bs, num_dimensions] coarse pose (continuous)
            
        Returns:
            predicted_velocity: [bs, num_dimensions] predicted velocity field
        """
        bs = x_t.shape[0]
        
        # Make sure t has the correct shape
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(bs)
        elif t.shape[0] != bs:
            t = t.expand(bs)
        
        # 1. Encode coarse pose
        coarse_pose_emb = self.coarse_pose_embedder(coarse_pose)  # [bs, 256]
        
        # 2. Time embedding
        t_emb = self.time_embedder(t)  # [bs, 256]
        
        # 3. Feature fusion (time with point cloud features)
        cond_proj = self.cond_time_proj(pts_feat)
        t_emb_expanded = t_emb.unsqueeze(1)
        cond_proj_expanded = cond_proj.unsqueeze(1)
        t_emb_enhanced = self.cross_attention_fusion(
            t_emb_expanded, cond_proj_expanded
        ).squeeze(1)
        
        # 4. Concatenate all conditioning features
        # concat(x_t, t_emb, pts_feat, coarse_pose_emb)
        input_feat = torch.cat([
            x_t,
            t_emb_enhanced,
            pts_feat,
            coarse_pose_emb
        ], dim=1)
        
        # 5. Shared MLP
        shared_feat = self.mlp_shared(input_feat)  # [bs, 512]
        
        # 6. Rotation and translation branches
        rotation_feat = self.rotation_branch(shared_feat)
        translation_feat = self.translation_branch(shared_feat)
        
        # 7. Self-attention blocks
        rotation_feat = self.self_attention_rotation(rotation_feat)
        translation_feat = self.self_attention_translation(translation_feat)
        
        # 8. Conditional attention (cross-branch)
        rotation_feat_enhanced = self.conditional_attention_rotation(
            rotation_feat, translation_feat
        )
        translation_feat_enhanced = self.conditional_attention_translation(
            translation_feat, rotation_feat
        )
        
        # 9. Predict velocity
        rotation_velocity = self.rotation_head(rotation_feat_enhanced)
        translation_velocity = self.translation_head(translation_feat_enhanced)
        
        predicted_velocity = torch.cat([
            rotation_velocity, translation_velocity
        ], dim=1)
        
        return predicted_velocity
    
    def sample(self, pts_feat, coarse_pose, step_size=0.01, method='euler'):
        """
        Sample residual delta from noise (or from coarse pose in refinement mode).

        Args:
            pts_feat: [bs, pts_feat_dim]
            coarse_pose: [bs, num_dimensions] coarse pose
            step_size: ODE step size
            method: ODE solver method
            
        Returns:
            delta: [bs, num_dimensions] predicted residual
        """
        class VelocityModelWrapper(ModelWrapper):
            def __init__(self, model, pts_feat, coarse_pose):
                super().__init__(model)
                self.pts_feat = pts_feat
                self.coarse_pose = coarse_pose
            
            def forward(self, x, t, **extras):
                return self.model.model_predict(
                    x, t, self.pts_feat, self.coarse_pose
                )
        
        # Initialize ODE solver
        model_wrapper = VelocityModelWrapper(self, pts_feat, coarse_pose)
        solver = ODESolver(velocity_model=model_wrapper)
        
        bs = pts_feat.shape[0]
        
        # Always start from Gaussian noise
        x_init = torch.randn(bs, self.num_dimensions, device=self.device)
        
        # ODE sampling
        result = solver.sample(
            x_init=x_init,
            step_size=step_size,
            method=method,
            time_grid=torch.tensor([0.0, 1.0], device=self.device),
            return_intermediates=False
        )
        
        # Result is the predicted residual
        return result

    def loss(self, delta_gt, pts_feat, coarse_pose):
        """
        Compute the continuous flow matching loss.

        Args:
            delta_gt: [bs, num_dimensions] ground truth residual (x_1_gt - coarse_pose)
            pts_feat: [bs, pts_feat_dim]
            coarse_pose: [bs, num_dimensions] coarse pose
            
        Returns:
            total_loss, loss_description, loss_dict
        """
        bs = delta_gt.shape[0]
        
        # Sample time
        t = torch.rand(bs, device=self.device) * (1.0 - self.time_epsilon) + self.time_epsilon
        
        # x_0 = noise, x_1 = residual
        x_0 = torch.randn(bs, self.num_dimensions, device=self.device)
        x_1 = delta_gt
        
        # Sample along the path
        path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        x_t = path_sample.x_t
        dx_t = path_sample.dx_t  # True velocity (for AffinePath, always x_1 - x_0 = delta_gt)
        
        # Predicted velocity
        predicted_velocity = self.model_predict(x_t, t, pts_feat, coarse_pose)
        
        # Split rotation and translation parts
        pred_rot_vel = predicted_velocity[:, :self.rotation_dim]
        pred_trans_vel = predicted_velocity[:, self.rotation_dim:]
        
        # Scale ground truth to match the output scale
        true_rot_vel = dx_t[:, :self.rotation_dim]
        true_trans_vel = dx_t[:, self.rotation_dim:]
        
        # Time weighting
        time_weight = torch.ones(bs, 1, device=self.device)
        
        # Velocity loss
        rot_loss_raw = F.l1_loss(pred_rot_vel, true_rot_vel, reduction='none') # [bs, 3]
        trans_loss_raw = F.l1_loss(pred_trans_vel, true_trans_vel, reduction='none') # [bs, 3]
        
        rot_loss = (time_weight * rot_loss_raw).mean()
        trans_loss = (time_weight * trans_loss_raw).mean()
        
        
        pred_target = self.path.velocity_to_target_broadcast(
            velocity=predicted_velocity, x_t=x_t, t=t
        )
        
        # pred_target is the predicted delta
        pred_delta = pred_target
            
        pred_delta_rot = pred_delta[:, :self.rotation_dim]
        pred_delta_trans = pred_delta[:, self.rotation_dim:]
        true_delta_rot = delta_gt[:, :self.rotation_dim]
        true_delta_trans = delta_gt[:, self.rotation_dim:]
        
        # Rotation loss: use geodesic distance for axis-angle representation
        if self.rotation_type == 'axis_angle':
            # Convert axis-angle to rotation matrices for geodesic distance computation
            pred_rot_matrix = pytorch3d_transforms.so3_exp_map(pred_delta_rot)  # [bs, 3, 3]
            true_rot_matrix = pytorch3d_transforms.so3_exp_map(true_delta_rot)  # [bs, 3, 3]
            
            # Compute geodesic distance on SO(3) manifold
            # Geodesic distance = arccos((trace(R_pred^T @ R_true) - 1) / 2)
            # For loss, we use: 1 - cos(theta) = 1 - (trace(R_pred^T @ R_true) - 1) / 2
            # This is approximately theta^2 / 2 for small angles
            R_diff = torch.matmul(pred_rot_matrix.transpose(-2, -1), true_rot_matrix)
            trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)  # [bs]
            # Clamp to avoid numerical issues with arccos
            cos_angle = (trace - 1.0) / 2.0
            cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
            
            # Geodesic loss: arccos((trace - 1) / 2)
            geodesic_angle = torch.acos(cos_angle)  # [bs], angle in radians
            rot_pred_loss_raw = geodesic_angle.unsqueeze(1)  # [bs, 1]
            
            # Also add a small L1 regularization on axis-angle space for numerical stability
            axis_angle_l1 = F.l1_loss(pred_delta_rot, true_delta_rot, reduction='none')
            rot_pred_loss = (time_weight[:, :1] * (rot_pred_loss_raw + 0.1 * axis_angle_l1.mean(dim=1, keepdim=True))).mean()
        else:
            # For Euler, use l1 loss
            rot_pred_loss = F.l1_loss(pred_delta_rot, true_delta_rot, reduction='none')
            rot_pred_loss = (time_weight * rot_pred_loss).mean()
        
        trans_pred_loss = F.l1_loss(pred_delta_trans, true_delta_trans, reduction='none')
        trans_pred_loss = (time_weight * trans_pred_loss).mean()
        
        pose_pred_loss = self.rotation_weight*rot_pred_loss + self.translation_weight*trans_pred_loss
        
        total_loss = (
            self.rotation_weight * rot_loss +
            self.translation_weight * trans_loss +
            self.pose_prediction_weight * pose_pred_loss
        )
        
        loss_description = (
            f"Total: {total_loss:.4f}, "
            f"RotVel: {rot_loss:.4f}, "
            f"TransVel: {trans_loss:.4f}, "
            f"DeltaPred: {pose_pred_loss:.4f}"
        )
        
        # Return detailed loss dict for wandb logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'rot_velocity_loss': rot_loss.item(),
            'trans_velocity_loss': trans_loss.item(),
            'rot_pred_loss': rot_pred_loss.item(),
            'trans_pred_loss': trans_pred_loss.item(),
            'pose_pred_loss': pose_pred_loss.item(),
        }
        
        # Add pred_delta statistics for wandb logging
        loss_dict['pred_delta_rot_mean'] = pred_delta_rot.mean().item()
        loss_dict['true_delta_rot_mean'] = true_delta_rot.mean().item()
        loss_dict['pred_delta_trans_mean'] = pred_delta_trans.mean().item()
        loss_dict['true_delta_trans_mean'] = true_delta_trans.mean().item()
        
        return total_loss, loss_description, loss_dict