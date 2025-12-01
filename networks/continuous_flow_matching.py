import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.transforms as pytorch3d_transforms
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from networks.pts_encoder.pointnets import PointNetfeat
from networks.pts_encoder.pointnet2 import Pointnet2ClsMSG
from networks.model_modules import *
from flow_matching.path import CondOTProbPath,AffineProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from utils.metrics import rot_diff_degree


class ContinuousFlowMatching(nn.Module):
    """Continuous Flow Matching network for 6D pose estimation
    """
    
    def __init__(self, cfg, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.eps = 1e-8
        self.time_epsilon =  1e-3
        # 6D rotation: first two columns of rotation matrix (6D) + translation (3D) = 9D
        self.rotation_dim = 6  # 6D rotation representation
        self.translation_dim = 3
        self.num_dimensions = self.rotation_dim + self.translation_dim  # 9D
        
        # Loss weights
        self.velocity_weight = cfg.velocity_weight if hasattr(cfg, 'velocity_weight') else 1.0
        self.rotation_weight = cfg.rotation_weight if hasattr(cfg, 'rotation_weight') else 1.0
        self.translation_weight = cfg.translation_weight if hasattr(cfg, 'translation_weight') else 1.0
        # Weight for pose prediction loss (angle + translation)
        self.pose_prediction_weight = cfg.pose_prediction_weight if hasattr(cfg, 'pose_prediction_weight') else 0.1
        
        # Initialize Conditional OT Flow Matching
        scheduler = PolynomialConvexScheduler(n=1.5)
        self.path = AffineProbPath(scheduler)

        # Point cloud feature extractors
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

        # Timestep embedder
        self.time_embedder = TimestepEmbedder(hidden_size=256, frequency_embedding_size=128).to(device)

        # Attention modules for enhanced feature interaction
        self.cond_time_proj = nn.Linear(self.pts_feat_dim, 256).to(device)
        self.cross_attention_fusion = CrossAttentionFusion(embed_dim=256, num_heads=8, dropout=0.1).to(device)
        self.self_attention_angles = SelfAttentionBlock(embed_dim=256, num_heads=8, dropout=0.1).to(device)
        self.self_attention_trans = SelfAttentionBlock(embed_dim=256, num_heads=8, dropout=0.1).to(device)
        self.conditional_attention_angles = ConditionalAttention(embed_dim=256, condition_dim=256, num_heads=4, dropout=0.1).to(device)
        self.conditional_attention_trans = ConditionalAttention(embed_dim=256, condition_dim=256, num_heads=4, dropout=0.1).to(device)

        # Shared MLP to predict velocity field v(x_t, t, cond)
        # Input: x_t (9D: 6D rotation + 3D translation) + time_embedding (256D) + pts_feat (1024D)
        input_dim = self.num_dimensions + 256 + self.pts_feat_dim
        self.mlp_shared = SharedMLP(input_dim=input_dim, hidden_dim=1024, output_dim=512, dropout=0.1).to(device)

        # Rotation and translation branches (separate R/T heads, inspired by GenPose2)
        self.rotation_branch = PoseBranch(input_dim=512, hidden_dim=384, output_dim=256, dropout=0.1).to(device)
        self.translation_branch = PoseBranch(input_dim=512, hidden_dim=384, output_dim=256, dropout=0.1).to(device)

        # Rotation and translation output heads
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


    def extract_pts_feature(self, data):
        """Extract point cloud features"""
        pts = data['pts']
        if self.cfg.pts_encoder == 'pointnet':
            return self.pts_encoder(pts.permute(0,2,1))
        elif self.cfg.pts_encoder == 'pointnet2':
            return self.pts_encoder(pts)
        elif self.cfg.pts_encoder == 'pointnet_and_pointnet2':
            feat1 = self.pts_pointnet(pts.permute(0,2,1))
            feat2 = self.pts_pointnet2(pts)
            return self.fusion(torch.cat([feat1, feat2], dim=1))
        else:
            raise NotImplementedError

    def sample_noise(self, batch_size):

        return torch.randn(batch_size, self.num_dimensions, device=self.device)

    def model_predict(self, x_t, t, cond):
        """
        Predict velocity field v(x_t, t, cond)
        
        Args:
            x_t: Current state [bs, num_dimensions]
            t: Time step [bs] or scalar
            cond: Point cloud condition features [bs, pts_feat_dim]
            
        Returns:
            predicted_velocity: Predicted velocity field [bs, num_dimensions]
        """
        bs = x_t.shape[0]
        
        # Ensure t has correct shape
        if t.dim() == 0:  # scalar
            t = t.unsqueeze(0).expand(bs)
        elif t.shape[0] != bs:
            t = t.expand(bs)    
        t_emb = self.time_embedder(t)  # [bs, 256]

        # Enhanced feature fusion with attention
        cond_proj = self.cond_time_proj(cond)
        t_emb_expanded = t_emb.unsqueeze(1)  # [bs, 1, 256]
        cond_proj_expanded = cond_proj.unsqueeze(1)  # [bs, 1, 256]
        t_emb_enhanced = self.cross_attention_fusion(t_emb_expanded, cond_proj_expanded).squeeze(1)

        # Concatenate x_t, time embedding, and condition
        input_feat = torch.cat([x_t, t_emb_enhanced, cond], dim=1)
        shared_feat = self.mlp_shared(input_feat)  # [bs, 512]

        # Rotation and translation branches
        rotation_feat = self.rotation_branch(shared_feat)  # [bs, 256]
        trans_feat = self.translation_branch(shared_feat)  # [bs, 256]

        # Self attention for each branch
        rotation_feat = self.self_attention_angles(rotation_feat)
        trans_feat = self.self_attention_trans(trans_feat)

        # Conditional attention between branches (cross-branch interaction)
        rotation_feat_enhanced = self.conditional_attention_angles(rotation_feat, trans_feat)
        trans_feat_enhanced = self.conditional_attention_trans(trans_feat, rotation_feat)

        # Predict velocity using separated heads
        rotation_velocity = self.rotation_head(rotation_feat_enhanced)  # [bs, 6]
        trans_velocity = self.translation_head(trans_feat_enhanced)  # [bs, 3]

        predicted_velocity = torch.cat([rotation_velocity, trans_velocity], dim=1)
        return predicted_velocity

    def sample(self, pts_feat, step_size=0.01, method='euler'):
        """
        Sample 6D pose from noise using ODE solver
        
        Args:
            pts_feat: Point cloud features [bs, pts_feat_dim]
            step_size: Step size for ODE solver
            method: ODE solver method ('euler', 'dopri5', 'midpoint', etc.)
            
        Returns:
            Final sampling result [bs, num_dimensions]
        """
        # Create model wrapper to match flow_matching interface
        class VelocityModelWrapper(ModelWrapper):
            def __init__(self, model, cond):
                super().__init__(model)
                self.cond = cond

            def forward(self, x, t, **extras):

                return self.model.model_predict(x, t, self.cond)

        # Initialize solver
        model_wrapper = VelocityModelWrapper(self, pts_feat)
        solver = ODESolver(velocity_model=model_wrapper)

        # Sample from Gaussian as initial condition (x_0)
        bs = pts_feat.shape[0]
        x_init = self.sample_noise(bs)

        # Sample using the ODE solver
        # Integrate from t=0 to t=1
        result = solver.sample(
            x_init=x_init,
            step_size=step_size,
            method=method,
            time_grid=torch.tensor([0.0, 1.0], device=self.device),
            return_intermediates=False
        )
        return result

    def loss(self, x_1, pts_feat):
        """
        Compute continuous flow matching loss
        
        In Conditional OT Flow Matching:
        - Path: x_t = (1-t) * x_0 + t * x_1
        - True velocity: dx_t/dt = x_1 - x_0
        - Loss: ||v_theta(x_t, t, cond) - (x_1 - x_0)||^2
        
        Args:
            x_1: Target pose [bs, num_dimensions] (9D: 6D rotation + 3D translation, no normalization)
            pts_feat: Point cloud condition features [bs, pts_feat_dim]
            
        Returns:
            total_loss: Total loss
            loss_description: Loss description string
        """
        cond = pts_feat
        bs = x_1.shape[0]

        # Sample time t uniformly from (0,1)
        t = torch.rand(bs, device=self.device) * (1.0 - self.time_epsilon)+ self.time_epsilon
        
        # Sample x_0 from standard Gaussian (source distribution)
        x_0 = self.sample_noise(bs)

        # Sample x_t from the conditional path p_t(x_t|x_0,x_1)
        path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        x_t = path_sample.x_t  # [bs, num_dimensions]
        dx_t = path_sample.dx_t  # True velocity = x_1 - x_0

        # Predict velocity using the model
        predicted_velocity = self.model_predict(x_t, t, cond)  # [bs, num_dimensions]

        # Separate rotation and translation parts
        pred_rot_vel = predicted_velocity[:, :self.rotation_dim]  # [bs, 6]
        pred_trans_vel = predicted_velocity[:, self.rotation_dim:]  # [bs, 3]
        true_rot_vel = dx_t[:, :self.rotation_dim]  # [bs, 6]
        true_trans_vel = dx_t[:, self.rotation_dim:]  # [bs, 3]

        # Time-weighted strategy (inspired by GenPose2's weighting approach)
        # Later time steps (closer to target) have larger weights, focusing more on final prediction quality
        time_weight = torch.sqrt(t.unsqueeze(1) + 0.1)  # [bs, 1]
        
        # Rotation loss (with time weighting)
        rot_loss = F.mse_loss(pred_rot_vel, true_rot_vel, reduction='none')  # [bs, 6]
        rot_loss = (time_weight * rot_loss).mean()
        
        # Translation loss (with time weighting)
        trans_loss = F.mse_loss(pred_trans_vel, true_trans_vel, reduction='none')  # [bs, 3]
        trans_loss = (time_weight * trans_loss).mean()

        # Pose prediction loss: estimate x_1 from x_t and predicted velocity
        pred_x_1 = self.path.velocity_to_target_broadcast(
            velocity=predicted_velocity,
            x_t=x_t,
            t=t
        )
        pred_x_1_rot_6d = pred_x_1[:, :self.rotation_dim]  # [bs, 6]
        pred_x_1_trans = pred_x_1[:, self.rotation_dim:]  # [bs, 3]
        true_x_1_rot_6d = x_1[:, :self.rotation_dim]  # [bs, 6]
        true_x_1_trans = x_1[:, self.rotation_dim:]  # [bs, 3]
        
        # Rotation prediction loss: directly compare 6D vectors
        # Avoid rotation matrix conversion and acos gradient explosion
        # Use smooth L1 loss for better numerical stability
        rot_pred_loss = F.smooth_l1_loss(pred_x_1_rot_6d, true_x_1_rot_6d, reduction='none')  # [bs, 6]
        rot_pred_loss = (time_weight * rot_pred_loss).mean()  # scalar
        
        # Translation prediction loss
        trans_pred_loss = F.smooth_l1_loss(pred_x_1_trans, true_x_1_trans, reduction='none')  # [bs, 3]
        trans_pred_loss = (time_weight * trans_pred_loss).mean()  # scalar
        
        # Pose prediction loss
        pose_pred_loss = rot_pred_loss + trans_pred_loss

        # Total loss (weighted combination of velocity loss and pose prediction loss)
        total_loss = (
            self.rotation_weight * rot_loss + 
            self.translation_weight * trans_loss +
            self.pose_prediction_weight * pose_pred_loss
        )
        loss_description = (
            f"Total: {total_loss:.4f}, "
            f"RotVel: {rot_loss:.4f}, "
            f"TransVel: {trans_loss:.4f}, "
            f"PosePred: {pose_pred_loss:.4f} (Rot6D: {rot_pred_loss:.4f}, Trans: {trans_pred_loss:.4f}), "
        )
        return total_loss, loss_description

