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
        
        # Configuration for starting point
        self.use_coarse_as_x0 = cfg.use_coarse_as_x0 if hasattr(cfg, 'use_coarse_as_x0') else False
        if self.use_coarse_as_x0:
            print("AdaptiveContinuousFlowMatching: Using coarse_pose as x_0 (Refinement Mode)")
        else:
            print("AdaptiveContinuousFlowMatching: Using noise as x_0 (Standard Mode)")
        
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
        
        # 2. Top-K condition encoder (uses DFM condition dimensions - always 3D Euler)
        self.topk_encoder = TopKConditionEncoder(
            embed_dim=128,
            k=self.k,
            num_rot_dims=self.cond_rot_dim,  # DFM outputs 3D Euler angles
            num_trans_dims=3
        ).to(device)
        
        # 3. Entropy gate (uses DFM condition dimensions)
        self.entropy_gate = EntropyGate(
            num_dimensions=self.cond_num_dimensions
        ).to(device)
        
        # 4. Entropy encoder (uses DFM condition dimensions)
        self.entropy_embedder = nn.Sequential(
            nn.Linear(self.cond_num_dimensions, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 256)
        ).to(device)
        
        # 5. Exploration vector aggregation (uses DFM condition dimensions)
        self.exploration_aggregator = nn.Sequential(
            nn.Linear(self.cond_num_dimensions * 128, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256)
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
        # Input dim: x_t (num_dimensions) + time_emb (256D) + pts_feat (1024D) +
        #            coarse_pose_emb (256D) + exploration_agg (256D) + entropy_emb (256D)
        input_dim = self.num_dimensions + 256 + self.pts_feat_dim + 256 + 256 + 256
        
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
        """Extract point cloud features."""
        # pts = data['pts']
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
    
    def encode_topk_conditions(self, topk_info):
        """
        Encode Top-K conditions.

        Args:
            topk_info: dict with keys topk_offsets, topk_probs, entropy
            
        Returns:
            condition_dict: dict that contains all encoded conditioning tensors
        """
        topk_offsets = topk_info['topk_offsets']  # [bs, num_dims, k]
        topk_probs = topk_info['topk_probs']      # [bs, num_dims, k]
        entropy = topk_info['entropy']            # [bs, num_dims]
        
        # 1. Encode Top-K into exploration vectors
        exploration_vectors = self.topk_encoder(
            topk_offsets, topk_probs
        )  # [bs, num_dims, 128]
        
        # 2. Apply entropy gate
        gated_exploration = self.entropy_gate(
            entropy, exploration_vectors
        )  # [bs, num_dims, 128]
        
        # 3. Aggregate exploration vectors
        bs = gated_exploration.shape[0]
        exploration_flat = gated_exploration.view(bs, -1)  # [bs, num_dims*128]
        exploration_agg = self.exploration_aggregator(exploration_flat)  # [bs, 256]
        
        # 4. Encode entropy
        entropy_emb = self.entropy_embedder(entropy)  # [bs, 256]
        
        return {
            'exploration_agg': exploration_agg,
            'entropy_emb': entropy_emb,
            'gated_exploration': gated_exploration 
        }
    
    def model_predict(self, x_t, t, pts_feat, coarse_pose, topk_info):
        """
        Predict the velocity field (residual).

        Args:
            x_t: [bs, num_dimensions] current state in residual space
            t: [bs] time steps
            pts_feat: [bs, pts_feat_dim] point cloud features
            coarse_pose: [bs, num_dimensions] coarse pose (continuous)
            topk_info: dict with Top-K information
            
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
        
        # 2. Encode Top-K conditions
        condition_dict = self.encode_topk_conditions(topk_info)
        exploration_agg = condition_dict['exploration_agg']  # [bs, 256]
        entropy_emb = condition_dict['entropy_emb']          # [bs, 256]
        
        # 3. Time embedding
        t_emb = self.time_embedder(t)  # [bs, 256]
        
        # 4. Feature fusion (time with point cloud features)
        cond_proj = self.cond_time_proj(pts_feat)
        t_emb_expanded = t_emb.unsqueeze(1)
        cond_proj_expanded = cond_proj.unsqueeze(1)
        t_emb_enhanced = self.cross_attention_fusion(
            t_emb_expanded, cond_proj_expanded
        ).squeeze(1)
        
        # 5. Concatenate all conditioning features
        # concat(x_t, t_emb, pts_feat, coarse_pose_emb, exploration_agg, entropy_emb)
        input_feat = torch.cat([
            x_t,
            t_emb_enhanced,
            pts_feat,
            coarse_pose_emb,
            exploration_agg,
            entropy_emb
        ], dim=1)
        
        # 6. Shared MLP
        shared_feat = self.mlp_shared(input_feat)  # [bs, 512]
        
        # 7. Rotation and translation branches
        rotation_feat = self.rotation_branch(shared_feat)
        translation_feat = self.translation_branch(shared_feat)
        
        # 8. Self-attention blocks
        rotation_feat = self.self_attention_rotation(rotation_feat)
        translation_feat = self.self_attention_translation(translation_feat)
        
        # 9. Conditional attention (cross-branch)
        rotation_feat_enhanced = self.conditional_attention_rotation(
            rotation_feat, translation_feat
        )
        translation_feat_enhanced = self.conditional_attention_translation(
            translation_feat, rotation_feat
        )
        
        # 10. Predict velocity
        rotation_velocity = self.rotation_head(rotation_feat_enhanced)
        translation_velocity = self.translation_head(translation_feat_enhanced)
        
        predicted_velocity = torch.cat([
            rotation_velocity, translation_velocity
        ], dim=1)
        
        return predicted_velocity
    
    def sample(self, pts_feat, coarse_pose, topk_info, step_size=0.01, method='euler'):
        """
        Sample residual delta from noise (or from coarse pose in refinement mode).

        Args:
            pts_feat: [bs, pts_feat_dim]
            coarse_pose: [bs, num_dimensions] coarse pose
            topk_info: dict with Top-K information
            step_size: ODE step size
            method: ODE solver method
            
        Returns:
            delta: [bs, num_dimensions] predicted residual
        """
        class VelocityModelWrapper(ModelWrapper):
            def __init__(self, model, pts_feat, coarse_pose, topk_info):
                super().__init__(model)
                self.pts_feat = pts_feat
                self.coarse_pose = coarse_pose
                self.topk_info = topk_info
            
            def forward(self, x, t, **extras):
                return self.model.model_predict(
                    x, t, self.pts_feat, self.coarse_pose, self.topk_info
                )
        
        # Initialize ODE solver
        model_wrapper = VelocityModelWrapper(self, pts_feat, coarse_pose, topk_info)
        solver = ODESolver(velocity_model=model_wrapper)
        
        bs = pts_feat.shape[0]
        
        # Choose starting point according to configuration
        if self.use_coarse_as_x0:
            # Refinement Mode: start from coarse pose
            x_init = coarse_pose
        else:
            # Standard Mode: start from Gaussian noise
            x_init = torch.randn(bs, self.num_dimensions, device=self.device)
        
        # ODE sampling
        result = solver.sample(
            x_init=x_init,
            step_size=step_size,
            method=method,
            time_grid=torch.tensor([0.0, 1.0], device=self.device),
            return_intermediates=False
        )
        
        # Always return delta (residual)
        if self.use_coarse_as_x0:
            if self.rotation_type == 'axis_angle':
                # For axis-angle: result is absolute axis-angle, extract residual using log map
                result_axis_angle = result[:, :self.rotation_dim]
                result_trans = result[:, self.rotation_dim:]
                coarse_axis_angle = coarse_pose[:, :self.rotation_dim]
                coarse_trans = coarse_pose[:, self.rotation_dim:]
                
                # Convert to rotation matrices using exponential map
                result_rot_matrix = pytorch3d_transforms.so3_exp_map(result_axis_angle)
                coarse_rot_matrix = pytorch3d_transforms.so3_exp_map(coarse_axis_angle)
                
                # Compute relative rotation: R_delta = R_coarse^T @ R_result
                rot_delta_matrix = torch.matmul(
                    coarse_rot_matrix.transpose(-2, -1),
                    result_rot_matrix
                )
                
                # Convert back to axis-angle using logarithm map
                rot_residual = pytorch3d_transforms.so3_log_map(rot_delta_matrix)
                
                trans_residual = result_trans - coarse_trans
                return torch.cat([rot_residual, trans_residual], dim=1)
            else:
                # For Euler: simple subtraction is fine
                return result - coarse_pose
        else:
            # result itself is the residual
            return result

    def loss(self, delta_gt, pts_feat, coarse_pose, topk_info):
        """
        Compute the continuous flow matching loss.

        Args:
            delta_gt: [bs, num_dimensions] ground truth residual (x_1_gt - coarse_pose)
            pts_feat: [bs, pts_feat_dim]
            coarse_pose: [bs, num_dimensions] coarse pose
            topk_info: dict with Top-K information
            
        Returns:
            total_loss, loss_description
        """
        bs = delta_gt.shape[0]
        
        # Sample time
        t = torch.rand(bs, device=self.device) * (1.0 - self.time_epsilon) + self.time_epsilon
        
        if self.use_coarse_as_x0:
            # Refinement Mode: x_0 = coarse_pose, x_1 = GT pose
            x_0 = coarse_pose
            x_1 = coarse_pose + delta_gt  # GT Pose
        else:
            # Standard Mode: x_0 = noise, x_1 = residual
            x_0 = torch.randn(bs, self.num_dimensions, device=self.device)
            x_1 = delta_gt
        
        # Sample along the path
        path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        x_t = path_sample.x_t
        dx_t = path_sample.dx_t  # True velocity (for AffinePath, always x_1 - x_0 = delta_gt)
        
        # Predicted velocity
        predicted_velocity = self.model_predict(x_t, t, pts_feat, coarse_pose, topk_info)
        
        # Split rotation and translation parts
        pred_rot_vel = predicted_velocity[:, :self.rotation_dim]
        pred_trans_vel = predicted_velocity[:, self.rotation_dim:]
        
        # Scale ground truth to match the output scale
        true_rot_vel = dx_t[:, :self.rotation_dim]
        true_trans_vel = dx_t[:, self.rotation_dim:]
        
        # Time weighting
        #time_weight = torch.sqrt(t.unsqueeze(1) + 0.1)
        time_weight = torch.ones(bs, 1, device=self.device)
        # Velocity loss
        rot_loss_raw = F.l1_loss(pred_rot_vel, true_rot_vel, reduction='none') # [bs, 3]
        trans_loss_raw = F.l1_loss(pred_trans_vel, true_trans_vel, reduction='none') # [bs, 3]
        
        # Apply entropy-weighted loss: higher entropy (higher uncertainty) -> higher weight
        # Note: entropy is always in condition space (3D Euler + 3D trans)
        entropy_weight_desc = ""
        if 'entropy' in topk_info:
            entropy = topk_info['entropy'] # [bs, cond_num_dims] - always 6 (3 Euler + 3 trans)
            rot_entropy = entropy[:, :self.cond_rot_dim]  # [bs, 3]
            trans_entropy = entropy[:, self.cond_rot_dim:]  # [bs, 3]
            
            # For Euler and axis-angle (both 3D), use entropy directly
            rot_entropy_expanded = rot_entropy  # [bs, 3]
            
            # Weight computation: 1 + entropy (simple linear scaling)
            rot_loss_weight = 1.0 + rot_entropy_expanded
            trans_loss_weight = 1.0 + trans_entropy
            
            rot_loss_raw = rot_loss_raw * rot_loss_weight
            trans_loss_raw = trans_loss_raw * trans_loss_weight
            
            entropy_weight_desc = f", MeanEntRot: {rot_entropy.mean().item():.2f}, MeanEntTrans: {trans_entropy.mean().item():.2f}"

        rot_loss = (time_weight * rot_loss_raw).mean()
        trans_loss = (time_weight * trans_loss_raw).mean()
        
        
        pred_target = self.path.velocity_to_target_broadcast(
            velocity=predicted_velocity, x_t=x_t, t=t
        )
        
        if self.use_coarse_as_x0:
            if self.rotation_type == 'axis_angle':
                # For axis-angle: pred_target is absolute axis-angle, extract residual using log map
                pred_target_axis_angle = pred_target[:, :self.rotation_dim]
                pred_target_trans = pred_target[:, self.rotation_dim:]
                coarse_axis_angle = coarse_pose[:, :self.rotation_dim]
                coarse_trans = coarse_pose[:, self.rotation_dim:]
                
                # Convert to rotation matrices using exponential map
                pred_target_rot_matrix = pytorch3d_transforms.so3_exp_map(pred_target_axis_angle)
                coarse_rot_matrix = pytorch3d_transforms.so3_exp_map(coarse_axis_angle)
                
                # Compute relative rotation: R_delta = R_coarse^T @ R_pred
                rot_delta_matrix = torch.matmul(
                    coarse_rot_matrix.transpose(-2, -1),
                    pred_target_rot_matrix
                )
                
                # Convert back to axis-angle using logarithm map
                pred_rot_delta = pytorch3d_transforms.so3_log_map(rot_delta_matrix)
                
                pred_trans_delta = pred_target_trans - coarse_trans
                pred_delta = torch.cat([pred_rot_delta, pred_trans_delta], dim=1)
            else:
                # For Euler: simple subtraction is fine
                pred_delta = pred_target - coarse_pose
        else:
            # pred_target itself is the predicted delta
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
            f"{entropy_weight_desc}"
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
        
        # Add entropy info if available
        if 'entropy' in topk_info:
            entropy = topk_info['entropy']
            loss_dict['mean_entropy_rot'] = entropy[:, :self.rotation_dim].mean().item()
            loss_dict['mean_entropy_trans'] = entropy[:, self.rotation_dim:].mean().item()
            loss_dict['mean_entropy_total'] = entropy.mean().item()
        
        # Add pred_delta statistics for wandb logging
        loss_dict['pred_delta_rot_mean'] = pred_delta_rot.mean().item()
        loss_dict['true_delta_rot_mean'] = true_delta_rot.mean().item()
        loss_dict['pred_delta_trans_mean'] = pred_delta_trans.mean().item()
        loss_dict['true_delta_trans_mean'] = true_delta_trans.mean().item()
        
        return total_loss, loss_description, loss_dict