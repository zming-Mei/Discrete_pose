import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from networks.pts_encoder.pointnets import PointNetfeat
from networks.pts_encoder.pointnet2 import Pointnet2ClsMSG
from networks.model_modules import *
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper


class DiscreteFlowMatching(nn.Module):
    """Discrete Flow Matching network for 6D pose estimation"""
    
    def __init__(self, cfg, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.num_bins = cfg.num_bins
        self.angle_dimensions = 3
        self.translation_dimensions = 3
        self.num_dimensions = self.angle_dimensions + self.translation_dimensions
        self.device = device
        self.eps = 1e-8
        
        # Loss weights (from configuration)
        self.mse_weight = cfg.mse_weight
        self.kl_weight = cfg.kl_weight
        self.L1_weight = cfg.L1_weight

        # Initialize flow matching components
        scheduler = PolynomialConvexScheduler(n=2)  # Polynomial scheduler with n=2
        self.path = MixtureDiscreteProbPath(scheduler=scheduler)
        self.criterion = MixturePathGeneralizedKL(path=self.path)

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

        # Shared MLP to predict posterior probability p(x_1|x_t)
        input_dim = self.num_dimensions * self.num_bins + 256 + self.pts_feat_dim
        self.mlp_shared = SharedMLP(input_dim=input_dim, hidden_dim=1024, output_dim=512, dropout=0.1).to(device)

        # Angle and translation branches
        self.angles_branch = PoseBranch(input_dim=512, hidden_dim=384, output_dim=256, dropout=0.1).to(device)
        self.translation_branch = PoseBranch(input_dim=512, hidden_dim=384, output_dim=256, dropout=0.1).to(device)

        # Output heads to predict posterior logits p(x_1^i | x_t)
        self.angle_heads = nn.ModuleList([
            PredictionHead(input_dim=256, hidden_dim=256, num_bins=self.num_bins)
            for _ in range(self.angle_dimensions)
        ]).to(device)

        self.translation_heads = nn.ModuleList([
            PredictionHead(input_dim=256, hidden_dim=256, num_bins=self.num_bins)
            for _ in range(self.translation_dimensions)
        ]).to(device)


    def extract_pts_feature(self, data):

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
        return torch.randint(0, self.num_bins, (batch_size, self.num_dimensions), device=self.device)

    def model_predict(self, x_t, t, cond):
        bs = x_t.shape[0]
        x_t_onehot = F.one_hot(x_t, num_classes=self.num_bins).float()
        x_t_flat = x_t_onehot.view(bs, -1)
        t_emb = self.time_embedder(t)  # [bs, 256]

        # Enhanced feature fusion with attention
        cond_proj = self.cond_time_proj(cond)
        t_emb_expanded = t_emb.unsqueeze(1)  # [bs, 1, 256]
        cond_proj_expanded = cond_proj.unsqueeze(1)  # [bs, 1, 256]
        t_emb_enhanced = self.cross_attention_fusion(t_emb_expanded, cond_proj_expanded).squeeze(1)

        input_feat = torch.cat([x_t_flat, t_emb_enhanced, cond], dim=1)

        shared_feat = self.mlp_shared(input_feat)  # [bs, 512]

        # Angle and translation branches
        angles_feat = self.angles_branch(shared_feat)  # [bs, 256]
        trans_feat = self.translation_branch(shared_feat)  # [bs, 256]

        # Self attention for each branch
        angles_feat = self.self_attention_angles(angles_feat)
        trans_feat = self.self_attention_trans(trans_feat)

        # Conditional attention between branches (cross-branch interaction)
        angles_feat_enhanced = self.conditional_attention_angles(angles_feat, trans_feat)
        trans_feat_enhanced = self.conditional_attention_trans(trans_feat, angles_feat)

        # Predict posterior logits for each dimension
        angle_logits = [head(angles_feat_enhanced) for head in self.angle_heads]
        trans_logits = [head(trans_feat_enhanced) for head in self.translation_heads]

        # Stack logits: [bs, num_dimensions, num_bins]
        posterior_logits = torch.stack(angle_logits + trans_logits, dim=1)
        return posterior_logits

    def sample(self, pts_feat, step_size=0.01):
        """Sample using discrete flow matching solver
        
        Args:
            pts_feat: Point cloud features [bs, pts_feat_dim]
            step_size: Sampling step size
            
        Returns:
            Sampling results [bs, num_dimensions]
        """
        # Create model wrapper to match flow_matching interface
        class FlowMatchingWrapper(ModelWrapper):
            def __init__(self, model, cond):
                super().__init__(model)
                self.cond = cond

            def forward(self, x, t, **extras):
                # x: [batch_size, num_dimensions]
                # t: [batch_size] or scalar
                # cond: [batch_size, pts_feat_dim]
                logits = self.model.model_predict(x, t, self.cond)
                return torch.softmax(logits, dim=-1)

        # Initialize solver
        model_wrapper = FlowMatchingWrapper(self, pts_feat)
        solver = MixtureDiscreteEulerSolver(
            model=model_wrapper,
            path=self.path,
            vocabulary_size=self.num_bins
        )

        # Sample from uniform distribution as initial condition (x_0)
        bs = pts_feat.shape[0]
        x_init = self.sample_noise(bs)

        # Sample using the solver
        result = solver.sample(
            x_init=x_init,
            step_size=step_size,
            time_grid=torch.tensor([0.0, 1.0], device=self.device)
        )

        return result

    def loss(self, x_1, pts_feat):
        cond = pts_feat
        bs = x_1.shape[0]

        time_epsilon = 1e-3 
        # Sample time t uniformly from [0,1]
        t = torch.rand(bs, device=self.device) * (1.0 - time_epsilon)
        # Sample x_0 from uniform distribution (source distribution)
        x_0 = self.sample_noise(bs)

        # Sample x_t from the path p_t(x_t|x_0,x_1)
        path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        x_t = path_sample.x_t

        # Predict posterior logits p(x_1|x_t)
        posterior_logits = self.model_predict(x_t, t, cond)

        # Compute GeneralizedKL loss (following discrete diffusion model)
        kl_loss = self.criterion(
            logits=posterior_logits,
            x_1=x_1,
            x_t=x_t,
            t=t
        )

        # Separate losses for angles and translations
        angle_logits = posterior_logits[:, :self.angle_dimensions]
        trans_logits = posterior_logits[:, self.angle_dimensions:]
        angle_x1 = x_1[:, :self.angle_dimensions]
        angle_xt = x_t[:, :self.angle_dimensions]
        trans_x1 = x_1[:, self.angle_dimensions:]
        trans_xt = x_t[:, self.angle_dimensions:]

        angle_kl_loss = self.criterion(angle_logits, angle_x1, angle_xt, t)
        trans_kl_loss = self.criterion(trans_logits, trans_x1, trans_xt, t)

        # Compute MSE loss (following discrete diffusion model implementation)
        # Convert softmax probabilities to expected bin values
        probs = F.softmax(posterior_logits, dim=-1)  # [bs, num_dimensions, num_bins]
        bin_indices = torch.arange(self.num_bins, device=probs.device).float()  # [num_bins]
        pred_bins = torch.sum(probs * bin_indices, dim=-1)  # [bs, num_dimensions]
        true_bins = x_1.float()  # [bs, num_dimensions]
        
        mse_loss = F.mse_loss(pred_bins, true_bins)
        L1_loss = F.l1_loss(pred_bins, true_bins)

        # Combined loss (following discrete diffusion model weight configuration)
        total_loss = self.kl_weight * kl_loss + self.mse_weight * mse_loss + self.L1_weight * L1_loss

        loss_description = (
            f"KL Loss: {self.kl_weight * kl_loss:.4f}, "
            f"MSE Loss: {self.mse_weight * mse_loss:.4f}, "
            f"L1 Loss: {self.L1_weight * L1_loss:.4f}, "
            f"Angle KL: {angle_kl_loss:.4f}, "
            f"Trans KL: {trans_kl_loss:.4f}"
        )

        return total_loss, loss_description
