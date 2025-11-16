import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import math
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from networks.pts_encoder.pointnets import PointNetfeat
from networks.pts_encoder.pointnet2 import Pointnet2ClsMSG
from configs.config import get_config
from utils.metrics import rot_diff_degree
from networks.gf_algorithms.discrete_angle import *
from datasets.dataloader import get_data_loaders_from_cfg, process_batch
from networks.gf_algorithms.discrete_number import *

# Import flow matching components
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler, ConvexScheduler
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels)
        )
        if not self.same_channels:
            self.proj = nn.Linear(in_channels, out_channels)
        
    def forward(self, x):
        if self.same_channels:
            return x + self.net(x)
        else:
            return self.proj(x) + self.net(x)
        
class DiscreteFlowMatching(nn.Module):
    def __init__(self, cfg, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.num_bins = cfg.num_bins
        self.angle_dimensions = 3
        self.translation_dimensions = 3
        self.num_dimensions = self.angle_dimensions + self.translation_dimensions
        self.device = device
        self.eps = 1e-8
        
        # Loss weights (following discrete diffusion model configuration)
        self.mse_weight = 0.01
        self.kl_weight = 0.2
        self.L1_weight = 1

        # Initialize flow matching components
        scheduler = PolynomialConvexScheduler(n=1.5)  # Polynomial scheduler with n=1
        self.path = MixtureDiscreteProbPath(scheduler=scheduler)
        self.criterion = MixturePathGeneralizedKL(path=self.path)

        # Point Cloud Feature Extractors
        if self.cfg.pts_encoder == 'pointnet':
            self.pts_encoder = PointNetfeat(num_points=self.cfg.num_points, out_dim=1024)
        elif self.cfg.pts_encoder == 'pointnet2':
            self.pts_encoder = Pointnet2ClsMSG(0)
        elif self.cfg.pts_encoder == 'pointnet_and_pointnet2':
            self.pts_pointnet = PointNetfeat(num_points=self.cfg.num_points, out_dim=1024)
            self.pts_pointnet2 = Pointnet2ClsMSG(0)
            self.fusion = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU()
            )
        else:
            raise NotImplementedError

        self.time_embedder = TimestepEmbedder(hidden_size=256, frequency_embedding_size=128).to(device)

        # Flow matching predicts posterior p(x_1|x_t) which will be converted to velocity
        self.mlp_shared = nn.Sequential(
            nn.Linear(self.num_dimensions*self.num_bins + 256 + 1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            ResidualBlock(1024, 1024),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(p=0.2)
        ).to(device)

        self.angles_branch = nn.Sequential(
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(p=0.1)
        ).to(device)

        self.translation_branch = nn.Sequential(
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(p=0.1)
        ).to(device)

        # Output heads predict posterior logits p(x_1^i | x_t)
        self.angle_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Linear(256, self.num_bins)
            ) for _ in range(self.angle_dimensions)
        ]).to(device)

        self.translation_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Linear(256, self.num_bins)
            ) for _ in range(self.translation_dimensions)
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
        """Sample noise x1 from uniform distribution"""
        return torch.randint(0, self.num_bins, (batch_size, self.num_dimensions), device=self.device)



    def model_predict(self, x_t, t, cond):
        """Predict posterior logits p(x_1 | x_t)"""
        bs = x_t.shape[0]
        x_t_onehot = F.one_hot(x_t, num_classes=self.num_bins).float()
        x_t_flat = x_t_onehot.view(bs, -1)
        t_emb = self.time_embedder(t)       # [bs, 256]
        input_feat = torch.cat([x_t_flat, t_emb, cond], dim=1)

        shared_feat = self.mlp_shared(input_feat)  # [bs, 512]

        angles_feat = self.angles_branch(shared_feat)  # [bs, 256]
        trans_feat = self.translation_branch(shared_feat)  # [bs, 256]

        # Predict posterior logits for each dimension
        angle_logits = [head(angles_feat) for head in self.angle_heads]
        trans_logits = [head(trans_feat) for head in self.translation_heads]

        # Stack logits: [bs, num_dimensions, num_bins]
        posterior_logits = torch.stack(angle_logits + trans_logits, dim=1)
        return posterior_logits


    def sample(self, pts_feat, step_size=0.01):
        """Sample using discrete flow matching solver"""
        # Create a wrapper for the model that matches the flow_matching interface
        class FlowMatchingWrapper(ModelWrapper):
            def __init__(self, model, cond):
                super().__init__(model)
                self.cond = cond

            def forward(self, x, t, **extras):
                # x: [batch_size, num_dimensions]
                # t: [batch_size] or scalar
                # cond: [batch_size, 1024]
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
        """Flow matching loss using mixed MSE + GeneralizedKL divergence"""
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
