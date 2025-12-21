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
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper


class AdaptiveDiscreteFlowMatching(nn.Module):
    """
    Adaptive Discrete Flow Matching network (Stage 2).

    Given the coarse pose from DFM and Top-K conditions, this network predicts
    the refined discrete pose using discrete flow matching.
    
    Key idea: Hierarchical bin refinement
    - Stage 1 (DFM): Predicts coarse bins (e.g., 36 bins covering full range)
    - Stage 2 (ADFM): Predicts fine bins WITHIN the Top-K coarse bins
    - Fine bins are distributed non-uniformly according to Top-K probabilities
    
    Example:
    - Stage 1: 36 coarse bins, Top-5 bins with probs [0.4, 0.3, 0.15, 0.1, 0.05]
    - Stage 2: 36 fine bins distributed as:
      - Coarse bin 1 (prob=0.4): 14 fine bins within its range
      - Coarse bin 2 (prob=0.3): 11 fine bins within its range
      - Coarse bin 3 (prob=0.15): 5 fine bins within its range
      - etc.
    """
    
    def __init__(self, cfg, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        # Coarse bins from Stage 1 (DFM)
        self.num_coarse_bins = cfg.num_bins  # e.g., 36
        # Fine bins for Stage 2 (ADFM) - can be different from coarse
        self.num_fine_bins = cfg.num_fine_bins if hasattr(cfg, 'num_fine_bins') else cfg.num_bins
        
        self.angle_dimensions = 3
        self.translation_dimensions = 3
        self.num_dimensions = self.angle_dimensions + self.translation_dimensions
        self.eps = 1e-8
        self.time_epsilon = 1e-3
        
        # Loss weights (from configuration)
        self.mse_weight = cfg.mse_weight if hasattr(cfg, 'mse_weight') else 1.0
        self.kl_weight = cfg.kl_weight if hasattr(cfg, 'kl_weight') else 1.0
        self.L1_weight = cfg.L1_weight if hasattr(cfg, 'L1_weight') else 1.0
        
        print(f"AdaptiveDiscreteFlowMatching: Hierarchical bin refinement")
        print(f"  - Coarse bins (Stage 1): {self.num_coarse_bins}")
        print(f"  - Fine bins (Stage 2): {self.num_fine_bins}")
        print(f"  - Loss weights - KL: {self.kl_weight}, MSE: {self.mse_weight}, L1: {self.L1_weight}")
        
        # Initialize flow matching components (same as DFM)
        scheduler = PolynomialConvexScheduler(n=2)  # Polynomial scheduler with n=2
        self.path = MixtureDiscreteProbPath(scheduler=scheduler)
        self.criterion = MixturePathGeneralizedKL(path=self.path)
        
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
        
        # === Time embedding ===
        self.time_embedder = TimestepEmbedder(
            hidden_size=256,
            frequency_embedding_size=128
        ).to(device)
        
        # === Feature fusion with attention ===
        self.cond_time_proj = nn.Linear(self.pts_feat_dim, 256).to(device)
        self.cross_attention_fusion = CrossAttentionFusion(
            embed_dim=256,
            num_heads=8,
            dropout=0.1
        ).to(device)
        
        # === Condition embedding for Top-K info ===
        # Embed topk_bins and topk_probs as additional condition
        self.topk_k = cfg.topk_k if hasattr(cfg, 'topk_k') else 10
        # TopK condition: [num_dimensions, k] bins + [num_dimensions, k] probs
        self.topk_cond_dim = self.num_dimensions * self.topk_k * 2
        self.topk_cond_proj = nn.Sequential(
            nn.Linear(self.topk_cond_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256)
        ).to(device)
        
        # === Main network ===
        # Input: x_t one-hot (num_dimensions * num_fine_bins) + time_emb (256) + pts_feat (1024) + topk_cond (256)
        input_dim = self.num_dimensions * self.num_fine_bins + 256 + self.pts_feat_dim + 256
        
        self.mlp_shared = SharedMLP(
            input_dim=input_dim,
            hidden_dim=1024,
            output_dim=512,
            dropout=0.1
        ).to(device)
        
        # Angle and translation branches
        self.angles_branch = PoseBranch(
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
        
        # Self-attention blocks
        self.self_attention_angles = SelfAttentionBlock(
            embed_dim=256,
            num_heads=8,
            dropout=0.1
        ).to(device)
        
        self.self_attention_trans = SelfAttentionBlock(
            embed_dim=256,
            num_heads=8,
            dropout=0.1
        ).to(device)
        
        # Conditional attention (cross-branch interaction)
        self.conditional_attention_angles = ConditionalAttention(
            embed_dim=256,
            condition_dim=256,
            num_heads=4,
            dropout=0.1
        ).to(device)
        
        self.conditional_attention_trans = ConditionalAttention(
            embed_dim=256,
            condition_dim=256,
            num_heads=4,
            dropout=0.1
        ).to(device)
        
        # Output heads to predict posterior logits p(x_1^i | x_t) over fine bins
        self.angle_heads = nn.ModuleList([
            PredictionHead(input_dim=256, hidden_dim=256, num_bins=self.num_fine_bins)
            for _ in range(self.angle_dimensions)
        ]).to(device)
        
        self.translation_heads = nn.ModuleList([
            PredictionHead(input_dim=256, hidden_dim=256, num_bins=self.num_fine_bins)
            for _ in range(self.translation_dimensions)
        ]).to(device)

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

    def sample_noise_uniform(self, batch_size):
        """Sample uniform random fine bin indices (used as fallback)."""
        return torch.randint(0, self.num_fine_bins, (batch_size, self.num_dimensions), device=self.device)
    
    def _build_fine_bin_allocation(self, topk_probs):
        """
        Build the allocation of fine bins to coarse bins based on probabilities.
        
        Args:
            topk_probs: [bs, num_dimensions, k] Top-K probabilities
            
        Returns:
            fine_bins_per_coarse: [bs, num_dimensions, k] Number of fine bins allocated to each coarse bin
            cumsum_fine_bins: [bs, num_dimensions, k] Cumulative sum of fine bins (for indexing)
        """
        bs, D, k = topk_probs.shape
        
        # Normalize probabilities
        probs_sum = topk_probs.sum(dim=-1, keepdim=True)
        norm_probs = topk_probs / (probs_sum + 1e-10)  # [bs, D, k]
        
        # Allocate fine bins proportionally to probabilities
        # At least 1 fine bin per coarse bin to ensure coverage
        raw_allocation = norm_probs * self.num_fine_bins  # [bs, D, k]
        
        # Round and ensure minimum of 1
        fine_bins_per_coarse = torch.clamp(torch.round(raw_allocation), min=1).long()  # [bs, D, k]
        
        # Adjust to ensure total equals num_fine_bins
        total_allocated = fine_bins_per_coarse.sum(dim=-1, keepdim=True)  # [bs, D, 1]
        diff = self.num_fine_bins - total_allocated  # [bs, D, 1]
        
        # Add/subtract from the highest probability bin
        fine_bins_per_coarse[:, :, 0] = fine_bins_per_coarse[:, :, 0] + diff.squeeze(-1)
        fine_bins_per_coarse = torch.clamp(fine_bins_per_coarse, min=1)
        
        # Compute cumulative sum for indexing
        cumsum_fine_bins = torch.cumsum(fine_bins_per_coarse, dim=-1)  # [bs, D, k]
        
        return fine_bins_per_coarse, cumsum_fine_bins
    
    def sample_noise_from_topk(self, topk_info):
        """
        Sample x0 (fine bin indices) based on Stage 1 top-k bins with hierarchical refinement.
        
        The sampling strategy (Hierarchical bin refinement):
        1. Allocate num_fine_bins to Top-K coarse bins proportionally to their probabilities
        2. For each dimension, first select a coarse bin according to probabilities
        3. Then sample a fine bin uniformly within that coarse bin's allocated range
        
        Example:
        - num_fine_bins = 36, Top-5 probs = [0.4, 0.3, 0.15, 0.1, 0.05]
        - Allocation: [14, 11, 5, 4, 2] fine bins for each coarse bin
        - Fine bin ranges: [0-13], [14-24], [25-29], [30-33], [34-35]
        - If coarse bin 2 is selected, sample uniformly from [14-24]
        
        Args:
            topk_info: dict containing:
                - topk_bins: [bs, num_dimensions, k] Top-K coarse bin indices per dimension
                - topk_probs: [bs, num_dimensions, k] Top-K probabilities per dimension
        
        Returns:
            x0: [bs, num_dimensions] Initial fine bin indices
        """
        topk_bins = topk_info['topk_bins']   # [bs, num_dimensions, k]
        topk_probs = topk_info['topk_probs'] # [bs, num_dimensions, k]
        
        bs, D, k = topk_bins.shape
        
        # Build fine bin allocation
        fine_bins_per_coarse, cumsum_fine_bins = self._build_fine_bin_allocation(topk_probs)
        
        # Normalize probabilities for sampling
        probs_sum = topk_probs.sum(dim=-1, keepdim=True)
        norm_probs = topk_probs / (probs_sum + 1e-10)  # [bs, D, k]
        
        # Step 1: Sample which coarse bin to use for each (batch, dimension)
        flat_probs = norm_probs.view(-1, k)  # [bs * D, k]
        selected_coarse_idx = torch.multinomial(flat_probs, num_samples=1).view(bs, D)  # [bs, D]
        
        # Step 2: Get the fine bin range for each selected coarse bin
        # Start index = cumsum[i-1] (or 0 if i=0)
        # End index = cumsum[i] - 1
        
        # Gather cumsum for selected indices
        selected_cumsum = torch.gather(cumsum_fine_bins, dim=2, 
                                       index=selected_coarse_idx.unsqueeze(-1)).squeeze(-1)  # [bs, D]
        selected_allocation = torch.gather(fine_bins_per_coarse, dim=2,
                                          index=selected_coarse_idx.unsqueeze(-1)).squeeze(-1)  # [bs, D]
        
        # Fine bin start = cumsum - allocation
        fine_bin_start = selected_cumsum - selected_allocation  # [bs, D]
        
        # Step 3: Sample uniformly within the fine bin range
        # random offset in [0, allocation)
        random_offset = (torch.rand(bs, D, device=self.device) * selected_allocation.float()).long()
        random_offset = torch.clamp(random_offset, max=selected_allocation - 1)
        
        x0 = fine_bin_start + random_offset  # [bs, D]
        
        return x0
    
    def fine_bin_to_continuous(self, fine_bins, topk_info, translation_status):
        """
        Convert fine bin indices to continuous values.
        
        Each fine bin corresponds to a sub-range within a coarse bin.
        
        Args:
            fine_bins: [bs, num_dimensions] Fine bin indices
            topk_info: dict with topk_bins and topk_probs
            translation_status: [x_min, x_max, y_min, y_max, z_min, z_max]
            
        Returns:
            continuous_values: [bs, num_dimensions] Continuous pose values
        """
        import math
        
        topk_bins = topk_info['topk_bins']   # [bs, D, k]
        topk_probs = topk_info['topk_probs'] # [bs, D, k]
        
        bs, D, k = topk_bins.shape
        
        # Build fine bin allocation
        fine_bins_per_coarse, cumsum_fine_bins = self._build_fine_bin_allocation(topk_probs)
        
        # Define coarse bin ranges for each dimension
        # Rotation: Euler angles
        angle_ranges = [
            (-math.pi, math.pi),      # Z
            (-math.pi/2, math.pi/2),  # Y
            (-math.pi, math.pi)       # X
        ]
        # Translation ranges from status
        trans_ranges = [
            (translation_status[0], translation_status[1]),  # x
            (translation_status[2], translation_status[3]),  # y
            (translation_status[4], translation_status[5])   # z
        ]
        all_ranges = angle_ranges + trans_ranges
        
        continuous_values = torch.zeros(bs, D, device=self.device)
        
        for d in range(D):
            range_min, range_max = all_ranges[d]
            coarse_bin_width = (range_max - range_min) / self.num_coarse_bins
            
            for b in range(bs):
                fine_bin = fine_bins[b, d].item()
                
                # Find which coarse bin this fine bin belongs to
                cumsum = cumsum_fine_bins[b, d].cpu().numpy()
                allocation = fine_bins_per_coarse[b, d].cpu().numpy()
                
                coarse_idx = 0
                for i in range(k):
                    if fine_bin < cumsum[i]:
                        coarse_idx = i
                        break
                
                # Get the coarse bin index from Stage 1
                coarse_bin = topk_bins[b, d, coarse_idx].item()
                
                # Get fine bin's position within this coarse bin
                fine_start = 0 if coarse_idx == 0 else cumsum[coarse_idx - 1]
                fine_offset = fine_bin - fine_start
                num_fine_in_coarse = allocation[coarse_idx]
                
                # Coarse bin range
                coarse_start = range_min + coarse_bin * coarse_bin_width
                coarse_end = coarse_start + coarse_bin_width
                
                # Fine bin value (center of the fine bin)
                fine_bin_width = (coarse_end - coarse_start) / num_fine_in_coarse
                value = coarse_start + (fine_offset + 0.5) * fine_bin_width
                
                continuous_values[b, d] = value
        
        return continuous_values
    
    def continuous_to_fine_bin(self, continuous_values, topk_info, translation_status):
        """
        Convert continuous values to fine bin indices.
        
        Args:
            continuous_values: [bs, num_dimensions] Continuous pose values
            topk_info: dict with topk_bins and topk_probs
            translation_status: [x_min, x_max, y_min, y_max, z_min, z_max]
            
        Returns:
            fine_bins: [bs, num_dimensions] Fine bin indices
        """
        import math
        
        topk_bins = topk_info['topk_bins']   # [bs, D, k]
        topk_probs = topk_info['topk_probs'] # [bs, D, k]
        
        bs, D, k = topk_bins.shape
        
        # Build fine bin allocation
        fine_bins_per_coarse, cumsum_fine_bins = self._build_fine_bin_allocation(topk_probs)
        
        # Define coarse bin ranges
        angle_ranges = [
            (-math.pi, math.pi),
            (-math.pi/2, math.pi/2),
            (-math.pi, math.pi)
        ]
        trans_ranges = [
            (translation_status[0], translation_status[1]),
            (translation_status[2], translation_status[3]),
            (translation_status[4], translation_status[5])
        ]
        all_ranges = angle_ranges + trans_ranges
        
        fine_bins = torch.zeros(bs, D, dtype=torch.long, device=self.device)
        
        for d in range(D):
            range_min, range_max = all_ranges[d]
            coarse_bin_width = (range_max - range_min) / self.num_coarse_bins
            
            for b in range(bs):
                value = continuous_values[b, d].item()
                
                # Find which coarse bin this value belongs to
                coarse_bin_idx = int((value - range_min) / coarse_bin_width)
                coarse_bin_idx = max(0, min(coarse_bin_idx, self.num_coarse_bins - 1))
                
                # Find if this coarse bin is in Top-K
                topk_bins_d = topk_bins[b, d].cpu().numpy()
                cumsum = cumsum_fine_bins[b, d].cpu().numpy()
                allocation = fine_bins_per_coarse[b, d].cpu().numpy()
                
                found_idx = -1
                for i in range(k):
                    if topk_bins_d[i] == coarse_bin_idx:
                        found_idx = i
                        break
                
                if found_idx == -1:
                    # Coarse bin not in Top-K, assign to nearest Top-K bin
                    distances = np.abs(topk_bins_d - coarse_bin_idx)
                    found_idx = np.argmin(distances)
                    coarse_bin_idx = topk_bins_d[found_idx]
                
                # Calculate fine bin index
                coarse_start = range_min + coarse_bin_idx * coarse_bin_width
                coarse_end = coarse_start + coarse_bin_width
                
                num_fine_in_coarse = allocation[found_idx]
                fine_bin_width = (coarse_end - coarse_start) / num_fine_in_coarse
                
                fine_offset = int((value - coarse_start) / fine_bin_width)
                fine_offset = max(0, min(fine_offset, num_fine_in_coarse - 1))
                
                fine_start = 0 if found_idx == 0 else cumsum[found_idx - 1]
                fine_bin = fine_start + fine_offset
                
                fine_bins[b, d] = fine_bin
        
        return fine_bins
    
    def _encode_topk_condition(self, topk_info):
        """
        Encode top-k information as a condition vector.
        
        Args:
            topk_info: dict containing topk_bins and topk_probs
            
        Returns:
            topk_cond: [bs, 256] Encoded top-k condition
        """
        topk_bins = topk_info['topk_bins'].float()   # [bs, num_dimensions, k]
        topk_probs = topk_info['topk_probs']          # [bs, num_dimensions, k]
        
        bs = topk_bins.shape[0]
        
        # Normalize bin indices to [0, 1] (topk_bins are coarse bins from Stage 1)
        topk_bins_normalized = topk_bins / self.num_coarse_bins
        
        # Flatten and concatenate
        bins_flat = topk_bins_normalized.view(bs, -1)  # [bs, num_dimensions * k]
        probs_flat = topk_probs.view(bs, -1)           # [bs, num_dimensions * k]
        
        topk_feat = torch.cat([bins_flat, probs_flat], dim=1)  # [bs, num_dimensions * k * 2]
        
        # Project to condition embedding
        topk_cond = self.topk_cond_proj(topk_feat)  # [bs, 256]
        
        return topk_cond

    def model_predict(self, x_t, t, pts_feat, topk_info):
        """
        Predict the posterior distribution p(x_1 | x_t) over fine bins.
        
        Args:
            x_t: [bs, num_dimensions] Current discrete state (fine bin indices)
            t: [bs] Time steps
            pts_feat: [bs, pts_feat_dim] Point cloud features
            topk_info: dict with top-k condition from DFM
            
        Returns:
            posterior_logits: [bs, num_dimensions, num_fine_bins] Predicted posterior logits
        """
        bs = x_t.shape[0]
        
        # Make sure t has the correct shape
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(bs)
        elif t.shape[0] != bs:
            t = t.expand(bs)
        
        # 1. Convert x_t to one-hot encoding over fine bins
        x_t_onehot = F.one_hot(x_t, num_classes=self.num_fine_bins).float()  # [bs, D, num_fine_bins]
        x_t_flat = x_t_onehot.view(bs, -1)  # [bs, D * num_fine_bins]
        
        # 2. Time embedding
        t_emb = self.time_embedder(t)  # [bs, 256]
        
        # 3. Enhanced feature fusion with cross-attention
        cond_proj = self.cond_time_proj(pts_feat)  # [bs, 256]
        t_emb_expanded = t_emb.unsqueeze(1)        # [bs, 1, 256]
        cond_proj_expanded = cond_proj.unsqueeze(1) # [bs, 1, 256]
        t_emb_enhanced = self.cross_attention_fusion(t_emb_expanded, cond_proj_expanded).squeeze(1)  # [bs, 256]
        
        # 4. Encode top-k condition
        topk_cond = self._encode_topk_condition(topk_info)  # [bs, 256]
        
        # 5. Concatenate all features
        input_feat = torch.cat([x_t_flat, t_emb_enhanced, pts_feat, topk_cond], dim=1)
        
        # 6. Shared MLP
        shared_feat = self.mlp_shared(input_feat)  # [bs, 512]
        
        # 7. Angle and translation branches
        angles_feat = self.angles_branch(shared_feat)
        trans_feat = self.translation_branch(shared_feat)
        
        # 8. Self-attention
        angles_feat = self.self_attention_angles(angles_feat)
        trans_feat = self.self_attention_trans(trans_feat)
        
        # 9. Conditional attention (cross-branch)
        angles_feat_enhanced = self.conditional_attention_angles(angles_feat, trans_feat)
        trans_feat_enhanced = self.conditional_attention_trans(trans_feat, angles_feat)
        
        # 10. Predict posterior logits for each dimension
        angle_logits = [head(angles_feat_enhanced) for head in self.angle_heads]
        trans_logits = [head(trans_feat_enhanced) for head in self.translation_heads]
        
        # Stack logits: [bs, num_dimensions, num_bins]
        posterior_logits = torch.stack(angle_logits + trans_logits, dim=1)
        
        return posterior_logits
    
    def sample(self, pts_feat, topk_info, step_size=0.01):
        """
        Sample pose using discrete flow matching solver over fine bins.
        
        Args:
            pts_feat: [bs, pts_feat_dim] Point cloud features
            topk_info: dict with top-k condition from DFM
            step_size: Sampling step size
            
        Returns:
            result: [bs, num_dimensions] Predicted fine bin indices
        """
        class FlowMatchingWrapper(ModelWrapper):
            def __init__(self, model, pts_feat, topk_info):
                super().__init__(model)
                self.pts_feat = pts_feat
                self.topk_info = topk_info
            
            def forward(self, x, t, **extras):
                logits = self.model.model_predict(x, t, self.pts_feat, self.topk_info)
                return torch.softmax(logits, dim=-1)
        
        # Initialize solver with fine bin vocabulary
        model_wrapper = FlowMatchingWrapper(self, pts_feat, topk_info)
        solver = MixtureDiscreteEulerSolver(
            model=model_wrapper,
            path=self.path,
            vocabulary_size=self.num_fine_bins
        )
        
        # Sample initial state from hierarchical fine bin distribution
        x_init = self.sample_noise_from_topk(topk_info)
        
        # Sample using the solver
        result = solver.sample(
            x_init=x_init,
            step_size=step_size,
            time_grid=torch.tensor([0.0, 1.0], device=self.device)
        )
        
        return result
    
    def loss(self, x_1_fine, pts_feat, topk_info):
        """
        Compute discrete flow matching loss over fine bins.
        
        Args:
            x_1_fine: [bs, num_dimensions] Target fine bin indices (ground truth)
            pts_feat: [bs, pts_feat_dim] Point cloud features
            topk_info: dict with top-k condition from DFM
            
        Returns:
            total_loss: scalar loss value
            loss_description: string describing the loss components
            loss_dict: dict with detailed loss values
        """
        bs = x_1_fine.shape[0]
        
        # Sample time t uniformly from [0, 1-epsilon]
        t = torch.rand(bs, device=self.device) * (1.0 - self.time_epsilon)
        
        # Sample x_0 from hierarchical fine bin distribution
        x_0 = self.sample_noise_from_topk(topk_info)
        
        # Sample x_t from the path p_t(x_t | x_0, x_1)
        path_sample = self.path.sample(x_0=x_0, x_1=x_1_fine, t=t)
        x_t = path_sample.x_t
        
        # Predict posterior logits p(x_1 | x_t) over fine bins
        posterior_logits = self.model_predict(x_t, t, pts_feat, topk_info)
        
        # Compute GeneralizedKL loss
        kl_loss = self.criterion(
            logits=posterior_logits,
            x_1=x_1_fine,
            x_t=x_t,
            t=t
        )
        
        # Separate losses for angles and translations
        angle_logits = posterior_logits[:, :self.angle_dimensions]
        trans_logits = posterior_logits[:, self.angle_dimensions:]
        angle_x1 = x_1_fine[:, :self.angle_dimensions]
        angle_xt = x_t[:, :self.angle_dimensions]
        trans_x1 = x_1_fine[:, self.angle_dimensions:]
        trans_xt = x_t[:, self.angle_dimensions:]
        
        angle_kl_loss = self.criterion(angle_logits, angle_x1, angle_xt, t)
        trans_kl_loss = self.criterion(trans_logits, trans_x1, trans_xt, t)
        
        # Compute MSE loss over fine bin indices
        probs = F.softmax(posterior_logits, dim=-1)  # [bs, num_dimensions, num_fine_bins]
        bin_indices = torch.arange(self.num_fine_bins, device=probs.device).float()
        pred_bins = torch.sum(probs * bin_indices, dim=-1)  # [bs, num_dimensions]
        true_bins = x_1_fine.float()
        
        mse_loss = F.mse_loss(pred_bins, true_bins)
        L1_loss = F.l1_loss(pred_bins, true_bins)
        
        # Combined loss
        total_loss = self.kl_weight * kl_loss + self.mse_weight * mse_loss + self.L1_weight * L1_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'kl_loss': kl_loss.item(),
            'mse_loss': mse_loss.item(),
            'L1_loss': L1_loss.item(),
            'angle_kl_loss': angle_kl_loss.item(),
            'trans_kl_loss': trans_kl_loss.item()
        }
        
        loss_description = (
            f"Total: {total_loss:.4f}, "
            f"KL: {self.kl_weight * kl_loss:.4f}, "
            f"MSE: {self.mse_weight * mse_loss:.4f}, "
            f"L1: {self.L1_weight * L1_loss:.4f}, "
            f"Angle KL: {angle_kl_loss:.4f}, "
            f"Trans KL: {trans_kl_loss:.4f}"
        )
        
        return total_loss, loss_description, loss_dict
