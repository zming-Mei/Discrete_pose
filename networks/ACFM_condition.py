import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class TopKConditionEncoder(nn.Module):
    """
    Encode Top-K information into exploration vectors.
    Optimization: use separate offset encoders for rotation and translation,
    and remove Sigmoid to better preserve the physical meaning of raw probabilities.
    """
    def __init__(self, embed_dim=128, k=10, num_rot_dims=3, num_trans_dims=3):
        super().__init__()
        self.k = k
        self.embed_dim = embed_dim
        self.num_rot_dims = num_rot_dims
        
        # 1. Define separate offset encoders for rotation and translation.
        # Rotation is usually periodic or bounded, while translation can have a larger dynamic range.
        self.rot_offset_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU()
        )
        
        self.trans_offset_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU()
        )
        
        # No Sigmoid here because probabilities are already in [0, 1].
        # The output can be any real value representing an "attention score".
        self.prob_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 1) 
        )
        
    def forward(self, topk_offsets, topk_probs):
        """
        Args:
            topk_offsets: [bs, num_dimensions, k]
            topk_probs: [bs, num_dimensions, k]
        """
        bs, num_dims, k = topk_offsets.shape
        
        rot_offsets = topk_offsets[:, :self.num_rot_dims, :]
        trans_offsets = topk_offsets[:, self.num_rot_dims:, :]
        
        rot_flat = rot_offsets.reshape(-1, 1).float() # ensure float
        rot_features = self.rot_offset_encoder(rot_flat)
        rot_features = rot_features.view(bs, self.num_rot_dims, k, self.embed_dim)

        trans_flat = trans_offsets.reshape(-1, 1).float()
        trans_features = self.trans_offset_encoder(trans_flat)
        trans_features = trans_features.view(bs, num_dims - self.num_rot_dims, k, self.embed_dim)
        
        # Merge rotation and translation features
        offset_features = torch.cat([rot_features, trans_features], dim=1) # [bs, num_dims, k, embed_dim]
        
        # Compute probability weights.
        # Option A: directly use topk_probs * features (most physically intuitive).
        # Option B: map probabilities with a small network (current logic).
        # Here we conceptually combine both: use topk_probs as the base and let a network rescale it.
        # probs_flat = topk_probs.reshape(-1, 1)
        # prob_scores = self.prob_encoder(probs_flat) # [..., 1]
        # prob_weights = F.softmax(prob_scores.view(bs, num_dims, k), dim=2).unsqueeze(-1) # 在K维度归一化
        
        # Or simply use raw probabilities directly (if you trust the DFM probabilities enough).
        # This design choice should ideally be validated with ablation experiments.
        prob_weights = topk_probs.unsqueeze(-1) 

        # Weighted sum: E^d = sum_k P_k * feature_k
        exploration_vectors = torch.sum(
            prob_weights * offset_features, dim=2
        )  # [bs, num_dims, embed_dim]
        
        return exploration_vectors

class EntropyGate(nn.Module):
    """
    Entropy-based gating mechanism.

    If entropy is high, increase the weight of exploration vectors
    (especially for translation dimensions).
    """
    def __init__(self, num_dimensions=6):
        super().__init__()
        self.num_dimensions = num_dimensions
        
        # Learn a gating weight for each dimension
        self.gate_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # output [0, 1] gate weights
        )
        
        self.register_buffer('dim_type_weights', torch.ones(num_dimensions))
        # self.dim_type_weights[3:] = 2.0  # give larger weights to translation dims
        
    def forward(self, entropy, exploration_vectors):
        """
        Args:
            entropy: [bs, num_dimensions] entropy for each dimension
            exploration_vectors: [bs, num_dimensions, embed_dim]
            
        Returns:
            gated_vectors: [bs, num_dimensions, embed_dim] gated exploration vectors
        """
        bs, num_dims, embed_dim = exploration_vectors.shape
        
        # Compute gating weights
        entropy_flat = entropy.view(bs * num_dims, 1)  # [bs*num_dims, 1]
        gate_weights = self.gate_net(entropy_flat)  # [bs*num_dims, 1]
        gate_weights = gate_weights.view(bs, num_dims, 1)  # [bs, num_dims, 1]
        
        # Apply dimension-type weights
        dim_weights = self.dim_type_weights.view(1, num_dims, 1)  # [1, num_dims, 1]
        gate_weights = gate_weights * dim_weights  # [bs, num_dims, 1]
        
        # Apply gate to exploration vectors
        gated_vectors = gate_weights * exploration_vectors  # [bs, num_dims, embed_dim]
        
        return gated_vectors


class CoarsePoseEmbedder(nn.Module):
    def __init__(self, num_dimensions=6, embed_dim=256, num_frequencies=64):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.num_dimensions = num_dimensions
        
        freq_bands = 2.0 ** torch.linspace(0., num_frequencies // 2 - 1, num_frequencies // 2)
        self.register_buffer('freq_bands', freq_bands)
        
        self.proj = nn.Sequential(
            nn.Linear(num_dimensions * (num_frequencies // 2) * 2, embed_dim), # adjust input dim to match
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, coarse_pose):
        bs, num_dims = coarse_pose.shape
        
        # [bs, num_dims, 1] * [1, 1, num_freqs/2] -> [bs, num_dims, num_freqs/2]
        x = coarse_pose.unsqueeze(-1) * self.freq_bands.view(1, 1, -1)
        
        # sin/cos encoding
        fourier_features = torch.cat([torch.sin(x * torch.pi), torch.cos(x * torch.pi)], dim=-1)
        
        # Flatten: [bs, num_dims * num_freqs]
        fourier_features = fourier_features.view(bs, -1)
        
        embedded = self.proj(fourier_features)
        return embedded

