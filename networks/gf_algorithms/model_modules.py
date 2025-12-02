"""
Network Components Module
Contains basic network components including time embedding, residual blocks, and attention mechanisms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimestepEmbedder(nn.Module):
    """Timestep embedding module"""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Generate sinusoidal positional encoding"""
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
    """Residual block with layer normalization and dropout"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        if self.same_channels:
            return x + self.net(x)
        else:
            return self.proj(x) + self.net(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projection and split heads
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(attn_output)

        return output


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion module for interaction between time embeddings and point cloud features"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        # Cross attention
        attn_output = self.attention(query, key_value, key_value)
        query = self.norm1(query + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.feedforward(query)
        query = self.norm2(query + self.dropout(ff_output))

        return query


class SelfAttentionBlock(nn.Module):
    """Self-attention block for internal feature enhancement"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, embed_dim], add sequence dimension for attention
        x_seq = x.unsqueeze(1)  # [batch_size, 1, embed_dim]

        # Self attention
        attn_output = self.attention(x_seq, x_seq, x_seq)
        x_seq = self.norm1(x_seq + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.feedforward(x_seq)
        x_seq = self.norm2(x_seq + self.dropout(ff_output))

        return x_seq.squeeze(1)  # [batch_size, embed_dim]


class ConditionalAttention(nn.Module):
    """Conditional attention for interaction between angle and translation branches"""
    def __init__(self, embed_dim, condition_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.condition_proj = nn.Linear(condition_dim, embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, condition):
        # x, condition: [batch_size, embed_dim], add sequence dimension for attention
        x_seq = x.unsqueeze(1)  # [batch_size, 1, embed_dim]
        condition_seq = condition.unsqueeze(1)  # [batch_size, 1, embed_dim]

        # Project condition to the same dimension
        condition_embed = self.condition_proj(condition_seq)

        # Conditional attention
        attn_output = self.attention(x_seq, condition_embed, condition_embed)
        x_seq = self.norm(x_seq + self.dropout(attn_output))

        return x_seq.squeeze(1)  # [batch_size, embed_dim]


class SharedMLP(nn.Module):
    """Shared MLP for processing combined features"""
    def __init__(self, input_dim, hidden_dim=1024, output_dim=512, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            ResidualBlock(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout * 2)
        )
    
    def forward(self, x):
        return self.network(x)


class PoseBranch(nn.Module):
    """Branch network for angle or translation prediction"""
    def __init__(self, input_dim=512, hidden_dim=384, output_dim=256, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, x):
        return self.network(x)


class PredictionHead(nn.Module):
    """Prediction head for individual dimension"""
    def __init__(self, input_dim=256, hidden_dim=256, num_bins=360):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_bins)
        )
    
    def forward(self, x):
        return self.network(x)

