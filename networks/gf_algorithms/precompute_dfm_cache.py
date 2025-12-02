"""
Precompute DFM predictions and cache them for faster ACFM training.

This script runs DFM inference on the entire training dataset once and saves
the results (coarse pose + top-K information) to disk. During ACFM training,
these cached results can be loaded directly, avoiding repeated DFM inference.

Usage:
    python tools/precompute_dfm_cache.py \
        --dfm_pretrained_path ckpts/DFM/model.pt \
        --data_path ../ArtImage-High-level/ArtImage \
        --output_cache_path ckpts/dfm_cache.pt \
        --topk_k 10
"""

import sys
import os
import torch
import argparse
from tqdm import tqdm
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import get_config
from networks.discrete_flow_matching import DiscreteFlowMatching
from datasets.dataloader import get_data_loaders_from_cfg, process_batch


def precompute_dfm_cache(cfg, dfm_model, dataloader, cache_path, k=10):
    """
    Precompute DFM predictions for all samples in the dataloader.
    
    Args:
        cfg: Configuration object
        dfm_model: Pretrained DFM model
        dataloader: DataLoader to process
        cache_path: Path to save the cache file
        k: Number of top-K bins to extract
    
    Returns:
        cache_dict: Dictionary mapping sample indices to DFM predictions
    """
    dfm_model.eval()
    cache_dict = {}
    
    print(f"Precomputing DFM predictions for {len(dataloader)} batches...")
    print(f"Top-K: {k}, Step size: {cfg.T_dfm if hasattr(cfg, 'T_dfm') else 0.01}")
    
    sample_idx = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Precomputing DFM")):
            # Process batch
            batch = process_batch(
                batch, 
                cfg.device, 
                cfg.pose_mode, 
                mini_batch_size=96, 
                PTS_AUG_PARAMS=None
            )
            
            # Extract point cloud features
            pts_feat = dfm_model.extract_pts_feature(batch).to(cfg.device)
            
            # Get DFM predictions (use_sampling=True for best quality cache)
            step_size = cfg.T_dfm if hasattr(cfg, 'T_dfm') else 0.01
            topk_info = dfm_model.predict_coarse_pose(
                pts_feat, 
                step_size=step_size, 
                k=k,
                use_sampling=True  # Use full ODE sampling for high-quality cache
            )
            
            # Store results for each sample in the batch
            batch_size = pts_feat.shape[0]
            for i in range(batch_size):
                cache_dict[sample_idx] = {
                    'coarse_pose': topk_info['coarse_pose'][i].cpu(),
                    'topk_bins': topk_info['topk_bins'][i].cpu(),
                    'topk_probs': topk_info['topk_probs'][i].cpu(),
                    'topk_offsets': topk_info['topk_offsets'][i].cpu(),
                    'entropy': topk_info['entropy'][i].cpu(),
                }
                sample_idx += 1
    
    # Save cache to disk
    print(f"\nSaving cache to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache_dict, cache_path)
    print(f"Cache saved successfully! Total samples: {sample_idx}")
    
    # Print statistics
    print("\n" + "="*80)
    print("Cache Statistics:")
    print("="*80)
    print(f"Total samples: {sample_idx}")
    print(f"Cache file size: {os.path.getsize(cache_path) / (1024**2):.2f} MB")
    
    # Sample entropy statistics
    all_entropy = torch.stack([cache_dict[i]['entropy'] for i in range(sample_idx)])
    print(f"\nEntropy statistics (per dimension):")
    print(f"  Mean: {all_entropy.mean(dim=0).numpy()}")
    print(f"  Std:  {all_entropy.std(dim=0).numpy()}")
    print(f"  Min:  {all_entropy.min(dim=0)[0].numpy()}")
    print(f"  Max:  {all_entropy.max(dim=0)[0].numpy()}")
    print("="*80)
    
    return cache_dict


def main():
    """Main function for precomputing DFM cache"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Precompute DFM cache for ACFM training')
    parser.add_argument('--dfm_pretrained_path', type=str, required=True,
                       help='Path to pretrained DFM model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--output_cache_path', type=str, required=True,
                       help='Path to save the cache file')
    parser.add_argument('--topk_k', type=int, default=10,
                       help='Number of top-K bins to extract')
    parser.add_argument('--batch_size', type=int, default=96,
                       help='Batch size for processing')
    parser.add_argument('--num_bins', type=int, default=72,
                       help='Number of bins (must match DFM training)')
    parser.add_argument('--T_dfm', type=float, default=0.01,
                       help='Step size for DFM sampling')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--cate_id', type=int, default=1,
                       help='Category ID')
    parser.add_argument('--joint_num', type=int, default=1,
                       help='Joint number')
    parser.add_argument('--num_parts', type=int, default=2,
                       help='Number of parts')
    parser.add_argument('--pts_encoder', type=str, default='pointnet2',
                       help='Point cloud encoder')
    parser.add_argument('--num_workers', type=int, default=16,
                       help='Number of workers for data loading')
    parser.add_argument('--dataset_split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Which dataset split to precompute')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create a minimal config object
    class MinimalConfig:
        pass
    
    cfg = MinimalConfig()
    cfg.device = args.device
    cfg.data_path = args.data_path
    cfg.batch_size = args.batch_size
    cfg.num_bins = args.num_bins
    cfg.T_dfm = args.T_dfm
    cfg.seed = args.seed
    cfg.cate_id = args.cate_id
    cfg.joint_num = args.joint_num
    cfg.num_parts = args.num_parts
    cfg.pts_encoder = args.pts_encoder
    cfg.num_workers = args.num_workers
    cfg.pose_mode = 'rot_matrix'
    cfg.num_points = 1024
    cfg.percentage_data_for_train = 1.0
    cfg.percentage_data_for_val = 1.0
    cfg.percentage_data_for_test = 1.0
    cfg.mse_weight = 0
    cfg.kl_weight = 1
    cfg.L1_weight = 0.1
    
    # Load DFM model
    print(f"Loading DFM model from {args.dfm_pretrained_path}...")
    dfm = DiscreteFlowMatching(cfg, device=cfg.device).to(cfg.device)
    dfm.load_state_dict(torch.load(args.dfm_pretrained_path, map_location=cfg.device))
    dfm.eval()
    print("DFM model loaded successfully!")
    
    # Load dataset
    print(f"\nLoading {args.dataset_split} dataset...")
    data_loaders = get_data_loaders_from_cfg(cfg, [args.dataset_split])
    dataloader = data_loaders[f'{args.dataset_split}_loader']
    print(f"Dataset loaded: {len(dataloader)} batches")
    
    # Precompute and save cache
    cache_dict = precompute_dfm_cache(
        cfg, 
        dfm, 
        dataloader, 
        args.output_cache_path,
        k=args.topk_k
    )
    
    print(f"\n✅ Precomputation complete! Cache saved to: {args.output_cache_path}")
    print(f"To use this cache during ACFM training, add the following argument:")
    print(f"  --dfm_cache_path {args.output_cache_path}")


if __name__ == "__main__":
    main()

