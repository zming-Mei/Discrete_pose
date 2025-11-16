import sys
import os
import torch
from tqdm import tqdm
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import get_config
from utils.metrics import rot_diff_degree
from networks.gf_algorithms.discrete_angle import *
from datasets.dataloader import (get_data_loaders_from_cfg, process_batch)
import pytorch3d.transforms as pytorch3d_transforms
from networks.gf_algorithms.discrete_number import *
from networks.discrete_flow_matching import DiscreteFlowMatching


class MetricsTracker:
    """Track and compute evaluation metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_angle_error = 0.0
        self.total_trans_error = 0.0
        self.total_samples = 0
        self.batch_count = 0
        
        # Angle threshold statistics (degrees)
        self.angle_thresholds = {5: 0, 10: 0, 20: 0}
        # Translation threshold statistics (cm)
        self.trans_thresholds = {2: 0, 5: 0, 10: 0}
    
    def update(self, angle_errors, trans_errors):
        """Update statistics with new batch results"""
        batch_size = angle_errors.size(0)
        self.total_samples += batch_size
        self.batch_count += 1
        
        # Accumulate errors
        self.total_angle_error += angle_errors.mean().item()
        self.total_trans_error += trans_errors.mean().item()
        
        # Count samples within thresholds
        for threshold in self.angle_thresholds.keys():
            self.angle_thresholds[threshold] += (angle_errors < threshold).sum().item()
        
        trans_errors_cm = trans_errors * 100
        for threshold in self.trans_thresholds.keys():
            self.trans_thresholds[threshold] += (trans_errors_cm < threshold).sum().item()
    
    def get_batch_stats(self, angle_errors, trans_errors):
        """Get statistics for current batch"""
        batch_size = angle_errors.size(0)
        trans_errors_cm = trans_errors * 100
        
        return {
            'avg_angle': angle_errors.mean().item(),
            'avg_trans': trans_errors.mean().item(),
            'angle_ratios': {t: (angle_errors < t).sum().item() / batch_size 
                           for t in [5, 10, 20]},
            'trans_ratios': {t: (trans_errors_cm < t).sum().item() / batch_size 
                           for t in [2, 5, 10]}
        }
    
    def get_summary(self):
        """Get overall statistics summary"""
        avg_angle = self.total_angle_error / self.batch_count
        avg_trans = self.total_trans_error / self.batch_count
        
        return {
            'total_samples': self.total_samples,
            'avg_angle': avg_angle,
            'avg_trans': avg_trans,
            'angle_ratios': {t: count / self.total_samples 
                           for t, count in self.angle_thresholds.items()},
            'trans_ratios': {t: count / self.total_samples 
                           for t, count in self.trans_thresholds.items()},
            'angle_counts': self.angle_thresholds,
            'trans_counts': self.trans_thresholds
        }
    
    def print_batch_stats(self, stats):
        """Print batch statistics"""
        angle_r = stats['angle_ratios']
        trans_r = stats['trans_ratios']
        print(f"Batch - Angle < 5°: {angle_r[5]:.2%} | < 10°: {angle_r[10]:.2%} | < 20°: {angle_r[20]:.2%}")
        print(f"Batch - Trans < 2cm: {trans_r[2]:.2%} | < 5cm: {trans_r[5]:.2%} | < 10cm: {trans_r[10]:.2%}")
        print(f"Batch - Avg Angle Error: {stats['avg_angle']:.4f}° | Avg Trans Error: {stats['avg_trans']:.4f}m\n")
    
    def print_summary(self):
        """Print final evaluation summary"""
        summary = self.get_summary()
        angle_r = summary['angle_ratios']
        trans_r = summary['trans_ratios']
        angle_c = summary['angle_counts']
        trans_c = summary['trans_counts']
        
        print("=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Total samples evaluated: {summary['total_samples']}")
        print(f"\nAverage Rotation Error: {summary['avg_angle']:.4f}°")
        print(f"Average Translation Error: {summary['avg_trans']:.4f}m ({summary['avg_trans']*100:.4f}cm)")
        print(f"\nRotation Error Distribution:")
        print(f"  Samples with error < 5°:  {angle_r[5]:.2%} ({angle_c[5]}/{summary['total_samples']})")
        print(f"  Samples with error < 10°: {angle_r[10]:.2%} ({angle_c[10]}/{summary['total_samples']})")
        print(f"  Samples with error < 20°: {angle_r[20]:.2%} ({angle_c[20]}/{summary['total_samples']})")
        print(f"\nTranslation Error Distribution:")
        print(f"  Samples with error < 2cm:  {trans_r[2]:.2%} ({trans_c[2]}/{summary['total_samples']})")
        print(f"  Samples with error < 5cm:  {trans_r[5]:.2%} ({trans_c[5]}/{summary['total_samples']})")
        print(f"  Samples with error < 10cm: {trans_r[10]:.2%} ({trans_c[10]}/{summary['total_samples']})")
        print("=" * 80)


class DiscreteFlowEvaluator:
    """Evaluator for Discrete Flow Matching model"""
    
    def __init__(self, cfg, trans_stats=None, pretrained_path=None):
        """Initialize evaluator
        
        Args:
            cfg: Configuration object
            trans_stats: Translation statistics for denormalization [min_x, max_x, min_y, max_y, min_z, max_z]
            pretrained_path: Path to pretrained model checkpoint
        """
        self.cfg = cfg
        self.device = cfg.device
        self.trans_stats = trans_stats
        
        # Initialize model
        self.model = DiscreteFlowMatching(cfg, device=self.device).to(self.device)
        
        # Load pretrained weights
        if pretrained_path is not None:
            print(f"Loading pretrained model from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print("Model loaded successfully")

    def test_step(self, batch_sample):
        """Perform single evaluation step
        
        Returns:
            angle_errors: Rotation errors in degrees
            trans_errors: Translation errors in meters
        """
        self.model.eval()
        
        # Extract ground truth pose
        gt_pose = batch_sample['zero_mean_gt_pose']
        rot_part, true_trans = gt_pose[:, :6], gt_pose[:, -3:]
        rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(rot_part)
        
        # Extract point cloud features and sample
        pts_feat = self.model.extract_pts_feature(batch_sample).to(self.device)
    
        with torch.no_grad():
            pred = self.model.sample(pts_feat, step_size=self.cfg.T)
            pred_rot_bins, pred_trans_bins = pred[:, :3], pred[:, 3:6]
            
            # Convert bins to angles and rotation matrix
            pred_euler_angles = euler_angles_from_bins(pred_rot_bins, self.model.num_bins)
            pred_euler_matrix = pytorch3d_transforms.euler_angles_to_matrix(pred_euler_angles, "ZYX")
            
            # Convert translation bins to actual values
            pred_trans = bins_to_numbers(pred_trans_bins, self.trans_stats, self.model.num_bins)
            
            # Compute errors
            angle_errors = rot_diff_degree(rot_matrix, pred_euler_matrix)
            trans_errors = torch.norm(pred_trans - true_trans, p=2, dim=-1)
            
            return angle_errors, trans_errors


def evaluate_model(cfg, test_loader, trans_stats):
    """Evaluate model performance"""
    # Initialize evaluator and metrics tracker
    evaluator = DiscreteFlowEvaluator(cfg, trans_stats, pretrained_path=cfg.pretrained_model_path_test)
    evaluator.model.eval()
    metrics = MetricsTracker()
    
    # Evaluation loop
    pbar = tqdm(test_loader, desc="Evaluating")
    for batch in pbar:
        # Process batch data
        test_batch = process_batch(batch, cfg.device, cfg.pose_mode, 
                                   mini_batch_size=96, PTS_AUG_PARAMS=None)
        
        with torch.no_grad():
            # Perform evaluation step
            angle_errors, trans_errors = evaluator.test_step(test_batch)
            
            # Update metrics
            metrics.update(angle_errors, trans_errors)
            
            # Print batch statistics
            batch_stats = metrics.get_batch_stats(angle_errors, trans_errors)
            metrics.print_batch_stats(batch_stats)
    
    # Print final summary
    metrics.print_summary()


def main():
    """Main evaluation function"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Load configuration and data
    cfg = get_config()
    data_loaders = get_data_loaders_from_cfg(cfg, ['test'])
    test_loader = data_loaders['test_loader'] 
    print(f'Test set size: {len(test_loader)} batches')
    
    # Translation statistics [min_x, max_x, min_y, max_y, min_z, max_z]
    trans_stats = [
        -0.3785014748573303, 0.39416784048080444,
        -0.4042277932167053, 0.39954620599746704,
        -0.30842161178588867, 0.7598943710327148
    ]
    #trans_stats = get_dataset_translation_min_max(train_loader, cfg)
    
    # Run evaluation
    evaluate_model(cfg, test_loader, trans_stats)

            
if __name__ == "__main__":
    main()

