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
        
        # Bin difference statistics
        self.total_bin_diff = {
            'rot_x': 0.0, 'rot_y': 0.0, 'rot_z': 0.0,
            'trans_x': 0.0, 'trans_y': 0.0, 'trans_z': 0.0
        }
        self.max_bin_diff = {
            'rot_x': 0, 'rot_y': 0, 'rot_z': 0,
            'trans_x': 0, 'trans_y': 0, 'trans_z': 0
        }
    
    def update(self, angle_errors, trans_errors, bin_diffs=None):
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
        
        # Update bin difference statistics
        if bin_diffs is not None:
            for key in self.total_bin_diff.keys():
                self.total_bin_diff[key] += bin_diffs[key].mean().item()
                self.max_bin_diff[key] = max(self.max_bin_diff[key], bin_diffs[key].max().item())
    
    def get_batch_stats(self, angle_errors, trans_errors, bin_diffs=None):
        """Get statistics for current batch"""
        batch_size = angle_errors.size(0)
        trans_errors_cm = trans_errors * 100
        
        stats = {
            'avg_angle': angle_errors.mean().item(),
            'avg_trans': trans_errors.mean().item(),
            'angle_ratios': {t: (angle_errors < t).sum().item() / batch_size 
                           for t in [5, 10, 20]},
            'trans_ratios': {t: (trans_errors_cm < t).sum().item() / batch_size 
                           for t in [2, 5, 10]}
        }
        
        if bin_diffs is not None:
            stats['bin_diffs'] = {key: val.mean().item() for key, val in bin_diffs.items()}
            stats['bin_diffs_max'] = {key: val.max().item() for key, val in bin_diffs.items()}
        
        return stats
    
    def get_summary(self):
        """Get overall statistics summary"""
        avg_angle = self.total_angle_error / self.batch_count
        avg_trans = self.total_trans_error / self.batch_count
        
        summary = {
            'total_samples': self.total_samples,
            'avg_angle': avg_angle,
            'avg_trans': avg_trans,
            'angle_ratios': {t: count / self.total_samples 
                           for t, count in self.angle_thresholds.items()},
            'trans_ratios': {t: count / self.total_samples 
                           for t, count in self.trans_thresholds.items()},
            'angle_counts': self.angle_thresholds,
            'trans_counts': self.trans_thresholds,
            'avg_bin_diff': {key: val / self.batch_count for key, val in self.total_bin_diff.items()},
            'max_bin_diff': self.max_bin_diff
        }
        
        return summary
    
    def print_batch_stats(self, stats):
        """Print batch statistics"""
        angle_r = stats['angle_ratios']
        trans_r = stats['trans_ratios']
        print(f"Batch - Angle < 5°: {angle_r[5]:.2%} | < 10°: {angle_r[10]:.2%} | < 20°: {angle_r[20]:.2%}")
        print(f"Batch - Trans < 2cm: {trans_r[2]:.2%} | < 5cm: {trans_r[5]:.2%} | < 10cm: {trans_r[10]:.2%}")
        print(f"Batch - Avg Angle Error: {stats['avg_angle']:.4f}° | Avg Trans Error: {stats['avg_trans']:.4f}m")
        
        if 'bin_diffs' in stats:
            bd = stats['bin_diffs']
            bd_max = stats['bin_diffs_max']
            print(f"Batch - Avg Bin Diff - Rot(X/Y/Z): {bd['rot_x']:.2f}/{bd['rot_y']:.2f}/{bd['rot_z']:.2f} | Trans(X/Y/Z): {bd['trans_x']:.2f}/{bd['trans_y']:.2f}/{bd['trans_z']:.2f}")
            print(f"Batch - Max Bin Diff - Rot(X/Y/Z): {bd_max['rot_x']:.0f}/{bd_max['rot_y']:.0f}/{bd_max['rot_z']:.0f} | Trans(X/Y/Z): {bd_max['trans_x']:.0f}/{bd_max['trans_y']:.0f}/{bd_max['trans_z']:.0f}")
        print()
    
    def print_summary(self):
        """Print final evaluation summary"""
        summary = self.get_summary()
        angle_r = summary['angle_ratios']
        trans_r = summary['trans_ratios']
        angle_c = summary['angle_counts']
        trans_c = summary['trans_counts']
        avg_bd = summary['avg_bin_diff']
        max_bd = summary['max_bin_diff']
        
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
        print(f"\nBin Difference Statistics:")
        print(f"  Average Bin Diff - Rotation(X/Y/Z):    {avg_bd['rot_x']:.2f} / {avg_bd['rot_y']:.2f} / {avg_bd['rot_z']:.2f}")
        print(f"  Average Bin Diff - Translation(X/Y/Z): {avg_bd['trans_x']:.2f} / {avg_bd['trans_y']:.2f} / {avg_bd['trans_z']:.2f}")
        print(f"  Maximum Bin Diff - Rotation(X/Y/Z):    {max_bd['rot_x']:.0f} / {max_bd['rot_y']:.0f} / {max_bd['rot_z']:.0f}")
        print(f"  Maximum Bin Diff - Translation(X/Y/Z): {max_bd['trans_x']:.0f} / {max_bd['trans_y']:.0f} / {max_bd['trans_z']:.0f}")
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

    def test_step(self, batch_sample, return_detailed_info=False):
        """Perform single evaluation step

        Returns:
            angle_errors: Rotation errors in degrees
            trans_errors: Translation errors in meters
            bin_diffs: Dictionary containing bin differences for each dimension
            detailed_info: (Optional) Dictionary with pred_bins, gt_bins, logits, and probabilities
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

            # Get final probabilities from model prediction at t=1 (final time step)
            bs = pred.shape[0]
            t_final = torch.ones(bs, device=self.device)
            final_logits = self.model.model_predict(pred, t_final, pts_feat)
            final_probs = torch.softmax(final_logits, dim=-1)  # [bs, 6, num_bins]
            
            # Convert bins to angles and rotation matrix
            pred_euler_angles = euler_angles_from_bins(pred_rot_bins, self.model.num_bins)
            pred_euler_matrix = pytorch3d_transforms.euler_angles_to_matrix(pred_euler_angles, "ZYX")
            
            # Convert translation bins to actual values
            pred_trans = bins_to_numbers(pred_trans_bins, self.trans_stats, self.model.num_bins)
            
            # Compute ground truth bins for comparison
            # Get ground truth euler angles from rotation matrix
            gt_euler_angles = pytorch3d_transforms.matrix_to_euler_angles(rot_matrix, "ZYX")
            gt_rot_bins = discretize_euler_angles(gt_euler_angles, self.model.num_bins)
            
            # Get ground truth translation bins
            gt_trans_bins = translation_to_bins(true_trans, self.trans_stats, self.model.num_bins)
            
            # Compute bin differences
            # For rotation (circular): use circular distance
            # For translation (linear): use absolute difference
            def circular_bin_diff(pred_bins, gt_bins, num_bins):
                """Compute circular distance between bins"""
                diff = torch.abs(pred_bins - gt_bins).float()
                # Take the minimum of direct distance and wrap-around distance
                circular_diff = torch.min(diff, num_bins - diff)
                return circular_diff
            
            bin_diffs = {
                'rot_x': circular_bin_diff(pred_rot_bins[:, 0], gt_rot_bins[:, 0], self.model.num_bins),
                'rot_y': circular_bin_diff(pred_rot_bins[:, 1], gt_rot_bins[:, 1], self.model.num_bins),
                'rot_z': circular_bin_diff(pred_rot_bins[:, 2], gt_rot_bins[:, 2], self.model.num_bins),
                'trans_x': torch.abs(pred_trans_bins[:, 0] - gt_trans_bins[:, 0]).float(),
                'trans_y': torch.abs(pred_trans_bins[:, 1] - gt_trans_bins[:, 1]).float(),
                'trans_z': torch.abs(pred_trans_bins[:, 2] - gt_trans_bins[:, 2]).float()
            }
            
            # Compute errors
            angle_errors = rot_diff_degree(rot_matrix, pred_euler_matrix)
            trans_errors = torch.norm(pred_trans - true_trans, p=2, dim=-1)
            
            if return_detailed_info:
                detailed_info = {
                    'pred_rot_bins': pred_rot_bins,
                    'pred_trans_bins': pred_trans_bins,
                    'gt_rot_bins': gt_rot_bins,
                    'gt_trans_bins': gt_trans_bins,
                    'final_probs': final_probs  # [bs, 6, num_bins] - probabilities for each dimension
                }
                return angle_errors, trans_errors, bin_diffs, detailed_info
            else:
                return angle_errors, trans_errors, bin_diffs


def get_top_k_bins(probs, k=5):
    """Get top k bins with highest probabilities for each dimension"""
    # probs: [6, num_bins] - probabilities for one sample
    top_probs, top_indices = torch.topk(probs, k, dim=-1)
    return top_probs, top_indices

def evaluate_model(cfg, test_loader, trans_stats):
    """Evaluate model performance"""
    # Initialize evaluator and metrics tracker
    evaluator = DiscreteFlowEvaluator(cfg, trans_stats, pretrained_path=cfg.pretrained_model_path_test)
    evaluator.model.eval()
    metrics = MetricsTracker()
    
    # Thresholds for printing detailed info
    ANGLE_THRESHOLD = 20.0  # degrees
    TRANS_THRESHOLD = 10  # 10cm = 0.1m
    
    # Open file for writing detailed results
    output_file = "tools/eval_bad_samples_detail.txt"
    f = open(output_file, 'w', encoding='utf-8')
    
    try:
        f.write(f"Evaluation Results - Samples with Angle Error > {ANGLE_THRESHOLD}° OR Trans Error > {TRANS_THRESHOLD*100}cm\n")
        f.write("="*100 + "\n\n")
        
        bad_sample_count = 0
        
        # Evaluation loop
        pbar = tqdm(test_loader, desc="Evaluating")
        sample_idx = 0
        for batch_idx, batch in enumerate(pbar):
            # Process batch data
            test_batch = process_batch(batch, cfg.device, cfg.pose_mode, 
                                       mini_batch_size=96, PTS_AUG_PARAMS=None)
            
            with torch.no_grad():
                # Perform evaluation step with detailed info
                angle_errors, trans_errors, bin_diffs, detailed_info = evaluator.test_step(test_batch, return_detailed_info=True)
                
                # Update metrics
                metrics.update(angle_errors, trans_errors, bin_diffs)
                
                # Find samples that exceed thresholds
                trans_errors_cm = trans_errors * 100
                bad_samples = (angle_errors > ANGLE_THRESHOLD) | (trans_errors_cm > TRANS_THRESHOLD)
            
                if bad_samples.any():
                    bad_indices = torch.where(bad_samples)[0]
                    for idx in bad_indices:
                        bad_sample_count += 1
                        f.write(f"\n{'='*100}\n")
                        f.write(f"Bad Sample #{bad_sample_count} - Global Sample #{sample_idx + idx.item()} (Batch {batch_idx}, Local idx {idx.item()})\n")
                        f.write(f"  Angle Error: {angle_errors[idx].item():.2f}° | Trans Error: {trans_errors_cm[idx].item():.2f}cm\n")
                        f.write(f"\n  Rotation Bins (X/Y/Z):\n")
                        f.write(f"    Predicted:     [{detailed_info['pred_rot_bins'][idx, 0].item():3d}, "
                              f"{detailed_info['pred_rot_bins'][idx, 1].item():3d}, "
                              f"{detailed_info['pred_rot_bins'][idx, 2].item():3d}]\n")
                        f.write(f"    Ground Truth:  [{detailed_info['gt_rot_bins'][idx, 0].item():3d}, "
                              f"{detailed_info['gt_rot_bins'][idx, 1].item():3d}, "
                              f"{detailed_info['gt_rot_bins'][idx, 2].item():3d}]\n")
                        f.write(f"    Bin Diff:      [{bin_diffs['rot_x'][idx].item():3.0f}, "
                              f"{bin_diffs['rot_y'][idx].item():3.0f}, "
                              f"{bin_diffs['rot_z'][idx].item():3.0f}]\n")

                        # Add top 5 probability bins for rotation
                        sample_probs = detailed_info['final_probs'][idx]  # [6, num_bins]
                        rot_top_probs, rot_top_indices = get_top_k_bins(sample_probs[:3])
                        f.write(f"    Top 5 Prob Bins: X[{rot_top_indices[0].tolist()}] P[{rot_top_probs[0].tolist()}] | "
                              f"Y[{rot_top_indices[1].tolist()}] P[{rot_top_probs[1].tolist()}] | "
                              f"Z[{rot_top_indices[2].tolist()}] P[{rot_top_probs[2].tolist()}]\n")

                        f.write(f"\n  Translation Bins (X/Y/Z):\n")
                        f.write(f"    Predicted:     [{detailed_info['pred_trans_bins'][idx, 0].item():3d}, "
                              f"{detailed_info['pred_trans_bins'][idx, 1].item():3d}, "
                              f"{detailed_info['pred_trans_bins'][idx, 2].item():3d}]\n")
                        f.write(f"    Ground Truth:  [{detailed_info['gt_trans_bins'][idx, 0].item():3d}, "
                              f"{detailed_info['gt_trans_bins'][idx, 1].item():3d}, "
                              f"{detailed_info['gt_trans_bins'][idx, 2].item():3d}]\n")
                        f.write(f"    Bin Diff:      [{bin_diffs['trans_x'][idx].item():3.0f}, "
                              f"{bin_diffs['trans_y'][idx].item():3.0f}, "
                              f"{bin_diffs['trans_z'][idx].item():3.0f}]\n")

                        # Add top 5 probability bins for translation
                        trans_top_probs, trans_top_indices = get_top_k_bins(sample_probs[3:])
                        f.write(f"    Top 5 Prob Bins: X[{trans_top_indices[0].tolist()}] P[{trans_top_probs[0].tolist()}] | "
                              f"Y[{trans_top_indices[1].tolist()}] P[{trans_top_probs[1].tolist()}] | "
                              f"Z[{trans_top_indices[2].tolist()}] P[{trans_top_probs[2].tolist()}]\n")

                        f.write(f"{'='*100}\n")
                        f.flush()  # Ensure data is written immediately
            
            sample_idx += angle_errors.size(0)
            
            # Print batch statistics
            batch_stats = metrics.get_batch_stats(angle_errors, trans_errors, bin_diffs)
            metrics.print_batch_stats(batch_stats)
        
        # Write summary to file
        f.write(f"\n\n{'='*100}\n")
        f.write(f"SUMMARY: Found {bad_sample_count} samples with Angle Error > {ANGLE_THRESHOLD}° OR Trans Error > {TRANS_THRESHOLD*100}cm\n")
        f.write(f"Total samples evaluated: {sample_idx}\n")
        f.write(f"Bad sample ratio: {bad_sample_count/sample_idx*100:.2f}%\n")
        f.write(f"{'='*100}\n")
        
        print(f"\n{'='*100}")
        print(f"Detailed results written to: {output_file}")
        print(f"Found {bad_sample_count} bad samples out of {sample_idx} total samples ({bad_sample_count/sample_idx*100:.2f}%)")
        print(f"{'='*100}\n")
        
        # Print final summary
        metrics.print_summary()
        
    finally:
        f.close()


def main():
    """Main evaluation function"""
    # Load configuration and data
    cfg = get_config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    data_loaders = get_data_loaders_from_cfg(cfg, ['val'])
    test_loader = data_loaders['val_loader'] 
    print(f'Validation set size: {len(test_loader)} batches')
    
    # Translation statistics [min_x, max_x, min_y, max_y, min_z, max_z]
    trans_stats = [
        -0.3785014748573303, 0.39416784048080444,
        -0.4042277932167053, 0.39954620599746704,
        -0.30842161178588867, 0.7598943710327148
    ]
    
    # Run evaluation
    evaluate_model(cfg, test_loader, trans_stats)

            
if __name__ == "__main__":
    main()

