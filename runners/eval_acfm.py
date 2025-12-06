import sys
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import get_config
from utils.metrics import rot_diff_degree
from utils.angle_utils import compute_angle_residual, add_angle_residual, normalize_angles
from networks.discrete_flow_matching import DiscreteFlowMatching
from networks.adaptive_continuous_flow_matching import AdaptiveContinuousFlowMatching
from datasets.dataloader import get_data_loaders_from_cfg, process_batch
import pytorch3d.transforms as pytorch3d_transforms
from networks.gf_algorithms.discrete_angle import euler_angles_from_bins
from networks.gf_algorithms.discrete_number import bins_to_numbers


class MetricsTracker:
    """Track and compute evaluation metrics for two-stage model"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_angle_error = 0.0
        self.total_trans_error = 0.0
        self.total_coarse_angle_error = 0.0
        self.total_coarse_trans_error = 0.0
        self.total_samples = 0
        self.batch_count = 0
        
        self.total_residual_rot_abs_sum = 0.0
        self.total_residual_trans_abs_sum = 0.0
        
        # Angle threshold statistics (degrees)
        self.angle_thresholds = {5: 0, 10: 0, 20: 0}
        self.coarse_angle_thresholds = {5: 0, 10: 0, 20: 0}
        # Translation threshold statistics (cm)
        self.trans_thresholds = {2: 0, 5: 0, 10: 0}
        self.coarse_trans_thresholds = {2: 0, 5: 0, 10: 0}
    
    def update(self, angle_errors, trans_errors, coarse_angle_errors, coarse_trans_errors, pred_delta_rot=None, pred_delta_trans=None):
        """Update statistics with new batch results"""
        batch_size = angle_errors.size(0)
        self.total_samples += batch_size
        self.batch_count += 1
        
        # Accumulate errors
        self.total_angle_error += angle_errors.mean().item()
        self.total_trans_error += trans_errors.mean().item()
        self.total_coarse_angle_error += coarse_angle_errors.mean().item()
        self.total_coarse_trans_error += coarse_trans_errors.mean().item()

        if pred_delta_rot is not None and pred_delta_trans is not None:
            self.total_residual_rot_abs_sum += pred_delta_rot.abs().sum().item()
            self.total_residual_trans_abs_sum += pred_delta_trans.abs().sum().item()
        
        # Count samples within thresholds (ACFM refined)
        for threshold in self.angle_thresholds.keys():
            self.angle_thresholds[threshold] += (angle_errors < threshold).sum().item()
        
        trans_errors_cm = trans_errors * 100
        for threshold in self.trans_thresholds.keys():
            self.trans_thresholds[threshold] += (trans_errors_cm < threshold).sum().item()
        
        # Count samples within thresholds (DFM coarse)
        for threshold in self.coarse_angle_thresholds.keys():
            self.coarse_angle_thresholds[threshold] += (coarse_angle_errors < threshold).sum().item()
        
        coarse_trans_errors_cm = coarse_trans_errors * 100
        for threshold in self.coarse_trans_thresholds.keys():
            self.coarse_trans_thresholds[threshold] += (coarse_trans_errors_cm < threshold).sum().item()
    
    def get_batch_stats(self, angle_errors, trans_errors, coarse_angle_errors, coarse_trans_errors):
        """Get statistics for current batch"""
        batch_size = angle_errors.size(0)
        trans_errors_cm = trans_errors * 100
        coarse_trans_errors_cm = coarse_trans_errors * 100
        
        return {
            'avg_angle': angle_errors.mean().item(),
            'avg_trans': trans_errors.mean().item(),
            'avg_coarse_angle': coarse_angle_errors.mean().item(),
            'avg_coarse_trans': coarse_trans_errors.mean().item(),
            'angle_ratios': {t: (angle_errors < t).sum().item() / batch_size 
                           for t in [5, 10, 20]},
            'trans_ratios': {t: (trans_errors_cm < t).sum().item() / batch_size 
                           for t in [2, 5, 10]},
            'coarse_angle_ratios': {t: (coarse_angle_errors < t).sum().item() / batch_size 
                                  for t in [5, 10, 20]},
            'coarse_trans_ratios': {t: (coarse_trans_errors_cm < t).sum().item() / batch_size 
                                  for t in [2, 5, 10]}
        }
    
    def get_summary(self):
        """Get overall statistics summary"""
        avg_angle = self.total_angle_error / self.batch_count
        avg_trans = self.total_trans_error / self.batch_count
        avg_coarse_angle = self.total_coarse_angle_error / self.batch_count
        avg_coarse_trans = self.total_coarse_trans_error / self.batch_count
        
        avg_residual_rot = self.total_residual_rot_abs_sum / (self.total_samples * 3) if self.total_samples > 0 else 0
        avg_residual_trans = self.total_residual_trans_abs_sum / (self.total_samples * 3) if self.total_samples > 0 else 0

        return {
            'total_samples': self.total_samples,
            'avg_angle': avg_angle,
            'avg_trans': avg_trans,
            'avg_coarse_angle': avg_coarse_angle,
            'avg_coarse_trans': avg_coarse_trans,
            'avg_residual_rot': avg_residual_rot,
            'avg_residual_trans': avg_residual_trans,
            'angle_ratios': {t: count / self.total_samples 
                           for t, count in self.angle_thresholds.items()},
            'trans_ratios': {t: count / self.total_samples 
                           for t, count in self.trans_thresholds.items()},
            'coarse_angle_ratios': {t: count / self.total_samples 
                                  for t, count in self.coarse_angle_thresholds.items()},
            'coarse_trans_ratios': {t: count / self.total_samples 
                                  for t, count in self.coarse_trans_thresholds.items()},
            'angle_counts': self.angle_thresholds,
            'trans_counts': self.trans_thresholds,
            'improvement_angle': avg_coarse_angle - avg_angle,
            'improvement_trans': avg_coarse_trans - avg_trans,
        }
    
    def print_batch_stats(self, stats):
        """Print batch statistics"""
        print(f"[DFM Coarse] Angle < 5°: {stats['coarse_angle_ratios'][5]:.2%} | < 10°: {stats['coarse_angle_ratios'][10]:.2%} | < 20°: {stats['coarse_angle_ratios'][20]:.2%}")
        print(f"[DFM Coarse] Trans < 2cm: {stats['coarse_trans_ratios'][2]:.2%} | < 5cm: {stats['coarse_trans_ratios'][5]:.2%} | < 10cm: {stats['coarse_trans_ratios'][10]:.2%}")
        print(f"[DFM Coarse] Avg Angle: {stats['avg_coarse_angle']:.4f}° | Avg Trans: {stats['avg_coarse_trans']:.4f}m")
        print(f"[ACFM Refined] Angle < 5°: {stats['angle_ratios'][5]:.2%} | < 10°: {stats['angle_ratios'][10]:.2%} | < 20°: {stats['angle_ratios'][20]:.2%}")
        print(f"[ACFM Refined] Trans < 2cm: {stats['trans_ratios'][2]:.2%} | < 5cm: {stats['trans_ratios'][5]:.2%} | < 10cm: {stats['trans_ratios'][10]:.2%}")
        print(f"[ACFM Refined] Avg Angle: {stats['avg_angle']:.4f}° | Avg Trans: {stats['avg_trans']:.4f}m\n")
    
    def print_summary(self):
        """Print final evaluation summary"""
        summary = self.get_summary()
        angle_r = summary['angle_ratios']
        trans_r = summary['trans_ratios']
        angle_c = summary['angle_counts']
        trans_c = summary['trans_counts']
        coarse_angle_r = summary['coarse_angle_ratios']
        coarse_trans_r = summary['coarse_trans_ratios']
        
        print("=" * 80)
        print("EVALUATION SUMMARY - TWO-STAGE MODEL (DFM + ACFM)")
        print("=" * 80)
        print(f"Total samples evaluated: {summary['total_samples']}")
        
        print(f"\n【Stage 1: DFM Coarse Estimation】")
        print(f"Average Rotation Error: {summary['avg_coarse_angle']:.4f}°")
        print(f"Average Translation Error: {summary['avg_coarse_trans']:.4f}m ({summary['avg_coarse_trans']*100:.4f}cm)")
        print(f"Rotation Error Distribution:")
        print(f"  < 5°:  {coarse_angle_r[5]:.2%}")
        print(f"  < 10°: {coarse_angle_r[10]:.2%}")
        print(f"  < 20°: {coarse_angle_r[20]:.2%}")
        print(f"Translation Error Distribution:")
        print(f"  < 2cm:  {coarse_trans_r[2]:.2%}")
        print(f"  < 5cm:  {coarse_trans_r[5]:.2%}")
        print(f"  < 10cm: {coarse_trans_r[10]:.2%}")
        
        print(f"\n【Stage 2: ACFM Refinement】")
        print(f"Average Rotation Error: {summary['avg_angle']:.4f}°")
        print(f"Average Translation Error: {summary['avg_trans']:.4f}m ({summary['avg_trans']*100:.4f}cm)")
        print(f"Rotation Error Distribution:")
        print(f"  < 5°:  {angle_r[5]:.2%} ({angle_c[5]}/{summary['total_samples']})")
        print(f"  < 10°: {angle_r[10]:.2%} ({angle_c[10]}/{summary['total_samples']})")
        print(f"  < 20°: {angle_r[20]:.2%} ({angle_c[20]}/{summary['total_samples']})")
        print(f"Translation Error Distribution:")
        print(f"  < 2cm:  {trans_r[2]:.2%} ({trans_c[2]}/{summary['total_samples']})")
        print(f"  < 5cm:  {trans_r[5]:.2%} ({trans_c[5]}/{summary['total_samples']})")
        print(f"  < 10cm: {trans_r[10]:.2%} ({trans_c[10]}/{summary['total_samples']})")
        
        print(f"\n【Improvement】")
        print(f"Rotation Error Improvement: {summary['improvement_angle']:.4f}° ({summary['improvement_angle']/summary['avg_coarse_angle']*100:.2f}%)")
        print(f"Translation Error Improvement: {summary['improvement_trans']:.4f}m ({summary['improvement_trans']/summary['avg_coarse_trans']*100:.2f}%)")
        
        print(f"\n【ACFM Predicted Residual Statistics】")
        print(f"Average Predicted Rotation Residual (Abs Mean per dim): {summary['avg_residual_rot']:.4f} rad ({np.degrees(summary['avg_residual_rot']):.4f}°)")
        print(f"Average Predicted Translation Residual (Abs Mean per dim): {summary['avg_residual_trans']:.4f} m")
        print("=" * 80)


class ACFMEvaluator:
    """Evaluator for Two-Stage Model (DFM + ACFM)"""
    
    def __init__(self, cfg, trans_stats=None, dfm_pretrained_path=None, acfm_pretrained_path=None):
        """Initialize evaluator
        
        Args:
            cfg: Configuration object
            trans_stats: Translation statistics for denormalization [min_x, max_x, min_y, max_y, min_z, max_z]
            dfm_pretrained_path: Path to pretrained DFM model checkpoint
            acfm_pretrained_path: Path to pretrained ACFM model checkpoint
        """
        self.cfg = cfg
        self.device = cfg.device
        self.trans_stats = trans_stats
        self.k = cfg.topk_k if hasattr(cfg, 'topk_k') else 10
        
        # Initialize DFM model
        self.dfm = DiscreteFlowMatching(cfg, device=self.device).to(self.device)
        if dfm_pretrained_path is not None:
            print(f"Loading DFM model from: {dfm_pretrained_path}")
            checkpoint = torch.load(dfm_pretrained_path, map_location=self.device)
            self.dfm.load_state_dict(checkpoint)
            print("DFM model loaded successfully")
        
        # Initialize ACFM model
        self.acfm = AdaptiveContinuousFlowMatching(cfg, device=self.device).to(self.device)
        if acfm_pretrained_path is not None:
            print(f"Loading ACFM model from: {acfm_pretrained_path}")
            checkpoint = torch.load(acfm_pretrained_path, map_location=self.device)
            self.acfm.load_state_dict(checkpoint)
            print("ACFM model loaded successfully")

    def test_step(self, batch_sample):
        """Perform single evaluation step (same as cotrainer.py eval_step)
        
        Returns:
            angle_errors: Rotation errors in degrees (ACFM refined)
            trans_errors: Translation errors in meters (ACFM refined)
            coarse_angle_errors: Rotation errors in degrees (DFM coarse)
            coarse_trans_errors: Translation errors in meters (DFM coarse)
        """
        self.dfm.eval()
        self.acfm.eval()
        
        with torch.no_grad():
            # GT pose
            rot_part_6d = batch_sample['zero_mean_gt_pose'][:, :6].to(self.device)
            trans_part = batch_sample['zero_mean_gt_pose'][:, -3:].to(self.device)
            gt_pose_continuous = torch.cat([rot_part_6d, trans_part], dim=1)
            
            # Point cloud features
            pts_feat = self.dfm.extract_pts_feature(batch_sample).to(self.device)
            
            # Stage 1: DFM
            topk_info = self.dfm.predict_coarse_pose(pts_feat, step_size=self.cfg.T_dfm, k=self.k)
            coarse_bins = topk_info['coarse_pose']
            
            # Convert angles using euler_angles_from_bins
            coarse_angle_bins = coarse_bins[:, :3]
            coarse_angles = euler_angles_from_bins(coarse_angle_bins, self.dfm.num_bins)
            coarse_angles = normalize_angles(coarse_angles)  # Normalize to [-pi, pi]
            
            # Convert translation using bins_to_numbers
            coarse_trans_bins = coarse_bins[:, 3:]
            coarse_trans = bins_to_numbers(
                coarse_trans_bins,
                self.trans_stats,
                self.dfm.num_bins
            )
            
            # Optimization: use 3D Euler angles consistently
            coarse_pose_continuous = torch.cat([coarse_angles, coarse_trans], dim=1)
            
            # Compute coarse errors (for comparison)
            coarse_rot_matrix = pytorch3d_transforms.euler_angles_to_matrix(
                coarse_angles, convention='ZYX'
            )
            gt_rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(rot_part_6d)
            coarse_angle_errors = rot_diff_degree(coarse_rot_matrix, gt_rot_matrix)
            coarse_trans_errors = torch.norm(coarse_trans - trans_part, dim=1)
            
            # Stage 2: ACFM
            delta_pred = self.acfm.sample(
                pts_feat, coarse_pose_continuous, topk_info,
                step_size=self.cfg.T_acfm, method='euler'
            )
            
            # 🔧 FIX: Properly add angle residual with wrapping
            pred_angles_delta = delta_pred[:, :3]
            pred_trans_delta = delta_pred[:, 3:]
            print(f"pred_angles_delta: {pred_angles_delta}")
            coarse_angles = coarse_pose_continuous[:, :3]
            coarse_trans = coarse_pose_continuous[:, 3:]
            
            # Use proper angle addition
            pred_angles = add_angle_residual(coarse_angles, pred_angles_delta)
            pred_trans = coarse_trans + pred_trans_delta
            
            # Final prediction (in Euler space)
            pred_pose = torch.cat([pred_angles, pred_trans], dim=1)
            
            # Compute errors
            pred_angles = pred_pose[:, :3]
            pred_trans = pred_pose[:, 3:]
            gt_rot_6d = gt_pose_continuous[:, :6]  # GT is originally 6D rotation + 3D translation
            gt_trans = gt_pose_continuous[:, 6:]
            
            # Euler -> Matrix -> 6D (for consistency check or just compare matrices directly)
            pred_rot_matrix = pytorch3d_transforms.euler_angles_to_matrix(
                pred_angles, convention='ZYX'  # Use ZYX to match DFM
            )
            gt_rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(gt_rot_6d)
            
            # Compute rotation error (degrees)
            angle_errors = rot_diff_degree(pred_rot_matrix, gt_rot_matrix)
            
            # Compute translation error (meters)
            trans_errors = torch.norm(pred_trans - gt_trans, dim=1)
            
            return angle_errors, trans_errors, coarse_angle_errors, coarse_trans_errors, pred_angles_delta, pred_trans_delta


def evaluate_model(cfg, test_loader, trans_stats):
    """Evaluate two-stage model performance"""
    # Get checkpoint paths from config
    dfm_ckpt = cfg.dfm_pretrained_path if hasattr(cfg, 'dfm_pretrained_path') else None
    acfm_ckpt = cfg.acfm_pretrained_path if hasattr(cfg, 'acfm_pretrained_path') else None
    
    if dfm_ckpt is None or acfm_ckpt is None:
        raise ValueError("Both dfm_pretrained_path and acfm_pretrained_path must be provided in config")
    
    # Initialize evaluator and metrics tracker
    evaluator = ACFMEvaluator(cfg, trans_stats, 
                             dfm_pretrained_path=dfm_ckpt,
                             acfm_pretrained_path=acfm_ckpt)
    evaluator.dfm.eval()
    evaluator.acfm.eval()
    metrics = MetricsTracker()
    
    # Evaluation loop
    pbar = tqdm(test_loader, desc="Evaluating")
    for batch in pbar:
        # Process batch data
        test_batch = process_batch(batch, cfg.device, cfg.pose_mode, 
                                   mini_batch_size=96, PTS_AUG_PARAMS=None)
        
        with torch.no_grad():
            # Perform evaluation step
            angle_errors, trans_errors, coarse_angle_errors, coarse_trans_errors, pred_delta_rot, pred_delta_trans = evaluator.test_step(test_batch)
            
            # Update metrics
            metrics.update(angle_errors, trans_errors, coarse_angle_errors, coarse_trans_errors, pred_delta_rot, pred_delta_trans)
            
            # Print batch statistics
            batch_stats = metrics.get_batch_stats(angle_errors, trans_errors, 
                                                 coarse_angle_errors, coarse_trans_errors)
            metrics.print_batch_stats(batch_stats)
    
    # Print final summary
    metrics.print_summary()


def main():
    """Main evaluation function"""
    # Load configuration and data
    cfg = get_config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    # Load data
    data_loaders = get_data_loaders_from_cfg(cfg, ['train'])
    test_loader = data_loaders['train_loader'] 
    print(f'Test set size: {len(test_loader)} batches')
    
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
