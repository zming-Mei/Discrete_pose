import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import get_config
from utils.metrics import rot_diff_degree
from networks.discrete_flow_matching import DiscreteFlowMatching
from networks.Adaptive_continuous_flow_matching import AdaptiveContinuousFlowMatching
from datasets.dataloader import get_data_loaders_from_cfg, process_batch
import pytorch3d.transforms as pytorch3d_transforms
from networks.gf_algorithms.discrete_angle import euler_angles_from_bins
from networks.gf_algorithms.discrete_number import bins_to_numbers
from networks.gf_algorithms.discrete_number import get_dataset_translation_min_max
from networks.gf_algorithms.ema import ExponentialMovingAverage

class TwoStageTrainer:
    """Two-stage trainer: DFM + ACFM"""
    
    def __init__(self, cfg, dfm_pretrained_path=None, acfm_pretrained_path=None, translation_status=None):
        self.cfg = cfg
        self.device = cfg.device
        self.k = cfg.topk_k if hasattr(cfg, 'topk_k') else 10
        self.filter_bad_data = cfg.filter_bad_data if hasattr(cfg, 'filter_bad_data') else False
        self.filter_angle_threshold = 10.0 # degrees
        self.filter_trans_threshold = 0.1 # meters (10cm)
        self.translation_status = translation_status
        self.pts_transform = cfg.pts_transform if hasattr(cfg, 'pts_transform') else False
        
        # Stage 1: DFM (can be frozen)
        self.dfm = DiscreteFlowMatching(cfg, device=self.device).to(self.device)
        if dfm_pretrained_path is not None:
            print(f"Loading DFM from {dfm_pretrained_path}")
            self.dfm.load_state_dict(torch.load(dfm_pretrained_path, map_location=self.device))
            print("DFM loaded successfully")
        
        # Stage 2: ACFM
        self.acfm = AdaptiveContinuousFlowMatching(
            cfg, device=self.device
        ).to(self.device)
        if acfm_pretrained_path is not None:
            print(f"Loading ACFM from {acfm_pretrained_path}")
            self.acfm.load_state_dict(torch.load(acfm_pretrained_path, map_location=self.device))
            print("ACFM loaded successfully")
        
        # Whether to freeze DFM
        self.freeze_dfm = cfg.freeze_dfm
        
        # Rotation representation (euler, axis_angle, 6d supported)
        self.rotation_type = cfg.acfm_rotation_type if hasattr(cfg, 'acfm_rotation_type') else 'euler'
        assert self.rotation_type in ['euler', 'axis_angle', '6d'], f"Unsupported rotation_type: {self.rotation_type}"
        print(f"TwoStageTrainer: rotation_type = {self.rotation_type}")
        self.acfm_predict_delta = cfg.acfm_predict_delta if hasattr(cfg, 'acfm_predict_delta') else False
        print(f"TwoStageTrainer: acfm_predict_delta = {self.acfm_predict_delta}")
        
        if self.freeze_dfm:
            for param in self.dfm.parameters():
                param.requires_grad = False
            self.dfm.eval()
            print("DFM is frozen")
        
        # Optimizer (only optimize ACFM if DFM is frozen)
        if self.freeze_dfm:
            self.optimizer = torch.optim.RAdam(
                self.acfm.parameters(),
                lr=cfg.lr,
                betas=(0.95, 0.999),
                weight_decay=1e-6
            )
        else:
            self.optimizer = torch.optim.RAdam(
                list(self.dfm.parameters()) + list(self.acfm.parameters()),
                lr=cfg.lr,
                betas=(0.95, 0.999),
                weight_decay=1e-6
            )
        total_steps = cfg.total_steps if hasattr(cfg, 'total_steps') else None
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=cfg.eta_min
        )
        
        # EMA
        self.ema = ExponentialMovingAverage(
            self.acfm.parameters() if self.freeze_dfm else list(self.dfm.parameters()) + list(self.acfm.parameters()),
            decay=cfg.ema_rate if hasattr(cfg, 'ema_rate') else 0.995
        )
    
    def _bins_to_continuous_pose(self, coarse_bins):
        """Convert bin indices to continuous pose values."""
        coarse_angle_bins = coarse_bins[:, :3]
        coarse_angles = euler_angles_from_bins(coarse_angle_bins, self.dfm.num_bins)
        
        coarse_trans_bins = coarse_bins[:, 3:]
        coarse_trans = bins_to_numbers(
            coarse_trans_bins, 
            self.translation_status, 
            self.dfm.num_bins
        )
        if self.rotation_type == 'axis_angle':
            coarse_rot_matrix = pytorch3d_transforms.euler_angles_to_matrix(coarse_angles, convention='ZYX')
            coarse_axis_angle = pytorch3d_transforms.so3_log_map(coarse_rot_matrix)
            return torch.cat([coarse_axis_angle, coarse_trans], dim=1)
        elif self.rotation_type == '6d':
            coarse_rot_matrix = pytorch3d_transforms.euler_angles_to_matrix(coarse_angles, convention='ZYX')
            coarse_rot_6d = pytorch3d_transforms.matrix_to_rotation_6d(coarse_rot_matrix)
            return torch.cat([coarse_rot_6d, coarse_trans], dim=1)
        else:  # euler
            return torch.cat([coarse_angles, coarse_trans], dim=1)

    def _get_gt_continuous_pose(self, rot_part_6d, trans_part):
        """Convert GT 6D pose to continuous representation (Euler/AxisAngle/6D + Trans)."""
        if self.rotation_type == 'axis_angle':
            gt_rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(rot_part_6d)
            gt_axis_angle = pytorch3d_transforms.so3_log_map(gt_rot_matrix)
            return torch.cat([gt_axis_angle, trans_part], dim=1)
        elif self.rotation_type == '6d':
            return torch.cat([rot_part_6d, trans_part], dim=1)
        else:  # euler
            gt_rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(rot_part_6d)
            gt_angles = pytorch3d_transforms.matrix_to_euler_angles(gt_rot_matrix, convention='ZYX')
            return torch.cat([gt_angles, trans_part], dim=1)
    
    def _transform_pointcloud_by_pose(self, pts, pose):

        if self.rotation_type == 'axis_angle':
            axis_angle = pose[:, :3]
            trans = pose[:, 3:]
            rot_matrix = pytorch3d_transforms.so3_exp_map(axis_angle)
        elif self.rotation_type == '6d':
            rot_6d = pose[:, :6]
            trans = pose[:, 6:]
            rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(rot_6d)
        elif self.rotation_type == 'euler':
            angles = pose[:, :3]
            trans = pose[:, 3:]
            rot_matrix = pytorch3d_transforms.euler_angles_to_matrix(
                angles, convention='ZYX'
            )
        else:
            raise ValueError(f"Unsupported rotation_type: {self.rotation_type}")
        # Forward transform: P' = R @ P + t
        # Note: For row vectors, this is P @ R^T + t
        pts_transformed = torch.matmul(pts, rot_matrix.transpose(-2, -1)) + trans.unsqueeze(1)
        
        return pts_transformed

    def _filter_bad_data(self, batch_sample, pts_feat, topk_info):

        # Extract GT pose from batch_sample
        rot_part_6d = batch_sample['zero_mean_gt_pose'][:, :6].to(self.device)
        trans_part = batch_sample['zero_mean_gt_pose'][:, -3:].to(self.device)
        pose_gt = self._get_gt_continuous_pose(rot_part_6d, trans_part)    
        gt_rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(rot_part_6d)
        coarse_pose_continuous = self._bins_to_continuous_pose(topk_info['coarse_pose'])
        # Compute errors for DFM
        if self.rotation_type == 'axis_angle':
            coarse_rot_matrix = pytorch3d_transforms.so3_exp_map(coarse_pose_continuous[:, :3])
            coarse_trans = coarse_pose_continuous[:, 3:]
        elif self.rotation_type == '6d':
            coarse_rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(coarse_pose_continuous[:, :6])
            coarse_trans = coarse_pose_continuous[:, 6:]
        else:  # euler
            coarse_rot_matrix = pytorch3d_transforms.euler_angles_to_matrix(
                coarse_pose_continuous[:, :3], convention='ZYX'
            )
            coarse_trans = coarse_pose_continuous[:, 3:]

        # Compute errors
        coarse_diff_angle = rot_diff_degree(coarse_rot_matrix, gt_rot_matrix)
        coarse_diff_trans = torch.norm(coarse_trans - trans_part, dim=1)
        
        # Mask: Good data only
        mask = (coarse_diff_angle <= self.filter_angle_threshold) & (coarse_diff_trans <= self.filter_trans_threshold)
        mask_sum = mask.sum()

        filtered_pose_gt = pose_gt[mask]
        filtered_pts_feat = pts_feat[mask]

        filtered_topk_info = {}
        for k, v in topk_info.items():
            if isinstance(v, torch.Tensor):
                filtered_topk_info[k] = v[mask]
            else:
                filtered_topk_info[k] = v
                
        return {
            'pose_gt': filtered_pose_gt,
            'pts_feat': filtered_pts_feat,
            'topk_info': filtered_topk_info,
            'mask_sum': mask_sum
        }

    def train_step(self, batch_sample):
        """Two-stage training step."""
        # Extract GT pose
        rot_part_6d = batch_sample['zero_mean_gt_pose'][:, :6].to(self.device)
        trans_part = batch_sample['zero_mean_gt_pose'][:, -3:].to(self.device)
        pose_gt = self._get_gt_continuous_pose(rot_part_6d, trans_part)

        # Extract point cloud for DFM
        pts = batch_sample['pts'].to(self.device)
        pts_original = batch_sample['zero_mean_pts'].to(self.device)
        dfm_pts_feat = self.dfm.extract_pts_feature(pts).to(self.device)

        # Stage 1: DFM prediction and Top-K extraction
        with torch.no_grad() if self.freeze_dfm else torch.enable_grad():
            topk_info = self.dfm.predict_coarse_pose(dfm_pts_feat, step_size=self.cfg.T_dfm, k=self.k)
            coarse_pose_continuous = self._bins_to_continuous_pose(topk_info['coarse_pose'])
        
        # Transform point cloud by coarse pose (inverse transform) for ACFM
        if self.pts_transform:
            pts_transformed = self._transform_pointcloud_by_pose(pts_original, coarse_pose_continuous)
        else:
            pts_transformed = pts_original

        # Extract point cloud features for ACFM
        acfm_pts_feat = self.acfm.extract_pts_feature(pts_transformed).to(self.device)
        
        # --- Data Filtering Logic ---
        if self.filter_bad_data:
            filter_result = self._filter_bad_data(
                batch_sample, acfm_pts_feat, topk_info
            )
        
            if filter_result['mask_sum'] == 0:
                print("Warning: All data filtered out in this batch!")
                return 0.0, "Skipped (Filtered)", {}

                
            pose_gt = filter_result['pose_gt']
            acfm_pts_feat = filter_result['pts_feat']
            topk_info = filter_result['topk_info']
        
        # ACFM loss: direct prediction from original points + topk condition
        acfm_loss, acfm_loss_desc, loss_dict = self.acfm.loss(pose_gt, acfm_pts_feat, topk_info, self.translation_status)
        
        # Backward and optimize
        self.optimizer.zero_grad()
        acfm_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.acfm.parameters(), 1.0)
        self.optimizer.step()
        self.ema.update(self.acfm.parameters() if self.freeze_dfm else list(self.dfm.parameters()) + list(self.acfm.parameters()))
        
        # Add training statistics to loss dict
        with torch.no_grad():
            loss_dict['grad_norm'] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        return acfm_loss.item(), acfm_loss_desc, loss_dict
    
    def eval_step(self, batch_sample):
        """Evaluation step: compute rotation and translation errors for both stages."""
        self.dfm.eval()
        self.acfm.eval()
        
        # Store current parameters and copy EMA parameters
        self.ema.store(self.acfm.parameters() if self.freeze_dfm else list(self.dfm.parameters()) + list(self.acfm.parameters()))
        self.ema.copy_to(self.acfm.parameters() if self.freeze_dfm else list(self.dfm.parameters()) + list(self.acfm.parameters()))
        
        try:
            with torch.no_grad():
                # Extract GT pose
                rot_part_6d = batch_sample['zero_mean_gt_pose'][:, :6].to(self.device)
                trans_part = batch_sample['zero_mean_gt_pose'][:, -3:].to(self.device)
                gt_rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(rot_part_6d)

                # Extract point cloud
                pts = batch_sample['pts'].to(self.device)
                pts_original = batch_sample['zero_mean_pts'].to(self.device)
                dfm_pts_feat = self.dfm.extract_pts_feature(pts).to(self.device)

                # Stage 1: DFM prediction
                topk_info = self.dfm.predict_coarse_pose(dfm_pts_feat, step_size=self.cfg.T_dfm, k=self.k)
                coarse_pose_continuous = self._bins_to_continuous_pose(topk_info['coarse_pose'])

                # Transform point cloud by coarse pose
                if self.pts_transform:
                    pts_transformed = self._transform_pointcloud_by_pose(pts_original, coarse_pose_continuous)
                else:
                    pts_transformed = pts_original
                
                # Extract point cloud features for ACFM
                acfm_pts_feat = self.acfm.extract_pts_feature(pts_transformed).to(self.device)
                
                # Compute Stage 1 errors
                if self.rotation_type == 'axis_angle':
                    coarse_rot_matrix = pytorch3d_transforms.so3_exp_map(coarse_pose_continuous[:, :3])
                    coarse_trans = coarse_pose_continuous[:, 3:]
                elif self.rotation_type == '6d':
                    coarse_rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(coarse_pose_continuous[:, :6])
                    coarse_trans = coarse_pose_continuous[:, 6:]
                else:  # euler
                    coarse_rot_matrix = pytorch3d_transforms.euler_angles_to_matrix(
                        coarse_pose_continuous[:, :3], convention='ZYX'
                    )
                    coarse_trans = coarse_pose_continuous[:, 3:]
                
                coarse_diff_angle = rot_diff_degree(coarse_rot_matrix, gt_rot_matrix)
                coarse_diff_trans = torch.norm(coarse_trans - trans_part, dim=1)
                
                # Stage 2: ACFM - Direct Prediction
                # Note: We use original pts_feat (not transformed)
                pred_pose = self.acfm.sample(acfm_pts_feat, topk_info, self.translation_status, step_size=self.cfg.T_acfm, method='euler')
                
                # Decompose predicted pose
                if self.rotation_type == 'axis_angle':
                    pred_rot_matrix = pytorch3d_transforms.so3_exp_map(pred_pose[:, :3])
                    pred_trans = pred_pose[:, 3:]
                elif self.rotation_type == '6d':
                    pred_rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(pred_pose[:, :6])
                    pred_trans = pred_pose[:, 6:]
                else: # Euler
                    pred_rot_matrix = pytorch3d_transforms.euler_angles_to_matrix(
                        pred_pose[:, :3], convention='ZYX'
                    )
                    pred_trans = pred_pose[:, 3:]
                
                # Compute Stage 2 errors
                refined_diff_angle = rot_diff_degree(pred_rot_matrix, gt_rot_matrix)
                refined_diff_trans = torch.norm(pred_trans - trans_part, dim=1)
                
                result = {
                    'stage1_angle': coarse_diff_angle.mean().item(),
                    'stage1_trans': coarse_diff_trans.mean().item(),
                    'stage2_angle': refined_diff_angle.mean().item(),
                    'stage2_trans': refined_diff_trans.mean().item()
                }
        finally:
            # Restore original parameters
            self.ema.restore(self.acfm.parameters() if self.freeze_dfm else list(self.dfm.parameters()) + list(self.acfm.parameters()))
        
        return result


def train_two_stage(cfg, train_loader, val_loader, test_loader, translation_status=None):
    """Two-stage training main function"""
    dfm_path = cfg.dfm_pretrained_path if hasattr(cfg, 'dfm_pretrained_path') else None
    acfm_path = cfg.acfm_pretrained_path if hasattr(cfg, 'acfm_pretrained_path') else None
    
    trainer = TwoStageTrainer(
        cfg, 
        dfm_pretrained_path=dfm_path, 
        acfm_pretrained_path=acfm_path,
        translation_status=translation_status
    )
    
    # Training mode
    use_steps = cfg.total_steps is not None if hasattr(cfg, 'total_steps') else False
    
    if use_steps:
        total_steps = cfg.total_steps
        eval_freq = cfg.eval_freq_steps if hasattr(cfg, 'eval_freq_steps') else 1000
        print(f"Training mode: steps-based, total_steps={total_steps}")
    else:
        total_epochs = cfg.n_epochs
        eval_freq = cfg.eval_freq if hasattr(cfg, 'eval_freq') else 5
        print(f"Training mode: epochs-based, total_epochs={total_epochs}")
        
    wandb.watch(trainer.acfm, log="all", log_freq=100)
    
    save_dir = cfg.output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    if use_steps:
        # Steps-based training
        current_step = 0
        epoch = 0
        train_losses = []
        
        while current_step < total_steps:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} (Step {current_step}/{total_steps})")
            epoch_losses = []
            
            for batch in pbar:
                if current_step >= total_steps:
                    break
                
                batch = process_batch(batch, cfg.device, cfg.pose_mode, mini_batch_size=96, PTS_AUG_PARAMS=None)
                loss, loss_desc, loss_dict = trainer.train_step(batch)
                train_losses.append(loss)
                epoch_losses.append(loss)
                
                trainer.scheduler.step()
                current_step += 1
                
                pbar.set_postfix({
                    "Step": f"{current_step}/{total_steps}",
                    "Details": loss_desc
                })
                
                # Log detailed losses to wandb
                if current_step % 10 == 0:  # Log every 10 steps
                    wandb_log_dict = {
                        "step": current_step,
                        "epoch": epoch,
                        "learning_rate": trainer.optimizer.param_groups[0]['lr']
                    }
                    # Add all detailed losses
                    for key, value in loss_dict.items():
                        wandb_log_dict[f"train/{key}"] = value
                    wandb.log(wandb_log_dict)
                
                if current_step % 100 == 0:
                    # Filter out zeros from train_losses for avg calc if needed, but usually loss > 0
                    valid_losses = [l for l in train_losses[-100:] if l > 0]
                    avg_loss = np.mean(valid_losses) if valid_losses else 0.0
                    wandb.log({"step": current_step, "train/avg_loss_100": avg_loss})
                
                if current_step % eval_freq == 0:
                    # Switch to eval mode
                    if not trainer.freeze_dfm:
                        trainer.dfm.eval()
                    trainer.acfm.eval()
                    
                    stage1_angles, stage1_trans, stage2_angles, stage2_trans = [], [], [], []
                    print(f"\nValidation at step {current_step}...")
                    for val_batch in tqdm(val_loader, desc="Validation"):
                        val_batch = process_batch(val_batch, cfg.device, cfg.pose_mode, mini_batch_size=96, PTS_AUG_PARAMS=None)
                        try:
                            result = trainer.eval_step(val_batch)
                            stage1_angles.append(result['stage1_angle'])
                            stage1_trans.append(result['stage1_trans'])
                            stage2_angles.append(result['stage2_angle'])
                            stage2_trans.append(result['stage2_trans'])
                        except Exception as e:
                            print(f"Eval error: {e}")
                            continue
                    
                    stage1_angle_mean = np.mean(stage1_angles) if stage1_angles else float('inf')
                    stage1_trans_mean = np.mean(stage1_trans) if stage1_trans else float('inf')
                    stage2_angle_mean = np.mean(stage2_angles) if stage2_angles else float('inf')
                    stage2_trans_mean = np.mean(stage2_trans) if stage2_trans else float('inf')
                    
                    print(f"Step {current_step} - Validation Results:")
                    print(f"  Stage 1 - Angle Error: {stage1_angle_mean:.4f}°, Trans Error: {stage1_trans_mean:.4f}m")
                    print(f"  Stage 2 - Angle Error: {stage2_angle_mean:.4f}°, Trans Error: {stage2_trans_mean:.4f}m")
                    print(f"  Improvement - Angle: {stage1_angle_mean - stage2_angle_mean:.4f}°, Trans: {stage1_trans_mean - stage2_trans_mean:.4f}m")
                    
                    wandb.log({
                        "step": current_step,
                        "val/stage1_angle_error": stage1_angle_mean,
                        "val/stage1_trans_error": stage1_trans_mean,
                        "val/stage2_angle_error": stage2_angle_mean,
                        "val/stage2_trans_error": stage2_trans_mean,
                        "val/angle_improvement": stage1_angle_mean - stage2_angle_mean,
                        "val/trans_improvement": stage1_trans_mean - stage2_trans_mean
                    })
                    
                    # Save EMA model state
                    save_path = os.path.join(
                        save_dir,
                        f"acfm_step_{current_step}_angle_{stage2_angle_mean:.4f}_trans_{stage2_trans_mean:.4f}.pt"
                    )
                    # Store current params and copy EMA params for saving
                    trainer.ema.store(trainer.acfm.parameters() if trainer.freeze_dfm else list(trainer.dfm.parameters()) + list(trainer.acfm.parameters()))
                    trainer.ema.copy_to(trainer.acfm.parameters() if trainer.freeze_dfm else list(trainer.dfm.parameters()) + list(trainer.acfm.parameters()))
                    torch.save(trainer.acfm.state_dict(), save_path)
                    trainer.ema.restore(trainer.acfm.parameters() if trainer.freeze_dfm else list(trainer.dfm.parameters()) + list(trainer.acfm.parameters()))
                    print(f"Model (EMA) saved at step {current_step}")
                    
                    # Switch back to train mode
                    if not trainer.freeze_dfm:
                        trainer.dfm.train()
                    trainer.acfm.train()
            
            # Print epoch average loss
            if len(epoch_losses) > 0:
                epoch_avg_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch} completed: Average Loss = {epoch_avg_loss:.4f}, Steps in epoch = {len(epoch_losses)}, Total steps = {current_step}/{total_steps}")
                wandb.log({"epoch": epoch, "epoch_avg_loss": epoch_avg_loss, "step": current_step})
            
            epoch += 1
    
    else:
        # Epochs-based training
        pass


def main():
    cfg = get_config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    # Load data
    data_loaders = get_data_loaders_from_cfg(cfg, ['train', 'val', 'test'])
    train_loader = data_loaders['train_loader']
    val_loader = data_loaders['val_loader']
    test_loader = data_loaders['test_loader']
    
    print('train_set: ', len(train_loader))
    print('val_set: ', len(val_loader))
    print('test_set: ', len(test_loader))
    
    # Compute translation statistics from training data
    print("Computing translation statistics from training data...")
    #translation_status = get_dataset_translation_min_max(train_loader, cfg)
    #translation_status = [-0.42640459537506104, 0.4034724235534668, -0.4079521894454956, 0.4090191125869751, -0.31017589569091797, 0.7824592590332031]
    translation_status = [-0.3785014748573303, 0.39416784048080444, -0.4042277932167053, 0.39954620599746704, -0.30842161178588867, 0.7598943710327148]
    print(f"Translation statistics: {translation_status}")
    print(f"  X range: [{translation_status[0]:.4f}, {translation_status[1]:.4f}]")
    print(f"  Y range: [{translation_status[2]:.4f}, {translation_status[3]:.4f}]")
    print(f"  Z range: [{translation_status[4]:.4f}, {translation_status[5]:.4f}]")
    
    # Wandb configuration
    wandb_config = {
        "model_type": "two_stage_dfm_acfm",
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "freeze_dfm": cfg.freeze_dfm if hasattr(cfg, 'freeze_dfm') else True,
        "topk_k": cfg.topk_k if hasattr(cfg, 'topk_k') else 10,
        "translation_status": translation_status,
        "use_coarse_as_x0": cfg.use_coarse_as_x0 if hasattr(cfg, 'use_coarse_as_x0') else False,
        "acfm_rotation_type": cfg.acfm_rotation_type if hasattr(cfg, 'acfm_rotation_type') else 'euler',
        "acfm_predict_delta": cfg.acfm_predict_delta if hasattr(cfg, 'acfm_predict_delta') else False,
        "filter_bad_data": cfg.filter_bad_data if hasattr(cfg, 'filter_bad_data') else False,
    }
    
    wandb.init(project="two_stage_6d_pose", config=wandb_config)
    
    # Start training
    train_two_stage(cfg, train_loader, val_loader, test_loader, translation_status=translation_status)


if __name__ == "__main__":
    main()
