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
from networks.Adaptive_discrete_flow_matching import AdaptiveDiscreteFlowMatching
from datasets.dataloader import get_data_loaders_from_cfg, process_batch
import pytorch3d.transforms as pytorch3d_transforms
from networks.gf_algorithms.discrete_angle import euler_angles_from_bins, discretize_euler_angles
from networks.gf_algorithms.discrete_number import bins_to_numbers, translation_to_bins
from networks.gf_algorithms.discrete_number import get_dataset_translation_min_max
from networks.gf_algorithms.ema import ExponentialMovingAverage


class TwoStageADFMTrainer:
    """Two-stage trainer: DFM (Stage 1) + ADFM (Stage 2)
    
    Both stages use discrete flow matching with hierarchical bin refinement:
    - Stage 1 (DFM): Predicts coarse pose bins with uniform noise initialization
    - Stage 2 (ADFM): Predicts fine pose bins within Top-K coarse bins
      - Fine bins are distributed non-uniformly according to Top-K probabilities
    """
    
    def __init__(self, cfg, dfm_pretrained_path=None, adfm_pretrained_path=None, translation_status=None):
        self.cfg = cfg
        self.device = cfg.device
        self.k = cfg.topk_k if hasattr(cfg, 'topk_k') else 10
        self.translation_status = translation_status
        self.num_coarse_bins = cfg.num_bins  # Coarse bins from Stage 1
        self.num_fine_bins = cfg.num_fine_bins if hasattr(cfg, 'num_fine_bins') else cfg.num_bins  # Fine bins for Stage 2
        
        # Stage 1: DFM (can be frozen)
        self.dfm = DiscreteFlowMatching(cfg, device=self.device).to(self.device)
        if dfm_pretrained_path is not None:
            print(f"Loading DFM from {dfm_pretrained_path}")
            self.dfm.load_state_dict(torch.load(dfm_pretrained_path, map_location=self.device))
            print("DFM loaded successfully")
        
        # Stage 2: ADFM (Adaptive Discrete Flow Matching)
        self.adfm = AdaptiveDiscreteFlowMatching(
            cfg, device=self.device
        ).to(self.device)
        if adfm_pretrained_path is not None:
            print(f"Loading ADFM from {adfm_pretrained_path}")
            self.adfm.load_state_dict(torch.load(adfm_pretrained_path, map_location=self.device))
            print("ADFM loaded successfully")
        
        # Whether to freeze DFM
        self.freeze_dfm = cfg.freeze_dfm
        
        if self.freeze_dfm:
            for param in self.dfm.parameters():
                param.requires_grad = False
            self.dfm.eval()
            print("DFM is frozen")
        
        # Optimizer (only optimize ADFM if DFM is frozen)
        if self.freeze_dfm:
            self.optimizer = torch.optim.RAdam(
                self.adfm.parameters(),
                lr=cfg.lr,
                betas=(0.95, 0.999),
                weight_decay=1e-6
            )
        else:
            self.optimizer = torch.optim.RAdam(
                list(self.dfm.parameters()) + list(self.adfm.parameters()),
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
            self.adfm.parameters() if self.freeze_dfm else list(self.dfm.parameters()) + list(self.adfm.parameters()),
            decay=cfg.ema_rate if hasattr(cfg, 'ema_rate') else 0.995
        )
    
    def _pose_to_coarse_bins(self, rot_part_6d, trans_part):
        """Convert continuous pose to coarse bin indices (for Stage 1).
        
        Args:
            rot_part_6d: [bs, 6] 6D rotation representation
            trans_part: [bs, 3] translation
            
        Returns:
            bins: [bs, 6] coarse bin indices for each dimension
        """
        # Convert 6D rotation to Euler angles
        gt_rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(rot_part_6d)
        gt_angles = pytorch3d_transforms.matrix_to_euler_angles(gt_rot_matrix, convention='ZYX')
        
        # Convert Euler angles to coarse bins
        angle_bins = discretize_euler_angles(gt_angles, self.num_coarse_bins)
        
        # Convert translation to coarse bins
        trans_bins = translation_to_bins(trans_part, self.translation_status, self.num_coarse_bins)
        
        # Concatenate
        bins = torch.cat([angle_bins, trans_bins], dim=1)
        
        return bins
    
    def _coarse_bins_to_continuous_pose(self, bins):
        """Convert coarse bin indices to continuous pose values (for Stage 1 evaluation).
        
        Args:
            bins: [bs, 6] coarse bin indices
            
        Returns:
            pose: [bs, 6] continuous pose (Euler angles + translation)
        """
        angle_bins = bins[:, :3]
        angles = euler_angles_from_bins(angle_bins, self.num_coarse_bins)
        
        trans_bins = bins[:, 3:]
        trans = bins_to_numbers(trans_bins, self.translation_status, self.num_coarse_bins)
        
        return torch.cat([angles, trans], dim=1)
    
    def _pose_to_continuous(self, rot_part_6d, trans_part):
        """Convert pose to continuous values (Euler angles + translation).
        
        Args:
            rot_part_6d: [bs, 6] 6D rotation representation
            trans_part: [bs, 3] translation
            
        Returns:
            continuous_pose: [bs, 6] (Euler angles + translation)
        """
        gt_rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(rot_part_6d)
        gt_angles = pytorch3d_transforms.matrix_to_euler_angles(gt_rot_matrix, convention='ZYX')
        return torch.cat([gt_angles, trans_part], dim=1)

    def train_step(self, batch_sample):
        """Two-stage training step with hierarchical discrete flow matching."""
        # Extract GT pose
        rot_part_6d = batch_sample['zero_mean_gt_pose'][:, :6].to(self.device)
        trans_part = batch_sample['zero_mean_gt_pose'][:, -3:].to(self.device)
        
        # Extract point cloud for DFM
        pts = batch_sample['pts'].to(self.device)
        dfm_pts_feat = self.dfm.extract_pts_feature(pts).to(self.device)
        
        # Stage 1: DFM prediction and Top-K extraction
        with torch.no_grad() if self.freeze_dfm else torch.enable_grad():
            topk_info = self.dfm.predict_coarse_pose(dfm_pts_feat, step_size=self.cfg.T_dfm, k=self.k)
        
        # Extract point cloud features for ADFM
        pts_original = batch_sample['zero_mean_pts'].to(self.device)
        adfm_pts_feat = self.adfm.extract_pts_feature(pts_original).to(self.device)
        
        # Convert GT pose to continuous values, then to fine bins
        gt_continuous = self._pose_to_continuous(rot_part_6d, trans_part)
        gt_fine_bins = self.adfm.continuous_to_fine_bin(gt_continuous, topk_info, self.translation_status)
        
        # ADFM loss: predict fine bins using hierarchical top-k guided noise
        adfm_loss, adfm_loss_desc, loss_dict = self.adfm.loss(gt_fine_bins, adfm_pts_feat, topk_info)
        
        # Backward and optimize
        self.optimizer.zero_grad()
        adfm_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.adfm.parameters(), 1.0)
        self.optimizer.step()
        self.ema.update(self.adfm.parameters() if self.freeze_dfm else list(self.dfm.parameters()) + list(self.adfm.parameters()))
        
        # Add training statistics to loss dict
        with torch.no_grad():
            loss_dict['grad_norm'] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        
        return adfm_loss.item(), adfm_loss_desc, loss_dict
    
    def eval_step(self, batch_sample):
        """Evaluation step: compute rotation and translation errors for both stages."""
        self.dfm.eval()
        self.adfm.eval()
        
        # Store current parameters and copy EMA parameters
        self.ema.store(self.adfm.parameters() if self.freeze_dfm else list(self.dfm.parameters()) + list(self.adfm.parameters()))
        self.ema.copy_to(self.adfm.parameters() if self.freeze_dfm else list(self.dfm.parameters()) + list(self.adfm.parameters()))
        
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
                
                # Stage 1: DFM prediction (coarse bins)
                topk_info = self.dfm.predict_coarse_pose(dfm_pts_feat, step_size=self.cfg.T_dfm, k=self.k)
                
                # Convert coarse bins to continuous pose
                coarse_pose = self._coarse_bins_to_continuous_pose(topk_info['coarse_pose'])
                coarse_rot_matrix = pytorch3d_transforms.euler_angles_to_matrix(
                    coarse_pose[:, :3], convention='ZYX'
                )
                coarse_trans = coarse_pose[:, 3:]
                
                # Compute Stage 1 errors
                coarse_diff_angle = rot_diff_degree(coarse_rot_matrix, gt_rot_matrix)
                coarse_diff_trans = torch.norm(coarse_trans - trans_part, dim=1)
                
                # Stage 2: ADFM prediction (fine bins within Top-K coarse bins)
                adfm_pts_feat = self.adfm.extract_pts_feature(pts_original).to(self.device)
                refined_fine_bins = self.adfm.sample(adfm_pts_feat, topk_info, step_size=self.cfg.T_adfm)
                
                # Convert fine bins to continuous pose using hierarchical mapping
                refined_pose = self.adfm.fine_bin_to_continuous(refined_fine_bins, topk_info, self.translation_status)
                refined_rot_matrix = pytorch3d_transforms.euler_angles_to_matrix(
                    refined_pose[:, :3], convention='ZYX'
                )
                refined_trans = refined_pose[:, 3:]
                
                # Compute Stage 2 errors
                refined_diff_angle = rot_diff_degree(refined_rot_matrix, gt_rot_matrix)
                refined_diff_trans = torch.norm(refined_trans - trans_part, dim=1)
                
                result = {
                    'stage1_angle': coarse_diff_angle.mean().item(),
                    'stage1_trans': coarse_diff_trans.mean().item(),
                    'stage2_angle': refined_diff_angle.mean().item(),
                    'stage2_trans': refined_diff_trans.mean().item()
                }
        finally:
            # Restore original parameters
            self.ema.restore(self.adfm.parameters() if self.freeze_dfm else list(self.dfm.parameters()) + list(self.adfm.parameters()))
        
        return result


def train_two_stage_adfm(cfg, train_loader, val_loader, test_loader, translation_status=None):
    """Two-stage ADFM training main function"""
    dfm_path = cfg.dfm_pretrained_path if hasattr(cfg, 'dfm_pretrained_path') else None
    adfm_path = cfg.adfm_pretrained_path if hasattr(cfg, 'adfm_pretrained_path') else None
    
    trainer = TwoStageADFMTrainer(
        cfg, 
        dfm_pretrained_path=dfm_path, 
        adfm_pretrained_path=adfm_path,
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
        
    wandb.watch(trainer.adfm, log="all", log_freq=100)
    
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
                if current_step % 10 == 0:
                    wandb_log_dict = {
                        "step": current_step,
                        "epoch": epoch,
                        "learning_rate": trainer.optimizer.param_groups[0]['lr']
                    }
                    for key, value in loss_dict.items():
                        wandb_log_dict[f"train/{key}"] = value
                    wandb.log(wandb_log_dict)
                
                if current_step % 100 == 0:
                    valid_losses = [l for l in train_losses[-100:] if l > 0]
                    avg_loss = np.mean(valid_losses) if valid_losses else 0.0
                    wandb.log({"step": current_step, "train/avg_loss_100": avg_loss})
                
                if current_step % eval_freq == 0:
                    # Switch to eval mode
                    if not trainer.freeze_dfm:
                        trainer.dfm.eval()
                    trainer.adfm.eval()
                    
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
                    print(f"  Stage 1 (DFM) - Angle Error: {stage1_angle_mean:.4f}°, Trans Error: {stage1_trans_mean:.4f}m")
                    print(f"  Stage 2 (ADFM) - Angle Error: {stage2_angle_mean:.4f}°, Trans Error: {stage2_trans_mean:.4f}m")
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
                        f"adfm_step_{current_step}_angle_{stage2_angle_mean:.4f}_trans_{stage2_trans_mean:.4f}.pt"
                    )
                    trainer.ema.store(trainer.adfm.parameters() if trainer.freeze_dfm else list(trainer.dfm.parameters()) + list(trainer.adfm.parameters()))
                    trainer.ema.copy_to(trainer.adfm.parameters() if trainer.freeze_dfm else list(trainer.dfm.parameters()) + list(trainer.adfm.parameters()))
                    torch.save(trainer.adfm.state_dict(), save_path)
                    trainer.ema.restore(trainer.adfm.parameters() if trainer.freeze_dfm else list(trainer.dfm.parameters()) + list(trainer.adfm.parameters()))
                    print(f"Model (EMA) saved at step {current_step}")
                    
                    # Switch back to train mode
                    if not trainer.freeze_dfm:
                        trainer.dfm.train()
                    trainer.adfm.train()
            
            # Print epoch average loss
            if len(epoch_losses) > 0:
                epoch_avg_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch} completed: Average Loss = {epoch_avg_loss:.4f}, Steps in epoch = {len(epoch_losses)}, Total steps = {current_step}/{total_steps}")
                wandb.log({"epoch": epoch, "epoch_avg_loss": epoch_avg_loss, "step": current_step})
            
            epoch += 1
    
    else:
        # Epochs-based training (placeholder)
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
    translation_status = [-0.3785014748573303, 0.39416784048080444, -0.4042277932167053, 0.39954620599746704, -0.30842161178588867, 0.7598943710327148]
    print(f"Translation statistics: {translation_status}")
    print(f"  X range: [{translation_status[0]:.4f}, {translation_status[1]:.4f}]")
    print(f"  Y range: [{translation_status[2]:.4f}, {translation_status[3]:.4f}]")
    print(f"  Z range: [{translation_status[4]:.4f}, {translation_status[5]:.4f}]")
    
    # Wandb configuration
    num_fine_bins = cfg.num_fine_bins if hasattr(cfg, 'num_fine_bins') else cfg.num_bins
    wandb_config = {
        "model_type": "two_stage_dfm_adfm_hierarchical",
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "freeze_dfm": cfg.freeze_dfm if hasattr(cfg, 'freeze_dfm') else True,
        "topk_k": cfg.topk_k if hasattr(cfg, 'topk_k') else 10,
        "translation_status": translation_status,
        "num_coarse_bins": cfg.num_bins,
        "num_fine_bins": num_fine_bins,
    }
    
    wandb.init(project="two_stage_6d_pose_adfm", config=wandb_config)
    
    # Start training
    train_two_stage_adfm(cfg, train_loader, val_loader, test_loader, translation_status=translation_status)


if __name__ == "__main__":
    main()

