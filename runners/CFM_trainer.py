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
from networks.gf_algorithms.discrete_angle import *
from datasets.dataloader import (get_data_loaders_from_cfg, process_batch)
import pytorch3d.transforms as pytorch3d_transforms
from networks.gf_algorithms.discrete_number import *
from networks.continuous_flow_matching import ContinuousFlowMatching
from runners.CFM_eval import ContinuousFlowEvaluator


class ContinuousFlowTrainer:
    """Trainer for Continuous Flow Matching model"""
    
    def __init__(self, cfg, trans_stats=None, pretrained_path=None, total_steps=None):
        self.cfg = cfg
        self.device = cfg.device
        # Note: trans_stats no longer used (normalization removed), kept for backward compatibility
        self.trans_stats = None
        
        # Initialize continuous flow matching model
        self.model = ContinuousFlowMatching(
            cfg,
            device=self.device
        ).to(self.device)
        
        # Load pretrained model if provided
        if pretrained_path is not None:
            print(f"Loading pretrained model from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print("Model loaded successfully")
        
        # Create evaluator instance for validation/testing during training
        self.evaluator = ContinuousFlowEvaluator(cfg, trans_stats, pretrained_path=None)
        self.evaluator.model = self.model  # Share the same model instance
        
        # Optimizer & Scheduler
        self.optimizer = torch.optim.RAdam(
            self.model.parameters(),
            lr=cfg.lr,  
            betas=(0.95, 0.999), 
            weight_decay=1e-6  
        )
        
        # Setup scheduler based on training mode (steps vs epochs)
        if total_steps is not None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=cfg.eta_min
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.n_epochs,
                eta_min=cfg.eta_min
            )

    def train_step(self, batch_sample):
        """
        Execute one training step
        Args:
            batch_sample: Batch data
        Returns:
            loss: Loss value
            loss_description: Loss description string
        """
        self.model.train()
        
        # Extract rotation and translation from ground truth pose
        # Use 6D rotation representation directly, no normalization needed
        rot_part_6d = batch_sample['zero_mean_gt_pose'][:, :6]  # 6D rotation
        trans_part = batch_sample['zero_mean_gt_pose'][:, -3:]  # Translation
        
        # Combine target pose (9D: 6D rotation + 3D translation)
        # Note: No normalization, learn directly in original space
        gt_pose = torch.cat([rot_part_6d, trans_part], dim=1).to(self.device)  # [bs, 9]
        
        # Extract point cloud features
        pts_feat = self.model.extract_pts_feature(batch_sample).to(self.device)
        
        # Compute loss
        loss, loss_description = self.model.loss(gt_pose, pts_feat)
        
        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item(), loss_description

    def eval_step(self, batch_sample):
        """
        Execute one evaluation step
        Returns:
            diff_angle: Average rotation error (degrees)
            diff_trans: Average translation error (meters)
        """
        self.model.eval()
        
        try:
            with torch.no_grad():
                # Use the evaluator's test_step for consistent evaluation
                diff_angle, diff_trans = self.evaluator.test_step(batch_sample)
                # Return mean values for logging
                return diff_angle.mean().item(), diff_trans.mean().item()
            
        except Exception as e:
            print(f"Error in eval_step: {e}")
            return float('inf'), float('inf')


def train_data(cfg, train_loader, val_loader, test_loader, trans_stats=None):
    """
    Training function - supports both steps-based and epochs-based training modes
    
    Note: trans_stats parameter is kept for backward compatibility but no longer used (normalization removed)
    """
    pretrained_path = cfg.pretrained_model_path if hasattr(cfg, 'pretrained_model_path') else None
    
    # Determine training mode
    use_steps = cfg.total_steps is not None if hasattr(cfg, 'total_steps') else False
    
    if use_steps:
        total_steps = cfg.total_steps
        eval_freq = cfg.eval_freq_steps if hasattr(cfg, 'eval_freq_steps') else 1000
        trainer = ContinuousFlowTrainer(cfg, trans_stats=None, pretrained_path=pretrained_path, total_steps=total_steps)
        print(f"Training mode: steps-based, total_steps={total_steps}, eval_freq={eval_freq} steps")
    else:
        total_epochs = cfg.n_epochs
        eval_freq = cfg.eval_freq if hasattr(cfg, 'eval_freq') else 5
        trainer = ContinuousFlowTrainer(cfg, trans_stats=None, pretrained_path=pretrained_path, total_steps=None)
        print(f"Training mode: epochs-based, total_epochs={total_epochs}, eval_freq={eval_freq} epochs")
    
    # Watch model with wandb
    wandb.watch(trainer.model, log="all", log_freq=100)
    
    # Setup save directory
    save_dir = cfg.output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    if use_steps:
        # Steps-based training loop
        current_step = 0
        epoch = 0
        train_losses = []
        
        while current_step < total_steps:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} (Step {current_step}/{total_steps})")
            epoch_losses = []
            
            for batch in pbar:
                if current_step >= total_steps:
                    break
                    
                # Process batch
                batch = process_batch(batch, cfg.device, cfg.pose_mode, mini_batch_size=96, PTS_AUG_PARAMS=None)
                loss, loss_description = trainer.train_step(batch)
                train_losses.append(loss)
                epoch_losses.append(loss)
                
                # Update scheduler at each step
                trainer.scheduler.step()
                current_step += 1
                
                pbar.set_postfix({
                    "Step": f"{current_step}/{total_steps}",
                    "Loss": f"{loss:.4f}",  
                    "Details": loss_description   
                })
                
                # Log training loss every 100 steps
                if current_step % 100 == 0:
                    avg_train_loss = np.mean(train_losses[-100:]) if len(train_losses) >= 100 else np.mean(train_losses)
                    wandb.log({"step": current_step, "train_loss": avg_train_loss, "epoch": epoch})
                
                # Evaluation and saving
                if current_step % eval_freq == 0:
                    trainer.model.eval()
                    val_diff_angles = []
                    val_diff_trans = []
                    
                    print(f"\nRunning validation at step {current_step}...")
                    for val_batch in tqdm(val_loader, desc=f"Step {current_step} Validation"):
                        val_batch = process_batch(val_batch, cfg.device, cfg.pose_mode, mini_batch_size=96, PTS_AUG_PARAMS=None)
                        try:
                            diff_angle, diff_trans = trainer.eval_step(val_batch)
                            if diff_angle != float('inf') and diff_trans != float('inf'):
                                val_diff_angles.append(diff_angle)
                                val_diff_trans.append(diff_trans)
                        except Exception as e:
                            print(f"Error during validation: {e}")
                            continue
                    
                    val_mean_angle_diff = sum(val_diff_angles) / len(val_diff_angles) if val_diff_angles else float('inf')
                    val_mean_trans_diff = sum(val_diff_trans) / len(val_diff_trans) if val_diff_trans else float('inf')
                    
                    print(f"Step {current_step} validation mean angle diff: {val_mean_angle_diff:.4f} degrees")
                    print(f"Step {current_step} validation mean trans diff: {val_mean_trans_diff:.4f} meters")

                    # Test set evaluation
                    test_diff_angles = []
                    test_diff_trans = []
                    
                    for test_batch in tqdm(test_loader, desc=f"Step {current_step} Testing"):
                        test_batch = process_batch(test_batch, cfg.device, cfg.pose_mode, mini_batch_size=96, PTS_AUG_PARAMS=None)
                        try:
                            diff_angle, diff_trans = trainer.eval_step(test_batch)
                            if diff_angle != float('inf') and diff_trans != float('inf'):
                                test_diff_angles.append(diff_angle)
                                test_diff_trans.append(diff_trans)
                        except Exception as e:
                            print(f"Error during testing: {e}")
                            continue
                    
                    test_mean_angle_diff = sum(test_diff_angles) / len(test_diff_angles) if test_diff_angles else float('inf')
                    test_mean_trans_diff = sum(test_diff_trans) / len(test_diff_trans) if test_diff_trans else float('inf')
                    
                    print(f"Step {current_step} test mean angle diff: {test_mean_angle_diff:.4f} degrees")
                    print(f"Step {current_step} test mean trans diff: {test_mean_trans_diff:.4f} meters")

                    # Log to wandb
                    wandb.log({
                        "step": current_step,
                        "epoch": epoch,
                        "val_mean_angle_diff": val_mean_angle_diff,
                        "val_mean_trans_diff": val_mean_trans_diff,
                        "test_mean_angle_diff": test_mean_angle_diff,
                        "test_mean_trans_diff": test_mean_trans_diff
                    })
                    
                    # Save model checkpoint
                    save_path = os.path.join(
                        save_dir,
                        f"cfm_step_{current_step}_angle_{val_mean_angle_diff:.4f}_trans_{val_mean_trans_diff:.4f}.pt"
                    )
                    torch.save(trainer.model.state_dict(), save_path)
                    print(f"Model saved at step {current_step}")
                    
                    trainer.model.train()
            
            # Print epoch average loss
            if len(epoch_losses) > 0:
                epoch_avg_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch} completed: Average Loss = {epoch_avg_loss:.4f}, Steps in epoch = {len(epoch_losses)}, Total steps = {current_step}/{total_steps}")
                wandb.log({"epoch": epoch, "epoch_avg_loss": epoch_avg_loss, "step": current_step})
            
            epoch += 1
            
    else:
        # Original epochs-based training loop
        for epoch in range(cfg.n_epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            train_losses = []
            
            for batch in pbar:
                batch = process_batch(batch, cfg.device, cfg.pose_mode, mini_batch_size=96, PTS_AUG_PARAMS=None)
                loss, loss_description = trainer.train_step(batch)
                train_losses.append(loss)
                
                pbar.set_postfix({
                    "Loss": f"{loss:.4f}",  
                    "Details": loss_description   
                })
                
            avg_train_loss = np.mean(train_losses)
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss})
            
            trainer.model.eval()
            if (epoch + 1) % eval_freq == 0:
                val_diff_angles = []
                val_diff_trans = []
                
                print(f"Running full validation on epoch {epoch}...")
                for val_batch in tqdm(val_loader, desc=f"Epoch {epoch} Full Validation"):
                    val_batch = process_batch(val_batch, cfg.device, cfg.pose_mode, mini_batch_size=96, PTS_AUG_PARAMS=None)
                    try:
                        diff_angle, diff_trans = trainer.eval_step(val_batch)
                        if diff_angle != float('inf') and diff_trans != float('inf'):
                            val_diff_angles.append(diff_angle)
                            val_diff_trans.append(diff_trans)
                    except Exception as e:
                        print(f"Error during validation: {e}")
                        continue
                
                val_mean_angle_diff = sum(val_diff_angles) / len(val_diff_angles) if val_diff_angles else float('inf')
                val_mean_trans_diff = sum(val_diff_trans) / len(val_diff_trans) if val_diff_trans else float('inf')
                
                print(f"Epoch {epoch} Validation Mean Angle Difference: {val_mean_angle_diff:.4f} degrees")
                print(f"Epoch {epoch} Validation Mean Translation Difference: {val_mean_trans_diff:.4f} meters")

                # Test dataloader
                test_diff_angles = []
                test_diff_trans = []
                
                for test_batch in tqdm(test_loader, desc=f"Epoch {epoch} Testing"):
                    test_batch = process_batch(test_batch, cfg.device, cfg.pose_mode, mini_batch_size=96, PTS_AUG_PARAMS=None)
            
                    try:
                        diff_angle, diff_trans = trainer.eval_step(test_batch)
                        if diff_angle != float('inf') and diff_trans != float('inf'):
                            test_diff_angles.append(diff_angle)
                            test_diff_trans.append(diff_trans)
                    except Exception as e:
                        print(f"Error during testing: {e}")
                        continue
                
                test_mean_angle_diff = sum(test_diff_angles) / len(test_diff_angles) if test_diff_angles else float('inf')
                test_mean_trans_diff = sum(test_diff_trans) / len(test_diff_trans) if test_diff_trans else float('inf')
                
                print(f"Epoch {epoch} Test Mean Angle Difference: {test_mean_angle_diff:.4f} degrees")
                print(f"Epoch {epoch} Test Mean Translation Difference: {test_mean_trans_diff:.4f} meters")

                # Log to wandb
                wandb.log({
                    "epoch": epoch, 
                    "val_mean_angle_diff": val_mean_angle_diff,
                    "val_mean_trans_diff": val_mean_trans_diff,
                    "test_mean_angle_diff": test_mean_angle_diff,
                    "test_mean_trans_diff": test_mean_trans_diff
                })
                
                # Save model checkpoint
                save_path = os.path.join(
                    save_dir,
                    f"cfm_epoch_{epoch}_angle_{val_mean_angle_diff:.4f}_trans_{val_mean_trans_diff:.4f}.pt"
                )
                torch.save(trainer.model.state_dict(), save_path)
                print(f"Model saved at epoch {epoch}")
            else:
                wandb.log({"epoch": epoch})
                
            trainer.scheduler.step()

            if (epoch + 1) % eval_freq == 0:
                print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val Angle Diff {val_mean_angle_diff:.4f} | Val Trans Diff {val_mean_trans_diff:.4f}")
                print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Test Angle Diff {test_mean_angle_diff:.4f} | Test Trans Diff {test_mean_trans_diff:.4f}")
            else:
                print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Validation skipped")


def main():
    """Main training function"""
    cfg = get_config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    # Setup wandb config
    wandb_config = {
        "model_type": "continuous_flow_matching",
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.lr,
        "eta_min": cfg.eta_min,
        "encoder": cfg.pts_encoder,
        "T": cfg.T if hasattr(cfg, 'T') else 0.01,
        "velocity_weight": cfg.velocity_weight if hasattr(cfg, 'velocity_weight') else 1.0,
        "rotation_weight": cfg.rotation_weight if hasattr(cfg, 'rotation_weight') else 1.0,
        "translation_weight": cfg.translation_weight if hasattr(cfg, 'translation_weight') else 1.0,
        "pose_prediction_weight": cfg.pose_prediction_weight if hasattr(cfg, 'pose_prediction_weight') else 0.1,
        "rotation_representation": "6D_rotation",
        "normalization": "none",
        "pose_dim": 9,
    }
    
    # Determine training mode
    if hasattr(cfg, 'total_steps') and cfg.total_steps is not None:
        wandb_config["training_mode"] = "steps"
        wandb_config["total_steps"] = cfg.total_steps
        wandb_config["eval_freq_steps"] = cfg.eval_freq_steps if hasattr(cfg, 'eval_freq_steps') else 1000
    else:
        wandb_config["training_mode"] = "epochs"
        wandb_config["n_epochs"] = cfg.n_epochs
        wandb_config["eval_freq"] = cfg.eval_freq if hasattr(cfg, 'eval_freq') else 5
    
    wandb.init(project="continuous_flow_matching_6d_pose", config=wandb_config)
    
    # Load data
    data_loaders = get_data_loaders_from_cfg(cfg, ['train', 'val', 'test'])
    train_loader = data_loaders['train_loader'] 
    val_loader = data_loaders['val_loader']   
    test_loader = data_loaders['test_loader'] 
    print('train_set: ', len(train_loader))
    print('val_set: ', len(val_loader))
    print('test_set: ', len(test_loader))
    
    # Note: trans_stats no longer needed (normalization removed, inspired by GenPose2)
    # Start training
    train_data(cfg, train_loader, val_loader, test_loader, trans_stats=None)
            
if __name__ == "__main__":
    main()

