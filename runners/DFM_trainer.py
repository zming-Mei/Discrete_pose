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
from networks.discrete_flow_matching import DiscreteFlowMatching
from runners.DFM_eval import DiscreteFlowEvaluator

class DiscreteFlowTrainer:
    def __init__(self, cfg, trans_stats=None, pretrained_path=None, total_steps=None):
        self.cfg = cfg
        self.device = cfg.device
        self.trans_stats = trans_stats
        self.model = DiscreteFlowMatching(
            cfg,
            device=self.device
        ).to(self.device)
        if pretrained_path is not None:
            print(f"Loading pretrained model from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print("Model loaded successfully")
        
        # Create evaluator instance for validation/testing during training
        # The evaluator will share the same model as the trainer
        self.evaluator = DiscreteFlowEvaluator(cfg, trans_stats, pretrained_path=None)
        self.evaluator.model = self.model  # Share the same model instance
        
        # Optimizer & Scheduler
        self.optimizer = torch.optim.RAdam(
            self.model.parameters(),
            lr=cfg.lr,  
            betas = (0.95, 0.999), 
            weight_decay=1e-6  
        )
        
        # If using total_steps, create scheduler based on steps
        if total_steps is not None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=3e-6
            )
        else:
            # Keep the original epochs-based scheduler
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.n_epochs,
                eta_min=3e-6
            )

    def train_step(self, batch_sample):
        self.model.train()
        
        rot_part = batch_sample['zero_mean_gt_pose'][:, :6]
        trans_part = batch_sample['zero_mean_gt_pose'][:, -3:]

        rot_matrix = pytorch3d_transforms.rotation_6d_to_matrix(rot_part)
        euler_angles = pytorch3d_transforms.matrix_to_euler_angles(rot_matrix, "ZYX")
        spherical_angles = matrix_to_spherical_angles(rot_matrix)
        discretized_angles = get_bin_index(spherical_angles, self.model.num_bins, self.model.num_bins, self.model.num_bins)
        discretized_euler = discretize_euler_angles(euler_angles, self.model.num_bins)
        trans_bins = translation_to_bins(trans_part, self.trans_stats, self.model.num_bins)

        gt_pose = torch.cat([discretized_euler, trans_bins], dim=1)
        y = gt_pose.to(self.device)
        
        pts_feat = self.model.extract_pts_feature(batch_sample).to(self.device)
        loss,loss_description = self.model.loss(y, pts_feat)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # Lower gradient clipping threshold
        self.optimizer.step()
        
        return loss.item(),loss_description

    def eval_step(self, batch_sample):
        """
        Evaluate a single batch using the evaluator.
        This delegates to the DiscreteFlowEvaluator's test_step method.
        
        Args:
            batch_sample: Batch data dictionary
            
        Returns:
            diff_angle: Mean rotation error in degrees
            diff_trans: Mean translation error in meters
        """
        self.model.eval()
        
        try:
            with torch.no_grad():
                # Use the evaluator's test_step method for consistent evaluation
                diff_angle, diff_trans = self.evaluator.test_step(batch_sample)
                # Return mean values for logging
                return diff_angle.mean().item(), diff_trans.mean().item()
            
        except Exception as e:
            print(f"Error in eval_step: {e}")
            return float('inf'), float('inf')  


def train_data(cfg, train_loader, val_loader, test_loader,trans_stats):

    pretrained_path = cfg.pretrained_model_path
    
    # Determine whether to use steps or epochs mode
    use_steps = cfg.total_steps is not None
    if use_steps:
        total_steps = cfg.total_steps
        eval_freq = cfg.eval_freq_steps
        trainer = DiscreteFlowTrainer(cfg, trans_stats, pretrained_path=pretrained_path, total_steps=total_steps)
        print(f"Training mode: steps-based, total_steps={total_steps}, eval_freq={eval_freq} steps")
    else:
        total_epochs = cfg.n_epochs
        eval_freq = cfg.eval_freq
        trainer = DiscreteFlowTrainer(cfg, trans_stats, pretrained_path=pretrained_path, total_steps=None)
        print(f"Training mode: epochs-based, total_epochs={total_epochs}, eval_freq={eval_freq} epochs")
    
    wandb.watch(trainer.model, log="all", log_freq=100)
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

                    wandb.log({
                        "step": current_step,
                        "epoch": epoch,
                        "val_mean_angle_diff": val_mean_angle_diff,
                        "val_mean_trans_diff": val_mean_trans_diff,
                        "test_mean_angle_diff": test_mean_angle_diff,
                        "test_mean_trans_diff": test_mean_trans_diff
                    })
                    
                    save_path = os.path.join(
                        save_dir,
                        f"step_{current_step}_angle_{val_mean_angle_diff:.4f}_trans_{val_mean_trans_diff:.4f}.pt"
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
                loss,loss_description = trainer.train_step(batch)
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

                #Test dataloader
                test_diff_angles = []
                test_diff_trans = []
                
                for test_batch in tqdm(test_loader, desc=f"Epoch {epoch} Full Validation"):
                    test_batch = process_batch(test_batch, cfg.device, cfg.pose_mode, mini_batch_size=96, PTS_AUG_PARAMS=None)
            
                    try:
                        diff_angle, diff_trans = trainer.eval_step(test_batch)
                        if diff_angle != float('inf') and diff_trans != float('inf'):
                            test_diff_angles.append(diff_angle)
                            test_diff_trans.append(diff_trans)
                    except Exception as e:
                        print(f"Error during validation: {e}")
                        continue
                
                test_mean_angle_diff = sum(test_diff_angles) / len(test_diff_angles) if test_diff_angles else float('inf')
                test_mean_trans_diff = sum(test_diff_trans) / len(test_diff_trans) if test_diff_trans else float('inf')
                
                print(f"Epoch {epoch} testidation Mean Angle Difference: {test_mean_angle_diff:.4f} degrees")
                print(f"Epoch {epoch} testidation Mean Translation Difference: {test_mean_trans_diff:.4f} meters")

                wandb.log({
                    "epoch": epoch, 
                    "val_mean_angle_diff": val_mean_angle_diff,
                    "val_mean_trans_diff": val_mean_trans_diff,
                    "test_mean_angle_diff": test_mean_angle_diff,
                    "test_mean_trans_diff": test_mean_trans_diff
                })
                
                save_path = os.path.join(
                    save_dir,
                    f"epoch_model_epoch_{epoch}_angle_diff_{val_mean_angle_diff:.4f}_trans_diff_{val_mean_trans_diff:.4f}.pt"
                )
                torch.save(trainer.model.state_dict(), save_path)
                print(f"Model saved at epoch {epoch}")
            else:
                wandb.log({
                    "epoch": epoch
                })
                
            trainer.scheduler.step()

            if (epoch + 1) % eval_freq == 0:
                print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val Angle Diff {val_mean_angle_diff:.4f} | Val Trans Diff {val_mean_trans_diff:.4f}")
                print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | test Angle Diff {test_mean_angle_diff:.4f} | test Trans Diff {test_mean_trans_diff:.4f}")
            else:
                print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Validation skipped")



def main():

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    cfg = get_config()
    
    # Determine whether to use steps or epochs based on configuration
    wandb_config = {
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.lr,
        "encoder": cfg.pts_encoder,
        "num_bins": cfg.num_bins,
        "T": cfg.T,
        "mse_weight": cfg.mse_weight,
        "kl_weight": cfg.kl_weight,
        "L1_weight": cfg.L1_weight,
    }
    
    if cfg.total_steps is not None:
        wandb_config["training_mode"] = "steps"
        wandb_config["total_steps"] = cfg.total_steps
        wandb_config["eval_freq_steps"] = cfg.eval_freq_steps
    else:
        wandb_config["training_mode"] = "epochs"
        wandb_config["n_epochs"] = cfg.n_epochs
        wandb_config["eval_freq"] = cfg.eval_freq
    
    wandb.init(project="discrete_flow_matching", config=wandb_config)
    
    data_loaders = get_data_loaders_from_cfg(cfg, ['train', 'val', 'test'])
    train_loader = data_loaders['train_loader'] 
    val_loader = data_loaders['val_loader']   
    test_loader = data_loaders['test_loader'] 
    print('train_set: ', len(train_loader))
    print('val_set: ', len(val_loader))
    print('test_set: ', len(test_loader))
    
    # Translation statistics for normalization/denormalization
    trans_stats = [-0.3785014748573303, 0.39416784048080444, -0.4042277932167053, 0.39954620599746704, -0.30842161178588867, 0.7598943710327148]
    #trans_stats = get_dataset_translation_min_max(train_loader, cfg)

    # Start training
    train_data(cfg, train_loader, val_loader, test_loader, trans_stats)
            
if __name__ == "__main__":
    main()