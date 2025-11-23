CUDA_VISIBLE_DEVICES="0" python runners/DFM_trainer.py \
--data_path ../ArtImage-High-level/ArtImage \
--sampling_steps 100 \
--batch_size 96 \
--total_steps 30000 \
--lr 1e-4 \
--num_bins 360 \
--T 0.01 \
--mse_weight 0 \
--kl_weight 1 \
--L1_weight 0.1 \
--output_dir ckpts/DFM_lr3e-4_5w_360bins_0.0_1_0.1_new_resume \
--eval_freq_steps 2000 \
--seed 42 \
--cate_id 1 \
--num_workers 16 \
--saved_model_name DFM_Train11.09 \
--pts_encoder pointnet2 \
--pretrained_model_path /home/zming/diffpose/6D/code/DICArt/ckpts/DFM_lr3e-4_5w_360bins_0.0_1_0.1_new/step_50000_angle_inf_trans_inf.pt \
--is_train  


