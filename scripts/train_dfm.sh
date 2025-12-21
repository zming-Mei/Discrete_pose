CUDA_VISIBLE_DEVICES="0" python runners/DFM_trainer.py \
--data_path ../ArtImage-High-level/ArtImage \
--sampling_steps 100 \
--batch_size 96 \
--total_steps 80000 \
--lr 5e-4 \
--eta_min 1e-5 \
--mse_weight 0 \
--kl_weight 1 \
--L1_weight 0.1 \
--num_bins 72 \
--T_dfm 0.01 \
--output_dir ckpts/DFM_lr5e-4_8w_72bins_baseline \
--eval_freq_steps 2500 \
--seed 42 \
--cate_id 1 \
--joint_num 1 \
--num_parts 2 \
--num_workers 16 \
--pts_encoder pointnet2 \
--is_train  


