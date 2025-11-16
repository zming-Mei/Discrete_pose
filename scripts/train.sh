CUDA_VISIBLE_DEVICES="0" python runners/DFM_trainer.py \
--data_path ../ArtImage-High-level/ArtImage \
--sampling_steps 100 \
--batch_size 96 \
--total_steps 50000 \
--lr 3e-4 \
--num_bins 36 \
--T 0.01 \
--mse_weight 0.0 \
--kl_weight 0.8 \
--L1_weight 0.3 \  
--output_dir ckpts/DiscreteFlow_6D_KL_36bins \
--eval_freq_steps 2500 \
--seed 42 \
--cate_id 1 \
--num_workers 16 \
--saved_model_name DFM_Train11.09 \
--pts_encoder pointnet2 \
--is_train  


