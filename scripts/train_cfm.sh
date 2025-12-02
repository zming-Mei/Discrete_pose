CUDA_VISIBLE_DEVICES="0" python runners/CFM_trainer.py \
--data_path ../ArtImage-High-level/ArtImage \
--sampling_steps 100 \
--batch_size 96 \
--total_steps 80000 \
--eval_freq_steps 1000 \
--lr 5e-4 \
--eta_min 1e-5 \
--rotation_weight 1.0 \
--translation_weight 1.0 \
--pose_prediction_weight 5.0 \
--T_acfm 0.01 \
--output_dir ckpts/CFM_lr5e-4_8w_attention_new_loss \
--seed 42 \
--num_bins 360 \
--cate_id 1 \
--joint_num 1 \
--num_parts 2 \
--num_workers 16 \
--pts_encoder pointnet2 \
--is_train


