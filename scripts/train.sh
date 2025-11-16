CUDA_VISIBLE_DEVICES="0" python runners/DFM_trainer.py \
--data_path /home/zming/diffpose/6D/code/ArtImage-High-level/ArtImage \
--sampling_steps 100 \
--batch_size 96 \
--total_steps 50000 \
--lr 3e-4 \
--num_bins 36 \
--T 0.01 \
--output_dir ckpts/DiscreteFlow_6D_KL_36bins \
--eval_freq_steps 2500 \
--seed 42 \
--cate_id 1 \
--saved_model_name DFM_Train11.09 \
--pts_encoder pointnet2 \
#--pretrained_model_path /home/zming/diffpose/6D/code/DICArt/DiscreteFlow_6D_KL/epoch_model_epoch_289_angle_diff_18.2792_trans_diff_0.0899.pt \
--is_train  


