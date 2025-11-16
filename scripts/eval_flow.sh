CUDA_VISIBLE_DEVICES="0" python runners/DFM_eval.py \
--data_path /home/zming/diffpose/6D/code/ArtImage-High-level/ArtImage \
--sampling_steps 100 \
--batch_size 96 \
--seed 0 \
--cate_id 1 \
--num_bins 36 \
--T 0.01 \
--saved_model_name Train11.17 \
--pts_encoder pointnet2 \
--pretrained_model_path_test /home/zming/diffpose/6D/code/DICArt/DiscreteFlow_6D_KL_60bins/step_50000_angle_13.9912_trans_0.0757.pt \
--eval 

