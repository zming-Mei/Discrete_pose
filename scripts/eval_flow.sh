CUDA_VISIBLE_DEVICES="0" python runners/DFM_test.py \
--data_path ../ArtImage-High-level/ArtImage \
--sampling_steps 100 \
--batch_size 96 \
--seed 0 \
--cate_id 1 \
--num_bins 72 \
--T 0.01 \
--pts_encoder pointnet2 \
--pretrained_model_path_test ckpts/DFM_lr5e-4_8w_72bins_0_1_0.1_new_attention/step_80000_angle_4.8916_trans_0.0484.pt \
--eval 

