CUDA_VISIBLE_DEVICES="0" python runners/eval_acfm.py \
--data_path ../ArtImage-High-level/ArtImage \
--batch_size 96 \
--seed 42 \
--cate_id 1 \
--num_bins 72 \
--T_dfm 0.1 \
--T_acfm 0.01 \
--pts_encoder pointnet2 \
--topk_k 10 \
--use_coarse_as_x0 False \
--dfm_pretrained_path ckpts/DFM_lr5e-4_8w_72bins_0_1_0.1_new_attention/step_80000_angle_4.8916_trans_0.0484.pt \
--acfm_pretrained_path ckpts/ACFM_lr2e-4_5w/acfm_step_50000_angle_3.8731_trans_0.0366.pt \
--eval

