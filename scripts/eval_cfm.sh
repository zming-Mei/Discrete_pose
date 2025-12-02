CUDA_VISIBLE_DEVICES="0" python runners/CFM_eval.py \
--data_path ../ArtImage-High-level/ArtImage \
--sampling_steps 100 \
--batch_size 96 \
--seed 0 \
--cate_id 1 \
--saved_model_name CFM_Eval \
--pts_encoder pointnet2 \
--T_acfm 0.01 \
--pretrained_model_path_test ckpts/CFM_lr5e-4_8w_attention/cfm_step_80000.pt \
--eval 


