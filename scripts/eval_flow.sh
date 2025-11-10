CUDA_VISIBLE_DEVICES="0" python runners/DFM_trainer.py \
--data_path /home/zming/diffpose/6D/code/ArtImage-High-level/ArtImage \
--sampling_steps 100 \
--batch_size 96 \
--seed 0 \
--cate_id 1 \
--saved_model_name Train11.17 \
--pts_encoder pointnet2 \
--pretrained_model_path_test /home/zming/diffpose/6D/code/DICArt/DiscreteFlow_6D/epoch_model_epoch_9_angle_diff_inf_trans_diff_inf.pt \
--eval 

