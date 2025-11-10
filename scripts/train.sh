CUDA_VISIBLE_DEVICES="0" python runners/DFM_trainer.py \
--data_path /home/zming/diffpose/6D/code/ArtImage-High-level/ArtImage \
--sampling_steps 100 \
--batch_size 128 \
--eval_freq 10 \
--n_epochs 200 \
--seed 42 \
--cate_id 1 \
--saved_model_name DFM_Train11.09 \
--pts_encoder pointnet2 \
--is_train
# --pretrained_model_path  \

