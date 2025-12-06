import argparse

def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--synset_names', nargs='+', default=['dishwasher', 'drawer', 'eyeglasses', 'laptop', 'scissors'])
    parser.add_argument('--num_parts', type=int, default=2)
    parser.add_argument('--joint_num', type=int, default=1)
    parser.add_argument('--PTS_AUG_PARAMS', default = {
    'aug_bb_pro': 0.0,  
    'aug_rt_pro': 0.3,  
    'aug_pc_pro': 0.3,    
    'aug_pc_r': 0.01     
})
    parser.add_argument('--cate_id', type=int, default=1)
    
    """ dataset """
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_batch_size', type=int, default=192)  # 192
    parser.add_argument('--mini_bs', type=int, default=1)
    parser.add_argument('--pose_mode', type=str, default='rot_matrix')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--percentage_data_for_train', type=float, default=1.0) 
    parser.add_argument('--percentage_data_for_val', type=float, default=1.0) 
    parser.add_argument('--percentage_data_for_test', type=float, default=1.0) 
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0) # 32
    
    """ model """
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--sampling_steps', type=int)
    parser.add_argument('--pointnet2_params', type=str, default='light')
    parser.add_argument('--pts_encoder', type=str, default='pointnet2')
    parser.add_argument('--num_bins', type=int, default=360)
    parser.add_argument('--T_dfm', type=float, default=0.01)  # step_size for DFM sampling
    parser.add_argument('--T_acfm', type=float, default=0.01)  # step_size for ACFM sampling
    
    """ loss weights """
    parser.add_argument('--mse_weight', type=float, default=0)
    parser.add_argument('--kl_weight', type=float, default=1)
    parser.add_argument('--L1_weight', type=float, default=0.1) 
    parser.add_argument('--velocity_weight', type=float, default=1.0)
    parser.add_argument('--rotation_weight', type=float, default=1.0)
    parser.add_argument('--translation_weight', type=float, default=1.0)
    parser.add_argument('--pose_prediction_weight', type=float, default=0.1)  # Weight for pose prediction loss (angle + translation)

    """ training """
    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--n_epochs', type=int, default=1000)  
    parser.add_argument('--total_steps', type=int, default=None)  
    parser.add_argument('--log_dir', type=str, default='debug')
    parser.add_argument('--output_dir', type=str, default='DFM')  
    parser.add_argument('--optimizer',  type=str, default='Adam')
    parser.add_argument('--eval_freq', type=int, default=100) 
    parser.add_argument('--eval_freq_steps', type=int, default=1000) 
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--ema_rate', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eta_min', type=float, default=1e-5)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--lr_decay', type=float, default=0.98)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--is_train', default=False, action='store_true')
    parser.add_argument('--saved_model_name', type=str, default=None)
    """ co-training """
    parser.add_argument('--use_coarse_as_x0', type=str2bool, default=False)
    parser.add_argument('--freeze_dfm', type=str2bool, default=True)
    parser.add_argument('--topk_k', type=int, default=10)
    parser.add_argument('--dfm_pretrained_path', type=str, default=None)
    parser.add_argument('--acfm_pretrained_path', type=str, default=None)
    parser.add_argument('--dfm_cache_path', type=str, default=None, help='Path to precomputed DFM cache for faster training')
    parser.add_argument('--acfm_rotation_type', type=str, default='euler', choices=['euler', '6d', 'axis_angle'], 
                        help='Rotation representation for ACFM output: euler (3D) or 6d (6D rotation)')
    parser.add_argument('--rotation_scale', type=float, default=100.0)
    parser.add_argument('--translation_scale', type=float, default=100.0)
    """ testing """
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--pred', default=False, action='store_true')
    parser.add_argument('--results_path', type=str, default='')
    parser.add_argument('--pretrained_model_path_test', type=str, default='')
   
    cfg = parser.parse_args()

    return cfg


