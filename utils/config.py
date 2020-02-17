import torch
from utils import variables

def init_config(args):
    config = {
        'train_from': '',
        'test_from': '',
        'log_root': '../',
        'data_path': variables.CNNDMPath,
        'vocab_file': variables.CNNDMPath + '/vocab_cnt.pkl',
        'model_path': '../saved_models_copy',
        'log_path': '../results',

        'max_src_ntokens': 400,
        'max_tgt_ntokens': 100,
        'max_dec_steps': 120,
        'min_dec_steps': 35,
        'vocab_size': 50000,
        'padding_idx': 0,

        'is_predicting': False,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'copy': False,
        'coverage': False,

        'hidden_size': 512,
        'emb_dim': 512,
        'd_ff': 1024,
        'num_layers': 4,
        'num_heads': 8,
        'label_smoothing': 0.1,
        'dropout': 0.2,
        'max_grad_norm': 5.0,

        'betas': [0.9, 0.99],

        'learning_rate': 0.04,
        'batch_size': 6,
        'gradient_accum': 8,
        'beam_size': 5,
        'validate_every': 50000,
        'report_every': 100,
        'save_every': 50000,
        'train_epoch': 10,

    }
    for key in args.keys():
        config[key] = args[key]
    return config
