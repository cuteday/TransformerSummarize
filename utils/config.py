import torch
from utils import variables

def init_config(args):
    config = {
        'train_from': '',
        'test_from': '',
        'vocab_file': variables.CNNDMPath + '/vocab_cnt.pkl',
        'model_path': '../saved_models',
        'log_path': '../results',

        'max_src_ntokens': 400,
        'max_tgt_ntokens': 100,
        'max_dec_steps': 100,
        'min_dec_steps': 35,
        'vocab_size': 50000,
        'padding_idx': 0,

        'is_predicting': False,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'copy': False,
        'coverage': False,

        'hidden_size': 512,
        'emb_dim': 768,
        'd_ff': 1024,
        'num_layers': 6,
        'num_heads': 8,
        'label_smoothing': 0.1,
        'dropout': 0.1,
        'max_grad_norm': 2.0,

        'learning_rate': 0.15,
        'batch_size': 32,
        'beam_size': 5,
        'validate_every': 1,
        'report_every': 1,
        'save_every': 1,

    }
    for key in args.keys():
        config[key] = args[key]
    return config
