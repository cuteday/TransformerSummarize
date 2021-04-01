import torch
from torch import nn
import random
import numpy as np

seed = 1       # assign a random seed for reproducing results

def init_seeds():
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_lstm_weight(lstm):
    for param in lstm.parameters():
        if len(param.shape) >= 2: # weights
            init_ortho_weight(param.data)
        else: # bias
            init_bias(param.data)

def init_gru_weight(gru):
    for param in gru.parameters():
        if len(param.shape) >= 2: # weights
            init_ortho_weight(param.data)
        else: # bias
            init_bias(param.data)

def init_linear_weight(linear):
    init_xavier_weight(linear.weight)
    if linear.bias is not None:
        init_bias(linear.bias)

def init_normal_weight(w):
    nn.init.normal_(w, mean=0, std=0.01)

def init_uniform_weight(w):
    nn.init.uniform_(w, -0.1, 0.1)

def init_ortho_weight(w):
    nn.init.orthogonal_(w)

def init_xavier_weight(w):
    nn.init.xavier_normal_(w)

def init_bias(b):
    nn.init.constant_(b, 0.)

def save_model(f, model, optimizer):
    torch.save({"model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict()},
            f)
 
def load_model(f, model, optimizer):
    checkpoint = torch.load(f)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer
