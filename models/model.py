import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.transformer import TransformerLayer, SinusoidalPositionalEmbedding, SelfAttentionMask
from models.modules import Embedding, WordProbLayer, LabelSmoothing
from utils.initialize import init_uniform_weight

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.device = config['device']
        self.vocab_size = config['vocab_size']
        self.emb_dim = config['emb_dim']
        self.hidden_size = config['hidden_size']
        self.padding_idx = config['padding_idx']
        self.num_layer = config['num_layer']
        self.num_head = config['num_head']
        self.dropout = config['dropout']
        self.confidence = config['label_smoothing']

        self.attn_mask = SelfAttentionMask(device=self.device)
        self.word_embed = nn.Embedding(self.vocab_size, self.emb_dim, self.padding_idx)
        self.pos_embed = SinusoidalPositionalEmbedding(self.emb_dim, device=self.device)
        self.enc_layers = nn.ModuleList()
        self.dec_layers = nn.ModuleList()
        self.emb_layer_norm = nn.LayerNorm(self.emb_dim)    # copy & coverage not implemented...
        self.word_prob = WordProbLayer(self.hidden_size, self.vocab_size, self.device, self.dropout)
        self.label_smoothing = LabelSmoothing(self.device, self.vocab_size, self.padding_idx, self.confidence)

    def init_parameters(self):
        #init_uniform_weight(self.word_embed.weight)
        pass

    def label_smoothing_loss(self):
        # KL散度需要预测概率过log...
        pass

    def forward(self):
        pass