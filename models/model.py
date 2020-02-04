import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import TransformerLayer, SinusoidalPositionalEmbedding, SelfAttentionMask
from models.modules import WordProbLayer, LabelSmoothing

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.device = config['device']
        self.vocab_size = config['vocab_size']
        self.emb_dim = config['emb_dim']
        self.hidden_size = config['hidden_size']
        self.d_ff = config['d_ff']
        self.padding_idx = config['padding_idx']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.smoothing = config['label_smoothing']
        self.is_predicting = config['is_predicting']

        self.attn_mask = SelfAttentionMask(device=self.device)
        self.word_embed = nn.Embedding(self.vocab_size, self.emb_dim, self.padding_idx)
        self.pos_embed = SinusoidalPositionalEmbedding(self.emb_dim, device=self.device)
        self.enc_layers = nn.ModuleList()
        self.dec_layers = nn.ModuleList()
        self.emb_layer_norm = nn.LayerNorm(self.emb_dim)    # copy & coverage not implemented...
        self.word_prob = WordProbLayer(self.hidden_size, self.vocab_size, self.device, self.dropout)
        self.label_smoothing = LabelSmoothing(self.device, self.vocab_size, self.padding_idx, self.smoothing)

        for _ in range(self.num_layers):
            self.enc_layers.append(TransformerLayer(self.hidden_size, self.d_ff,self.num_heads,self.dropout))
            self.dec_layers.append(TransformerLayer(self.hidden_size, self.d_ff,self.num_heads,self.dropout, with_external=True))

    def reset_parameters(self):
        #init_uniform_weight(self.word_embed.weight)
        pass

    def label_smoothing_loss(self, pred, gold, mask = None):
        """
            mask 0 表示忽略 
            gold: seqlen, bsz
        """
        if mask is None: mask = gold.ne(self.padding_idx)
        seq_len, bsz = gold.size()
        # KL散度需要预测概率过log...
        pred = torch.log(pred.clamp(min=1e-8))  # 方便实用的截断函数 (这名字让人想起CLAMP 
        # 本损失函数中, 每个词的损失不对seqlen作规范化
        return self.label_smoothing(pred.view(seq_len * bsz, -1),
                    gold.contiguous().view(seq_len * bsz, -1), mask.sum())
        
    def nll_loss(self, pred:torch.Tensor, gold, mask = None):
        """
            nll: 指不自带softmax的loss计算函数
            pred: seqlen, bsz, vocab
            gold: seqlen, bsz
        """
        if mask is None: mask = gold.ne(self.padding_idx)
        seqlen, bsz = gold.size()
        mask = mask.view(seqlen, bsz)
        gold_prob = pred.gather(dim=2, index=gold.view(seqlen, bsz, 1)).view(gold.size())   # cross entropy
        gold_prob = (gold_prob*mask).clamp(min=1e-8).log().sum(dim=-1) / mask.sum(dim=-1)   # batch内规范化
        return gold_prob.mean()

    def encode(self, inputs, padding_mask = None):
        if padding_mask is None: 
            padding_mask = inputs.eq(self.padding_idx)
        x = self.word_embed(inputs) + self.pos_embed(inputs)
        x = F.dropout(self.emb_layer_norm(x), self.dropout, self.training)  #embed dropout

        for layer in self.enc_layers:
            x, _, _ = layer(x, self_padding_mask=padding_mask)
        
        return x, padding_mask

    def decode(self, inputs, src, src_padding_mask, padding_mask = None):
        """ copy not implemented """
        seqlen, _ = inputs.size()
        if not self.is_predicting and padding_mask is None:
            padding_mask = inputs.eq(self.padding_idx)
        x = self.word_embed(inputs) + self.pos_embed(inputs)
        x = F.dropout(self.emb_layer_norm(x), self.dropout, self.training)
        
        self_attn_mask = self.attn_mask(seqlen)

        for layer in self.dec_layers:
            x,_,_ = layer(x, self_padding_mask=padding_mask, self_attn_mask=self_attn_mask,
                    external_memories=src, external_padding_mask=src_padding_mask)
        
        pred, _ = self.word_prob(x)
        return pred

    def forward(self, src, tgt, src_padding_mask = None, tgt_padding_mask = None):
        """
            src&tgt: seqlen, bsz
        """
        src_enc, src_padding_mask = self.encode(src, src_padding_mask)
        return self.decode(tgt, src_enc, src_padding_mask, tgt_padding_mask)