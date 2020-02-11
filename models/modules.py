import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import MultiheadAttention
from utils.initialize import init_linear_weight

class WordProbLayer(nn.Module):
    def __init__(self, hidden_size, dict_size, device, dropout, copy=False, coverage=False):
        super(WordProbLayer, self).__init__()
        self.hidden_size = hidden_size
        self.dict_size = dict_size
        self.device = device
        self.copy = copy
        self.coverage = coverage
        self.dropout = dropout
    
        if self.copy:
            # add an extra single headed att-layer to implement copy-mechanism ヽ(･ω･´ﾒ)
            self.external_attn = MultiheadAttention(self.hidden_size, 1, self.dropout, weights_dropout=False)
            self.proj = nn.Linear(self.hidden_size * 3, self.dict_size)
            self.prob_copy = nn.Linear(self.hidden_size * 3, 1, bias=True)
        else:
            self.proj = nn.Linear(self.hidden_size, self.dict_size)
        
        self.init_weights()

    def init_weights(self):
        init_linear_weight(self.proj)
        if self.copy: init_linear_weight(self.prob_copy)

    def forward(self, h, emb=None, memory=None, src_mask=None, tokens=None, extra_zeros=None):
        """
            h: final hidden layer output by decoder [ seqlen * bsz * hidden ]
            memory: output by encoder               
            emb: word embedd for current token...
            src_mask: padding mask
            tokens: indices of words from source text [include extended vocabs]
            max_ext_len: max len of extended vocab
            returns: softmaxed probabilities, copy attention distribs
        """
        if self.copy:
            # dists: seqlen * bsz * seqlen
            # pred: seqlen * bsz * vocab_size
            atts, dists = self.external_attn(query=h, key=memory, value=memory, key_padding_mask=src_mask, need_weights = True)
            pred = torch.softmax(self.proj(torch.cat([h, emb, atts], -1)), dim=-1)        #原词典上的概率分布
            if extra_zeros is not None:
                pred = torch.cat((pred, extra_zeros.repeat(pred.size(0),1,1)), -1)
            g = torch.sigmoid(self.prob_copy(torch.cat([h, emb, atts], -1)))              #计算生成概率g
            # tokens应与dists的大小保持一致, 并仅在最后一维大小与pred不同
            tokens = tokens.unsqueeze(0).repeat(pred.size(0), 1, 1)
            # 在最后一维(即预测概率分布)上scatter
            pred = (g * pred).scatter_add(2, tokens, (1 - g) * dists)
        else:
            pred = torch.softmax(self.proj(h), dim=-1)
            dists = None
        return pred, dists

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, device, size, padding_idx, label_smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = padding_idx
        self.size = size
        self.device = device

        self.smoothing_value = label_smoothing / (size - 2)    # not padding idx & gold
        self.one_hot = torch.full((1, size), self.smoothing_value).to(device)
        self.one_hot[0, self.padding_idx] = 0
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
            支持扩展词典, 比如copy机制使用的src词典
            input size: bsz*seq_en, vocab
            return: 0维tensor
        """
        real_size = output.size(1)  
        if real_size > self.size:
            real_size -= self.size  # real size即扩展词典的大小
        else:
            real_size = 0   

        model_prob = self.one_hot.repeat(target.size(0), 1) # -1 * vocab 
        if real_size > 0: 
            ext_zeros = torch.full((model_prob.size(0), real_size), self.smoothing_value).to(self.device)
            model_prob = torch.cat((model_prob, ext_zeros), -1)
        # @scatter 的正确使用方法
        # 只有被声明的那一维拥有与src和index不同的维数
        model_prob.scatter_(1, target, self.confidence)
        model_prob.masked_fill_((target == self.padding_idx), 0.)

        return F.kl_div(output, model_prob, reduction='sum')

