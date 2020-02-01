import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

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
            # transformer的拷贝机制 是在顶端添加额外的单头注意力层
            self.external_attn = MultiheadAttention(self.hidden_size, 1, self.dropout, weights_dropout=False)
            self.proj = nn.Linear(self.hidden_size * 3, self.dict_size)
            self.prob_copy = nn.Linear(self.hidden_size * 3, 1, bias=True)
        else:
            self.proj = nn.Linear(self.hidden_size, self.dict_size)
        
        self.init_weights()

    def init_weights(self):
        init_linear_weight(self.proj)
        if self.copy: init_linear_weight(self.prob_copy)

    def forward(self, h, y_emb=None, memory=None, mask_x=None, xids=None, max_ext_len=None):
        """
            h: final hidden layer output by decoder [ seqlen * bsz * hidden ]
            memory: output by encoder               
            y_emb: 
            mask_x: padding mask
            xids: indices of words from source text [words2ids(src)]
            max_ext_len: max len of extended vocab
            return: softmax probabilities
        """
        if self.copy:
            # dists: seqlen * bsz * seqlen
            atts, dists = self.external_attn(query=h, key=memory, value=memory, key_padding_mask=mask_x, need_weights = True)
            pred = torch.softmax(self.proj(torch.cat([h, y_emb, atts], -1)), dim=-1)        #原词典上的概率分布
            if max_ext_len > 0:
                ext_zeros = Variable(torch.zeros(pred.size(0), pred.size(1), max_ext_len)).to(self.device)
                pred = torch.cat((pred, ext_zeros), -1)
            g = torch.sigmoid(self.prob_copy(torch.cat([h, y_emb, atts], -1)))              #计算生成概率g
            # xids应与dists的大小保持一致
            xids = xids.transpose(0, 1).unsqueeze(0).repeat(pred.size(0), 1, 1)
            # 在最后一维(即预测概率分布)上scatter
            pred = (g * pred).scatter_add(2, xids, (1 - g) * dists)
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

    def forward(self, output, target, normalize=1):
        """
            支持扩展词典, 比如copy机制使用的src词典
            input size: bsz*seq_en, vocab
            normalize: 一般是词的数量 即每个词的重要性相同
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

        return F.kl_div(output, model_prob, reduction='sum').div(float(normalize))   

class LayerNorm(nn.Module):
    """
        LayerNorm的原型函数... !资料仅供学习使用!
        说的那么麻烦...其实就是沿最后一维作标准化
        为了不让取值集中在0附近(失去激活函数的线性性质), 它还非常贴心地添加了平移和缩放功能...!
    """
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, 1.)
        nn.init.constant_(self.bias, 0.)
    
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias

def gelu(x):
    """
        GeLU(x) = x * \\phi(x)  
        phi(x)是正态概率分布函数, 即error function
    """
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return cdf*x
