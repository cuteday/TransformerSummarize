import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math

class TransformerLayer(nn.Module):
    
    def __init__(self, embed_dim, ff_embed_dim, num_heads, dropout, with_external=False, weights_dropout = True):
        """
            external: 外部注意力(target to source)
        """
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout, weights_dropout)
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim)
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim)
        self.attn_layer_norm = LayerNorm(embed_dim, eps = 1e-12)
        self.ff_layer_norm = LayerNorm(embed_dim, eps = 1e-12)
        self.with_external = with_external
        self.dropout = nn.Dropout(p=dropout)
        if self.with_external:
            self.external_attn = MultiheadAttention(embed_dim, num_heads, dropout, weights_dropout)
            self.external_layer_norm = LayerNorm(embed_dim, eps = 1e-12)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, 
                self_padding_mask = None, self_attn_mask = None,
                external_memories = None, external_padding_mask=None,
                need_weights=False):
        """ returns: x, self_att or src_att """
        # x: seq_len x bsz x embed_dim
        residual = x
        x, self_attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask, need_weights = need_weights)

        x = self.dropout(x)
        x = self.attn_layer_norm(residual + x)  # norm前都接dropout嗷

        if self.with_external:
            residual = x
            x, external_attn = self.external_attn(query=x, key=external_memories, value=external_memories, key_padding_mask=external_padding_mask, need_weights = need_weights)
            x = self.dropout(x)
            x = self.external_layer_norm(residual + x)
        else:
            external_attn = None

        # Position-wise FF
        residual = x
        #x = self.dropout(gelu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        x = self.ff_layer_norm(residual + x)

        return x, self_attn, external_attn
    
class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., weights_dropout=True):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        # in_proj: | q | k | v |, 这些参数给F.linear用 
        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.weights_dropout = weights_dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=False):
        """ 
            key_padding_mask: seqlen x batch
            attn_mask:  tgt_len x src_len
            mask 1 为忽略项
            returns: attn[tgtlen * bsz * srclen]
        """

        # 通过数据指针判断是自注意力还是...
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()   # py支持连等号的奥
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert key.size() == value.size()

        if qkv_same: # 合在一起是能加快速度么...
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling
        
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)
        # k,v: bsz*heads x src_len x dim
        # q: bsz*heads x tgt_len x dim 

        attn_weights = torch.bmm(q, k.transpose(1, 2))      # Q * K^T
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        
        if attn_mask is not None:   # tgt self-att mask (triu)
            attn_weights.masked_fill_(
                attn_mask.unsqueeze(0), # masked_fill expects num of dim tobe same
                float('-inf')
            )

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) # extends
            attn_weights.masked_fill_(
                # mask: bsz, 1, 1, src_len
                key_padding_mask.transpose(0, 1).unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        if self.weights_dropout:    # !!! attention 的 dropout...?
            attn_weights = self.dropout(attn_weights)

        attn = torch.bmm(attn_weights, v)
        if not self.weights_dropout:
            attn = self.dropout(attn)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            #attn_weights, _ = attn_weights.max(dim=1)  # max pooling
            #attn_weights = attn_weights[:, 0, :, :]    # 只拿第k个head > <
            attn_weights = attn_weights.mean(dim=1)    # mean pooling
            attn_weights = attn_weights.transpose(0, 1)
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        # chunk: splits a tensor into a specific number of chunks.
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, inputs, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(inputs, weight, bias)

class SelfAttentionMask(nn.Module):
    def __init__(self, init_size = 100, device = 0):
        super(SelfAttentionMask, self).__init__()
        self.weights = SelfAttentionMask.get_mask(init_size)
        self.device = device
    
    @staticmethod
    def get_mask(size):
        weights = torch.triu(torch.ones((size, size), dtype = torch.uint8), 1)
        return weights

    def forward(self, size):
        if self.weights is None or size > self.weights.size(0):
            self.weights = SelfAttentionMask.get_mask(size)
        res = self.weights[:size,:size].to(self.device).detach()
        return res

class LearnedPositionalEmbedding(nn.Module):
    """
        This module produces LearnedPositionalEmbedding.
    """
    def __init__(self, embedding_dim, init_size=405, device=0):
        super(LearnedPositionalEmbedding, self).__init__()
        self.weights = nn.Embedding(init_size, embedding_dim)   # nn.embedding 默认finetune
        self.device = device
        self.reset_parameters()
    
    def reset_parameters(self):
        """ 跟词向量采用了相同的初始化方式...! """
        nn.init.normal_(self.weights.weight, std=0.02)

    def forward(self, inputs, offset=0):
        """Input is expected to be of size [seq_len x bsz]."""
        seq_len, bsz = inputs.size()
        positions = (offset + torch.arange(seq_len)).to(self.device)
        res = self.weights(positions).unsqueeze(1).expand(-1, bsz, -1)
        return res

class SinusoidalPositionalEncoding(nn.Module):
    """
        Attention is All You Need ver.
        Positional Encoding 的计算!
        PE(pos, 2i) = sin(pos / (10000 ^ (2 * i / d_model)))
    """
    def __init__(self, d_model, max_size = 512, device=0):
        super(SinusoidalPositionalEncoding, self).__init__()

        pe = torch.zeros(max_size, d_model)
        position = torch.arange(0, max_size).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                        - (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div)
        pe[:, 1::2] = torch.cos(position.float() * div)
        pe.unsqueeze_(1)
        self.register_buffer('pe', pe)
        self.device = device
        
    def forward(self, x):
        seq_len, bsz = x.size()
        return self.pe[:seq_len,:,:].expand(-1, bsz, -1).to(self.device).detach()


def gelu(x):
    """
        GeLU(x) = x * \\phi(x)  
        phi(x)是正态概率分布函数, 即error function
    """
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return cdf*x


class LayerNorm(nn.Module):
    """
        LayerNorm的原型函数... 
        说的那么麻烦...其实就是沿最后一维作标准化
        为了不让取值集中在0附近(失去激活函数的非线性性质), 它还非常贴心地添加了平移和缩放功能...!
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
