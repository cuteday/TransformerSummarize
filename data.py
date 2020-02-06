import json
import re
import os
import pickle as pkl
import torch

from torch.utils.data import Dataset
from utils.data_utils import *

class Example:
    def __init__(self, config, vocab, data):
        
        article = ' '.join(data['article'])
        abstract = ' '.join(data['abstract'])

        src_words = article.split()[:config['max_src_ntokens']]
        self.enc_inp = [vocab.word2id(w) for w in src_words]

        abstract_words = abstract.split()
        abs_ids = [vocab.word2id(w) for w in abstract_words]
        self.dec_inp, self.dec_tgt = self.get_dec_inp_tgt(abs_ids, config['max_tgt_ntokens'])

        self.art_extend_vocab, self.art_oovs = article2ids(src_words, vocab)
        abs_extend_vocab = abstract2ids(abstract_words, vocab, self.art_oovs)

        if config['copy']:      
            # 改写目标输出 反映COPY OOV
            _, self.dec_tgt = self.get_dec_inp_tgt(abs_extend_vocab, config['max_tgt_ntokens'])

        self.original_article = article
        self.original_abstract = abstract

    def get_dec_inp_tgt(self, sequence, max_len, start_id = START, stop_id = END):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            # 如果需要截断，就不保留End Token
            inp = inp[:max_len]
            target = target[:max_len]
        else: # no truncation
            target.append(stop_id) # end token
        assert len(inp) == len(target)  
        return inp, target

class Batch:
    def __init__(self, batch):
        # 过滤掉他们（我有特别的filter技巧~）
        batch = list(filter(lambda poi: len(poi.enc_inp)>0, batch))

        dec_inp = [poi.dec_inp for poi in batch]
        dec_tgt = [poi.dec_tgt for poi in batch]
        enc_inp = [poi.enc_inp for poi in batch]
        #print(dec_inp)

        art_extend_vocab = [poi.art_extend_vocab for poi in batch]

        self.enc_lens = [len(src) for src in enc_inp]
        self.dec_lens = [len(tgt) for tgt in dec_inp]
        self.art_oovs = [poi.art_oovs for poi in batch]

        self.dec_inp = torch.tensor(pad_sequence(dec_inp, PAD))
        self.dec_tgt = torch.tensor(pad_sequence(dec_tgt, PAD))
        self.enc_inp = torch.tensor(pad_sequence(enc_inp, PAD))
        self.art_batch_extend_vocab = torch.tensor(pad_sequence(art_extend_vocab, PAD))
        self.max_art_oovs = max([len(oovs)for oovs in self.art_oovs])

        self.enc_pad_mask = self.enc_inp.eq(PAD)
        self.dec_pad_mask = self.dec_inp.eq(PAD)

        self.original_abstract = [poi.original_abstract for poi in batch]
        self.original_article = [poi.original_article for poi in batch]


class CNNDMDataset(Dataset):
    def __init__(self, split: str, path: str, config, vocab) -> None:
        assert split in ['train', 'val', 'test']
        self._data_path = os.path.join(path, split)
        self._n_data = _count_data(self._data_path)
        self.config = config
        self.vocab = vocab
        print('cnn-dm %s set loaded! %d examples found.'%(split, self._n_data))
        
    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        with open(os.path.join(self._data_path, '{}.json'.format(i))) as f:
            js = json.loads(f.read())
        return Example(self.config, self.vocab, js)

def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data

class Vocab:

    def __init__(self, vocab_file, vocab_size):
        with open(vocab_file, 'rb') as f:
            wc = pkl.load(f)
        self._word2id, self._id2word = make_vocab(wc, vocab_size)

    def word2id(self, word):
        return self._word2id['<unk>'] if word not in self._word2id else self._word2id[word]

    def id2word(self, idx):
        return '<unk>' if idx >= self.size else self._id2word[idx]

    @property
    def size(self):
        return len(self._word2id)


class Collate():
    def __init__(self, beam_size = 1):
        self.beam = beam_size

    def _collate(self, batch):
        return Batch(batch * self.beam)

    def __call__(self, batch):
        return self._collate(batch)
