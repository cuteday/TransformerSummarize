import os

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adagrad, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CNNDMDataset, Collate, Vocab
from utils.data_utils import get_input_from_batch, get_output_from_batch
from models.model import Model
from utils.train_utils import logging, calc_running_avg_loss

class Trainer:
    def __init__(self, config):
        
        self.config = config
        self.step = 0
        self.vocab = Vocab(config['vocab_file'], config['vocab_size'])
        self.train_data = CNNDMDataset('train', config['data_path'], config, self.vocab)
        self.validate_data = CNNDMDataset('val', config['data_path'], config, self.vocab)
        # self.model = Model(config).to(device)
        # self.optimizer = None
        self.setup(config)
        self.device = config['device']

    def setup(self, config):
        
        self.model = Model(config).to(config['device'])
        self.optimizer = Adagrad(self.model.parameters(),lr = config['learning_rate'], initial_accumulator_value=0.1)
        #self.optimizer = Adam(self.model.parameters(),lr = config['learning_rate'],betas = config['betas'])
        checkpoint = None
        
        if config['train_from'] != '':  # Counter在两次mostCommon间, 相同频率的元素可能以不同的次序输出...!
            logging('Train from %s'%config['train_from'])
            checkpoint = torch.load(config['train_from'], map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            self.step = checkpoint['step']
            self.vocab = checkpoint['vocab']
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train_one(self, batch):
        """ coverage not implemented """
        config = self.config
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, config, self.device)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, self.device)
        pred = self.model(enc_batch, dec_batch, enc_padding_mask, dec_padding_mask, enc_batch_extend_vocab, extra_zeros)
        
        #print(pred.max(dim=-1)[1][:,0])    # 
        #loss = self.model.nll_loss(pred, target_batch, dec_lens_var)
        loss = self.model.label_smoothing_loss(pred, target_batch)
        return loss

    def train(self):

        config = self.config
        train_loader = DataLoader(self.train_data, batch_size=config['batch_size'], shuffle=False, collate_fn=Collate())

        running_avg_loss = 0
        self.model.train()

        for _ in range(config['train_epoch']):
            for batch in train_loader:
                self.step += 1
                
                loss = self.train_one(batch)
                running_avg_loss = calc_running_avg_loss(loss.item(), running_avg_loss)
                loss.div(float(config['gradient_accum'])).backward()

                if self.step % config['gradient_accum'] == 0:   # gradient accumulation
                    clip_grad_norm_(self.model.parameters(), config['max_grad_norm'])
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.step % config['report_every'] == 0:
                    logging("Step %d Train loss %.3f"%(self.step, running_avg_loss))    
                if self.step % config['save_every'] == 0:
                    self.save()
                if self.step % config['validate_every'] == 0:
                    self.validate()


    @torch.no_grad()
    def validate(self):
        self.model.eval()
        validate_loader = DataLoader(self.validate_data, batch_size=self.config['batch_size'], shuffle=False, collate_fn=Collate())
        losses = []
        for batch in tqdm(validate_loader):
            loss = self.train_one(batch)
            losses.append(loss.item())
        self.model.train()
        ave_loss = sum(losses) / len(losses)
        logging('Validate loss : %f'%ave_loss)

    def save(self):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'vocab': self.vocab
        }
        save_path = os.path.join(self.config['model_path'], 'model_s%d.pt'%self.step)
        logging('Saving model step %d to %s...'%(self.step, save_path))
        torch.save(state, save_path)