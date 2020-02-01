import os

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adagrad
from torch.utils.data import DataLoader

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
        # transformer use ada, test for now...
        self.optimizer = Adagrad(self.model.parameters(),lr = config['learning_rate'], initial_accumulator_value=0.1)
        checkpoint = None
        if config['train_from'] != '':
            logging('Train from %s'%config['train_from'])
            checkpoint = torch.load(config['train_from'], map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            self.step = checkpoint['step']
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train_one(self, batch):
        """ copy and coverage not implemented """

        config = self.config
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, config, self.device)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, self.device)

        pred = self.model(enc_batch, dec_batch)
        loss = self.model.label_smoothing_loss(pred, target_batch)
        return loss

    def train(self):

        config = self.config
        train_loader = DataLoader(self.train_data, batch_size=config['batch_size'], shuffle=True, collate_fn=Collate())

        running_avg_loss = 0
        self.model.train()

        for _ in range(config['train_epoch']):
            for batch in train_loader:
                self.step += 1
                self.optimizer.zero_grad()
                loss = self.train_one(batch)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), config['max_grad_norm'])
                self.optimizer.step()
                #print(loss.item())
                running_avg_loss = calc_running_avg_loss(loss.item(), running_avg_loss)

                if self.step % config['report_every'] == 0:
                    logging("Step %d Train loss %.3f"%(self.step, running_avg_loss))    
                if self.step % config['validate_every'] == 0:
                    self.validate()
                if self.step % config['save_every'] == 0:
                    self.save(self.step)
                if self.step % config['test_every'] == 0:
                    pass

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        validate_loader = DataLoader(self.validate_data, batch_size=self.config['batch_size'], shuffle=False, collate_fn=Collate())
        losses = []
        for batch in validate_loader:
            loss = self.train_one(batch)
            losses.append(loss.item())
        self.model.train()
        ave_loss = sum(losses) / len(losses)
        logging('Validate loss : %f'%ave_loss)

    def save(self, step):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': step
        }
        save_path = os.path.join(self.config['model_path'], 'model_s%d.pt'%step)
        logging('Saving model step %d to %s...'%(step, save_path))
        torch.save(state, save_path)

def train(config):
    trainer = Trainer(config)
    trainer.train()