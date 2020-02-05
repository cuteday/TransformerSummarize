import argparse
import torch
from utils.initialize import init_seeds
from utils.config import init_config
from train import Trainer
from models.model import Model
from models.decode import BeamSearch

def train(config):
    trainer = Trainer(config)
    trainer.train()
    trainer.save()

def test(config):
    if config['test_from'] is not '':
        config['is_predicting'] = True
        model = Model(config)
        saved_model = torch.load(config['test_from'], map_location='cpu')
        model.load_state_dict(saved_model['model'])
        step = saved_model['step']

    predictor = BeamSearch(model, config, step)
    predictor.decode()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='urara')
    parser.add_argument('-train_from', default='', type=str)
    parser.add_argument('-test_from', default='', type=str)
    init_seeds()
    args = parser.parse_args()
    config_ = init_config(vars(args))
    train(config_)
    #test(config_)