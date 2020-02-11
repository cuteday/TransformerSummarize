import argparse
from utils.initialize import init_seeds
from utils.config import init_config
from train import Trainer
from models.decode import BeamSearch

def train(config):
    trainer = Trainer(config)
    trainer.train()
    trainer.save()

def test(config):
    config['is_predicting'] = True
    predictor = BeamSearch(config)
    predictor.decode()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='urara')
    parser.add_argument('-train_from', default='', type=str)
    parser.add_argument('-test_from', default='', type=str)
    init_seeds()
    args = parser.parse_args()
    config_ = init_config(vars(args))
    #train(config_)
    #test(config_)
