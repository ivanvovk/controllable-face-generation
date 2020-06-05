import os

import torch
from torch.utils.tensorboard import SummaryWriter


class Logger(SummaryWriter):
    def __init__(self, logdir):
        if os.path.exists(logdir):
            raise RuntimeError(f'Logdir `{logdir}` already exists. Remove it before training.')
        os.makedirs(logdir)
        super(Logger, self).__init__(logdir)
        
    def log(self, iteration, loss_stats):
        for key, value in loss_stats.items():
            self.add_scalar(key, value, iteration)
    
    def save_checkpoint(self, iteration, model):
        filename = f'{self.log_dir}/checkpoint_{iteration}.pt'
        torch.save(model.state_dict(), filename)
