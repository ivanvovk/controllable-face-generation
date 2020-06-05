import numpy as np
import torch


class ModelTrainer(object):
    def __init__(
        self,
        config=None,
        optimizers={'backbone_model_opt': None, 'duration_model_opt': None},
        logger=None,
        criterion=None
    ):
        self._config = config
        self.optimizers = optimizers
        self.logger = logger
        self.criterion = criterion
        
    def compute_losses(self, model, batch, training=True):
        model.train() if training else model.eval()
        outputs = model.forward(batch)
        losses = self.criterion(outputs, batch)
        loss_stats = self.criterion.loss_stats
        if training:
            return losses, loss_stats
        return losses, loss_stats, outputs
    
    def run_backward(self, model, losses):
        for loss in losses:
            loss.backward(retain_graph=True)
        self.gradient_apply_(model)
    
    def gradient_apply_(self, model):
        for key in self.optimizers.keys():
            self.optimizers[key].step()
        model.zero_grad()
        
    def log_training(self, iteration, loss_stats):
        self.logger.log(iteration, loss_stats={f'training/{key}': value
                                               for key, value in loss_stats.items()})
    
    def log_validating(self, iteration, loss_stats):
        self.logger.log(iteration, loss_stats={f'validating/{key}': value
                                               for key, value in loss_stats.items()})

    def _should_save_checkpoint(self, iteration):
        return (iteration % self._config.TRAIN.CHECKPOINT_SAVE_STEP) == 0

    def save_checkpoint(self, iteration, model):
        if self._should_save_checkpoint(iteration):
            self.logger.save_checkpoint(iteration, model)
