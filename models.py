import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import utils

import torchvision

from layers import BN_block, CondBN_block


class Generator(nn.Module):
    def __init__(self, hidden_size=256, num_classes=4):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        
        self.block1 = CondBN_block(hidden_size, hidden_size, num_classes)
        self.block2 = CondBN_block(hidden_size, hidden_size, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        # self.init_weights([self.block1, self.block4])
    
    def init_weights(self, blocks):
        for block in blocks:
            for layer in block.children():
                if type(layer) == nn.Linear:
                    init.xavier_uniform_(layer.weight)
                    layer.bias.data.fill_(0.01)

    def forward(self, x, y):
        x = self.block1(x, y)
        x = self.dropout(self.block2(x, y))
        return x


class ConcatGenerator(nn.Module):
    def __init__(self, hidden_size=256, num_classes=4):
        super(ConcatGenerator, self).__init__()
        self.hidden_size = hidden_size
        
        self.upsample = nn.Sequential(
            nn.Linear(num_classes, hidden_size),
            nn.ReLU()
        )
        self.model = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.5)
        self.init_weights([self.upsample, self.model])
    
    def init_weights(self, blocks):
        for block in blocks:
            for layer in block.children():
                if type(layer) == nn.Linear:
                    init.xavier_uniform_(layer.weight)
                    layer.bias.data.fill_(0.01)

    def forward(self, x, y):
        y = self.upsample(y.float())
        x = self.dropout(torch.cat([x, y], axis=1))
        x = self.model(x)
        return x
    
    
class ProjectionDiscriminator(nn.Module):
    def __init__(self, hidden_size=256, num_classes=4):
        super(ProjectionDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.mlp = nn.Sequential(
            utils.spectral_norm(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU()
        )
        self.last_linear = utils.spectral_norm(nn.Linear(hidden_size, 1))
        self.proj = utils.spectral_norm(nn.Embedding(num_classes, hidden_size))
        self.dropout = nn.Dropout(0.5)
        
        init.xavier_uniform_(self.last_linear.weight)


    def forward(self, x, y):
        x = self.mlp(x)
        x_fc = self.dropout(self.last_linear(x))
        inner_product = torch.sum(self.proj(y) * x, dim=1, keepdim=True)
        return x_fc + inner_product

            
class Autoencoder(nn.Module):
    def __init__(self, scales, depth, latent, colors, batch_norm=False):
        super().__init__()
             
        self.encoder = self._make_network(scales, depth, latent, colors, part='encoder', bn=batch_norm)
        self.decoder = self._make_network(scales, depth, latent, colors, part='decoder', bn=batch_norm)
        
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    @staticmethod
    def _make_network(scales, depth, latent, colors, part=None, bn = True):
        """
        input:
        part - encoder/decoder, str
        
        Following the structure of reimplementation by authors paper at
        kylemcdonald/ACAI (PyTorch).ipynb
        """
        activation = nn.LeakyReLU(0.01) 
        
        sub_network = []
        
        if part == 'encoder':
            sub_network += [nn.Conv2d(colors, depth, 1, padding=1)]
            
            input_channels = depth
            transformation = nn.AvgPool2d(2)
            
        elif part == 'decoder':
            
            input_channels = latent
            transformation = nn.Upsample(scale_factor=2)
        
        # joint part
        for scale in range(scales):
            k = depth * np.power(2,scale)
                        
            if bn:
                sub_network.extend([nn.Conv2d(input_channels, k, 3, padding=1), nn.BatchNorm2d(k), activation, 
                                    transformation])
                
            else:
                sub_network.extend([nn.Conv2d(input_channels, k, 3, padding=1), activation,
                    transformation])
            
            input_channels = k
        
        if part == 'encoder':
            k = depth << scales
            sub_network.extend([nn.Conv2d(input_channels, k, 3, padding=1), activation, nn.Conv2d(k, latent, 3, padding=1)])
        
        elif part == 'decoder':
            sub_network.extend([nn.Conv2d(input_channels, depth, 3, padding=1), activation, nn.Conv2d(depth, colors, 3, padding=1)])
        
        
        # Same initialization as in paper
        slope = 0.2
        for layer in sub_network:
            if hasattr(layer, 'weight'):
                layer.weight.data.normal_(std=(1/((1 + slope**2) * np.prod(layer.weight.data.shape[:-1])))**2)

            if hasattr(layer, 'bias'):
                layer.bias.data.zero_()
        
        return nn.Sequential(*sub_network)    
    
    def track_gradient(self):
        def get_grads(block):
            cache = []
            for layer in block:
                if isinstance(layer, nn.Conv2d):
                    cache.append(torch.cat([layer.weight.grad.reshape(-1,1), layer.bias.grad.reshape(-1,1)]))
            return torch.cat(cache)
        
        return {'encoder_grads': torch.norm(get_grads(self.encoder)),
                'decoder_grads': torch.norm(get_grads(self.decoder))}


class Critic(nn.Module):
    def __init__(self, scales, depth, latent, colors):
        super().__init__()
        
        self.flatten = nn.Flatten()
        self.critic = Autoencoder._make_network(scales, depth, latent, colors, part='encoder',bn=False)
        
        
    def forward(self, x):

        return self.flatten(self.critic(x)).mean(dim=1)
    
    def descriptor(self, x):
        return self.critic(x)
    
    def track_gradient(self):
        def get_grads(block):
            cache = []
            for layer in block:
                if isinstance(layer, nn.Conv2d):
                    cache.append(torch.cat([layer.weight.grad.reshape(-1,1), layer.bias.grad.reshape(-1,1)]))
            return torch.cat(cache)
        
        return {'critic_grads': torch.norm(get_grads(self.critic))}