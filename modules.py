import torchvision
import torch
from torch import nn
import numpy as np

            
class Autoencoder(nn.Module):
    def __init__(self, scales, depth, latent, colors, batch_norm=False, init=True):
        super().__init__()
             
        self.init = init
            
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
        
        if 5 > 10:
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