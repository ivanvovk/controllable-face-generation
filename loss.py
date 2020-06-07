import torch
from torch import nn


class DisLoss(nn.Module):
    def __init__(self, loss_type):
        super(DisLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'hinge':
            self.critetion = self.hinge_loss
        elif loss_type == 'dcgan':
            self.critetion = self.dcgan_loss
        elif loss_type == 'lsgan':
            self.critetion = self.lsgan_loss
        else:
            raise ValueError('Not supported! Choose from [\'hinge\', \'dcgan\', \'lsgan\'].')
        
    def hinge_loss(self, dis_real, dis_fake):
        return (1. - dis_real).relu().mean() + (1. + dis_fake).relu().mean()
    
    def dcgan_loss(self, dis_real, dis_fake):
        return -torch.log(F.softplus(-dis_real).mean().sigmoid()) - \
                torch.log(1.0 - F.softplus(dis_fake).mean().sigmoid())
    
    def lsgan_loss(self, dis_real, dis_fake):
        return 0.5 * ((dis_real - 1).pow(2).mean() + dis_fake.pow(2).mean())
        
    def forward(self, dis_real, dis_fake):
        """
        PARAMS:
            dis_real (Tensor): discriminator scores for real data
            dis_fake (Tensor): discriminator scores for fake outputs of generator
        """
        return self.critetion(dis_real, dis_fake)
    
    
class GenLoss(nn.Module):
    def __init__(self, loss_type):
        super(GenLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'hinge':
            self.critetion = self.hinge_loss
        elif loss_type == 'dcgan':
            self.critetion = self.dcgan_loss
        elif loss_type == 'lsgan':
            self.critetion = self.lsgan_loss
        else:
            raise ValueError('Not supported! Choose from [\'hinge\', \'dcgan\', \'lsgan\'].')
        
    def hinge_loss(self, dis_fake):
        return -dis_fake.mean()
    
    def dcgan_loss(self, dis_fake):
        return -torch.log(F.softplus(-dis_fake).mean().sigmoid())
    
    def lsgan_loss(self, dis_fake):
        return 0.5 * (dis_fake - 1).pow(2).mean()
        
    def forward(self, dis_fake):
        """
        PARAMS: 
            dis_fake (Tensor): discriminator scores for fake outputs of generator
        """
        return self.critetion(dis_fake)