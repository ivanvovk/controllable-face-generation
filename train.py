from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from torch import nn, optim
import torch.nn.functional as F
import torch

import os
import yaml

from modules import Autoencoder, Critic
from utils import load_dataset
from visualize import random_interpolation, uniform_interpolation, visualization
from utils import load_dataset
from evaluation import fit_FC


def train_baseline(args, batch_norm=False):
    """
    args - dict with AE parameters
    batch_norm - bool, whether to use BN as a part of block
    
    Trains parameters of AE w.r.t the Reconstruction Loss
    """
    
    if not os.path.exists(args['log_dir']):
        os.mkdir(args['log_dir'])
    
    max_test_acc = 0
    
    # Define AE 
    scales = int(round(np.log2(args['width'] // args['latent_width'])))
    autoencoder = Autoencoder(scales=scales,depth=args['depth'],latent=args['latent'],colors=args['colors'],
                              batch_norm=batch_norm).to(args['device'])
    print(autoencoder)
    
    # Define optimizer
    opt_ae = optim.Adam(autoencoder.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    
    train_loader, test_loader = load_dataset(args['dataset'], args['batch_size'])
    losses = defaultdict(list)
    
    for epoch in range(args['epochs']):
        
        # Train during one epoch
        print(f'Epoch {epoch} started')        
        
        for index, (X, _) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, desc=f'Epoch: {epoch}'):
            X = X.to(args['device'])


            latent_code = autoencoder.encoder(X)
            reconstruction = autoencoder.decoder(latent_code)

            reconstruction_loss = F.mse_loss(X, reconstruction)
            ae_loss = reconstruction_loss


            # AE's parameters update
            opt_ae.zero_grad()
            ae_loss.backward(retain_graph=True)
            
            if index%5==0:
                ae_grads = autoencoder.track_gradient()
                
            opt_ae.step()
                
#         train(autoencoder, opt_ae, critic, opt_c, train_loader, args, losses, clip=None)
            
            if index%10==0:
                for val, name in zip([ae_grads['encoder_grads'], ae_grads['decoder_grads'], reconstruction_loss], 
                                      ['encoder_gradients', 'decoder_gradients', 'reconstruction_loss']):

                    losses[name].append(val.item())
    
        # Evaluate on test
        if epoch % args['eval_each'] == 0:
            np.save(args['log_dir'] + 'Stats.npy', losses)
            
            losses['test_accuracy'].append(fit_FC(autoencoder, (train_loader, test_loader), args))
            
            if max(losses['test_accuracy'][-1]) > max_test_acc:
                
                max_test_acc = max(losses['test_accuracy'][-1])
                torch.save(autoencoder.state_dict(), args['log_dir'] + 'Autoencoder.torch')
                
                print(f'Best epoch: {epoch}!')
            
        # Visualize reconstructions and interpolations
        if epoch % args['eval_each'] == 0:
            visualization(autoencoder, (train_loader, test_loader), args, epoch)
        
    return losses


def train_acai(args, batch_norm=False):
    """
    args - dict with AE and Critic parameters
    batch_norm - bool, whether to use BN as a part of block
    
    Trains parameters of ACAI and Critic w.r.t two separate losses
    """
    
    if not os.path.exists(args['log_dir']):
        os.mkdir(args['log_dir'])
    
    max_test_acc = 0
    
    # Define AE & Critic 
    scales = int(round(np.log2(args['width'] // args['latent_width'])))
    autoencoder = Autoencoder(scales=scales,depth=args['depth'],latent=args['latent'],colors=args['colors'],
                              batch_norm=batch_norm).to(args['device'])
    
    print(autoencoder)
    critic = Critic(scales=scales, depth=args['advdepth'], latent=args['latent'], colors=args['colors']).to(args['device'])
    
    # Define optimizers
    opt_ae = optim.Adam(autoencoder.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    opt_c = optim.Adam(critic.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    
    train_loader, test_loader = load_dataset(args['dataset'], args['batch_size'])
    losses = defaultdict(list)
    
    for epoch in range(args['epochs']):
        
        # Train during one epoch
        print(f'Epoch {epoch} started')        
        
        for index, (X, _) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, desc=f'Epoch: {epoch}'):
            X = X.to(args['device'])

            alpha = 0.5 * torch.rand(args['batch_size'], 1, 1, 1).to(args['device'])

            latent_code = autoencoder.encoder(X)
            reconstruction = autoencoder.decoder(latent_code)

            # Here we shift all objects in batch by 1
            shifted_index = torch.arange(0, args['batch_size']) - 1
            interpolated_code = latent_code + alpha * (latent_code[shifted_index] - latent_code)

            # Decode interpolated latent code and calculate Critic's predictions
            reconstruction_interpolated = autoencoder.decoder(interpolated_code)
            alpha_reconstruction = critic(reconstruction_interpolated).reshape(args['batch_size'], 1, 1, 1)

            # Term1: Reconstruction loss
            # Term2: Trying to fool the Critic via Lowering it's predicted values on interpolated samples

            reconstruction_loss = F.mse_loss(X, reconstruction)
            critic_fooling_loss = (alpha_reconstruction**2).sum()
            ae_loss = reconstruction_loss + args['lmbda'] * critic_fooling_loss

            # Term1: Critic is trying to guess actual alpha
            # Term2: Critic is trying to assing "high realistic score" to samples which are linear interpolations (in data spcae)
            #        of original images and their reconstructions. Thus we are trying to encode the information about real samples
            #        to help Critic to distinguish between original and interpolated samples. (REGULARIZATION, optional)
            #        In case if our AE is perfect, it is just the critic(X) -> 0, w.r.t. Critic parameters

            alpha_guessing_loss = F.mse_loss(alpha_reconstruction, alpha)
            realistic_loss = (critic(args['gamma'] * X + (1 - args['gamma']) * reconstruction)**2).sum()
            critic_loss = alpha_guessing_loss + realistic_loss

            # AE's parameters update
            opt_ae.zero_grad()
            ae_loss.backward(retain_graph=True)
            
            if index%5==0:
                ae_grads = autoencoder.track_gradient()
                
            opt_ae.step()

            # Critic's parameters update
            opt_c.zero_grad()
            critic_loss.backward(retain_graph=True)
            
            if index%5==0:
                critic_grads = critic.track_gradient()
            # Clip gradients of a Critic
    #         nn.utils.clip_grad_norm_(critic.parameters(), 4)
            opt_c.step()
                
            
            if index%10==0:
                for val, name in zip([ae_grads['encoder_grads'], ae_grads['decoder_grads'], critic_grads['critic_grads'], \
                                  torch.std(alpha_reconstruction), alpha_guessing_loss, realistic_loss, \
                                  critic_loss, reconstruction_loss, critic_fooling_loss, ae_loss], 

                                  ['encoder_gradients', 'decoder_gradients', 'critic_gradients', 'std(alphas_prediction)', \
                                   'alpha_guessing_loss', 'realistic_loss', 'critic_loss', 'reconstruction_loss', \
                                   'critic_fooling_loss', 'ae_loss']):

                    losses[name].append(val.item())
    
        # Evaluate on test
        if epoch % args['eval_each'] == 0:
            np.save(args['log_dir'] + 'Stats.npy', losses)
            
            losses['test_accuracy'].append(fit_FC(autoencoder, (train_loader, test_loader), args))
            
            if max(losses['test_accuracy'][-1]) > max_test_acc:
                
                max_test_acc = max(losses['test_accuracy'][-1])
                torch.save(autoencoder.state_dict(), args['log_dir'] + 'Autoencoder.torch')
                torch.save(critic.state_dict(), args['log_dir'] + 'Critic.torch')
                
                print(f'Best epoch: {epoch}!')
            
        # Visualize reconstructions and interpolations
        visualization(autoencoder, (train_loader, test_loader), args, epoch)
        
    return losses