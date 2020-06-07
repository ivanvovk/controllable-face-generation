import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip

from data_utils import CelebaDataset, EmbeddingDataset, BatchCollate
from models import Generator, ConcatGenerator, ProjectionDiscriminator

from loss import DisLoss, GenLoss


def train(config, output_directory, device)
    features = config['features']
    train_dataset = EmbeddingDataset(torch.load(config['emb_path']))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              num_workers=0, shuffle=True,
                              collate_fn=BatchCollate())
    
    DEVICE = device
    num_classes = len(features)
    G = Generator(hidden_size=config['hidden_size'], num_classes=num_classes).to(DEVICE)
    D = ProjectionDiscriminator(hidden_size=config['hidden_size'], num_classes=num_classes).to(DEVICE)

    args =  {'dataset': 'MNIST',
             'eval_each': 10,
             'epochs': 101,
             'log_dir': 'CelebA64_256_v2/',
             'device': DEVICE,
             'weight_decay': 1e-05,
             'depth': 16,
             'gamma': 0.2,
             'lmbda': 0.5,
             'batch_norm': False,
             'batch_size': 64,
             'colors': 3,
             'latent_width': 4, # Bottleneck HW
             'width': 128, # Means 4 downsampling blocks
             'latent': 32, # Bottleneck channels
             'n_classes': 10,
             'advdepth': 16,
             'lr': 0.0001}

    scales = int(round(math.log(args['width'] // args['latent_width'], 2)))
    ae = Autoencoder(scales=scales,depth=args['depth'],latent=args['latent'],colors=args['colors']).to(args['device']).eval()

    ae.load_state_dict(torch.load('acai_64.pt', map_location=args['device']))

    optimizer_g = torch.optim.Adam(G.parameters(), lr=config['learning_rate'], betas=config['betas'])
    optimizer_d = torch.optim.Adam(D.parameters(), lr=config['learning_rate'], betas=config['betas'])

    criterion_d = DisLoss(config['loss_type'])
    criterion_g = GenLoss(config['loss_type'])

    G.train();
    D.train();

    BCE = nn.BCEWithLogitsLoss()
    
    iteration = 0
    for epoch in range(config['num_epochs']):
        for _ in tqdm_notebook(range(len(train_loader))):
            # =================================================================== #
            #                         1. Get new batch                            #
            # =================================================================== #

            latent_real, y_real = next(iter(train_loader))
            y_fake = sample_target_labels(y_real, num_classes)
            latent_real, y_real, y_fake = latent_real.to(DEVICE), y_real.to(DEVICE), y_fake.to(DEVICE)

            bs = latent_real.size(0)

            # =================================================================== #
            #                        2. Train Discriminator                       #
            # =================================================================== #
            # for _ in range(5):
            dis_real = D(latent_real, y_real)

            latent_real, y_real = next(iter(train_loader))
            y_fake = sample_target_labels(y_real, num_classes)
            latent_real, y_real, y_fake = latent_real.to(DEVICE), y_real.to(DEVICE), y_fake.to(DEVICE)

            latent_fake = G(latent_real, y_fake)
            dis_fake = D(latent_fake, y_fake)

            if loss_type == 'bce':
                d_loss = BCE(dis_real, torch.ones_like(dis_real).to(DEVICE)) + \
                         BCE(dis_fake, torch.zeros_like(dis_fake).to(DEVICE))
            else:
                d_loss = criterion_d(dis_real, dis_fake)

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # =================================================================== #
            #                         3. Train Generator                          #
            # =================================================================== #
            latent_real, y_real = next(iter(train_loader))
            y_fake = sample_target_labels(y_real, num_classes)
            latent_real, y_real, y_fake = latent_real.to(DEVICE), y_real.to(DEVICE), y_fake.to(DEVICE)

            latent_fake = G(latent_real, y_fake)
            dis_fake = D(latent_fake, y_fake)

            latent_cyclic =  G(latent_fake, y_real)
            cyclic_loss = F.l1_loss(latent_cyclic, latent_real)

            dis_fakes.append(dis_fake)
            dis_reals.append(dis_real)

            # latent_id =  G(latent_real, y_real)
            # identity_loss = F.l1_loss(latent_id, latent_real)

            if loss_type == 'bce':
                g_loss = BCE(dis_fake, torch.ones_like(dis_fake).to(DEVICE)) + cyclic_loss
            else:
                g_loss = criterion_g(dis_fake) + cyclic_loss

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            if iteration % 50 == 0:
                print("Step: {} | D_loss: {:.3f} | G_loss: {:.3f} | Cyclic loss: {:.3f}".format(iteration, 
                                                                            d_loss.item(), 
                                                                            g_loss.item(),
                                                                            cyclic_loss.item()))


            if iteration % 1000 == 0:
                torch.save({'iteration': iteration,
                            'generator': G.state_dict(), 
                            'discriminator': D.state_dict()}, 
                           os.path.join(output_directory, f'model_{itetarion}.pt'))

            iteration += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", required=False, 
                        default='config.json', help='configuration JSON file')
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-d', "--device", required=False, 
                        default='cuda:0')
    args = parser.parse_args()

    train(args.config, args.output_directory, arg.device)
