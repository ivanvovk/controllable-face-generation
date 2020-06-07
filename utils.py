import torchvision
import torch
from tqdm.notebook import tqdm
import torch.nn.functional as F
from torch import nn

def load_dataset(dataset, bs, shuffle=True, drop_last=True):
    '''
    Dataset list: SVHN/ MNIST/ CIFAR10/ CelebA32
    '''
    
    if dataset == 'SVHN':
    
        train_set = torchvision.datasets.SVHN(
            root = './SVHN',
            split = 'train',
            download = True,
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

        test_set = torchvision.datasets.SVHN(
            root = './SVHN',
            split = 'test',
            download = True,
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
        
    elif dataset == 'MNIST':
        
        train_set = torchvision.datasets.MNIST(
            root = './MNIST',
            train = True,
            download = True,
            transform = torchvision.transforms.Compose([torchvision.transforms.Pad(2), torchvision.transforms.ToTensor()]))

        test_set = torchvision.datasets.MNIST(
            root = './MNIST',
            train = False,
            download = True,
            transform = torchvision.transforms.Compose([torchvision.transforms.Pad(2), torchvision.transforms.ToTensor()]))
        
        
    elif dataset == 'CIFAR10':
        
        train_set = torchvision.datasets.CIFAR10(
            root = './CIFAR10',
            train = True,
            download = True,
            transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                        torchvision.transforms.ToTensor()]))

        test_set = torchvision.datasets.CIFAR10(
            root = './CIFAR10',
            train = False,
            download = True,
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
        
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = bs, shuffle=shuffle, drop_last=drop_last)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = bs * 4, shuffle=shuffle, drop_last=drop_last)
    
    return train_loader, test_loader
