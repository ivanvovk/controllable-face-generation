import torchvision
import torch
from torch import nn
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cluster import KMeans
from torch import nn, optim
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import LambdaLR


def latent_space_quality(autoencoder, dataloaders, device='cpu'):
    cdn = lambda x: x.cpu().detach().numpy()
    
    train, test = dataloaders
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    
    def inference(model, loader):
        descriptor=[]
        for idx, (X, y) in enumerate(loader):
            prediction, target = cdn(model.encoder(X.to(device))), cdn(y).reshape(-1,1)
            descriptor.append(np.hstack([prediction.reshape(prediction.shape[0], -1), target]))
        return np.vstack(descriptor)
            
    descriptor_train, descriptor_test = inference(autoencoder, train), inference(autoencoder, test)
    
    autoencoder.train()
    
    return (descriptor_train, descriptor_test)

class DescriptorDataset(torch.utils.data.Dataset):
    def __init__(self, descriptor):
        self.descriptor = descriptor
        
    def __len__(self):
        return len(self.descriptor)
    
    def __getitem__(self, idx):
        obj = torch.Tensor(self.descriptor[idx])
        return (obj[:-1].float(), obj[-1])
    
    def fit_logistic_regression(self):
        lr = LogisticRegressionCV(Cs=10, cv=5, max_iter=500)
        lr.fit(descriptor[:, :-1], descriptor[:, -1])
        
        print(f'baseline acc: {[lr.scores_[k].mean() for k in lr.scores_.keys()]}')
        
    def fit_kmeans(self):
        self.kmeans = KMeans(10)
        self.kmeans.fit(self.descriptor[:, :-1])
        
        
class SingleLayer(nn.Module):
    def __init__(self, latent_dim, n_classes, dropout=0):
        super().__init__()
             
        self.FC = nn.Linear(latent_dim, n_classes)
        
        if dropout!=0:
            self.FC = nn.Sequential(nn.Dropout(dropout), self.FC)
        
    def forward(self, x):
        return self.FC(x)
    
    
def fit_FC(autoencoder, loaders, args):
    
    descr_train, descr_test = latent_space_quality(autoencoder, loaders, device=args['device'])    
    train_descriptor_dataset, test_descriptor_dataset = DescriptorDataset(descr_train), DescriptorDataset(descr_test)
    
    train_descr_loader = torch.utils.data.DataLoader(train_descriptor_dataset, batch_size = 64, shuffle=True, drop_last=False)
    test_descr_loader = torch.utils.data.DataLoader(test_descriptor_dataset, batch_size = 1000, shuffle=True, drop_last=False)
    
    criterion_CE = nn.CrossEntropyLoss()
    test_accuracy = []    
    
    latent_dim = train_descriptor_dataset[0][0].shape[0]
    
    fc_layer = SingleLayer(latent_dim=latent_dim, n_classes=10, dropout=0).to(args['device'])
    opt_fc = optim.Adam(fc_layer.parameters(), lr=1e-3, weight_decay=1e-5)
#     scheduler = LambdaLR(opt_fc, lr_lambda=lambda epoch: 0.9 ** epoch)
        
    for epoch in range(20):
        for index, (X, y) in tqdm(enumerate(train_descr_loader), total=len(train_descr_loader),
                                  leave=False, desc=f'Fit FC, Epoch: {epoch}'):

            y_hat = fc_layer(X.to(args['device']))
            loss = criterion_CE(y_hat, y.to(args['device']).long())

            opt_fc.zero_grad()
            loss.backward()
            opt_fc.step()

        # Test Step
        fc_layer.eval()

        acc = 0
        for index, (X, y) in enumerate(test_descr_loader):
            y_hat = fc_layer(X.to(args['device'])).cpu().detach()
            acc += (y == y_hat.argmax(dim=1)).sum().item()/y_hat.shape[0]
        test_accuracy.append(acc/len(test_descr_loader))

        fc_layer.train()
        
#         if epoch%2==0:
#             scheduler.step()
        
    return test_accuracy