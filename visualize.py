import torch
import numpy as np
import matplotlib.pyplot as plt


def random_interpolation(autoencoder, loader, savepath=None):
    cdn = lambda x: x.cpu().detach().numpy()
    
    # Get the device on which AE is located
    device = list(autoencoder.parameters())[0].device
    try:
        batch, _ = next(iter(loader))
    except:
        batch = next(iter(loader))
        
    batch = batch.to(device)
    bs = batch.shape[0]
    
    alphas = 0.5 * torch.rand(bs, 1, 1, 1).to(device)
    
    latent_code = autoencoder.encoder(batch)
    reconstruction = autoencoder.decoder(latent_code)
    
    shifted_index = torch.arange(0, bs) - 1
    interpolated_code = latent_code + alphas * (latent_code[shifted_index] - latent_code)
        
    interpolation = cdn(autoencoder.decoder(interpolated_code))
    
    def img_reshape(img):
        n_c, h, w = img.shape
        if n_c == 1:
            return img.reshape(h, w)
        elif n_c == 3:
            return np.transpose(img, (1,2,0))
        
    top, bottom = cdn(batch[:5]), cdn(batch[shifted_index][:5])
    top_reconstruction, bottom_reconstruction = cdn(reconstruction[:5]), cdn(reconstruction[shifted_index][:5]) 
        
    fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(25, 25))
    
    for col in range(5):
        alpha = np.round(alphas[col].cpu().numpy().item(), 3)
        
        ax[0, col].set_title(f'Alpha: {alpha}', size=30)
        
        for v in range(5):
            ax[v, col].axis('off')

        ax[0, col].imshow(img_reshape(top[col]), cmap=plt.cm.gray)
        ax[1, col].imshow(img_reshape(top_reconstruction[col]), cmap=plt.cm.gray)
        
        ax[2, col].imshow(img_reshape(interpolation[col]), cmap=plt.cm.gray)
        
        ax[3, col].imshow(img_reshape(bottom_reconstruction[col]), cmap=plt.cm.gray)
        ax[4, col].imshow(img_reshape(bottom[col]), cmap=plt.cm.gray)
        
    plt.tight_layout(pad=0.22)
    
    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()
        
        
def uniform_interpolation(autoencoder, loader, N=11, savepath=None):
    cdn = lambda x: x.cpu().detach().numpy()
    
    # Get the device on which AE is located
    device = list(autoencoder.parameters())[0].device
    
    try:
        batch, _ = next(iter(loader))
    except:
        batch = next(iter(loader))
        
    batch = batch.to(device)
    bs = batch.shape[0]
    
    # Grid of 11 uniform alphas [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] by default
    alphas = torch.arange(0, 1.1, step=0.1, dtype=torch.float32).reshape(-1, 1).to(device)
    
    latent_code = autoencoder.encoder(batch)
    reconstruction = autoencoder.decoder(latent_code)
    
    shifted_index = torch.arange(0, bs) - 1
    
    def normalize(X):
        return (X - X.min())/(X.max() - X.min())
    
    pair_interpolations = []
    for pair in range(N):
        pair_code = torch.cat([latent_code[pair] + alpha.reshape(1, 1, 1, 1) * \
                              (latent_code[shifted_index][pair] - latent_code[pair]) for alpha in alphas])
        pair_interpolations.append(normalize(cdn(autoencoder.decoder(pair_code))))
            
    def img_reshape(img):
        n_c, h, w = img.shape
        if n_c == 1:
            return img.reshape(h, w)
        elif n_c == 3:
            return np.transpose(img, (1,2,0))
            
    fig, ax = plt.subplots(ncols=len(alphas), nrows=N, figsize=(25, int(25//(11/N))))
    
    for row in range(N):
        for col in range(len(alphas)):
            
            if N!=1:
                ax[row, col].axis('off')
                ax[row, col].imshow(img_reshape(pair_interpolations[row][col]), cmap=plt.cm.gray)
            else:
                ax[col].axis('off')
                ax[col].imshow(img_reshape(pair_interpolations[row][col]), cmap=plt.cm.gray)

        
    plt.tight_layout(pad=0.22)
    
    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()
        
        
def visualization(autoencoder, loaders, args, epoch):
    train_loader, test_loader = loaders
    
    random_interpolation(autoencoder, train_loader, savepath=args['log_dir'] + f'Train_RandomInterpolation_{epoch}.png')
    random_interpolation(autoencoder, test_loader, savepath=args['log_dir'] + f'Test_RandomInterpolation_{epoch}.png')
    
    uniform_interpolation(autoencoder, train_loader, N=11, savepath=args['log_dir'] + f'Train_UniformInterpolation_{epoch}.png')
    uniform_interpolation(autoencoder, test_loader, N=11, savepath=args['log_dir'] + f'Test_UniformInterpolation_{epoch}.png')
