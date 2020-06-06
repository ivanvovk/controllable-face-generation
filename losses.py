import torch
from torch.nn.functional import softplus
from torch.autograd import grad

def zero_centered_gradient_penalty(real_samples, real_prediction):
    """
    Computes zero-centered gradient penalty for E, D
    """    
    grad_outputs = torch.ones_like(real_prediction, requires_grad=True)    
    squared_grad_wrt_x = grad(outputs=real_prediction, inputs=real_samples, grad_outputs=grad_outputs,\
                              create_graph=True, retain_graph=True)[0].pow(2)
    
    return squared_grad_wrt_x.view(squared_grad_wrt_x.shape[0], -1).sum(dim=1).mean()

def loss_discriminator(E, D, alpha, real_samples, fake_samples, gamma=10):
    real_prediction, fake_prediction = D(E(real_samples, alpha)), D(E(fake_samples, alpha))
    # Minimize negative = Maximize positive (Minimize incorrect D predictions for real data,
    #                                        minimize incorrect D predictions for fake data)

    loss = (softplus(-real_prediction) + softplus(fake_prediction)).mean()
    return loss
    if gamma > 0:
        loss += zero_centered_gradient_penalty(real_samples, real_prediction).mul(gamma/2)
    return loss

def loss_generator(E, D, alpha, fake_samples):
    # Minimize negative = Maximize positive (Minimize correct D predictions for fake data)
    return softplus(-D(E(fake_samples, alpha))).mean()
    
def loss_autoencoder(F, G, E, scale, alpha, z):
    latent_codes = F(z, scale, z2=None, p_mix=0)
    # Autoencoding loss in latent space
    return (latent_codes[:, 0, :] - E(G(latent_codes, scale, alpha), alpha)).pow(2).mean()

def reconstruction_loss(x, x_reconstruction):
    return (x - x_reconstruction).pow(2).mean()