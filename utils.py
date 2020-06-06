import torch
from torchvision.utils import save_image

def sample_noise(bs, code=512, device='cpu'):
    return torch.randn(bs, code).to(device)

def find_alpha(tracked, limit):
    return min(tracked/limit, 1)

def allow_gradient(module, permission=True):
    for block in module.parameters():
        block.requires_grad = permission

def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult
        
def linear_scale_lr(tracked, total_items, start=5e-6, end=1.5e-4):
    coef = tracked/total_items
    return (1 - coef) * start + coef * end

def save_batch(name, fake, real, nrows=6):
    fake, real = fake.split(4), real.split(2)
    save_image(torch.cat([torch.cat([fake[i], real[i]], dim=0) for i in range(nrows)], dim=0), name, nrow=nrows, padding=1,
               normalize=True, range=(-1, 1))
    
def save_reconstructions(name, original, reconstruction, nrows=6):
    """
    original, reconstruction - type: list, e.g. original = [x, x_hat], reconstruction = [G(E(x)), G(E(x_hat))]
    
    [[orig_x, rec_x], [orig_x, rec_x], [orig_x, rec_x]]
    [[orig_x_hat, rec_x_hat], [orig_x_hat, rec_x_hat], [orig_x_hat, rec_x_hat]]
    
    """
    tensor = []
    for orig, rec in zip(original, reconstruction):        
        tensor.append(torch.cat([torch.cat([orig.split(1)[i], rec.split(1)[i]], dim=0) for i in range(nrows//2)], dim=0))
    
    save_image(torch.cat(tensor, dim=0), name, nrow=nrows, padding=1, normalize=True, range=(-1, 1))