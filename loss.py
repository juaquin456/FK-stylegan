import torch
import torch.nn.functional as F
import math

def d_loss(real_logits, fake_logits):
    real_loss = F.softplus(-real_logits).mean()
    fake_loss = F.softplus(fake_logits).mean()
    return real_loss + fake_loss

def r1_reg(real_images, real_logits, gamma=10):
    grad_real = torch.autograd.grad(
        outputs=real_logits.sum(), inputs=real_images, create_graph=True, retain_graph=True,
    )[0]
    grad_penalty = grad_real.square().sum(dim=[1,2,3]).mean()
    return 0.5*gamma*grad_penalty


def g_loss(fake_logits):
    return F.softplus(-fake_logits).mean()

def path_length_regularization(w, images, pl_mean):
    noise = torch.randn_like(images) / math.sqrt(images.shape[2] * images.shape[3])
    output = (images * noise) . sum()
    grad = torch.autograd.grad(outputs=output, inputs=w, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(dim=1) + 1e-8)
    pl_mean = pl_mean + 0.01* (path_lengths.mean() - pl_mean)
    pl_penalty = ((path_lengths - pl_mean)**2).mean()
    return pl_penalty, pl_mean.detach(), path_lengths