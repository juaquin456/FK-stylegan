import torch
import torch.nn.functional as F

def d_loss(real_logits, fake_logits):
    real_loss = F.softplus(-real_logits).mean()
    fake_loss = F.softplus(fake_logits).mean()
    return real_loss + fake_loss

def r1_reg(discriminator, real_images, classes, gamma=10):
    real_images.requires_grad = True
    real_logits = discriminator(real_images, classes)
    grad_real = torch.autograd.grad(
        outputs=real_logits.sum(), inputs=real_images, create_graph=True
    )[0]
    grad_penalty = grad_real.square().sum(dim=[1,2,3]).mean()
    return 0.5*gamma*grad_penalty


def g_loss(fake_logits):
    return F.softplus(-fake_logits).mean()

def path_length_regularization(generator, w, images, pl_mean):
    ...