from net import Generator, Discriminator
from dataset import PokeData
from loss import d_loss, g_loss, r1_reg, path_length_regularization
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, flush=True)

def sample_labels(b_sz, c_dim):
    labels = torch.zeros(b_sz, c_dim, device=device)
    for i in range(b_sz):
        num_active = np.random.randint(1, 3)
        active_classes = np.random.choice(c_dim, num_active, replace=False)
        labels[i, active_classes] = 1
    return labels

epochs = 10000
batch_size = 16 
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])
ds = PokeData(sys.argv[1], transform=data_transform)
print(len(ds), flush=True)
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

G = Generator(512, 18).to(device)
D = Discriminator(c_dim=18).to(device)

optim_g = optim.Adam(G.parameters(), lr=0.0001, betas=(0.0, 0.99))
optim_d = optim.Adam(D.parameters(), lr=0.0004, betas=(0.0, 0.99))

current_res = 4
global_step = 0
pl_mean = torch.zeros(1, device=device)
for epoch in range(epochs):
    for real_images, real_labels in dataloader:
        real_images, real_labels = real_images.to(device), real_labels.to(device)
        z = torch.randn(batch_size, 512, device=device)
        fake_labels = sample_labels(batch_size, 18)
        fake_images, _ = G(z, fake_labels)
        print(fake_images.shape)
        if current_res < 256:
            fake_images = torch.nn.functional.interpolate(fake_images, scale_factor=256 // current_res)

        real_logits = D(real_images, real_labels)
        fake_logits = D(fake_images.detach(), fake_labels)

        loss_d = d_loss(real_logits, fake_logits) + r1_reg(D, real_images, real_labels)
        optim_d.zero_grad()
        loss_d.backward()
        optim_d.step()

        # GENERATOR
        optim_g.zero_grad()
        z = torch.randn(batch_size, 512, device=device)
        fake_images, w = G(z, fake_labels)
        if current_res < 256:
            fake_images = torch.nn.functional.interpolate(fake_images, scale_factor=256 // current_res)
        fake_labels = sample_labels(batch_size, 18)
        fake_logits = D(fake_images, fake_labels)

        pl_penalty, pl_mean, path_lengths = path_length_regularization(w, fake_images, pl_mean)

        lambda_pl = 2.0
        loss_g = g_loss(fake_logits) + lambda_pl * pl_penalty

        loss_g.backward()
        optim_g.step()

        writer.add_scalar("Loss/Discriminator", loss_d.item(), global_step)
        writer.add_scalar("Loss/Generator", loss_g.item(), global_step)

        global_step += 1

        if global_step % 10000 == 0 and current_res < 256:
            G.grow()
            current_res *= 2
            print("current_res:", current_res, flush=True)
    if epoch % 10 == 0:
        torch.save(G.state_dict(), f"./weights/{epoch}.pth")
writer.close()