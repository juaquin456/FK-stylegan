from net import Generator, Discriminator
from dataset import PokeData
from loss import d_loss, g_loss, r1_reg

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def sample_labels(b_sz, c_dim):
    labels = torch.zeros(b_sz, c_dim)
    for i in range(b_sz):
        num_active = np.random.randint(1, 3)
        active_classes = np.random.choice(c_dim, num_active, replace=False)
        labels[i, active_classes] = 1
    return labels

epochs = 10000
batch_size = 64
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])
ds = PokeData("/home/juaquin/Documents/playground/cds/images", transform=data_transform)
dataloader = DataLoader(ds, batch_size=64, shuffle=True)

G = Generator(512, 18)
D = Discriminator(c_dim=18)
optim_g = optim.Adam(G.parameters(), lr=0.001)
optim_d = optim.Adam(D.parameters(), lr=0.002)

global_step = 0

for epoch in range(epochs):
    for real_images, real_labels in dataloader:
        z = torch.randn(batch_size, 512)
        fake_labels = sample_labels(batch_size, 18)
        fake_images = G(z, fake_labels)

        real_logits = D(real_images, real_labels)
        fake_logits = D(fake_images.detach(), fake_labels)

        loss_d = d_loss(real_logits, fake_logits) + r1_reg(D, real_images, real_labels)
        optim_d.zero_grad()
        loss_d.backward()
        optim_d.step()

        # GENERATOR
        optim_g.zero_grad()
        z = torch.randn(batch_size, 512)
        fake_images = G(z, fake_labels)
        fake_logits = D(fake_images, fake_labels)
        loss_g = g_loss(fake_logits)
        loss_g.backward()
        optim_g.step()

        writer.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
        writer.add_scalar("Loss/Generator", g_loss.item(), global_step)

        global_step += 1
    if epoch % 10 == 0:
        torch.save(G.state_dict(), f"./weights/{epoch}.pth")
writer.close()