import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class Mapping(nn.Module):
    def __init__(self, z_dim, c_dim, n_layers):
        super().__init__()
        self.pixel_norm = PixelNorm()
        layers = []
        self.c_mpl = nn.Sequential(nn.Linear(c_dim, z_dim), nn.LeakyReLU(inplace=True))
        self.z_mpl = nn.Sequential(nn.Linear(z_dim, z_dim), nn.LeakyReLU(inplace=True))
        for _ in range(n_layers):
            layers.extend([nn.Linear(z_dim, z_dim), nn.LeakyReLU(0.2, inplace=True), PixelNorm()])
        self.mapping = nn.Sequential(*layers)
    def forward(self, z, c):
        z = self.pixel_norm(z)
        z = self.z_mpl(z)
        c = self.c_mpl(c)
        z = z + c
        return self.mapping(z)

class AdaIN(nn.Module):
    def __init__(self, n_chan, w_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_chan, affine=False)
        self.affine = nn.Linear(w_dim, 2*n_chan)

    def forward(self, x, w):
        batch, channels = x.size(0), x.size(1)
        style = self.affine(w) # [batch, 2*n_chan]
        ys = style[:, :channels].view(batch, channels, 1, 1)
        yb = style[:, channels:].view(batch, channels, 1, 1)
        x_norm = self.norm(x)
        return ys*x_norm + yb

class UP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x
class SynthesisBlock(nn.Module):
    def __init__(self, in_chan, out_chan, w_dim, is_const = False):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.w_dim = w_dim
        if is_const:
            self.const = nn.Parameter(torch.randn(1, in_chan, 4, 4))
        self.conv = nn.Conv2d(in_chan, out_chan, 3, padding=1)
        self.AdaIN = AdaIN(out_chan, w_dim)
        self.noise_scale = nn.Parameter(torch.zeros(1, out_chan, 1, 1))

    def forward(self, x, w):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        x = self.conv(x)
        x = x + self.noise_scale * noise
        x = self.AdaIN(x, w)
        return F.leaky_relu(x, 0.2)

class ToRGB(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, 3, kernel_size=1)
    def forward(self, logits):
        logits = self.conv(logits)
        return torch.tanh(logits)


class Stynthesis(nn.Module):
    def __init__(self, w_dim):
        super().__init__()
        self.w_dim = w_dim
        self.cur_res = 4
        self.blocks = nn.ModuleList([
            SynthesisBlock(512, 512, w_dim, True),
            SynthesisBlock(512, 512, w_dim),
       ])
        self.to_rgb = ToRGB(512)
    
    def grow(self):
        self.cur_res *= 2
        in_chan = self.blocks[-1].out_chan
        out_chan = in_chan // 2

        self.blocks.append(UP(in_chan, out_chan))

        self.blocks.append(SynthesisBlock(out_chan, out_chan, self.wdim))
        self.blocks.append(SynthesisBlock(out_chan, out_chan, self.wdim))

        self.to_rgb = ToRGB(out_chan)

    def forward(self, w):
        x = self.blocks[0].const.expand(w.size(0), -1, -1, -1)
        for block in self.blocks[1:]:
            if isinstance(block, UP):
                x = block(x)
            else:
                x = block(x, w)
        return self.to_rgb(x) 

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.mapping = Mapping(z_dim, c_dim, 8)
        self.synt = Stynthesis(z_dim)

    def forward(self, z, c):
        w = self.mapping(z, c)
        img = self.synt(w)
        return img, w

class MinibatchStdDev(nn.Module):
    def __init__(self, group_size=4, eps=1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x):
        batch_size, C, H, W = x.shape
        if batch_size % self.group_size != 0:
            group_size = batch_size
        else:
            group_size = self.group_size
        num_groups = batch_size // group_size
        y = x.view(group_size, num_groups, C, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.square().mean(dim=0) + self.eps)
        y = y.mean(dim=[1, 2, 3], keepdim=True)
        y = y.mean(dim=0, keepdim=True)
        y = y.repeat(batch_size, 1, H, W)
        return torch.cat([x, y], dim=1)

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.downsample(x)
        return x 

class Discriminator(nn.Module):
    def __init__(self, base_channels=64, c_dim = 10):
        super().__init__()
        self.blocks = nn.Sequential(
            #256
            DiscriminatorBlock(3, base_channels),
            #128
            DiscriminatorBlock(base_channels, base_channels*2),
            #64
            DiscriminatorBlock(base_channels*2, base_channels*4),
            #32
            DiscriminatorBlock(base_channels*4, base_channels*8),
            #16
            DiscriminatorBlock(base_channels*8, base_channels*16),
            #8
            DiscriminatorBlock(base_channels*16, base_channels*32),
            MinibatchStdDev(),
            #4
            DiscriminatorBlock(base_channels*32+1, base_channels*64),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(base_channels*64, 1)
        self.c_proj = nn.Linear(c_dim, base_channels*64)
    def forward(self, x, c):
        # x: [batch, 3, 256, 256]
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        c_embed = self.c_proj(c)
        projection = torch.sum(x*c_embed, dim=1, keepdim=True)
        out += projection
        return out

if __name__ == "__main__":
    G = Generator(512, 10)
    t = torch.randn((1, 512))
    c = torch.zeros((1, 10))
    D = Discriminator(c_dim=10)
    images, w = G(t, c)
    images = F.interpolate(images, scale_factor=64)
    print(D(images, c), c)