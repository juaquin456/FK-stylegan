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
            layers.extend([nn.Linear(z_dim, z_dim), nn.LeakyReLU(0.2, inplace=True)])
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

class Stynthesis(nn.Module):
    def __init__(self, w_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            #4
            SynthesisBlock(512, 512, w_dim, True),
            SynthesisBlock(512, 256, w_dim),
            
            #8
            nn.Upsample(scale_factor=2),
            SynthesisBlock(256, 256, w_dim),
            SynthesisBlock(256, 128, w_dim),

            #16
            nn.Upsample(scale_factor=2),
            SynthesisBlock(128, 128, w_dim),
            SynthesisBlock(128, 64, w_dim),

            #32
            nn.Upsample(scale_factor=2),
            SynthesisBlock(64, 64, w_dim),
            SynthesisBlock(64, 32, w_dim),

            #64
            nn.Upsample(scale_factor=2),
            SynthesisBlock(32, 32, w_dim),
            SynthesisBlock(32, 16, w_dim),

            #128
            nn.Upsample(scale_factor=2),
            SynthesisBlock(16, 16, w_dim),
            SynthesisBlock(16, 8, w_dim),

            #256
            nn.Upsample(scale_factor=2),
            SynthesisBlock(8, 8, w_dim),
            SynthesisBlock(8, 4, w_dim),
        ])
    def forward(self, w):
        x = self.blocks[0].const.expand(w.size(0), -1, -1, -1)
        for block in self.blocks[1:]:
            if isinstance(block, nn.Upsample):
                x = block(x)
            else:
                x = block(x, w)
        return x

class ToRGB(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, 3, kernel_size=1)
    def forward(self, logits):
        logits = self.conv(logits)
        return torch.tanh(logits)

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.mapping = Mapping(z_dim, c_dim, 8)
        self.synt = Stynthesis(z_dim)
        self.to_rgb = ToRGB(4)    

    def forward(self, z, c):
        w = self.mapping(z, c)
        img = self.synt(w)
        return self.to_rgb(img)

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


        y = x.view(group_size, -1, C, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.square().mean(dim=0) + self.eps)
        y = y.mean(dim=[1, 2, 3], keepdim=True)
        y = y.mean(dim=0, keepdim=True)
        y = y.repeat(batch_size, 1, H, W)
        return torch.cat([x, y], dim=1)

class Discriminator(nn.Module):
    def __init__(self, base_channels=64, c_dim = 10):
        super().__init__()
        # 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 64x64
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 32x32
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 16x16
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 8x8
        self.conv5 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(base_channels * 16, base_channels * 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 4x4
        self.conv6 = nn.Sequential(
            nn.Conv2d(base_channels * 16, base_channels * 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down6 = nn.Sequential(
            nn.Conv2d(base_channels * 32, base_channels * 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.stddev = MinibatchStdDev()
        self.conv7 = nn.Sequential(
            nn.Conv2d(base_channels * 32 + 1, base_channels * 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 64, base_channels * 64, kernel_size=4, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Linear(base_channels * 64, 1)
        self.cond_proj = nn.Linear(c_dim, base_channels*64)
    def forward(self, x, c):
        # x: [batch, 3, 256, 256]
        x = self.conv1(x)
        x = self.down1(x)    # 128x128
        x = self.conv2(x)
        x = self.down2(x)    # 64x64
        x = self.conv3(x)
        x = self.down3(x)    # 32x32
        x = self.conv4(x)
        x = self.down4(x)    # 16x16
        x = self.conv5(x)
        x = self.down5(x)    # 8x8
        x = self.conv6(x)
        x = self.down6(x)    # 4x4
        x = self.stddev(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        c_embed = self.cond_proj(c)
        projection = torch.sum(x*c_embed, dim=1, keepdim=True)
        out += projection
        return out

if __name__ == "__main__":
    G = Generator(512, 10)
    t = torch.randn((1, 512))
    c = torch.zeros((1, 10))
    D = Discriminator(c_dim=10)
    print(D(G(t, c), c))