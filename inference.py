from net import Generator
import torch
import cv2

G = Generator(512, 18)

z = torch.randn((1, 512))
print(z)
c = torch.zeros(18)
c[0] = 1
G.load_state_dict(torch.load("1600.pth", weights_only=True, map_location="cpu"))
G.eval()

img = G(z, c)
print(img.min(), img.max())
img = ((img + 1) / 2)*255
img = img.squeeze().permute(1, 2, 0).detach().numpy()
cv2.imwrite("x.png", img)