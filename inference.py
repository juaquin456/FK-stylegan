from net import Generator
import torch
import cv2

G = Generator(512, 18)

z = torch.randn((1, 512))
c = torch.zeros(18)
c[0] = 1
#G.load_state_dict(torch.load("100.pth", weights_only=True, map_location="cpu"))
G.eval()

img, _ = G(z, c)
print(img)
img = ((img + 1) / 2)*255
img = img.squeeze().permute(1, 2, 0).detach().numpy()
cv2.imwrite("x.png", img)