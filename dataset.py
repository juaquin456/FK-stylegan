import PIL.Image
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
import PIL
import os
import json

class PokeData(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        PIL.Image.init()
        self.path = path
        self.img_names = [os.path.relpath(os.path.join(root, fname), start=path) for root, _, files in os.walk(path) for fname in files if os.path.splitext(fname)[1] in PIL.Image.EXTENSION]
        self.labels = json.load(open(os.path.join(path, "dataset.json")))
        self.transform = transform
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.path, img_name)
        img =  decode_image(img_path)/255 * 2 - 1
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels["labels"][img_name])
        return img, label
