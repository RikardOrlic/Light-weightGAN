import os
import numpy as np
import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image

def InfiniteSampler(n):
    """Data sampler"""
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(data.sampler.Sampler):
    """Data sampler wrapper"""
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31
    

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImageDataset, self).__init__()
        self.root_dir = root
        self.transform = transform
   
        self.frame = self._parse_frame()

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root_dir)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root_dir, img_names[i])
            if image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg': 
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
            
        if self.transform:
            img = self.transform(img) 

        return img

