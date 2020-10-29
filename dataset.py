import os
import random
import multiprocessing
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class Dacon(Dataset):
    def __init__(self, dir, mode, transform=None):
        print(f'creating data loader - {mode}')
        assert mode in ['train', 'test']
        self.mode = mode
        self.image_paths = sorted(list(Path(dir+mode).rglob('*.JPG')))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        label = int(image_path.parents[0].name)
        return image, label