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



def get_parameters(path: str):
    parameter ={
        'mean': [0.0, 0.0, 0.0],
        'std': [0.0, 0.0, 0.0]
    }
    return parameter

class Dacon(Dataset):
    def __init__(self, mode: str):
        print(f'creating data loader - {mode}')
        assert mode in ['train', 'test']
        self.mode = mode
        self.image_paths = sorted(list(Path(os.getcwd()+f'/../data/{mode}').rglob('*.JPG')))
        transforms_list = []

        if self.mode == 'train':
            self.csv = pd.read_csv(os.getcwd()+f'/../data/{mode}.csv', header=0)
            transforms_list = [
                '''
                augmentation in here
                '''
            ]
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[,,],
                                 std=[,,]),
        ])

        self.transforms = transforms.Compose(transforms_list)
        assert len(self.image_paths) == len(self.csv.index)
        self.length = len(self.image_paths)
        
    
    def __len__(self):
        return len(self.length)
    
    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        label = int(image_path.parents[0].name)
        return image, label