import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms

from dataset import Dacon

'''
setting hyperparameters in here

BATCH_SIZE = 16
NUM_WORKERS = multiprocessing.cpu_count()
torch.manual_seed(777)
torch.cuda.manual_seed(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(777)
random.seed(777)
'''

if __name__ == '__main__':
    dacon = Dacon()