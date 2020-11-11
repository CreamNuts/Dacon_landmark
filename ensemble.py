import timm
import random
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss

'''
main에서 돌리지 말고
그냥 이 파일 자체롤 앙상블을 돌리는게 나을 듯.
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('use: ',device)

torch.manual_seed(777)
torch.cuda.manual_seed(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(777)
random.seed(777)

LEARNING_RATE = 1e-3
LR_STEP = 3
LR_FACTOR = 0.5
NUM_CLASSES = 1049


class Ensemble_1(nn.Module):
    '''
    이건 cat으로 병합
    '''
    def __init__(self, model_A, model_B, NUM_CLASSES):
        super(Ensemble_1, self).__init__()
        self.model_A = model_A
        self.model_B = model_B
        #self.model_C = model_C

        #Remove Last Linear Layer
        self.model_A.fc = nn.Identity()
        self.model_B.fc = nn.Identity()
        #self.model_C.fc = nn.Identity()

        #Create new classifier
        self.classifier = nn.Linear(2048, NUM_CLASSES)

    def forward(self, x):
        x1 = self.model_A(x.clone())
        x1 = x1.view(x1.size(0), -1)
        x2 = self.model_B(x.clone())
        x2 = x2.view(x2.size(0), -1)
        output = torch.cat((x1, x2), dim=1)

        return self.classifier(F.relu(output))

class Ensemble_2(nn.Module):
    '''
    이건 그냥 아웃풋 더하기
    '''
    def __init__(self, model_A, model_B, NUM_CLASSES):
        super(Ensemble_2, self).__init__()
        self.model_A = model_A
        self.model_B = model_B

        self.classifier = nn.Linear(2048, NUM_CLASSES)
    def forward(self, x):
        output_1 = self.model_A(x)
        output_2 = self.model_B(x)
        output = output_1 + output_2 

        return self.classifier(F.relu(output))


model_1 = EfficientNet.from_pretrained("efficientnet-b0", num_classes=NUM_CLASSES)        
model_2 = EfficientNet.from_pretrained("efficientnet-b0", num_classes=NUM_CLASSES)        

optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=LEARNING_RATE)
optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=LEARNING_RATE)

lr_scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_1, T_0=3)
lr_scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=LR_STEP, gamma=LR_FACTOR)

path1 = './b1_cos.pt'
path2 = './b1_steplr.pt'

check_epoch = 0
train_acc_list = []
valid_acc_list = []
train_loss_list = []
valid_loss_list = []


def inference(path, model, optimizer, lr_scheduler):
    checkpoint = torch.load(path)
    check_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    train_acc_list = checkpoint['train_accuracy']
    valid_acc_list = checkpoint['val_accuracy']
    train_loss_list = checkpoint['train_loss']
    valid_loss_list = checkpoint['val_loss']

    model.to(device)

