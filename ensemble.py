import os
import time
import timm
import random
import argparse
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from efficientnet_pytorch import EfficientNet
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
from dataset import Dacon

'''
main에서 돌리지 말고
그냥 이 파일 자체롤 앙상블을 돌리는게 나을 듯.
'''

#########setting hyperparameters in here########
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('use: ',device)

torch.manual_seed(777)
torch.cuda.manual_seed(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(777)
random.seed(777)

DIR = os.path.join(os.getcwd(),'..', 'data') + '/'
LEARNING_RATE = 1e-3
LR_STEP = 3
LR_FACTOR = 0.5
NUM_WORKERS = 4
NUM_CLASSES = 1049

TRAIN_PARAMETERS = {
    'mean':[0.4461, 0.4447, 0.4490],
    'std':[0.2598, 0.2591, 0.2590]
}

################################################

class Ensemble_1(nn.Module):
    '''
    이건 cat으로 병합
    '''
    def __init__(self, model_A, model_B, model_C, model_D, NUM_CLASSES):
        super(Ensemble_1, self).__init__()
        self.model_A = model_A
        self.model_B = model_B
        self.model_C = model_C
        self.model_D = model_D
        #self.model_C = model_C

        #Remove Last Linear Layer
        self.model_A.fc = nn.Identity()
        self.model_B.fc = nn.Identity()
        self.model_C.fc = nn.Identity()
        self.model_D.fc = nn.Identity()

        #Create new classifier
        self.classifier = nn.Linear(5195, NUM_CLASSES)

    def forward(self, x):
        output_1 = self.model_A(x)
        output_1 = output_1.view(output_1.size(0), -1)
        output_2 = self.model_B(x)
        output_2 = output_2.view(output_2.size(0), -1)
        output_3 = self.model_C(x)
        output_3 = output_3.view(output_3.size(0), -1)
        output_4 = self.model_D(x)
        output_4 = output_4.view(output_4.size(0), -1)

        output = torch.cat((output_1, output_2, output_3, output_4), dim=1)
        #import pdb;pdb.set_trace()
        return self.classifier(output)

class Ensemble_2(nn.Module):
    '''
    이건 그냥 아웃풋 더하기
    '''
    def __init__(self, model_A, model_B, model_C, model_D, NUM_CLASSES):
        super(Ensemble_2, self).__init__()
        self.model_A = model_A
        self.model_B = model_B
        self.model_C = model_C
        self.model_D = model_D

    def forward(self, x):
        output_1 = self.model_A(x)
        output_2 = self.model_B(x)
        output_3 = self.model_C(x)
        output_4 = self.model_D(x)
        output = output_1 + output_2 + output_3 +output_4

        return output

def ck_load(path, model):
    ckpoint = torch.load(path)
    model.load_state_dict(ckpoint['model_state_dict'])
    for param in model.parameters():
        param.requires_grad=False
    
    return model

def submission(test_loader, model):
    model.eval()
    landmark_id = []
    conf = []
    for images in tqdm(testloader):
        pred = model(images.to(device))
        pred = nn.Softmax(dim=1)(pred)
        pred = pred.detach().cpu()
        landmark_id.append(torch.argmax(pred, dim=1))
        conf.append(torch.max(pred, dim=1)[0])
    submission = test_loader.dataset.submission
    submission.landmark_id = torch.cat(landmark_id).numpy()
    submission.conf = torch.cat(conf).numpy()
    submission.to_csv(os.path.join(os.getcwd(), 'ensemble.csv'), index=False)


path1 = os.getcwd()+'/b1_cos.pt'
path2 = os.getcwd()+'/b5.pt'
path3 = os.getcwd()+'/vit_base_val.pt'
path4 = os.getcwd()+'/resnext.pt'

model_1 = EfficientNet.from_name("efficientnet-b1", num_classes=NUM_CLASSES)        
model_2 = EfficientNet.from_name("efficientnet-b5", num_classes=NUM_CLASSES) 
model_3 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)     
model_4 = timm.create_model('skresnext50_32x4d', pretrained=False, num_classes=NUM_CLASSES)

print('load checkpoint')
t1 = time.time()
model_1 = ck_load(path1, model_1)
model_2 = ck_load(path2, model_2)
model_3 = ck_load(path3, model_3)
model_4 = ck_load(path4, model_4)
print(f'time:{time.time()-t1:.3f}')

ensemble = Ensemble_2(model_1, model_2, model_3, model_4, NUM_CLASSES)
ensemble.to(device)

transforms_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_PARAMETERS['mean'], TRAIN_PARAMETERS['std'])
        ])

testset = Dacon(dir=DIR, mode='test', transform=transforms_train)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)
submission(testloader, ensemble)



