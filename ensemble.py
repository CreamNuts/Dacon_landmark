import os
import time
import timm
import random
import argparse
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from utils import *
from torch.utils.data import DataLoader, random_split
from efficientnet_pytorch import EfficientNet
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
from dataset import Dacon

'''
main에서 돌리지 말고
그냥 이 파일 자체롤 앙상블을 돌리는게 나을 듯.
'''

parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-m', default='val', choices=['train', 'val', 'test'], help='Train : Use Total Dataset, Val : Use Hold-Out, Test : Make Submission')
parser.add_argument('--ensemble', default=1)
parser.add_argument('--save', '-s', default='./Checkpoint.pt', help='Save Directory. if Checkpoint exists, Save Checkpoint in Checkpoint Dir')
parser.add_argument('--checkpoint', '-c', default=None, help='Checkpoint Directory')
parser.add_argument('--cutmix', default=True, help="If True, Use Cutmix Aug in Training")
parser.add_argument('--flooding', type=float, default=0.0)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--gpu', default='0')
args = parser.parse_args()

#########setting hyperparameters in here########
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('use:',device)
print('mode:',args.mode)
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

class Ensemble_0(nn.Module):
    '''
    Tensor와 Softmax 이용
    '''
    def __init__(self, model_A, model_B, model_C, model_D, NUM_CLASSES):
        super(Ensemble_0, self).__init__()
        self.model_A = model_A
        self.model_B = model_B
        self.model_C = model_C
        self.model_D = model_D

        #Create new classifier
        self.weight = nn.Parameter(torch.randn((4, 1049), requires_grad=True)) 

    def forward(self, x):
        output_1 = self.model_A(x)
        output_2 = self.model_B(x)
        output_3 = self.model_C(x)
        output_4 = self.model_D(x)

        output = torch.stack((output_1, output_2, output_3, output_4), dim=1)
        output = output * nn.Softmax(dim=0)(self.weight)
        output = torch.sum(output, dim=1)
        return output

class Ensemble_1(nn.Module):
    '''
    Tensor 이용
    '''
    def __init__(self, model_A, model_B, model_C, model_D, NUM_CLASSES):
        super(Ensemble_1, self).__init__()
        self.model_A = model_A
        self.model_B = model_B
        self.model_C = model_C
        #self.model_D = model_D
        #self.model_E = model_E

        #Create new classifier
        self.weight = nn.Parameter(torch.randn((4, 1049), requires_grad=True)) 
        #torch.randn(4*1049, requires_grad=True, device=device)

    def forward(self, x):
        output_1 = self.model_A(x)
        output_2 = self.model_B(x)
        output_3 = self.model_C(x)
        output_4 = self.model_D(x)

        output = torch.stack((output_1, output_2, output_3, output_4), dim=1)
        output = output * self.weight
        output = torch.sum(output, dim=1)
        return output

class Ensemble_2(nn.Module):
    '''
    이건 그냥 아웃풋 더하기
    '''
    #def __init__(self, model_A, model_B, model_C, model_D, model_E, NUM_CLASSES):
    def __init__(self, model_A, model_B, model_C, NUM_CLASSES):
        super(Ensemble_2, self).__init__()
        self.model_A = model_A
        self.model_B = model_B
        self.model_C = model_C
        #self.model_D = model_D
        #self.model_E = model_E

    def forward(self, x):
        output_0 = self.model_A(x)
        output_1 = self.model_B(x)
        output_2 = self.model_C(x)
       #output_3 = self.model_D(x)
       #output_4 = self.model_E(x)
        output = output_0 + output_1 + output_2 #+ output_3 +output_4

        return output

def ck_load(path, model):
    ckpoint = torch.load(path)
    model.load_state_dict(ckpoint['model_state_dict'])
    for param in model.parameters():
        param.requires_grad=False
    
    return model

def train(train_loader, model, criterion, optimizer):
    model.train()
    acc = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images.to(device))
        loss = criterion(output, labels.to(device))
        loss = (loss-args.flooding).abs()+args.flooding
        loss.backward()
        optimizer.step()
        if args.cutmix is True:
            acc += (output.softmax(dim=1)*labels.to(device)).sum(dim=1).float().mean().detach().cpu()
        else:
            acc += (output.argmax(1)==labels.to(device)).float().mean()
        pbar.set_description("Loss : %.3f" % loss)
    return acc, loss

def validation(valid_loader, model, criterion):
    model.eval()
    acc = 0
    for images, labels in valid_loader:
        with torch.no_grad():
            output = model(images.to(device))
            loss = criterion(output, labels.to(device))
            acc += (output.argmax(1)==labels.to(device)).float().mean()
    return acc, loss

def submission(test_loader, model):
    model.eval()
    landmark_id = []
    conf = []
    for images in tqdm(testloader):
        with torch.no_grad():
            pred = model(images.to(device))
            pred = nn.Softmax(dim=1)(pred)
            pred = pred.detach().cpu()
            landmark_id.append(torch.argmax(pred, dim=1))
            conf.append(torch.max(pred, dim=1)[0])
    submission = test_loader.dataset.submission
    submission.landmark_id = torch.cat(landmark_id).numpy()
    submission.conf = torch.cat(conf).numpy()
    submission.to_csv(os.path.join(os.getcwd(), 'ensemble.csv'), index=False)

model_0 = timm.create_model('skresnext50_32x4d', pretrained=False, num_classes=NUM_CLASSES)
model_1 = EfficientNet.from_name("efficientnet-b1", num_classes=NUM_CLASSES)        
model_2 = EfficientNet.from_name("efficientnet-b5", num_classes=NUM_CLASSES) 
model_3 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)     
#model_4 = EfficientNet.from_name("efficientnet-b0", num_classes=NUM_CLASSES)

if args.ensemble == '0':
    model = Ensemble_0(model_0, model_1, model_2, model_3, NUM_CLASSES)
elif args.ensemble == '1':
    model = Ensemble_1(model_0, model_1, model_2, model_3, NUM_CLASSES)
else:
    #model = Ensemble_2(model_0, model_1, model_2, model_3, model_4, NUM_CLASSES)
    model = Ensemble_2(model_2, model_3, model_4, NUM_CLASSES)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3)



if args.checkpoint is None:   
    path0 = os.getcwd()+'/resnext.pt'
    path1 = os.getcwd()+'/b1_cos.pt'
    path2 = os.getcwd()+'/b5.pt'
    path3 = os.getcwd()+'/vit.pt'
    path4 = os.getcwd()+'/b0.pt'

    print('load checkpoint')
    model_0 = ck_load(path0, model_0)
    model_1 = ck_load(path1, model_1)
    model_2 = ck_load(path2, model_2)
    model_3 = ck_load(path3, model_3)
    #model_4 = ck_load(path4, model_4)

    check_epoch = 0
    train_acc_list = []
    valid_acc_list = []
    train_loss_list = []
    valid_loss_list = []
else:
    checkpoint = torch.load(args.checkpoint)
    check_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    train_acc_list = checkpoint['train_accuracy']
    valid_acc_list = checkpoint['val_accuracy']
    train_loss_list = checkpoint['train_loss']
    valid_loss_list = checkpoint['val_loss']


model.to(device)

if args.cutmix is True:
    criterion = CutMixCrossEntropyLoss(True)
else:
    criterion = nn.CrossEntropyLoss()



transforms_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_PARAMETERS['mean'], TRAIN_PARAMETERS['std'])
        ])

if args.mode == 'train':
    trainset = Dacon(dir=DIR, mode=args.mode, transform=transforms_train)
    if args.cutmix is True:
        trainset = CutMix(trainset, num_class=NUM_CLASSES, num_mix=2, prob=0.5, beta=1)
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=NUM_WORKERS)
    with trange(args.epoch, initial=check_epoch, desc='Loss : 0', leave=True) as pbar:
        for epoch in pbar:
            train_acc, train_loss = train(trainloader, model, criterion, optimizer)
            lr_scheduler.step()
            train_acc_list.append(train_acc/len(trainloader))
            train_loss_list.append(train_loss.detach().cpu().numpy())
            save(model, epoch, check_epoch, optimizer, lr_scheduler, train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, args)
    if args.checkpoint is None:
        visualize(args.save, args)
    else:
        visualize(args.checkpoint, args)

elif args.mode == 'val':
    dacon = Dacon(dir=DIR, mode=args.mode, transform=transforms_train)
    num_train = int(len(dacon) * 0.8)
    num_valid = len(dacon) - num_train
    trainset, validset = random_split(dacon, [num_train, num_valid])
    if args.cutmix is True:
        trainset = CutMix(trainset, num_class=NUM_CLASSES, num_mix=2, prob=0.5, beta=1)
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=NUM_WORKERS)
    validloader = DataLoader(validset, batch_size=args.batchsize, shuffle=True, num_workers=NUM_WORKERS)
    with trange(args.epoch, initial=check_epoch, desc='Loss : 0', leave=True) as pbar:
        for epoch in pbar:
            train_acc, train_loss = train(trainloader, model, criterion, optimizer)
            lr_scheduler.step()
            train_acc_list.append(train_acc/len(trainloader))
            train_loss_list.append(train_loss.detach().cpu().numpy())

            valid_acc, valid_loss = validation(validloader, model, criterion)
            valid_acc_list.append(valid_acc/len(validloader))
            valid_loss_list.append(valid_loss.detach().cpu().numpy())
            if valid_loss_list[-1].item() == min(valid_loss_list).item():
                save(model, epoch, check_epoch, optimizer, lr_scheduler, train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, args)
    if args.checkpoint is None:
        visualize(args.save, args)
    else:
        visualize(args.checkpoint, args)

else:
    testset = Dacon(dir=DIR, mode='test', transform=transforms_train)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)
    submission(testloader, model)


