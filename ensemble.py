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
parser.add_argument('--cutmix', default=True, help="If True, Use Cutmix Aug in Training")
parser.add_argument('--flooding', type=float, default=0.001)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=10)
args = parser.parse_args()

#########setting hyperparameters in here########
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

        #Create new classifier
        self.classifier = nn.Linear(4196, NUM_CLASSES)

    def forward(self, x):
        output_1 = self.model_A(x)
        output_2 = self.model_B(x)
        output_3 = self.model_C(x)
        output_4 = self.model_D(x)

        output = torch.cat((output_1, output_2, output_3, output_4), dim=1)
        output.requires_grad = True
        output = self.classifier(output)
        return output

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
        pred = model(images.to(device))
        pred = nn.Softmax(dim=1)(pred)
        pred = pred.detach().cpu()
        landmark_id.append(torch.argmax(pred, dim=1))
        conf.append(torch.max(pred, dim=1)[0])
    submission = test_loader.dataset.submission
    submission.landmark_id = torch.cat(landmark_id).numpy()
    submission.conf = torch.cat(conf).numpy()
    submission.to_csv(os.path.join(os.getcwd(), 'ensemble.csv'), index=False)

def save(model, epoch, check_epoch, optimizer, train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, args):
    if args.checkpoint is None:
        torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'train_loss' : train_loss_list,
                'val_loss': valid_loss_list, 
                'train_accuracy': train_acc_list,
                'val_accuracy' : valid_acc_list,
                #'learning_rate' : get_learing_rate(optimizer),
                'batch_size' : args.batchsize,
                'flooding_level' : args.flooding
            }, args.save)
    else:            
        epoch += check_epoch
        torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'train_loss' : train_loss_list,
                'val_loss': valid_loss_list, 
                'train_accuracy': train_acc_list,
                'val_accuracy' : valid_acc_list,
                #'learning_rate' : get_learing_rate(optimizer),
                'batch_size' : args.batchsize,
                'flooding_level' : args.flooding
            }, args.checkpoint)

def visualize(checkpoint_dir, args):
    checkpoint = torch.load(checkpoint_dir)
    train_acc_list = checkpoint['train_accuracy']
    train_loss_list = checkpoint['train_loss']
    val_acc_list = checkpoint['val_accuracy']
    val_loss_list = checkpoint['val_loss']
    learning_rate = checkpoint['learning_rate']
    batch_size = checkpoint['batch_size']
    if args.mode == 'train':
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].set_title("Training Accuracy")
        ax[0].set_ylabel("Accuracy")
        ax[0].set_xlabel("Epoch")
        ax[0].plot(range(1, len(train_acc_list)+1), train_acc_list)
        ax[0].legend(['Train'])
        #ax[0].set_xlim(left=15)

        ax[1].set_title("Training Loss")
        ax[1].set_ylabel("Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].plot(range(1, len(train_loss_list)+1), train_loss_list)
        ax[1].legend(['Train'])
        #ax[1].set_xlim(left=15)
        #ax[1].set_ylim(top=1.5)
        print(f"LR : {learning_rate}, Batch Size : {batch_size}, 현재 Epoch : {checkpoint['epoch']}")
        print(f"Train 정확도 : {train_acc_list[-1]}, Train Loss : {train_loss_list[-1]}")
        plt.savefig(f'{args.model}_{args.scheduler}_train_fig.png')

    elif args.mode == 'val':
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].set_title("Training/Valid Accuracy")
        ax[0].set_ylabel("Accuracy")
        ax[0].set_xlabel("Epoch")
        ax[0].plot(range(1, len(train_acc_list)+1), train_acc_list)
        ax[0].plot(range(1, len(val_acc_list)+1), val_acc_list)
        ax[0].legend(['Train', 'Valid'])
        #ax[0].set_xlim(left=15)

        ax[1].set_title("Training/Valid Loss")
        ax[1].set_ylabel("Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].plot(range(1, len(train_loss_list)+1), train_loss_list)
        ax[1].plot(range(1, len(val_loss_list)+1), val_loss_list)
        ax[1].legend(['Train', 'Valid'])
        #ax[1].set_xlim(left=15)
        #ax[1].set_ylim(top=1.5)
        
        print(f"LR : {learning_rate}, Batch Size : {batch_size}, 현재 Epoch : {checkpoint['epoch']}")
        print(f"Train 정확도 : {train_acc_list[-1]}, Train Loss : {train_loss_list[-1]}")
        print(f"Val 정확도 : {val_acc_list[-1]}, Val Loss : {val_loss_list[-1]}")
        print(f"가장 높은 Val 정확도 : {max(val_acc_list)}")
        plt.savefig(f'{args.model}_{args.scheduler}_val_fig.png')


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

if args.ensemble == '1':
    model = Ensemble_1(model_1, model_2, model_3, model_4, NUM_CLASSES)
else:
    model = Ensemble_2(model_1, model_2, model_3, model_4, NUM_CLASSES)

model.to(device)

if args.cutmix is True:
    criterion = CutMixCrossEntropyLoss(True)
else:
    criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

check_epoch = 0
train_acc_list = []
valid_acc_list = []
train_loss_list = []
valid_loss_list = []

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
    with trange(10, initial=check_epoch, desc='Loss : 0', leave=True) as pbar:
        for epoch in pbar:
            train_acc, train_loss = train(trainloader, model, criterion, optimizer)
            #lr_scheduler.step()
            train_acc_list.append(train_acc/len(trainloader))
            train_loss_list.append(train_loss.detach().cpu().numpy())
            save(model, epoch, check_epoch, optimizer, train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, args)
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
    with trange(10, initial=check_epoch, desc='Loss : 0', leave=True) as pbar:
        for epoch in pbar:
            train_acc, train_loss = train(trainloader, model, criterion, optimizer)
            #lr_scheduler.step()
            train_acc_list.append(train_acc/len(trainloader))
            train_loss_list.append(train_loss.detach().cpu().numpy())

            valid_acc, valid_loss = validation(validloader, model, criterion)
            valid_acc_list.append(valid_acc/len(validloader))
            valid_loss_list.append(valid_loss.detach().cpu().numpy())
            if valid_loss_list[-1].item() == min(valid_loss_list).item():
                save(model, epoch, check_epoch, optimizer, train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, args)
    if args.checkpoint is None:
        visualize(args.save, args)
    else:
        visualize(args.checkpoint, args)

else:
    testset = Dacon(dir=DIR, mode='test', transform=transforms_train)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)
    submission(testloader, model)


