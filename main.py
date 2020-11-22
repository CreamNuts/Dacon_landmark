import os
import json
import random
import argparse
import multiprocessing
import numpy as np
import torch
import torch.nn as nn 
import timm
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from pathlib import Path
from utils import *
from dataset import Dacon
from efficientnet_pytorch import EfficientNet
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss

#########setting hyperparameters in here########
parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-m', default='val', choices=['train', 'val', 'test'], help='Train : Use Total Dataset, Val : Use Hold-Out, Test : Make Submission')
parser.add_argument('--calculator', default='False', help='Cacluate Dataset Mean and Std')
parser.add_argument('--checkpoint', '-c', default=None, help='Checkpoint Directory')
parser.add_argument('--save', '-s', default='./Checkpoint.pt', help='Save Directory. if Checkpoint exists, Save Checkpoint in Checkpoint Dir')
parser.add_argument('--model', default='b1', choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'l2', 'vit_base', 'vit_base_hybrid', 'vit_large', 'skresnext50'])
parser.add_argument('--gpu', default='0')
parser.add_argument('--cutmix', default=True, help="If True, Use Cutmix Aug in Training")
parser.add_argument('--scheduler', default='StepLR', choices=['StepLR', 'Cos'])
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--flooding', type=float, default=0.01)
args = parser.parse_args()

DIR = os.path.join(os.getcwd(),'..', 'data') + '/'
LR_STEP = 3
LR_FACTOR = 0.5
NUM_CLASSES = 1049
NUM_WORKERS = 4 #multiprocessing.cpu_count() #24
FLOODING_LEVEL = 0.01
TRAINING_EPOCH = 10

torch.manual_seed(777)
torch.cuda.manual_seed(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(777)
random.seed(777)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('use: ',device)
'''
parameter 구하는거 매번 하기 귀찮으니까 
일단 밑에거 긁어서 바로 넣으셈
이미지 224X224로 새로 계산
'''
TRAIN_PARAMETERS = {
    'mean':[0.4461, 0.4447, 0.4490],
    'std':[0.2598, 0.2591, 0.2590]
}

TEST_PARAMETERS = {
    'mean':[0.4456, 0.4462, 0.4468],
    'std':[0.2757, 0.2769, 0.2763]
}

################################################
calculation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

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
    submission.to_csv(os.path.join(os.getcwd(), 'submission.csv'), index=False)

if __name__ == '__main__':
    if args.calculator is True:
        cal_dataset = Dacon(dir=DIR, mode=args.mode, transform=calculation)
        PARAMETERS = get_parameters(cal_dataset, args)
    
        transforms_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(PARAMETERS['mean'], PARAMETERS['std'])
        ])

    else:
        transforms_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_PARAMETERS['mean'], TRAIN_PARAMETERS['std'])
        ])
    '''
    #TensorBoard code
    
    writer = SummaryWriter('runs/dacon')
    dacon = Dacon(dir=lab_dir, mode=args.mode, transform=transforms_train)
    dataloader = DataLoader(dacon, batch_size=args.batchsize, shuffle=True, num_workers=NUM_WORKERS)
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    img_grid = torchvision.utils.make_grid(images)
    visualize(img_grid)
    writer.add_image('dacon_image', img_grid)
    '''
    if args.model == 'vit_base':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_CLASSES)
    elif args.model == 'vit_base_hybrid':
        model = timm.create_model('vit_base_resnet26d_224', pretrained=True, num_classes=NUM_CLASSES)
    elif args.model == 'vit_large':
        model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=NUM_CLASSES)
    elif args.model == 'skresnext50':
        model = timm.create_model('skresnext50_32x4d', pretrained=True, num_classes=NUM_CLASSES)
    else:
        model = EfficientNet.from_pretrained(f"efficientnet-{args.model}", num_classes=NUM_CLASSES)        
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_FACTOR)
    elif args.scheduler == 'Cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3)
    
    if args.checkpoint is None:   
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
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_acc_list = checkpoint['train_accuracy']
        valid_acc_list = checkpoint['val_accuracy']
        train_loss_list = checkpoint['train_loss']
        valid_loss_list = checkpoint['val_loss']
        
    model.to(device)

    if args.cutmix is True:
        criterion = CutMixCrossEntropyLoss(True)
    else:
        criterion = nn.CrossEntropyLoss()
    
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

    elif args.mode == 'test':
        testset = Dacon(dir=DIR, mode=args.mode, transform=transforms_train)
        testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=NUM_WORKERS)
        submission(testloader, model)
