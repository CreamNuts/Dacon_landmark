import os
import random
import multiprocessing
from tqdm import tqdm, trange
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from dataset import Dacon


#########setting hyperparameters in here########
DIR = os.getcwd()+'/data/'
pbar = trange(100, desc='Loss : 0', leave=True, position=0)
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LR_STEP = 3
LR_FACTOR = 0.5
NUM_CLASSES = 1049
NUM_WORKERS = multiprocessing.cpu_count() #24
torch.manual_seed(777)
torch.cuda.manual_seed(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(777)
random.seed(777)
PARAMETERS = {
    'mean':[0.4451, 0.4457, 0.4464],
    'std':[0.2679, 0.2682, 0.2686]
}
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('use: ',device)
################################################
train_acc_list = []
test_acc_list = []
train_loss_list = []
test_loss_list = []

calculation = transforms.Compose([
    transforms.ToTensor()
])

def get_parameters(dataset):
    '''
    It takes about 5 minutes
    '''
    parameters ={
        'mean': [0.0, 0.0, 0.0],
        'std': [0.0, 0.0, 0.0]
    }
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, num_workers=NUM_WORKERS)
    count = 0
    for images, _ in tqdm(dataloader):
        for i in range(3):
            var = images[:,:,:,i].view(-1)
            parameters['mean'][i] += var.mean()
            parameters['std'][i] += var.std()
        count += 1

    for i in range(3):
        parameters['mean'][i] /= count
        parameters['std'][i] /= count

    print('Calculation Completed')
    print(parameters)
    return parameters

def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler):
    print(f'epoch: {epoch}')
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images.to(device))
        loss = criterion(output, labels.to(device))
        loss.barckward()
        optimizer.step()
        train_acc += (output.argmax(1)==labels).float().mean()
        pbar.set_description("Loss : %.3f" % loss)
    return loss
def inference():
    pass

def generate_submission():
    pass

if __name__ == '__main__':
    #cal_dataset = Dacon(dir=DIR, mode='train', transform=calculation)
    #parameters = get_parameters(cal_dataset)
    transforms_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*PARAMETERS)
    ])
    dacon = Dacon(dir=DIR, mode='train', transform=transforms_train)
    trainloader = DataLoader(dacon, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    model = torchvision.models.resnet50(pretrained=True)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_FACTOR)

    for epoch in pbar:
        train_acc = 0
        test_acc = 0
        train_loss = train(trainloader, model. criterion, optimizer, epoch, lr_scheduler)
        lr_scheduler.step()
        train_acc_list.append(train_acc/len(trainloader))
        train_loss_list.append(train_loss.detach.cpu().numpy())

        test_loss = inference()
