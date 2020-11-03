import os
import random
import argparse
import multiprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from dataset import Dacon
from model import ResNet18, ResNet34
from efficientnet_pytorch import EfficientNet


#########setting hyperparameters in here########
parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-m', default='train', choices=['train', 'test'])
parser.add_argument('--calculator', default='False')
parser.add_argument('--checkpoint', '-c', default=None, help='Checkpoint Directory')
parser.add_argument('--gpu', default='0')
args = parser.parse_args()

DIR = os.path.join(os.getcwd(),'..', 'data') + '/'
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
LR_STEP = 3
LR_FACTOR = 0.5
NUM_CLASSES = 1049
NUM_WORKERS = multiprocessing.cpu_count() #24
FLOODING_LEVEL = 0.01
TRAINING_EPOCH = 50

torch.manual_seed(777)
torch.cuda.manual_seed(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(777)
random.seed(777)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('use: ',device)

PARAMETERS = {
    'mean':[0, 0, 0],
    'std': [0, 0, 0]
}
'''
parameter 구하는거 매번 하기 귀찮으니까
일단 밑에거 긁어서 바로 넣으셈
'''
TRAIN_PARAMETERS = {
    'mean':[0.4451, 0.4457, 0.4464],
    'std':[0.2679, 0.2682, 0.2686]
}
TEST_PARAMETERS = {
    'mean':[0.4456, 0.4462, 0.4468],
    'std':[0.2757, 0.2769, 0.2763]
}

################################################
calculation = transforms.Compose([
    transforms.ToTensor()
])

def get_parameters(dataset):
    '''
    It takes about 5 minutes
    '''

    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, num_workers=NUM_WORKERS)
    count = 0
    if args.mode == 'train':
        for images, _ in tqdm(dataloader):
            for i in range(3):
                var = images[:,:,:,i].view(-1)
                PARAMETERS['mean'][i] += var.mean()
                PARAMETERS['std'][i] += var.std()
            count += 1
    else:
        print('In Test, Use Train mean and std')
        return None

    for i in range(3):
        PARAMETERS['mean'][i] /= count
        PARAMETERS['std'][i] /= count

    print('Calculation Completed')
    print(PARAMETERS)
    return PARAMETERS

def train(train_loader, model, criterion, optimizer, lr_scheduler):
    model.train()
    acc = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images.to(device))
        loss = criterion(output, labels.to(device))
        loss = (loss-FLOODING_LEVEL).abs()+FLOODING_LEVEL
        loss.backward()
        optimizer.step()
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
    submission.to_csv(os.path.join(os.getcwd(), 'submission.csv'), index=False)


def save(model, epoch, check_epoch, optimizer, train_loss_list, valid_loss_list, train_acc_list, valid_acc_list):
    if args.checkpoint is not None:
        epoch += check_epoch

    torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss' : train_loss_list,
            'val_loss': valid_loss_list, 
            'train_accuracy': train_acc_list,
            'val_accuracy' : valid_acc_list
        }, './Lab0.01_checkpoint.pt')

def visualize():
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].set_title("Training/Valid Accuracy")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].plot(range(1, len(train_acc_list)+1), train_acc_list)
    ax[0].plot(range(1, len(valid_acc_list)+1), valid_acc_list)
    ax[0].legend(['Train', 'Valid'])
    ax[0].set_xlim(left=15)
    ax[0].set_ylim(bottom=0.6)

    ax[1].set_title("Training/Valid Loss")
    ax[1].set_ylabel("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].plot(range(1, len(train_loss_list)+1), train_loss_list)
    ax[1].plot(range(1, len(valid_loss_list)+1), valid_loss_list)
    ax[1].legend(['Train', 'Valid'])
    ax[1].set_xlim(left=15)
    ax[1].set_ylim(top=1.5)
    print("Train 정확도 : %f, Train Loss : %f\n Valid 정확도 : %f, Valid Loss : %f " %(train_acc_list[-1], train_loss_list[-1], valid_acc_list[-1], valid_loss_list[-1]))
    print("가장 높은 Valid 정확도 : %f" %max(valid_acc_list))
    plt.show()

if __name__ == '__main__':
    #cal_dataset = Dacon(dir=lab_dir, mode=args.mode, transform=calculation)
    #PARAMETERS = get_parameters(cal_dataset)
    '''
    transforms_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(PARAMETERS['mean'], PARAMETERS['std'])
    ])
    '''
    transforms_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.4451, 0.4457, 0.4464], [0.2679, 0.2682, 0.2686])
    ])
    '''
    #TensorBoard code
    
    writer = SummaryWriter('runs/dacon')
    dacon = Dacon(dir=lab_dir, mode=args.mode, transform=transforms_train)
    dataloader = DataLoader(dacon, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    img_grid = torchvision.utils.make_grid(images)
    visualize(img_grid)
    writer.add_image('dacon_image', img_grid)
    '''
    #model = ResNet34()
    #for param in model.parameters():
    #    param.requires_grad = False
    #model._fc = nn.Linear(model._fc.in_features, NUM_CLASSES)
    if args.checkpoint is None:   
        model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=NUM_CLASSES)        
        check_epoch = 0
        train_acc_list = []
        valid_acc_list = []
        train_loss_list = []
        valid_loss_list = []
    else:
        model = EfficientNet.from_name("efficientnet-b0", num_classes=NUM_CLASSES)
        checkpoint = torch.load(args.checkpoint)
        check_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        train_acc_list = checkpoint['train_accuracy']
        valid_acc_list = checkpoint['val_accuracy']
        train_loss_list = checkpoint['train_loss']
        valid_loss_list = checkpoint['val_loss']
    model.to(device)

    if args.mode == 'train':
        dacon = Dacon(dir=DIR, mode=args.mode, transform=transforms_train)
        num_train = int(len(dacon) * 0.8)
        num_valid = len(dacon) - num_train
        trainset, validset = random_split(dacon, [num_train, num_valid])
        
        trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        validloader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_FACTOR)
        with trange(TRAINING_EPOCH, initial=check_epoch, desc='Loss : 0', leave=True) as pbar:
            for epoch in pbar:
                train_acc, train_loss = train(trainloader, model, criterion, optimizer, lr_scheduler)
                lr_scheduler.step()
                train_acc_list.append(train_acc/len(trainloader))
                train_loss_list.append(train_loss.detach().cpu().numpy())

                valid_acc, valid_loss = validation(validloader, model, criterion)
                valid_acc_list.append(valid_acc/len(validloader))
                valid_loss_list.append(valid_loss.detach().cpu().numpy())
                if (valid_loss_list[-1].item() == min(valid_loss_list).item()) or (epoch == TRAINING_EPOCH - check_epoch - 1):
                    save(model, epoch, check_epoch, optimizer, train_loss_list, valid_loss_list, train_acc_list, valid_acc_list)
        #visualize()

    elif args.mode == 'test':
        testset = Dacon(dir=DIR, mode=args.mode, transform=transforms_train)
        testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        submission(testloader, model)