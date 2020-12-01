import os, torch, timm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
#import multiprocessing

NUM_WORKERS = 4 #multiprocessing.cpu_count()

def save(model, epoch, check_epoch, optimizer, lr_scheduler, train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, args):
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
                'learning_rate' : get_learing_rate(optimizer),
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
                'learning_rate' : get_learing_rate(optimizer),
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

        ax[1].set_title("Training Loss")
        ax[1].set_ylabel("Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].plot(range(1, len(train_loss_list)+1), train_loss_list)
        ax[1].legend(['Train'])
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

        ax[1].set_title("Training/Valid Loss")
        ax[1].set_ylabel("Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].plot(range(1, len(train_loss_list)+1), train_loss_list)
        ax[1].plot(range(1, len(val_loss_list)+1), val_loss_list)
        ax[1].legend(['Train', 'Valid'])
        
        print(f"LR : {learning_rate}, Batch Size : {batch_size}, 현재 Epoch : {checkpoint['epoch']}")
        print(f"Train 정확도 : {train_acc_list[-1]}, Train Loss : {train_loss_list[-1]}")
        print(f"Val 정확도 : {val_acc_list[-1]}, Val Loss : {val_loss_list[-1]}")
        print(f"가장 높은 Val 정확도 : {max(val_acc_list)}")
        plt.savefig(f'{args.model}_{args.scheduler}_val_fig.png')

def get_parameters(dataset, args):
    PARAMETERS = {
    'mean':[0, 0, 0],
    'std': [0, 0, 0]
    }
    dataloader = DataLoader(dataset, batch_size = args.batchsize, num_workers=NUM_WORKERS)
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

def get_learing_rate(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        return lr

def train(train_loader, model, criterion, optimizer, device, args):
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
    return acc, loss

def validation(valid_loader, model, criterion, device):
    model.eval()
    acc = 0
    for images, labels in valid_loader:
        with torch.no_grad():
            output = model(images.to(device))
            loss = criterion(output, labels.to(device))
            acc += (output.argmax(1)==labels.to(device)).float().mean()
    return acc, loss

def submission(test_loader, model, device):
    model.eval()
    landmark_id = []
    conf = []
    for images in tqdm(test_loader):
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

class Ensemble_weighted_sum(nn.Module):
    '''
    Weighted Sum per Class
    '''
    def __init__(self, model_list, num_classes):
        super(Ensemble_weighted_sum, self).__init__()
        for idx, model in enumerate(model_list):
            setattr(self, f'model_{idx}', model)
            for param in getattr(self, f'model_{idx}').parameters():
                param.requires_grad=False
        self.numofmodel = len(model_list)
        #Create new classifier
        self.weight = nn.Parameter(torch.randn((len(model_list), num_classes), requires_grad=True)) 

    def forward(self, x):
        output = []
        for idx in range(self.numofmodel):
            output.append(getattr(self, f'model_{idx}')(x))

        output = torch.stack(output, dim=1)
        output = output * nn.Softmax(dim=0)(self.weight)
        output = torch.sum(output, dim=1)
        return output

class Ensemble_sum(nn.Module):
    '''
    Sum of Output
    '''
    def __init__(self, model_list, num_classes):
        super(Ensemble_sum, self).__init__()
        for idx, model in enumerate(model_list):
            setattr(self, f'model_{idx}', model)
            for param in getattr(self, f'model_{idx}').parameters():
                param.requires_grad=False
        self.numofmodel = len(model_list)

    def forward(self, x):
        output = []
        for idx in range(self.numofmodel):
            output.append(getattr(self, f'model_{idx}')(x))
        return sum(output)

def load_model(args):
    #Ensemble model
    if args.model in ['sum', 'weighted_sum']:
        path0 = os.getcwd()+'/resnext.pt'
        path1 = os.getcwd()+'/b1_cos.pt'
        path2 = os.getcwd()+'/b5.pt'
        path3 = os.getcwd()+'/vit.pt'
        model_0 = timm.create_model('skresnext50_32x4d', pretrained=False, num_classes=args.NUM_CLASSES)
        model_0.load_state_dict(torch.load(path0)['model_state_dict'])
        model_1 = timm.create_model("efficientnet_b1", pretrained=False, num_classes=args.NUM_CLASSES)
        model_1.load_state_dict(torch.load(path1)['model_state_dict'])
        model_2 = timm.create_model("efficientnet_b5", pretrained=False, num_classes=args.NUM_CLASSES)
        model_2.load_state_dict(torch.load(path2)['model_state_dict'])
        model_3 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=args.NUM_CLASSES)
        model_3.load_state_dict(torch.load(path3)['model_state_dict'])     
        model = globals()[f'Ensemble_{args.model}']([model_0, model_1, model_2, model_3], num_classes=args.NUM_CLASSES) 
    #One model
    elif args.model == 'vit_base':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=args.NUM_CLASSES)
    elif args.model == 'vit_base_hybrid':
        model = timm.create_model('vit_base_resnet26d_224', pretrained=True, num_classes=args.NUM_CLASSES)
    elif args.model == 'vit_large':
        model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=args.NUM_CLASSES)
    elif args.model == 'skresnext50':
        model = timm.create_model('skresnext50_32x4d', pretrained=True, num_classes=args.NUM_CLASSES)
    else:
        model = timm.create_model(f"efficientnet_{args.model}", pretrained=True, num_classes=args.NUM_CLASSES)        
    return model