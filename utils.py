import torch
import matplotlib.pyplot as plt
import multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

def save(model, epoch, check_epoch, optimizer, train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, args):
    if args.checkpoint is None:
        torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss' : train_loss_list,
                'val_loss': valid_loss_list, 
                'train_accuracy': train_acc_list,
                'val_accuracy' : valid_acc_list,
                'learning_rate' : args.lr,
                'batch_size' : args.batchsize,
                'flooding_level' : args.flooding
            }, args.save)
    else:            
        epoch += check_epoch
        torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss' : train_loss_list,
                'val_loss': valid_loss_list, 
                'train_accuracy': train_acc_list,
                'val_accuracy' : valid_acc_list,
                'learning_rate' : args.lr,
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
        ax[0].set_xlim(left=15)

        ax[1].set_title("Training Loss")
        ax[1].set_ylabel("Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].plot(range(1, len(train_loss_list)+1), train_loss_list)
        ax[1].legend(['Train'])
        ax[1].set_xlim(left=15)
        ax[1].set_ylim(top=1.5)
        print(f"LR : {learning_rate}, Batch Size : {batch_size}, 현재 Epoch : {checkpoint['epoch']}")
        print(f"Train 정확도 : {train_acc_list[-1]}, Train Loss : {train_loss_list[-1]}")
        plt.savefig(f'{args.model}_train_fig.png')

    elif args.mode == 'val':
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].set_title("Training/Valid Accuracy")
        ax[0].set_ylabel("Accuracy")
        ax[0].set_xlabel("Epoch")
        ax[0].plot(range(1, len(train_acc_list)+1), train_acc_list)
        ax[0].plot(range(1, len(val_acc_list)+1), val_acc_list)
        ax[0].legend(['Train', 'Valid'])
        ax[0].set_xlim(left=15)

        ax[1].set_title("Training/Valid Loss")
        ax[1].set_ylabel("Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].plot(range(1, len(train_loss_list)+1), train_loss_list)
        ax[1].plot(range(1, len(val_loss_list)+1), val_loss_list)
        ax[1].legend(['Train', 'Valid'])
        ax[1].set_xlim(left=15)
        ax[1].set_ylim(top=1.5)
        
        print(f"LR : {learning_rate}, Batch Size : {batch_size}, 현재 Epoch : {checkpoint['epoch']}")
        print(f"Train 정확도 : {train_acc_list[-1]}, Train Loss : {train_loss_list[-1]}")
        print(f"Val 정확도 : {val_acc_list[-1]}, Val Loss : {val_loss_list[-1]}")
        print(f"가장 높은 Val 정확도 : {max(val_acc_list)}")
        plt.savefig(f'{args.model}_val_fig.png')

def get_parameters(dataset, args):
    '''
    It takes about 5 minutes
    '''
    PARAMETERS = {
    'mean':[0, 0, 0],
    'std': [0, 0, 0]
    }
    NUM_WORKERS = multiprocessing.cpu_count()
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